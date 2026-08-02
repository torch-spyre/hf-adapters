"""Abstract result sink for the weekly Spyre test suite.

Two implementations:
- ``ClickHouseResultSink`` — inserts rows into ClickHouse; the skip guard runs
  a single bulk SELECT on construction so all per-row checks are O(1), and rows
  are buffered in memory then flushed in a single bulk INSERT on ``close()``.
- ``CsvResultSink`` — write-only, for runs with no database access. Writes one
  run's rows to a new file and never reads one back, so the skip rule below does
  not apply to it: it has no history to consult and reports nothing as blocking.

Skip rule (ClickHouse): insert a row for *model_name* when either

    * no prior row for *model_name* exists, OR
    * the most-recent prior row has ``failure_category == 'hardware_exception'``
      (accelerator problem, worth retrying now) OR the snapshot is older than
      ``_SKIP_WINDOW_DAYS`` days (long enough since last run).

Rows with `hardware_exception` are always re-run because the accelerator's
availability is a transient property — a failure yesterday says nothing about
today. Everything else (verified success, non-implemented adapter, quantized
model, moe, cpu load/generate failure, …) is treated as terminal for the
window.

``add_entry`` applies that rule automatically unless the sink is constructed with
``dedup_guard=False``, which every stage of the weekly pipeline does. The rule is
evaluated once, upstream, while the model list is built; from then on a write is
the recorded outcome of a decision already taken, and re-checking could only
discard a row the caller deliberately chose to write — leaving the run with fewer
rows than the models it handled and no accounting for the difference.

``should_insert_row`` remains callable with the guard off. That is how the
producers consult the rule in the first place.

``clickhouse_db.import_csv`` is the one caller that wants the guard: it reads an
arbitrary CSV with no upstream filter and derives its ``(inserted, skipped)``
return from the guard's verdict. Hence the default is True.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import (
    DATABASE,
    EMBEDDING_CREATE_TABLE_SQL,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_CREATE_TABLE_SQL,
    GENERATIVE_TABLE_NAME,
    TABLE_COLUMNS,
    get_client,
    table_exists,
)
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION as _HARDWARE_EXCEPTION_CATEGORY,
)

# Constant value
_SKIP_WINDOW_DAYS: int = 10


def _within_skip_window(existing_snapshot: date, today: date) -> bool:
    return (today - existing_snapshot).days < _SKIP_WINDOW_DAYS


_SNAPSHOT_DATE_FORMATS = (
    "%Y-%m-%d",  # ISO 8601 — primary format written by this module
    "%d/%m/%Y",  # DD/MM/YYYY
    "%m/%d/%Y",  # MM/DD/YYYY
    "%Y/%m/%d",  # YYYY/MM/DD
)


def _coerce_snapshot(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        for fmt in _SNAPSHOT_DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _require_non_empty(value: str, field_name: str) -> str:
    stripped: str = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


class ResultSink(ABC):
    """Abstract destination for weekly-test result rows.

    Implementations must be usable as a context manager; ``__exit__`` should
    release any external resources (CSV file handle, DB client).
    """

    _today: date
    _dedup_guard: bool

    def __init__(self, today: date | None = None, *, dedup_guard: bool = True) -> None:
        """Store the reference *today* used by the skip-window guard.

        Subclasses must call ``super().__init__(today=today)`` before touching
        anything that depends on ``self._today``.

        *dedup_guard* controls whether ``add_entry`` consults
        ``should_insert_row`` before every write. It defaults to True, which is
        what ``clickhouse_db.import_csv`` relies on — that path has no upstream
        filter, and derives its ``(inserted, skipped)`` return from the guard's
        verdict.

        The weekly pipeline passes False throughout. The skip-window decision is
        made once while the model list is built, so every later write records the
        outcome of a decision already taken; a second check could only reject a
        row the caller chose to write, silently and with no accounting. In
        ``weekly_test`` specifically the guard would consult a skip set
        snapshotted when that job started — newer than the producer's — so a row
        written in between by a concurrent run would suppress a result the job was
        asked to produce.

        Keyword-only, and positioned after *today*, because
        ``clickhouse_db.py`` constructs ``ClickHouseResultSink(mode)``
        positionally — a new positional parameter would misbind there.
        """
        self._today = today or date.today()
        self._dedup_guard = dedup_guard

    @abstractmethod
    def get_recent_blocking_entries(self, model_name: str) -> list[dict[str, Any]]:
        """Return prior rows for *model_name* that block a new insert.

        A row blocks re-insertion when BOTH:

        * its ``snapshot_date`` is within the skip window
          (``today - snapshot_date < _SKIP_WINDOW_DAYS``), AND
        * its ``failure_category`` is NOT ``hardware_exception`` — hardware
          failures are treated as transient and always re-run.

        Sorted by ``snapshot_date`` descending. Each row is a dict keyed by
        column name. Empty list when no blocking prior entry exists — the
        caller can treat empty as "insert away" without inspecting the rows.
        """

    @abstractmethod
    def get_all_models(self) -> list[dict[str, Any]]:
        """Return one row per known ``model_name``, reflecting its most recent snapshot.

        When a model appears in multiple rows (one per weekly run), only the row
        with the greatest ``snapshot_date`` is returned. The result is a flat
        list of dicts keyed by column name (same keys as ``TABLE_COLUMNS``), one
        dict per distinct model, in no guaranteed order.
        """

    @abstractmethod
    def _insert_entry(
        self,
        *,
        model_name: str,
        config_class: str,
        adapter_name: str,
        added_date: date | None,
        snapshot_date: date,
        verified_on_cpu: bool,
        verified_on_gpu: bool,
        verified_on_spyre: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        """Storage-specific write of one row's normalized fields.

        Called by ``add_entry`` after the skip guard has passed. Subclasses must
        not perform any deduplication here — that is the responsibility of
        ``should_insert_row``.
        """

    def add_entry(
        self,
        *,
        model_name: str,
        config_class: str,
        adapter_name: str,
        added_date: date | None,
        snapshot_date: date,
        verified_on_cpu: bool,
        verified_on_gpu: bool,
        verified_on_spyre: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> bool:
        """Persist one row when the skip guard allows it.

        Returns True if the row was written, False if ``should_insert_row``
        rejected it. Idempotent to call for every row in the driver loop.

        When the sink was constructed with ``dedup_guard=False`` the check is
        skipped entirely and every call writes, always returning True.
        """
        model_name = _require_non_empty(model_name, "model_name")
        if self._dedup_guard and not self.should_insert_row(model_name):
            return False
        self._insert_entry(
            model_name=model_name,
            config_class=config_class,
            adapter_name=adapter_name,
            added_date=added_date,
            snapshot_date=snapshot_date,
            verified_on_cpu=verified_on_cpu,
            verified_on_gpu=verified_on_gpu,
            verified_on_spyre=verified_on_spyre,
            num_downloads=num_downloads,
            family=family,
            architecture=architecture,
            parameters_number=parameters_number,
            failure_category=failure_category,
            error=error,
        )
        return True

    def should_insert_row(self, model_name: str) -> bool:
        """Return True when *model_name* should be re-run.

        See module docstring for the full rule. In short: absent OR the most
        recent prior row is a ``hardware_exception`` (retry) OR the prior
        snapshot has aged past the skip window.
        """
        model_name: str = _require_non_empty(model_name, "model_name")
        return not self.get_recent_blocking_entries(model_name)

    def __enter__(self) -> ResultSink:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def flush(self) -> None:
        """Persist any rows buffered in memory. Default is a no-op.

        Subclasses that buffer rows (e.g. ``ClickHouseResultSink``) override
        this so callers can force a durable write at safe checkpoints — for
        example, between batches in the weekly-test driver, so a hard crash
        of the parent loses at most one batch instead of the whole run.
        """

    def close(self) -> None:
        """Release resources. Default is a no-op; subclasses override."""


class CsvResultSink(ResultSink):
    """Write rows to a fresh CSV file. Write-only, for no-database test runs.

    Nothing is read back: the file must not already exist, so there is no prior
    history to consult and every model is due for a run. That is what makes
    ``get_recent_blocking_entries`` unconditionally empty here, rather than an
    approximation of the ClickHouse behaviour.

    The alternative — treating the CSV as its own skip-window index, as an earlier
    version did — meant a run's row count could silently disagree with the models
    it evaluated, and made the CSV and ClickHouse paths behave differently for the
    same input. A single-run output file has neither problem.
    """

    def __init__(
        self, path: Path, today: date | None = None, *, dedup_guard: bool = False
    ) -> None:
        """Open *path* for writing. Raises if it already exists and is non-empty.

        *dedup_guard* defaults to False here — unlike the base class — because a
        fresh file cannot contain a blocking row, so the guard would be pure
        overhead. Passing True is accepted (``get_recent_blocking_entries``
        returns empty, so it never rejects anything) but pointless.
        """
        super().__init__(today=today, dedup_guard=dedup_guard)
        self._path: Path = path
        if path.exists() and path.stat().st_size > 0:
            raise FileExistsError(
                f"'{path}' already exists and is not empty. This sink writes a "
                f"single run's results to a new file and never reads one back; "
                f"choose a different path or remove the existing file."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(TABLE_COLUMNS))
        self._writer.writeheader()
        self._fh.flush()

    def get_recent_blocking_entries(self, model_name: str) -> list[dict[str, Any]]:
        """Always empty — a fresh output file has no prior rows to block on."""
        _require_non_empty(model_name, "model_name")
        return []

    def get_all_models(self) -> list[dict[str, Any]]:
        """Always empty — this sink does not read its file back."""
        return []

    def _insert_entry(
        self,
        *,
        model_name: str,
        config_class: str,
        adapter_name: str,
        added_date: date | None,
        snapshot_date: date,
        verified_on_cpu: bool,
        verified_on_gpu: bool,
        verified_on_spyre: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        rec: dict[str, Any] = {
            "model_name": model_name,
            "config_class": config_class,
            "adapter_name": adapter_name,
            "added_date": added_date,
            "snapshot_date": snapshot_date,
            "verified_on_cpu": verified_on_cpu,
            "verified_on_gpu": verified_on_gpu,
            "verified_on_spyre": verified_on_spyre,
            "num_downloads": num_downloads,
            "family": family,
            "architecture": architecture,
            "parameters_number": parameters_number,
            "failure_category": failure_category,
            "error": error,
        }
        self._writer.writerow(rec)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class ClickHouseResultSink(ResultSink):
    """Insert rows into ClickHouse.

    On construction the table is created if missing, and a single bulk SELECT
    pre-fetches every model name that currently blocks a re-run (see the
    module docstring for the rule) so that all subsequent
    ``should_insert_row`` calls are O(1) with no network I/O.

    Rows accepted by the skip guard are accumulated in ``_pending`` and flushed
    to ClickHouse in one bulk INSERT on ``close()`` (or ``__exit__``), so the
    total network round-trips for N rows is 2 (one SELECT, one INSERT) instead
    of 2 × N.
    """

    def __init__(
        self,
        embedding_generative: EmbeddingGenerativeMode,
        today: date | None = None,
        *,
        dedup_guard: bool = True,
    ) -> None:
        super().__init__(today=today, dedup_guard=dedup_guard)
        self._embedding_generative = embedding_generative
        if embedding_generative is EmbeddingGenerativeMode.EMBEDDING:
            self._table_name = EMBEDDING_TABLE_NAME
            create_sql = EMBEDDING_CREATE_TABLE_SQL
        else:
            self._table_name = GENERATIVE_TABLE_NAME
            create_sql = GENERATIVE_CREATE_TABLE_SQL
        self._client = get_client()
        if not table_exists(self._client, self._table_name):
            self._client.command(create_sql)
            print(f"ClickHouse: table '{self._table_name}' created.\n")
        else:
            print(f"ClickHouse: table '{self._table_name}' already exists.\n")

        # Bulk pre-fetch: model names whose most-recent-in-window row blocks a
        # re-run. Populated once here; used by get_recent_blocking_entries()
        # for O(1) per-row checks.
        self._skip_model_names: set[str] = self._fetch_blocking_names()
        print(
            f"ClickHouse: {len(self._skip_model_names)} model(s) already have a "
            f"non-hardware-exception snapshot within the last "
            f"{_SKIP_WINDOW_DAYS} days — will be skipped.\n"
        )

        # Rows waiting to be flushed; each entry is a list matching TABLE_COLUMNS order.
        self._pending: list[list[Any]] = []

    def _fetch_blocking_names(self) -> set[str]:
        """One SELECT to get all model names that block a re-run.

        A model blocks iff it has any row in the skip window whose
        ``failure_category`` is not ``hardware_exception`` — matching the
        semantics of ``get_recent_blocking_entries``. Rows with
        ``failure_category = 'hardware_exception'`` (including a lone row
        older than the window) do NOT block.
        """
        cutoff: date = self._today - timedelta(days=_SKIP_WINDOW_DAYS - 1)
        result = self._client.query(
            "SELECT DISTINCT model_name "
            "FROM {db:Identifier}.{tbl:Identifier} "
            "WHERE snapshot_date >= {cutoff:Date} "
            "AND (failure_category IS NULL OR failure_category != {hw:String})",
            parameters={
                "db": DATABASE,
                "tbl": self._table_name,
                "cutoff": cutoff,
                "hw": _HARDWARE_EXCEPTION_CATEGORY,
            },
        )
        return {row[0] for row in result.result_rows}

    def get_recent_blocking_entries(self, model_name: str) -> list[dict[str, Any]]:
        """Return a non-empty sentinel list when *model_name* is in the skip set.

        The actual row data is not needed by the caller — it only tests
        ``bool(result)`` — so we return a lightweight placeholder instead of
        re-querying ClickHouse.
        """
        key: str = _require_non_empty(model_name, "model_name")
        if key in self._skip_model_names:
            # Return a truthy non-empty list so should_insert_row returns False.
            return [{"model_name": key}]
        return []

    def get_all_models(self) -> list[dict[str, Any]]:
        columns_sql: str = ", ".join(
            f"argMax({col}, snapshot_date) AS {col}" if col != "model_name" else col
            for col in TABLE_COLUMNS
        )
        result = self._client.query(
            f"SELECT {columns_sql} "
            "FROM {db:Identifier}.{tbl:Identifier} "
            "GROUP BY model_name",
            parameters={"db": DATABASE, "tbl": self._table_name},
        )
        return [dict(zip(TABLE_COLUMNS, row)) for row in result.result_rows]

    def _insert_entry(
        self,
        *,
        model_name: str,
        config_class: str,
        adapter_name: str,
        added_date: date | None,
        snapshot_date: date,
        verified_on_cpu: bool,
        verified_on_gpu: bool,
        verified_on_spyre: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        """Buffer the row; the actual INSERT happens in ``close()``."""
        # failure_category/error are non-nullable in the live table (String/
        # LowCardinality(String) DEFAULT '') — None must become '' here, or
        # clickhouse_connect raises DataError on the bulk insert for any
        # fully-passing row (which always has both fields as None).
        self._pending.append(
            [
                model_name,
                config_class,
                adapter_name,
                added_date,
                snapshot_date,
                verified_on_cpu,
                verified_on_gpu,
                verified_on_spyre,
                num_downloads,
                family,
                architecture,
                parameters_number,
                failure_category or "",
                error or "",
            ]
        )

    def flush(self) -> None:
        """Flush all buffered rows to ClickHouse in a single bulk INSERT.

        Safe to call repeatedly; a no-op when the buffer is empty. The driver
        loop calls this after every batch so a hard parent crash loses at most
        one batch of rows rather than the entire run.
        """
        if not self._pending:
            return
        print(f"ClickHouse: flushing {len(self._pending)} buffered row(s)…", flush=True)
        self._client.insert(
            self._table_name,
            self._pending,
            column_names=list(TABLE_COLUMNS),
        )
        print("ClickHouse: bulk insert complete.")
        self._pending.clear()

    def close(self) -> None:
        """Flush any remaining buffered rows on shutdown."""
        self.flush()


class EmbeddingGenerativeMode(str, Enum):
    EMBEDDING = "embedding"
    GENERATIVE = "generative"
