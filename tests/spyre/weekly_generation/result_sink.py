"""Abstract result sink for the weekly Spyre test suite.

Two implementations:
- ``CsvResultSink`` — write-only, for runs with no database access. Writes one
  run's rows to a new file and never reads one back, so the skip rule below does
  not apply to it: it has no history to consult and reports nothing as blocking.
- ``ClickHouseResultSink`` — inserts rows into ClickHouse; the skip guard runs
  a single bulk SELECT on construction so all per-row checks are O(1), and rows
  are buffered in memory then flushed in a single bulk INSERT on ``close()``.

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

``should_insert_row`` is how the rule gets consulted, and the producers call it
directly while building the model list. ``add_entry`` does NOT re-check it: the
decision is made once, upstream, so every later write records an outcome already
decided. Re-checking at write time would discard rows the caller chose to write —
leaving a run with fewer rows than the models it handled, and no accounting for
the difference.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import date, timedelta
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

    def __init__(self, today: date | None = None) -> None:
        """Store the reference *today* used by the skip-window rule.

        Subclasses must call ``super().__init__(today=today)`` before touching
        anything that depends on ``self._today``.
        """
        self._today = today or date.today()

    @abstractmethod
    def should_insert_row(self, model_name: str) -> bool:
        """Return True when *model_name* is due for a run.

        False only when a prior row blocks it, which requires BOTH:

        * its ``snapshot_date`` is within the skip window
          (``today - snapshot_date < _SKIP_WINDOW_DAYS``), AND
        * its ``failure_category`` is NOT ``hardware_exception`` — hardware
          failures are treated as transient and always re-run.
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
    ) -> None:
        """Persist one row. Always writes.

        The skip-window rule is applied upstream, when the model list is built
        (see the module docstring), so a row reaching here is one the caller
        decided to record.
        """
        model_name = _require_non_empty(model_name, "model_name")
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

    Nothing is read back and the file must not already exist, so there is no prior
    history to consult: ``should_insert_row`` is unconditionally True.
    """

    def __init__(self, path: Path, today: date | None = None) -> None:
        """Open *path* for writing. Raises if it already exists and is non-empty."""
        super().__init__(today=today)
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

    def should_insert_row(self, model_name: str) -> bool:
        """Always True — a fresh output file has no prior rows to block on."""
        _require_non_empty(model_name, "model_name")
        return True

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
        self, embedding_generative: EmbeddingGenerativeMode, today: date | None = None
    ) -> None:
        super().__init__(today=today)
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
        # re-run. Populated once here; used by should_insert_row()
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
        semantics of ``should_insert_row``. Rows with
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

    def should_insert_row(self, model_name: str) -> bool:
        """Membership test against the skip set pre-fetched in ``__init__``.

        O(1) with no network I/O — the single bulk SELECT there already resolved
        the skip-window rule for every model that currently blocks a re-run.
        """
        key: str = _require_non_empty(model_name, "model_name")
        return key not in self._skip_model_names

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
