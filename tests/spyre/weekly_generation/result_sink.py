"""Abstract result sink for the weekly Spyre test suite.

Two implementations:
- ``CsvResultSink`` — appends rows to a CSV file; loads existing rows once so
  the skip guard is O(1) per call.
- ``ClickHouseResultSink`` — inserts rows into ClickHouse; the skip guard runs
  a single SELECT per model.

Skip rule (both sinks):
    Skip when an existing entry for ``model_name`` has ``snapshot_date`` within
    the last 7 days AND ``verified_on_cpu`` is True. In other words: don't re-run
    a model that was successfully CPU-verified less than a week ago.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import (
    CREATE_TABLE_SQL,
    DATABASE,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_CREATE_TABLE_SQL,
    GENERATIVE_TABLE_NAME,
    TABLE_COLUMNS,
    get_client,
    insert_model_row,
    table_exists,
)

# Constant value
_SKIP_WINDOW_DAYS: int = 7


def _rec_is_cpu_verified(rec: dict[str, Any]) -> bool:
    value: Any = rec.get("verified_on_cpu", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


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

    def __init__(self, today: date | None = None) -> None:
        """Store the reference *today* used by the skip-window guard.

        Subclasses must call ``super().__init__(today=today)`` before touching
        anything that depends on ``self._today``.
        """
        self._today = today or date.today()

    @abstractmethod
    def get_recent_cpu_verified_entries(self, model_name: str) -> list[dict[str, Any]]:
        """Return prior CPU-verified rows for *model_name* within the skip window.

        Filters to entries where ``verified_on_cpu`` is True AND
        ``today - snapshot_date < _SKIP_WINDOW_DAYS``. Sorted by
        ``snapshot_date`` descending. Each row is a dict keyed by column name.
        Empty list when no such prior entries exist — the caller can treat
        empty as "no skip needed" without inspecting the rows further.
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
        """
        model_name = _require_non_empty(model_name, "model_name")
        if not self.should_insert_row(model_name):
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
        """Return False when *model_name* has a CPU-verified entry in the window.

        Callers can invoke this directly to short-circuit expensive work when
        the sink will reject the row anyway (see ``weekly_test.py``).
        """
        model_name: str = _require_non_empty(model_name, "model_name")
        return not self.get_recent_cpu_verified_entries(model_name)

    def __enter__(self) -> ResultSink:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Release resources. Default is a no-op; subclasses override."""


class CsvResultSink(ResultSink):
    """Append rows to a CSV file.

    On construction, any existing rows are read once into an in-memory index of
    ``{model_name: list[dict]}`` so lookups are O(1).
    """

    def __init__(self, path: Path, today: date | None = None) -> None:
        super().__init__(today=today)
        self._path: Path = path
        self._rows_by_model: dict[str, list[dict[str, Any]]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists: bool = path.exists() and path.stat().st_size > 0
        if file_exists:
            self._load_index()
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(TABLE_COLUMNS))
        if not file_exists:
            self._writer.writeheader()
            self._fh.flush()

    def _load_index(self) -> None:
        with open(self._path, newline="") as fh:
            reader = csv.DictReader(fh)
            for raw_row in reader:
                model_name: str = (raw_row.get("model_name") or "").strip()
                if not model_name:
                    print(
                        f"    warn: skipping CSV row with empty model_name in "
                        f"'{self._path}' (line {reader.line_num})"
                    )
                    continue
                self._rows_by_model.setdefault(model_name, []).append(dict(raw_row))
        loaded = sum(len(v) for v in self._rows_by_model.values())
        print(
            f"    index: loaded {loaded} row(s) for "
            f"{len(self._rows_by_model)} model(s) from '{self._path}'"
        )

    def get_recent_cpu_verified_entries(self, model_name: str) -> list[dict[str, Any]]:
        key: str = _require_non_empty(model_name, "model_name")
        rows: list[dict[str, Any]] = list(self._rows_by_model.get(key, ()))
        filtered: list[tuple[date, dict[str, Any]]] = []
        for row in rows:
            if not _rec_is_cpu_verified(row):
                continue
            snap: date | None = _coerce_snapshot(row.get("snapshot_date"))
            if snap is None:
                continue
            if not _within_skip_window(snap, self._today):
                continue
            filtered.append((snap, row))
        filtered.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in filtered]

    def get_all_models(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for rows in self._rows_by_model.values():
            best: dict[str, Any] = max(
                rows,
                key=lambda r: _coerce_snapshot(r.get("snapshot_date")) or date.min,
            )
            result.append(best)
        return result

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
        self._rows_by_model.setdefault(model_name, []).append(dict(rec))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class ClickHouseResultSink(ResultSink):
    """Insert rows into ClickHouse. Table is created on construction if missing."""

    def __init__(
        self, embedding_generative: EmbeddingGenerativeMode, today: date | None = None
    ) -> None:
        super().__init__(today=today)
        self._embedding_generative = embedding_generative
        if embedding_generative is EmbeddingGenerativeMode.EMBEDDING:
            self._table_name = EMBEDDING_TABLE_NAME
            create_sql = CREATE_TABLE_SQL
        else:
            self._table_name = GENERATIVE_TABLE_NAME
            create_sql = GENERATIVE_CREATE_TABLE_SQL
        self._client = get_client()
        if not table_exists(self._client, self._table_name):
            self._client.command(create_sql)
            print(f"ClickHouse: table '{self._table_name}' created.\n")
        else:
            print(f"ClickHouse: table '{self._table_name}' already exists.\n")

    def get_recent_cpu_verified_entries(self, model_name: str) -> list[dict[str, Any]]:
        key: str = _require_non_empty(model_name, "model_name")
        cutoff: date = self._today - timedelta(days=_SKIP_WINDOW_DAYS - 1)
        columns_sql: str = ", ".join(TABLE_COLUMNS)
        result = self._client.query(
            f"SELECT {columns_sql} "
            "FROM {db:Identifier}.{tbl:Identifier} "
            "WHERE model_name = {model:String} "
            "AND verified_on_cpu = 1 "
            "AND snapshot_date >= {cutoff:Date} "
            "ORDER BY snapshot_date DESC",
            parameters={
                "db": DATABASE,
                "tbl": self._table_name,
                "model": key,
                "cutoff": cutoff,
            },
        )
        return [dict(zip(TABLE_COLUMNS, row)) for row in result.result_rows]

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
        insert_model_row(
            self._client,
            table_name=self._table_name,
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


class EmbeddingGenerativeMode(str, Enum):
    EMBEDDING = "embedding"
    GENERATIVE = "generative"
