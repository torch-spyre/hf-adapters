from __future__ import annotations

from datetime import date, timedelta
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
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.result_sink import (
    _SKIP_WINDOW_DAYS,
    ResultSink,
    _require_non_empty,
)


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
        self, model_type: ModelType, today: date | None = None
    ) -> None:
        super().__init__(today=today)
        self._model_type = model_type
        if model_type is ModelType.EMBEDDING:
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
