"""ClickHouse result sink: the real destination for weekly-scan rows.

The only module in the weekly pipeline that reaches ``clickhouse_connect`` (via
``clickhouse_db``), which is why it lives apart from the ABC in ``result_sink``:
``--write-to-csv`` runs must reach that ABC, and the CSV sink, without the driver.
Note the two imports below are deliberately split — the client and credentials
come from ``clickhouse_db``, the table shape from the dependency-free
``table_schema``, so a schema consumer never drags in the driver.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import get_client, table_exists
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION as _HARDWARE_EXCEPTION_CATEGORY,
)
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.sink.result_sink import ResultSink
from tests.spyre.weekly_generation.table_schema import (
    DATABASE,
    EMBEDDING_CREATE_TABLE_SQL,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_CREATE_TABLE_SQL,
    GENERATIVE_TABLE_NAME,
    TABLE_COLUMNS,
)


class ClickHouseResultSink(ResultSink):
    """Insert rows into ClickHouse.

    On construction the table is created if it does not already exist.

    Rows are accumulated in ``_pending`` and flushed to ClickHouse in one bulk
    INSERT on ``close()`` (or ``__exit__``), so N rows cost one round-trip rather
    than N. ``flush()`` forces that write early, which the weekly driver uses at
    batch boundaries so a crash loses at most one batch.
    """

    def __init__(self, model_type: ModelType) -> None:
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

        # Rows waiting to be flushed; each entry is a list matching TABLE_COLUMNS order.
        self._pending: list[list[Any]] = []

    def fetch_hw_failure_models(self, snapshot_date: date) -> set[str]:
        """Model names whose row on *snapshot_date* is a ``hardware_exception``.

        Groundwork for recovery runs: when a scan aborts because the accelerator
        went away (see ``weekly_test.HardwareExceptionAbortError``), the affected
        models are the only ones worth re-testing — the failure says nothing about
        the checkpoint, so every other verdict from that run still stands.

        Not called yet. Nothing upstream exposes a recovery run, so this is
        reachable only by hand until that is wired up.
        """
        result = self._client.query(
            "SELECT DISTINCT model_name "
            "FROM {db:Identifier}.{tbl:Identifier} "
            "WHERE snapshot_date = {snapshot_date:Date} "
            "AND (failure_category = {hw:String})",
            parameters={
                "db": DATABASE,
                "tbl": self._table_name,
                "snapshot_date": snapshot_date,
                "hw": _HARDWARE_EXCEPTION_CATEGORY,
            },
        )
        return {row[0] for row in result.result_rows}

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
        curated: bool,
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
                curated,
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
