"""ClickHouse result sink: the real destination for weekly-scan rows.

Writes schema-v2 capability rows only (see ``capability_write``); it never creates or alters a
table, so the login it runs as needs INSERT and SELECT on the v2 database and nothing more.
The ingest library is imported inside the methods, so importing this module needs no driver,
and the CSV path never reaches it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION as _HARDWARE_EXCEPTION_CATEGORY,
)
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.sink import capability_write
from tests.spyre.weekly_generation.sink.result_sink import ResultSink
from tests.spyre.weekly_generation.table_schema import TABLE_COLUMNS


class ClickHouseResultSink(ResultSink):
    """Insert rows into the v2 capability tables.

    Rows are accumulated in ``_pending`` and flushed in one bulk INSERT on ``close()`` (or
    ``__exit__``), so N rows cost one round-trip rather than N. ``flush()`` forces that write
    early, which the weekly driver uses at batch boundaries so a crash loses at most one batch.
    A failed flush raises and keeps the rows buffered for the next attempt.
    """

    def __init__(
        self, model_type: ModelType, model_list_file: str | None = None
    ) -> None:
        from spyre_clickhouse_ingest import get_client, tables_present, target_database
        from spyre_clickhouse_ingest.schema import CAPABILITIES, CAPABILITY_RUNS

        self._model_type = model_type
        # Names this process's slice of a sharded scan: the dedup scope.
        self._model_list_file = model_list_file
        self._db = target_database()
        if not self._db:
            raise RuntimeError("CLICKHOUSE_DB_V2 is unset: no v2 database to write to.")
        self._client = get_client()
        if not tables_present(self._client, self._db, (CAPABILITIES, CAPABILITY_RUNS)):
            raise RuntimeError(
                f"{self._db} lacks the capability tables; the schema is applied by torch-spyre, not CI."
            )
        print(f"ClickHouse: writing {model_type} verdicts to {self._db}.\n")

        # Rows waiting to be flushed; each entry is a list matching TABLE_COLUMNS order.
        self._pending: list[list[Any]] = []

    def fetch_hw_failure_models(self, snapshot_date: date) -> set[str]:
        """Model names whose verdict on *snapshot_date* is a ``hardware_exception``.

        Groundwork for recovery runs: when a scan aborts because the accelerator went away
        (see ``weekly_test.HardwareExceptionAbortError``), the affected models are the only
        ones worth re-testing. Not called yet; reachable only by hand.
        """
        return capability_write.recorded_for_snapshot(
            self._client,
            self._db,
            snapshot_date.isoformat(),
            str(self._model_type),
            fail_reason=_HARDWARE_EXCEPTION_CATEGORY,
        )

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
        # None becomes '' so a passing row carries no failure text into fail_reason/props.
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
        """Write all buffered rows in one bulk INSERT.

        Safe to call repeatedly; a no-op when the buffer is empty. The driver loop calls this
        after every batch so a hard parent crash loses at most one batch of rows.
        """
        if not self._pending:
            return
        print(f"ClickHouse: flushing {len(self._pending)} buffered row(s)…", flush=True)
        rows = [
            {**row, "model_type": str(self._model_type)}
            for row in capability_write.rows_from_pending(self._pending, TABLE_COLUMNS)
        ]
        capability_write.write(
            rows,
            self._client,
            self._db,
            model_list_file=self._model_list_file,
        )
        self._pending.clear()

    def get_models_at_snapshot_date(self, *, snapshot_date: date) -> set[str]:
        """Return every distinct model name recorded for *snapshot_date* and this model type.

        Lets the caller build a skip-set without knowing the storage backend. Empty names
        cannot appear (the base class rejects them at ``add_entry`` time).
        """
        return capability_write.recorded_for_snapshot(
            self._client, self._db, snapshot_date.isoformat(), str(self._model_type)
        )

    def close(self) -> None:
        """Flush any remaining buffered rows on shutdown."""
        self.flush()
