"""Abstract result sink for the weekly Spyre test suite.

This module holds only the ABC. The two concrete implementations live in the
``sink`` package, and callers build one through ``sink.sink_factory.create_sink``
rather than importing them directly:

- ``sink.csv_sink.CsvResultSink`` — write-only, for runs with no database access.
  Writes one run's rows to a new file and never reads one back.
- ``sink.clickhouse_sink.ClickHouseResultSink`` — inserts rows into ClickHouse;
  rows are buffered in memory then flushed in a single bulk INSERT on ``close()``.

Keeping the ABC here, apart from its subclasses, is what lets callers type against
``ResultSink`` without dragging in a storage backend. Together with
``table_schema`` holding the column list, it is also what makes ``--write-to-csv``
genuinely runnable on a host with no ``clickhouse_connect`` installed: only
``clickhouse_sink`` reaches the driver.

A sink is write-only: every row handed to ``add_entry`` is recorded. Deciding
*which* models to evaluate happens upstream, in ``model_prefilter``, so a row
reaching here is one the caller already decided to record. Filtering at write
time would discard rows the caller chose to write, leaving a run with fewer rows
than the models it handled and no accounting for the difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date


def _require_non_empty(value: str, field_name: str) -> str:
    stripped: str = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


class ResultSink(ABC):
    """Abstract destination for weekly-test result rows.

    Implementations must be usable as a context manager; ``__exit__`` should
    release any external resources (CSV file handle, DB client).

    Closing is the owner's job, and only the owner's. A sink is closed exactly
    once, by whoever constructed it — a helper that is *handed* a sink must not
    wrap it in ``with``, because ``close()`` is not idempotent in the way that
    would need: the CSV sink closes its file handle, so a later ``add_entry``
    raises ``ValueError: I/O operation on closed file``. ``weekly_test.main``
    owns the sink for a whole run and passes it down to the pre-filter, so this
    is a live constraint rather than a hypothetical one. Use ``flush()`` for
    intermediate durability instead.
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
        curated: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        """Storage-specific write of one row's normalized fields.

        Called by ``add_entry`` once *model_name* has been validated. Subclasses
        must not deduplicate or drop rows here: a run's row count is meant to
        match the models it handled.
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
        curated: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        """Persist one row. Always writes; rejects an empty *model_name*."""
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
            curated=curated,
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

    @abstractmethod
    def get_models_at_snapshot_date(self, *, snapshot_date: date) -> set[str]:
        """Read-back hook: return the model names already recorded for a date.

        Enumerates the row keys the sink has for *snapshot_date* so callers
        can build a skip-set on resume without needing to know the storage
        backend. The returned set contains no duplicates and no empty names.
        """

    def close(self) -> None:
        """Release resources. Default is a no-op; subclasses override."""
