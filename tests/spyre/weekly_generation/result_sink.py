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

from abc import ABC, abstractmethod
from datetime import date

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

