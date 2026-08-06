"""Record the verdict rows for models the pre-filter rejected.

Split from ``model_prefilter`` so that module stays free of database imports:
this one touches a ``ResultSink``, that one is pure.

Every ``SkippedModel`` in ``PrefilterResult.skipped`` represents a terminal
verdict — no adapter, too large, MoE — and gets exactly one row written here.
The field mapping mirrors the ``add_entry`` signature on ``ResultSink``.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-cycle-free typing only
    from tests.spyre.weekly_generation.model_prefilter import SkippedModel
    from tests.spyre.weekly_generation.sink.result_sink import ResultSink


def write_skipped_rows(
    sink: ResultSink,
    skipped: list[SkippedModel],
    *,
    snapshot_date: date,
    verbose: bool = True,
) -> int:
    """Write one terminal row per entry in *skipped*. Returns rows written.

    Takes ``PrefilterResult.skipped``; every entry there is a terminal verdict and
    gets a row.
    """
    written = 0
    for item in skipped:
        row = item.row
        model_id = str(row["model_id"])
        sink.add_entry(
            model_name=model_id,
            config_class=str(row.get("config_class") or ""),
            adapter_name="",
            added_date=None,
            snapshot_date=snapshot_date,
            verified_on_cpu=False,
            verified_on_gpu=False,
            verified_on_spyre=False,
            curated=bool(row["curated"]),
            num_downloads=int(row.get("downloads") or 0),
            family=str(row.get("model_type") or ""),
            architecture=str(row.get("architectures") or ""),
            parameters_number=int(row.get("parameters") or 0),
            failure_category=item.failure_category,
            error=None,
        )
        written += 1
        if verbose:
            print(
                f"    skip-row: '{model_id}' → {item.failure_category} "
                f"({item.reason})"
            )
    return written
