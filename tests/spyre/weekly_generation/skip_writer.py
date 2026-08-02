"""Record the verdict rows for models the pre-filter rejected.

Split from ``model_prefilter`` so that module stays free of database imports:
this one touches a ``ResultSink``, that one is pure.

The field mapping below is the one the three skip branches in
``weekly_test.main`` used before the filter moved upstream — all three wrote an
identical shape and differed only in ``failure_category``.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-cycle-free typing only
    from tests.spyre.weekly_generation.model_prefilter import SkippedModel
    from tests.spyre.weekly_generation.result_sink import ResultSink


def write_skipped_rows(
    sink: ResultSink,
    skipped: list[SkippedModel],
    *,
    snapshot_date: date,
    verbose: bool = True,
) -> int:
    """Write one terminal row per entry in *skipped*. Returns rows written.

    Pass only ``PrefilterResult.skipped`` — never ``window_skipped``, whose
    members already have a recent row and are meant to produce no write.

    The return value can be lower than ``len(skipped)`` when *sink* was built
    with ``dedup_guard=True``, since ``add_entry`` then rejects rows inside the
    skip window. Producers construct their sink with ``dedup_guard=False``
    (they have already applied that rule), so for them the two match.
    """
    written = 0
    for item in skipped:
        row = item.row
        model_id = str(row["model_id"])
        if sink.add_entry(
            model_name=model_id,
            config_class=str(row.get("config_class") or ""),
            adapter_name="",
            added_date=None,
            snapshot_date=snapshot_date,
            verified_on_cpu=False,
            verified_on_gpu=False,
            verified_on_spyre=False,
            num_downloads=int(row.get("downloads") or 0),
            family=str(row.get("model_type") or ""),
            architecture=str(row.get("architectures") or ""),
            parameters_number=int(row.get("parameters") or 0),
            failure_category=item.failure_category,
            error=None,
        ):
            written += 1
            if verbose:
                print(
                    f"    skip-row: '{model_id}' → {item.failure_category} "
                    f"({item.reason})"
                )
        elif verbose:
            print(
                f"    skip-row: '{model_id}' not written — sink's dedup guard "
                f"rejected it ({item.failure_category})"
            )
    return written
