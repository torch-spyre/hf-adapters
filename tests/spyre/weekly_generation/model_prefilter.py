"""Remove models that are too large to hand to a Spyre worker.

Running this **upstream of sharding** keeps known oversized checkpoints from
occupying shard slots when they cannot fit on Spyre. Adapter resolution and MoE
classification happen in the worker so every supported and unsupported model
contributes to the runtime failure breakdown.

``weekly_test --fetch`` calls this too, for manual runs with no shard file, so
both entry points apply the same parameter limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from tests.spyre.weekly_generation import model_fetcher
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
)
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.sink.result_sink import ResultSink
from utils.utilities import concat_and_dedup_dicts


@dataclass(frozen=True)
class SkippedModel:
    """A model rejected for a terminal reason, with the row to record for it."""

    row: dict
    failure_category: str
    reason: str
    """Human-readable detail for the log line; never persisted."""


@dataclass
class PrefilterResult:
    """Two-way partition of the fetched models: work to do, and verdicts to record.

    Every fetched model lands in exactly one of the two lists, which is what lets
    ``counts`` reconcile against the input length.
    """

    keep: list[dict] = field(default_factory=list)
    skipped: list[SkippedModel] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """Per-category tallies, for one-line run summaries."""
        tally: dict[str, int] = {
            "keep": len(self.keep),
        }
        for item in self.skipped:
            tally[item.failure_category] = tally.get(item.failure_category, 0) + 1
        return tally


def _parameter_count(row: dict) -> int | None:
    """Parameter count for *row*, or None when the fetcher could not size it.

    Mirrors ``weekly_test``'s original guard, which tested
    ``params not in (None, "")`` before coercing, so an unsizable row is neither
    treated as zero-parameter nor assumed oversized — it goes to a worker and is
    judged by whether it actually loads.

    Note there is no worker-side size check to fall back on. A comment in
    ``weekly_test`` used to claim ``_process_batch`` kept one "as a defensive
    backstop for rows where parameters were unknown at fetch time"; no such check
    existed. An oversized model the fetcher could not size therefore surfaces as
    cpu_load_failed or a worker timeout rather than model_too_large — which is
    what happened before this filter moved upstream, too.
    """
    params = row.get("parameters")
    if params in (None, ""):
        return None
    try:
        return int(params)
    except (TypeError, ValueError):
        return None


def prefilter_models(
    models: list[dict],
    max_params: int,
) -> PrefilterResult:
    """Partition *models* into work to do and verdicts to record.

    Pure: decides, and writes nothing. Recording the terminal verdicts is
    ``write_skipped_rows``' job, so this needs no sink and its tests need no
    storage backend.

    Args:
        models: one dict per model as fetched from the HuggingFace Hub by
            ``build_catalog``, including ``model_id``, ``downloads``,
            ``parameters``, ``is_moe``, ``config_class``, ``model_type``, and
            ``architectures``. Callers should ``pop("model_info")`` first — the
            returned lists hold the same dict objects, and that field is not
            JSON-serializable.
        max_params: parameter ceiling above which a model cannot be brought up
            on Spyre.

    Returns:
        A ``PrefilterResult``. ``keep`` preserves the input order, which the
        tier router and shard chunker both depend on being downloads-descending.
    """
    result = PrefilterResult()

    for row in models:
        params = _parameter_count(row)
        if params is not None and params > max_params:
            result.skipped.append(
                SkippedModel(
                    row=row,
                    failure_category=FAILURE_CATEGORY_MODEL_TOO_LARGE,
                    reason=(f"{params:,} parameters exceeds the {max_params:,} limit"),
                )
            )
            continue

        result.keep.append(row)

    return result


def fetch_and_filter(
    model_type: ModelType,
    snapshot_date: date,
    top_k: int,
    sink: ResultSink,
    max_params: int,
) -> list[dict]:
    """Fetch *model_type*'s top-*top_k* catalog, filter it, record the verdicts.

    The one entry point both producers share, which is what keeps "worth handing
    to a Spyre worker" a single definition: ``generate_weekly_shards`` calls it
    before chunking (see the module docstring for why order matters there), and
    ``weekly_test --fetch`` calls it instead of reading a shard file.

    Terminal verdicts are written to *sink* as they are decided, so a model
    dropped here still gets its row and is not silently absent from the run's
    output.

    Does NOT close *sink*: the caller constructed it and keeps writing
    evaluation results to it afterwards. ``weekly_test.main`` in particular
    hands in the same sink it uses for the rest of the run, and closing it here
    left that path writing to a closed file.

    Returns:
        The models to evaluate, in the fetched (downloads-descending) order that
        the tier router and shard chunker both rely on.
    """
    # Deferred so that importing this module — and running prefilter_models,
    # which is pure — needs neither skip_writer nor anything it pulls in.
    from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

    # obtain the top-k models from hugging face
    fetched_models: list[dict] = model_fetcher.fetch(model_type=model_type, top_k=top_k)

    # obtain the curated models
    curated_models: list[dict] = model_fetcher.load_curated(model_type=model_type)

    # concatenate and dedup while giving precedence to the curated
    models = concat_and_dedup_dicts(first=curated_models, second=fetched_models)

    # Filter checkpoints known to exceed Spyre's parameter limit.
    result: PrefilterResult = prefilter_models(models, max_params=max_params)

    # write the skipped models to the sink; these won't be tested
    written: int = write_skipped_rows(
        sink, result.skipped, snapshot_date=snapshot_date, verbose=False
    )

    print(
        f"{model_type}: {len(models)} collected -> {len(result.keep)} to evaluate "
        f"({written} terminal row(s) written for skipped models) {result.counts}"
    )
    return result.keep
