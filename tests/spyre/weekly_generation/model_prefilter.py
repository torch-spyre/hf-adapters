"""Decide which fetched models are worth handing to a Spyre worker.

Four checks, applied in the order ``weekly_test.main`` used to apply them
in-process. Three of them are terminal properties of the checkpoint itself
(no adapter, too large, mixture-of-experts) and produce a row recording that
verdict; the fourth is the skip window, which produces no row because one
already exists.

Running this **upstream of sharding** is the point. ``generate_weekly_shards``
chunks a downloads-ordered list into fixed-size shards, and filtered-out models
cluster heavily — config-class families cluster by download count, so a single
unsupported family can hollow out one shard while leaving the next untouched.
Filtering after chunking left shards with wildly different amounts of real work:
some CI jobs finished in minutes, others ran for hours. Filtering first means
shard size maps to evaluations.

``weekly_test --fetch`` calls this too, for manual runs with no shard file, so
both entry points share one definition of "worth handing to a Spyre worker".

Deliberately free of database imports: the skip-window decision arrives as an
injected ``IsDueForScan`` callable rather than a sink, so this module (and its
tests) import cleanly with no ``clickhouse_connect`` installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from tests.spyre.weekly_generation import model_fetcher
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_MOE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
)
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.result_sink import ResultSink

IsDueForScan = Callable[[str], bool]
"""``model_id -> bool``: False when a recent row already covers this model.

In practice always ``sink.should_insert_row``. Taken as a callable rather than the
sink itself because one bit per model is the entire dependency — a sink parameter
would also type this module as able to call ``add_entry``/``flush``/``close``,
none of which it should touch — and because it keeps ``model_prefilter`` free of
database imports, so it and its unit tests run with no ``clickhouse_connect``
installed.
"""


@dataclass(frozen=True)
class SkippedModel:
    """A model rejected for a terminal reason, with the row to record for it."""

    row: dict
    failure_category: str
    reason: str
    """Human-readable detail for the log line; never persisted."""


@dataclass
class PrefilterResult:
    """Three-way partition of the fetched models.

    ``window_skipped`` is kept separate from ``skipped`` on purpose: those models
    already have a recent row, so writing another would either duplicate it or
    be silently swallowed by the sink's own guard. Only ``skipped`` should be
    handed to ``write_skipped_rows``.
    """

    keep: list[dict] = field(default_factory=list)
    skipped: list[SkippedModel] = field(default_factory=list)
    window_skipped: list[dict] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """Per-category tallies, for one-line run summaries."""
        tally: dict[str, int] = {
            "keep": len(self.keep),
            "window_skipped": len(self.window_skipped),
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
    sink: ResultSink,
    max_params: int,
) -> PrefilterResult:
    """Partition *models* into work to do, verdicts to record, and models to skip.

    Args:
        models: one dict per model as fetched from the HuggingFace Hub by
            ``build_catalog``, keyed as the catalog CSV header is (``model_id``,
            ``downloads``, ``parameters``, ``is_supported``, ``is_moe``,
            ``config_class``, ``model_type``, ``architectures``). Callers should
            ``pop("model_info")`` first — the returned lists hold the same dict
            objects, and that field is not JSON-serializable.
        sink: a sink to use for checking if row should be inserted.
        max_params: parameter ceiling above which a model cannot be brought up
            on Spyre.

    Returns:
        A ``PrefilterResult``. ``keep`` preserves the input order, which the
        tier router and shard chunker both depend on being downloads-descending.
    """
    result = PrefilterResult()

    for row in models:
        model_id = str(row["model_id"])

        # Checked first: a recent row already exists, so this model needs
        # neither evaluation nor a new row. Must precede the terminal checks —
        # otherwise a model that is both unsupported and recently recorded would
        # get a duplicate not-implemented-adapter row every run.
        if not sink.should_insert_row(model_id):
            result.window_skipped.append(row)
            continue

        # No adapter registered for this config class — the same terminal
        # decision resolve_adapter_module_for_test would reach in the worker,
        # reached without spawning one. `is False` rather than falsy: a missing
        # key or None means the fetcher could not determine the config class,
        # which is not the same as knowing it is unsupported.
        if row.get("is_supported") is False:
            result.skipped.append(
                SkippedModel(
                    row=row,
                    failure_category=FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
                    reason=f"no adapter for config_class={row.get('config_class')!r}",
                )
            )
            continue

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

        # MoE models aren't supported on Spyre yet. is_moe is precomputed at
        # fetch time (utils/hf_model_catalog.py) so it survives the JSON
        # round-trip through the model-list file.
        if row.get("is_moe"):
            result.skipped.append(
                SkippedModel(
                    row=row,
                    failure_category=FAILURE_CATEGORY_MOE,
                    reason="MoE model",
                )
            )
            continue

        result.keep.append(row)

    return result


def _prefilter_for_mode(
    models: list[dict],
    model_type: ModelType,
    snapshot_date: date,
    sink: ResultSink,
    max_params: int
) -> PrefilterResult:
    """Apply the four pre-filters to *models* and record the terminal verdicts.

    See the module docstring for why this runs before chunking. The sink is built
    per mode because it binds one table (or one file) per instance.

    With *write_to_csv* the verdicts go to a new CSV instead of ClickHouse, which
    needs no credentials. That sink is write-only, so it reports nothing as
    already-scanned and the emitted shards include every model that clears the
    other three filters.
    """
    # Imported here, not at module scope, so --write-to-csv works on a host with
    # no clickhouse_connect installed (result_sink pulls it in transitively).
    from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

    with sink:
        result = prefilter_models(models, sink=sink, max_params=max_params)
        written = write_skipped_rows(sink, result.skipped, snapshot_date=snapshot_date)

    print(
        f"{model_type}: {len(models)} fetched -> {len(result.keep)} to evaluate "
        f"({len(result.window_skipped)} already scanned within the skip window, "
        f"{written} terminal row(s) written) {result.counts}"
    )
    return result


def fetch_and_filter(model_type: ModelType, snapshot_date, top_k: int, sink: ResultSink, max_params: int) -> list[dict]:
    # fetch the models
    models: list[dict] = model_fetcher.fetch(
        model_type=model_type,
        top_k=top_k)

    # Filter BEFORE routing and chunking.
    return _prefilter_for_mode(
        models=models,
        model_type=model_type,
        snapshot_date=snapshot_date,
        sink=sink,
        max_params=max_params).keep
