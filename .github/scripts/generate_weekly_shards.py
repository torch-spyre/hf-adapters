#!/usr/bin/env python3
"""
Fetch the weekly top-K generative and embedding model lists once, then split
each into fixed-size shards for parallel GitHub Actions jobs.

Mirrors the pattern in generate_test_matrix.py: this script does the one-time
fetch/compute step and emits a JSON matrix via GITHUB_OUTPUT for a downstream
`strategy.matrix.include` job. Each shard is written to its own JSON file
under --output-dir; downstream jobs load their shard via
`weekly_test.py --model-list-file`.

Models are routed to one of three runner tiers by parameter count, each with
more memory (and more cards) than the last:

  - runner="x1" (spyre_pf_x1, 1 card):  parameters <  --x1-max-params
  - runner="x2" (spyre_pf_x2, 2 cards): --x1-max-params <= parameters <= --x2-max-params
  - runner="x4" (spyre_pf_x4, 4 cards): parameters >  --x2-max-params

so a model doesn't share a batch with (and inflate the memory footprint
of) much smaller ones. See push-to-clickhouse.yaml's weekly-model-scan job
for how `matrix.runner` selects the actual runs-on label.

Each mode's list is PRE-FILTERED before any chunking: models already scanned
inside the skip window, models with no adapter for their config class, models
too large for Spyre, and MoE models are all removed here, and the terminal
verdicts among those are written straight to ClickHouse (so this script needs
the CLICKHOUSE_* env vars; use --dry-run to skip both the read and the writes).

That ordering is the point. The dropped models cluster heavily — config-class
families cluster by download count — so filtering inside each worker, as
weekly_test.py used to, left surviving counts wildly uneven: some CI jobs
finished in minutes while others ran for hours. Filtering first means a shard's
size is a count of real evaluations, and shard durations become comparable.

Per-tier shard sizes do NOT need to be small: weekly_test.py already
re-chunks whatever list it's given into fresh-OS-process batches of
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS/EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS
regardless of shard size, which is what actually bounds how many models'
memory can accumulate in one process before a clean restart. A tiny shard
size buys no extra safety over a large one — it only multiplies GitHub
Actions job count, and matrices are hard-capped at 256 jobs total. Since the
sizes now apply to filtered lists, the same values yield fewer, fuller jobs
than they used to.

Usage (called by the GHA workflow):
    python .github/scripts/generate_weekly_shards.py \
        --top-k 10000 \
        --shard-size-generative 250 \
        --shard-size-embedding 500 \
        --x1-max-params 7000000000 \
        --x2-max-params 12000000000 \
        --x2-shard-size 100 \
        --x4-shard-size 50 \
        --output-dir shards
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Add the project root to the Python path so we can import from utils/ and from
# tests/spyre/weekly_generation/ (both resolve as namespace packages from here).
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    PrefilterResult,
    prefilter_models,
)
from utils.fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402
from utils.fetch_top_generative_models import fetch_top_generative_models  # noqa: E402

MODEL_TYPES = ("generative", "embedding")


def _chunk(rows: list[dict], shard_size: int) -> list[list[dict]]:
    """Split *rows* into consecutive sub-lists of length *shard_size* (the
    last shard may be shorter). Empty input yields zero shards.
    """
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


def _tier_for(row: dict, x1_max_params: int, x2_max_params: int) -> str:
    """Return "x1"/"x2"/"x4" for *row* by parameter count.

    Unknown/non-numeric parameter counts default to "x1" (the smallest,
    least risky tier) rather than being treated as large.
    """
    params = row.get("parameters")
    if not isinstance(params, (int, float)):
        return "x1"
    if params < x1_max_params:
        return "x1"
    if params <= x2_max_params:
        return "x2"
    return "x4"


def _prefilter_for_mode(
    rows: list[dict],
    mode: str,
    snapshot_date: date,
    dry_run: bool,
) -> PrefilterResult:
    """Apply the four pre-filters to *rows* and record the terminal verdicts.

    See the module docstring for why this runs before chunking. The sink is opened
    per mode because it binds one table per instance, and with
    ``dedup_guard=False`` because ``prefilter_models`` has just consulted the same
    skip set through ``should_insert_row``.
    """
    if dry_run:
        print(f"{mode}: dry run — skipping the ClickHouse skip-window check")
        result = prefilter_models(rows, should_scan=lambda _model_id: True)
        print(
            f"{mode}: {len(rows)} fetched -> {len(result.keep)} to evaluate "
            f"(no rows written) {result.counts}"
        )
        return result

    # Imported lazily so --dry-run works with no database driver installed.
    from tests.spyre.weekly_generation.result_sink import (
        ClickHouseResultSink,
        EmbeddingGenerativeMode,
    )
    from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

    with ClickHouseResultSink(
        embedding_generative=EmbeddingGenerativeMode(mode),
        today=snapshot_date,
        dedup_guard=False,
    ) as sink:
        result = prefilter_models(rows, should_scan=sink.should_insert_row)
        written = write_skipped_rows(sink, result.skipped, snapshot_date=snapshot_date)

    print(
        f"{mode}: {len(rows)} fetched -> {len(result.keep)} to evaluate "
        f"({len(result.window_skipped)} already scanned within the skip window, "
        f"{written} terminal row(s) written) {result.counts}"
    )
    return result


def generate_shards(
    top_k: int,
    shard_size_generative: int,
    shard_size_embedding: int,
    x1_max_params: int,
    x2_max_params: int,
    x2_shard_size: int,
    x4_shard_size: int,
    output_dir: Path,
    model_types: tuple[str, ...] = MODEL_TYPES,
    snapshot_date: date | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Fetch each requested mode's top-K list once, write shard JSON files,
    and return the combined matrix (list of {mode, shard_index, shard_file,
    runner} dicts).

    Each mode's list is pre-filtered (see ``_prefilter_for_mode``) before any
    chunking, so a shard's size is a count of real evaluations rather than of
    fetched candidates. Within each mode, the survivors are then split into
    three parameter-count tiers (see module docstring), each chunked at its own
    shard size and tagged with the runner ("x1"/"x2"/"x4") that handles it.

    *model_types* restricts which of MODEL_TYPES to fetch/shard — used by
    workflow_dispatch's model_type input so a manual run can scan just
    embedding models (much quicker, less resource-hungry) without the
    schedule-triggered full scan having to change.

    *dry_run* skips both the ClickHouse skip-window read and the skipped-row
    writes, so the whole fetch → filter → route → chunk path can be exercised
    without credentials. The emitted shards then include models that a real run
    would have dropped as recently-scanned.
    """
    snapshot_date = snapshot_date or date.today()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_fetchers = {
        "generative": (fetch_top_generative_models, shard_size_generative),
        "embedding": (fetch_top_embedding_models, shard_size_embedding),
    }
    fetchers = {model_type: all_fetchers[model_type] for model_type in model_types}

    matrix: list[dict] = []
    for mode, (fetch_fn, shard_size) in fetchers.items():
        rows: list[dict] = fetch_fn(limit=top_k)
        # model_info is a live huggingface_hub.ModelInfo object attached by
        # build_catalog — not JSON-serializable, and no longer needed since
        # is_moe is precomputed onto each row (see utils/hf_model_catalog.py).
        for row in rows:
            row.pop("model_info", None)

        # Filter BEFORE routing and chunking. Must stay after the pop above:
        # prefilter_models returns the same dict objects it was given, and a
        # surviving ModelInfo would break the shard JSON dump below.
        rows = _prefilter_for_mode(rows, mode, snapshot_date, dry_run).keep

        by_tier: dict[str, list[dict]] = {"x1": [], "x2": [], "x4": []}
        for row in rows:
            by_tier[_tier_for(row, x1_max_params, x2_max_params)].append(row)

        tier_shard_sizes = {"x1": shard_size, "x2": x2_shard_size, "x4": x4_shard_size}
        mode_shard_count = 0
        for runner, group_rows in by_tier.items():
            group_shard_size = tier_shard_sizes[runner]
            shards = _chunk(group_rows, group_shard_size)
            mode_shard_count += len(shards)
            print(
                f"{mode} ({runner}): {len(group_rows)} model(s), split into "
                f"{len(shards)} shard(s) of up to {group_shard_size} each"
            )
            for shard_index, shard_rows in enumerate(shards):
                shard_file = f"{mode}-{runner}-shard-{shard_index:03d}.json"
                (output_dir / shard_file).write_text(json.dumps(shard_rows))
                matrix.append(
                    {
                        "mode": mode,
                        "shard_index": shard_index,
                        "shard_file": shard_file,
                        "runner": runner,
                    }
                )
        print(f"{mode}: {len(rows)} model(s) total, {mode_shard_count} shard(s)")

    return matrix


def write_github_output(outputs: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        print("Not running in GitHub Actions. Output would be:")
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    with open(github_output, "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="Number of top models to fetch per mode (by downloads).",
    )
    parser.add_argument(
        "--shard-size-generative",
        type=int,
        default=250,
        help="Models per generative shard.",
    )
    parser.add_argument(
        "--shard-size-embedding",
        type=int,
        default=500,
        help="Models per embedding shard.",
    )
    parser.add_argument(
        "--x1-max-params",
        type=int,
        default=7_000_000_000,
        help="Models with < this many parameters stay on spyre_pf_x1 (1 card).",
    )
    parser.add_argument(
        "--x2-max-params",
        type=int,
        default=12_000_000_000,
        help=(
            "Models with parameters in [--x1-max-params, --x2-max-params] go "
            "to spyre_pf_x2 (2 cards); above this go to spyre_pf_x4 (4 cards)."
        ),
    )
    parser.add_argument(
        "--x2-shard-size",
        type=int,
        default=100,
        help=(
            "Models per x2-tier shard. Doesn't need to be small — "
            "weekly_test.py's own per-process batching already bounds memory "
            "accumulation regardless of shard size; this just controls "
            "GitHub Actions job count (matrices cap at 256 jobs total)."
        ),
    )
    parser.add_argument(
        "--x4-shard-size",
        type=int,
        default=50,
        help="Models per x4-tier shard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shards"),
        help="Directory to write shard JSON files into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the ClickHouse skip-window read and the skipped-row writes, so "
            "the fetch/filter/route/chunk path can be exercised without "
            "credentials. The emitted shards then include models a real run "
            "would have dropped as recently scanned — do not feed them to a "
            "production scan."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("all", *MODEL_TYPES),
        default="all",
        help="Restrict the scan to one mode (e.g. 'embedding' for a quick, "
        "low-resource manual run). 'all' (the default, and what the "
        "scheduled run always uses) fetches/shards both model-types.",
    )
    args = parser.parse_args()

    model_types = MODEL_TYPES if args.model_type == "all" else (args.model_type,)

    matrix = generate_shards(
        top_k=args.top_k,
        shard_size_generative=args.shard_size_generative,
        shard_size_embedding=args.shard_size_embedding,
        x1_max_params=args.x1_max_params,
        x2_max_params=args.x2_max_params,
        x2_shard_size=args.x2_shard_size,
        x4_shard_size=args.x4_shard_size,
        output_dir=args.output_dir,
        model_types=model_types,
        dry_run=args.dry_run,
    )

    print(f"\nTotal shards across both model-types: {len(matrix)}")

    # Split by runner tier so push-to-clickhouse.yaml's three per-tier jobs
    # can each cap strategy.max-parallel in cards (x1=1, x2=2, x4=4/shard).
    matrix_by_tier: dict[str, list[dict]] = {"x1": [], "x2": [], "x4": []}
    for item in matrix:
        matrix_by_tier[item["runner"]].append(item)

    write_github_output(
        {
            "matrix_x1": json.dumps(matrix_by_tier["x1"]),
            "matrix_x2": json.dumps(matrix_by_tier["x2"]),
            "matrix_x4": json.dumps(matrix_by_tier["x4"]),
        }
    )


if __name__ == "__main__":
    main()
