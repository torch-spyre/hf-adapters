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

Each model type's list is PRE-FILTERED before any chunking: models with no
adapter for their config class, models too large for Spyre, and MoE models are
all removed here, and a terminal verdict row for each is written straight to
ClickHouse (so this script needs the CLICKHOUSE_* env vars, or --write-to-csv to
record them in a file instead).

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

# Add the project root to sys.path BEFORE importing from tests/ or utils/ — this
# script lives in .github/scripts/, so neither is importable from its own
# directory, and it is run as a plain script (not `python -m`), which puts that
# directory on sys.path rather than the repo root. Must stay above the imports
# below; the workflow happens to invoke it from the repo root, which would mask
# a wrong order here until someone ran it from anywhere else.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.spyre.weekly_generation.failure_categories import (  # noqa: E402
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    fetch_and_filter,
)
from tests.spyre.weekly_generation.model_type import ModelType  # noqa: E402
from tests.spyre.weekly_generation.sink.result_sink import ResultSink  # noqa: E402
from tests.spyre.weekly_generation.sink.sink_factory import create_sink  # noqa: E402


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


def generate_shards(
    top_k: int,
    max_params: int,
    shard_size_generative: int,
    shard_size_embedding: int,
    x1_max_params: int,
    x2_max_params: int,
    x2_shard_size: int,
    x4_shard_size: int,
    output_dir: Path,
    model_types: list[ModelType],
    snapshot_date: date,
    write_to_csv: Path | None = None,
) -> list[dict]:
    """Fetch each requested model type's top-K list once, write shard JSON files,
    and return the combined matrix (list of {mode, shard_index, shard_file,
    runner} dicts).

    Each model type's list is pre-filtered (see ``fetch_and_filter``) before any
    chunking, so a shard's size is a count of real evaluations rather than of
    fetched candidates. Within each type, the survivors are then split into three
    parameter-count tiers (see module docstring), each chunked at its own shard
    size and tagged with the runner ("x1"/"x2"/"x4") that handles it.

    *model_types* restricts which ``ModelType`` members to fetch/shard — used by
    workflow_dispatch's model_type input so a manual run can scan just embedding
    models (much quicker, less resource-hungry) without the schedule-triggered
    full scan having to change.

    *max_params* is the ceiling above which a model is rejected outright as too
    large for Spyre — distinct from *x1_max_params*/*x2_max_params*, which only
    route surviving models between runner tiers.

    *write_to_csv* records the terminal verdicts in a new CSV (one per model type)
    instead of ClickHouse, so the whole fetch → filter → route → chunk path can be
    exercised without credentials. That sink is write-only, so the emitted shards
    then include models a real run would have dropped as recently-scanned.

    The matrix's ``mode`` key keeps its name because push-to-clickhouse.yaml
    reads ``matrix.mode`` and passes it to ``weekly_test.py --mode``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Only the x1 tier's shard size is per-model-type; x2/x4 hold far fewer,
    # larger models, so one size each is enough.
    x1_shard_sizes = {
        ModelType.GENERATIVE: shard_size_generative,
        ModelType.EMBEDDING: shard_size_embedding,
    }

    matrix: list[dict] = []
    for model_type in model_types:
        # One sink per model type — each binds a single table (or file) — and
        # closed here because this function is what constructed it. Closing is
        # what flushes the ClickHouse sink's buffered verdict rows.
        sink: ResultSink = create_sink(
            model_type=model_type,
            write_to_csv=write_to_csv,
        )
        with sink:
            rows: list[dict] = fetch_and_filter(
                model_type=model_type,
                snapshot_date=snapshot_date,
                top_k=top_k,
                sink=sink,
                max_params=max_params,
            )

        by_tier: dict[str, list[dict]] = {"x1": [], "x2": [], "x4": []}
        for row in rows:
            by_tier[_tier_for(row, x1_max_params, x2_max_params)].append(row)

        tier_shard_sizes = {
            "x1": x1_shard_sizes[model_type],
            "x2": x2_shard_size,
            "x4": x4_shard_size,
        }
        model_type_shard_count = 0
        for runner, group_rows in by_tier.items():
            group_shard_size = tier_shard_sizes[runner]
            shards = _chunk(group_rows, group_shard_size)
            model_type_shard_count += len(shards)
            print(
                f"{model_type} ({runner}): {len(group_rows)} model(s), split into "
                f"{len(shards)} shard(s) of up to {group_shard_size} each"
            )
            for shard_index, shard_rows in enumerate(shards):
                shard_file = f"{model_type}-{runner}-shard-{shard_index:03d}.json"
                (output_dir / shard_file).write_text(json.dumps(shard_rows))
                matrix.append(
                    {
                        "mode": model_type,
                        "shard_index": shard_index,
                        "shard_file": shard_file,
                        "runner": runner,
                    }
                )
        print(
            f"{model_type}: {len(rows)} model(s) total, "
            f"{model_type_shard_count} shard(s)"
        )

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
        help="Number of top models to fetch per model type (by downloads).",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=MAX_NUMBER_PARAMS,
        help=(
            "Reject models above this parameter count "
            f"(default: {MAX_NUMBER_PARAMS:,})."
        ),
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
        "--write-to-csv",
        type=Path,
        default=None,
        metavar="VERDICTS_CSV",
        help=(
            "Record the terminal verdicts in a new CSV per mode (suffixed "
            "-generative / -embedding) instead of ClickHouse, so this script can "
            "run without credentials."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("all", *(model_type.value for model_type in ModelType)),
        default="all",
        help="Restrict the scan to one model type (e.g. 'embedding' for a quick, "
        "low-resource manual run). 'all' (the default, and what the "
        "scheduled run always uses) fetches/shards every model type.",
    )
    parser.add_argument(
        "--snapshot-date",
        type=date.fromisoformat,
        required=True,
        metavar="YYYY-MM-DD",
        help=(
            "Date to record as the snapshot date for all rows written in this run. "
            "Use $(date -u +%Y-%m-%d) for today when invoking manually."
        ),
    )
    args = parser.parse_args()

    model_types = (
        list(ModelType) if args.model_type == "all" else [ModelType(args.model_type)]
    )

    print("Generating shards started")
    print("Snapshot date:", args.snapshot_date)

    matrix = generate_shards(
        top_k=args.top_k,
        max_params=args.max_params,
        shard_size_generative=args.shard_size_generative,
        shard_size_embedding=args.shard_size_embedding,
        x1_max_params=args.x1_max_params,
        x2_max_params=args.x2_max_params,
        x2_shard_size=args.x2_shard_size,
        x4_shard_size=args.x4_shard_size,
        output_dir=args.output_dir,
        model_types=model_types,
        write_to_csv=args.write_to_csv,
        snapshot_date=args.snapshot_date,
    )

    print(f"\nTotal shards across {len(model_types)} model type(s): {len(matrix)}")

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
