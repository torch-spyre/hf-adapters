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

Per-tier shard sizes do NOT need to be small: weekly_test.py already
re-chunks whatever list it's given into fresh-OS-process batches of
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS/EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS
regardless of shard size, which is what actually bounds how many models'
memory can accumulate in one process before a clean restart. A tiny shard
size buys no extra safety over a large one — it only multiplies GitHub
Actions job count, and matrices are hard-capped at 256 jobs total.

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
from pathlib import Path

# Add the project root to the Python path so we can import from utils/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402
from utils.fetch_top_generative_models import fetch_top_generative_models  # noqa: E402

MODES = ("generative", "embedding")


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
    shard_size_generative: int,
    shard_size_embedding: int,
    x1_max_params: int,
    x2_max_params: int,
    x2_shard_size: int,
    x4_shard_size: int,
    output_dir: Path,
) -> list[dict]:
    """Fetch both mode's top-K lists once, write shard JSON files, and return
    the combined matrix (list of {mode, shard_index, shard_file, runner} dicts).

    Within each mode, models are split into three parameter-count tiers (see
    module docstring), each chunked at its own shard size and tagged with
    the runner ("x1"/"x2"/"x4") that ends up handling it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fetchers = {
        "generative": (fetch_top_generative_models, shard_size_generative),
        "embedding": (fetch_top_embedding_models, shard_size_embedding),
    }

    matrix: list[dict] = []
    for mode, (fetch_fn, shard_size) in fetchers.items():
        rows: list[dict] = fetch_fn(limit=top_k)
        # model_info is a live huggingface_hub.ModelInfo object attached by
        # build_catalog — not JSON-serializable, and no longer needed since
        # is_moe is precomputed onto each row (see utils/hf_model_catalog.py).
        for row in rows:
            row.pop("model_info", None)

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
    args = parser.parse_args()

    matrix = generate_shards(
        top_k=args.top_k,
        shard_size_generative=args.shard_size_generative,
        shard_size_embedding=args.shard_size_embedding,
        x1_max_params=args.x1_max_params,
        x2_max_params=args.x2_max_params,
        x2_shard_size=args.x2_shard_size,
        x4_shard_size=args.x4_shard_size,
        output_dir=args.output_dir,
    )

    print(f"\nTotal shards across both modes: {len(matrix)}")
    write_github_output({"matrix": json.dumps(matrix)})


if __name__ == "__main__":
    main()
