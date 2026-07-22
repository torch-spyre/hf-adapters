#!/usr/bin/env python3
"""
Fetch the weekly top-K generative and embedding model lists once, then split
each into fixed-size shards for parallel GitHub Actions jobs.

Mirrors the pattern in generate_test_matrix.py: this script does the one-time
fetch/compute step and emits a JSON matrix via GITHUB_OUTPUT for a downstream
`strategy.matrix.include` job. Each shard is written to its own JSON file
under --output-dir; downstream jobs load their shard via
`weekly_test.py --model-list-file`.

Models at or above --large-model-threshold parameters are split into their
own small shards and tagged runner="x2" (spyre_pf_x2 — more memory, 2 cards)
instead of runner="x1" (spyre_pf_x1 — less memory, 1 card), so a shard that
happens to contain several large models doesn't have to share a single-card
runner's memory budget. See push-to-clickhouse.yaml's weekly-model-scan job
for how `matrix.runner` selects the actual runs-on label.

Usage (called by the GHA workflow):
    python .github/scripts/generate_weekly_shards.py \
        --top-k 10000 \
        --shard-size-generative 250 \
        --shard-size-embedding 500 \
        --large-model-threshold 7000000000 \
        --large-model-shard-size 2 \
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


def _is_large(row: dict, threshold: int) -> bool:
    params = row.get("parameters")
    return isinstance(params, (int, float)) and params >= threshold


def generate_shards(
    top_k: int,
    shard_size_generative: int,
    shard_size_embedding: int,
    large_model_threshold: int,
    large_model_shard_size: int,
    output_dir: Path,
) -> list[dict]:
    """Fetch both mode's top-K lists once, write shard JSON files, and return
    the combined matrix (list of {mode, shard_index, shard_file, runner} dicts).

    Within each mode, models are split into a "large" group (>= threshold
    parameters, chunked small and tagged runner="x2") and everyone else
    (chunked at the normal shard size, tagged runner="x1").
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

        large_rows = [r for r in rows if _is_large(r, large_model_threshold)]
        small_rows = [r for r in rows if not _is_large(r, large_model_threshold)]

        groups = (
            (small_rows, shard_size, "x1"),
            (large_rows, large_model_shard_size, "x2"),
        )
        mode_shard_count = 0
        for group_rows, group_shard_size, runner in groups:
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
        "--large-model-threshold",
        type=int,
        default=7_000_000_000,
        help="Models with >= this many parameters are routed to spyre_pf_x2 shards instead of spyre_pf_x1.",
    )
    parser.add_argument(
        "--large-model-shard-size",
        type=int,
        default=2,
        help="Models per large-model (x2) shard.",
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
        large_model_threshold=args.large_model_threshold,
        large_model_shard_size=args.large_model_shard_size,
        output_dir=args.output_dir,
    )

    print(f"\nTotal shards across both modes: {len(matrix)}")
    write_github_output({"matrix": json.dumps(matrix)})


if __name__ == "__main__":
    main()
