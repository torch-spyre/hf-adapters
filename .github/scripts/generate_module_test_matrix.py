#!/usr/bin/env python3
"""Build the model-module CI matrix, sharding only measured slow configs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# These were the only model-module jobs above 10 minutes in regression run
# 35877734714 (39m, 31m, 23m, and 16m respectively). The next-slowest config
# took 7m, so leave the remaining configs as one job each.
SHARDED_CONFIGS = frozenset(
    {
        "Ministral-3-14B-Instruct-2512.yaml",
        "gemma-4-12B-it.yaml",
        "gemma-4-26B-A4B-it.yaml",
        "gpt-oss-20b.yaml",
    }
)

# Only test_with_cpu is currently monitored by model-module CI. Split that test
# into mutually exclusive prefill and non-prefill shards for the slow configs.
_NOT_FORWARD_OR_EAGER = "not test_forward and not test_eager_vs_compile"
SHARDS = (
    {
        "shard": "with-cpu-prefill",
        "pytest_filter": f"{_NOT_FORWARD_OR_EAGER} and test_with_cpu and prefill",
    },
    {
        "shard": "with-cpu-other",
        "pytest_filter": f"{_NOT_FORWARD_OR_EAGER} and test_with_cpu and not prefill",
    },
)


def generate_matrix(config_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Return a GitHub Actions ``include`` matrix for module-test configs."""
    configs = sorted(path.name for path in config_dir.glob("*.yaml"))
    if not configs:
        raise ValueError(f"No module-test YAML configs found in {config_dir}")

    entries = []
    for config in configs:
        if config in SHARDED_CONFIGS:
            entries.extend({"config": config, **shard} for shard in SHARDS)
        else:
            entries.append(
                {"config": config, "shard": "all", "pytest_filter": "test_with_cpu"}
            )
    return {"include": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Directory containing model-module *.yaml configs",
    )
    args = parser.parse_args()

    matrix = generate_matrix(args.config_dir)
    encoded = json.dumps(matrix, separators=(",", ":"))
    print(f"Generated {len(matrix['include'])} model-module matrix entries:")
    print(json.dumps(matrix, indent=2))

    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"module_config_matrix={encoded}\n")
    else:
        print(f"module_config_matrix={encoded}")


if __name__ == "__main__":
    main()
