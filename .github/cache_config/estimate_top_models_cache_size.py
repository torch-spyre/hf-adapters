#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Estimate how much disk ``warm_top_models_cache.py`` would download, without downloading.

Builds the same model list as the warmer (the weekly scan's selection: by-name
models + top-K by downloads + top-K by trending, per model type, deduplicated,
models over the parameter limit or without an adapter dropped) and sizes each
model with one of these methods, best first:

* ``exact``: real file sizes from the Hub, after the warmer's ignore patterns.
  Only with ``--exact``, which costs one Hub request per model (paced by
  ``--sleep-seconds``).
* ``safetensors``: the per-dtype parameter counts the fetch already returned,
  times each dtype's width. Exact for the safetensors weights, with no extra
  requests. Tokenizers and configs are not counted; they are small.
* ``parameters``: parameter count x ``--fallback-bytes-per-param`` (default 2),
  for repos without safetensors metadata, usually ``.bin``-only repos.
* ``unknown``: nothing to size the model by. Counted as 0 bytes and reported.

The size estimate itself makes no Hub requests (except with ``--exact``); the
list fetch does, about two per candidate model, ``--catalog-workers`` at a time
(default 4). ``--save-model-list`` keeps that list so the warmer can reuse it
through ``--model-list-file`` instead of fetching it again, and
``--model-list-file`` here re-estimates a saved list with no fetch at all.

Usage::

    python .github/cache_config/estimate_top_models_cache_size.py
    python .github/cache_config/estimate_top_models_cache_size.py \\
        --top-k-downloads 2000 --top-k-trending 500 --output-csv sizes.csv \\
        --save-model-list model_list.json
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Runs as a plain script from .github/cache_config/: its own directory is on
# sys.path already (for the warmer); the project root is needed for utils/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.hf_api import ModelInfo  # noqa: E402
from warm_top_models_cache import (  # noqa: E402
    add_model_list_args,
    as_int,
    exact_bytes,
    get_model_list,
    human_bytes,
    ignore_patterns_for,
    list_settings,
    safetensors_bytes,
    save_model_list,
)

from utils.utilities import ts  # noqa: E402

_TOP_N_LARGEST: int = 20


@dataclass
class SizedModel:
    """One row of the estimate (and of ``--output-csv``)."""

    model_id: str
    mode: str
    curated: bool
    downloads: int | None
    parameters: int | None
    estimated_bytes: int
    method: str
    """exact | safetensors | parameters | unknown"""


def size_model(
    row: dict[str, object],
    fallback_bytes_per_param: float,
    api: HfApi | None,
) -> SizedModel:
    """Size one catalog row with the best method available (see module docstring)."""
    info: object = row.get("model_info")
    model_info: ModelInfo | None = info if isinstance(info, ModelInfo) else None
    repo_id: str = str(row["model_id"])
    params: int | None = as_int(row.get("parameters"))

    estimate: int | None = None
    method: str = "unknown"
    if api is not None:
        estimate = exact_bytes(api, repo_id, ignore_patterns_for(model_info))
        method = "exact"
    if estimate is None:
        estimate = safetensors_bytes(model_info)
        method = "safetensors"
    if estimate is None and params is not None:
        estimate = int(params * fallback_bytes_per_param)
        method = "parameters"
    if estimate is None:
        estimate, method = 0, "unknown"

    return SizedModel(
        model_id=repo_id,
        mode=str(row.get("mode") or ""),
        curated=bool(row.get("curated")),
        downloads=as_int(row.get("downloads")),
        parameters=params,
        estimated_bytes=estimate,
        method=method,
    )


def _summary_lines(models: list[SizedModel]) -> list[str]:
    total: int = sum(m.estimated_bytes for m in models)
    lines: list[str] = [
        f"Estimated download for {len(models):,} models: {human_bytes(total)}",
        "",
        "By model type:",
    ]
    for mode in sorted({m.mode for m in models}):
        group: list[SizedModel] = [m for m in models if m.mode == mode]
        lines.append(
            f"    {mode:<12} {len(group):>7,} models  "
            f"{human_bytes(sum(m.estimated_bytes for m in group)):>9}"
        )
    lines += ["", "By sizing method:"]
    for method in ("exact", "safetensors", "parameters", "unknown"):
        group = [m for m in models if m.method == method]
        if group:
            lines.append(
                f"    {method:<12} {len(group):>7,} models  "
                f"{human_bytes(sum(m.estimated_bytes for m in group)):>9}"
            )
    largest: list[SizedModel] = sorted(
        models, key=lambda m: m.estimated_bytes, reverse=True
    )[:_TOP_N_LARGEST]
    lines += ["", f"Largest {len(largest)}:"]
    lines += [
        f"    {human_bytes(m.estimated_bytes):>9}  {m.model_id} ({m.method})"
        for m in largest
    ]
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_list_args(parser)
    parser.add_argument(
        "--fallback-bytes-per-param",
        type=float,
        default=2.0,
        help=(
            "Bytes per parameter for models without safetensors metadata "
            "(default: 2, i.e. fp16/bf16; use 4 for an fp32 upper bound)."
        ),
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Ask the Hub for each model's real file sizes (one request per model).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="With --exact: pause between Hub requests (default: 0.5).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Also write one row per model to this CSV.",
    )
    parser.add_argument(
        "--save-model-list",
        type=Path,
        default=None,
        help=(
            "Save the fetched model list as JSON, for the warmer's "
            "--model-list-file, so it does not fetch the list again."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = _parse_args(argv)
    rows: list[dict[str, object]] = get_model_list(args)
    # Saved before sizing, so a failure or Ctrl+C there does not lose the fetch.
    if args.save_model_list is not None:
        save_model_list(args.save_model_list, rows, list_settings(args))

    api: HfApi | None = (
        HfApi(token=os.environ.get("HF_TOKEN") or None) if args.exact else None
    )
    sized: list[SizedModel] = []
    for index, row in enumerate(rows, start=1):
        sized.append(size_model(row, args.fallback_bytes_per_param, api))
        if api is not None:
            if index % 500 == 0:
                print(f"{ts()} sized {index:,}/{len(rows):,}", flush=True)
            time.sleep(args.sleep_seconds)

    lines: list[str] = _summary_lines(sized)
    print("\n" + "\n".join(lines), flush=True)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer: csv.DictWriter[str] = csv.DictWriter(
                f, fieldnames=list(SizedModel.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(m) for m in sized)
        print(f"\nWrote {len(sized):,} rows to {args.output_csv}")

    summary_path: str | None = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("## Top-models cache size estimate\n\n```\n")
            f.write("\n".join(lines) + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
