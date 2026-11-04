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
"""Build a ``--model-list-file`` from saved catalog CSVs, with no Hub requests.

Reads a generative and an embedding catalog CSV (the format
``utils/fetch_top_*_models.py`` writes, e.g. the ``*_DEPRECATED.csv`` snapshots
in ``resources/``) and writes the JSON list that ``warm_top_models_cache.py``
and ``estimate_top_models_cache_size.py`` accept through ``--model-list-file``.

The rules match ``build_model_list``: curated models first (unless
``--no-curated``), then both CSVs merged by downloads, duplicates dropped, and
models over ``--max-params`` removed. ``--top-k-*`` cut each CSV to its first N
rows and are recorded in the file, so the scripts reading it must be given the
same values.

The CSVs carry no repo file list and no per-dtype parameter counts, so the
rows hold neither. Consequences for the scripts that read the file:

* the estimator can size a model only from its parameter count (x 2 bytes by
  default), or with ``--exact``; models with no parameter count are ``unknown``;
* the warmer still asks the Hub for each model's real file sizes before
  downloading it. Without a file list, though, it cannot tell that a repo has
  safetensors, so it downloads ``.bin`` copies alongside them.

Usage::

    python .github/cache_config/model_list_from_csvs.py \\
        --generative-csv resources/top_generative_models_DEPRECATED.csv \\
        --embedding-csv resources/top_embedding_models_DEPRECATED.csv \\
        --output model_list.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Runs as a plain script from .github/cache_config/: its own directory is on
# sys.path already (for the warmer); the project root is needed for utils/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from warm_top_models_cache import (  # noqa: E402
    add_model_list_args,
    as_int,
    list_settings,
    save_model_list,
)

from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    PrefilterResult,
    prefilter_models,
)
from utils.hf_model_catalog import (  # noqa: E402
    load_curated_embedding_models,
    load_curated_generative_models,
)
from utils.utilities import concat_and_dedup_dicts, ts  # noqa: E402

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def read_catalog_csv(path: Path, mode: str, limit: int) -> list[dict[str, object]]:
    """The first *limit* rows of a catalog CSV, tagged with *mode*, in file order."""
    with open(path, newline="", encoding="utf-8") as f:
        reader: csv.DictReader[str] = csv.DictReader(f)
        missing: set[str] = {"model_id", "downloads", "parameters"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        rows: list[dict[str, object]] = []
        for record in reader:
            if len(rows) >= limit:
                break
            model_id: str = (record.get("model_id") or "").strip()
            if not model_id:
                continue
            # Only the fields the estimator, warmer and prefilter read, plus a
            # few for people reading the file.
            rows.append(
                {
                    "model_id": model_id,
                    "model_type": record.get("model_type") or None,
                    "config_class": record.get("config_class") or None,
                    "downloads": as_int(record.get("downloads")),
                    "parameters": as_int(record.get("parameters")),
                    "mode": mode,
                    "curated": False,
                }
            )
    return rows


def curated_rows() -> list[dict[str, object]]:
    """Curated ids as bare rows: no Hub request, so no downloads or size."""
    rows: list[dict[str, object]] = []
    for mode, ids in (
        ("generative", load_curated_generative_models()),
        ("embedding", load_curated_embedding_models()),
    ):
        rows.extend(
            {
                "model_id": model_id,
                "downloads": None,
                "parameters": None,
                "mode": mode,
                "curated": True,
            }
            for model_id in ids
        )
    return rows


def build_from_csvs(
    generative_csv: Path,
    embedding_csv: Path,
    top_k_generative: int,
    top_k_embedding: int,
    max_params: int,
    include_curated: bool,
) -> list[dict[str, object]]:
    """Same ordering and filters as ``build_model_list``, from local files."""
    fetched: list[dict[str, object]] = read_catalog_csv(
        generative_csv, "generative", top_k_generative
    ) + read_catalog_csv(embedding_csv, "embedding", top_k_embedding)
    fetched.sort(key=lambda row: as_int(row.get("downloads")) or 0, reverse=True)
    curated: list[dict[str, object]] = curated_rows() if include_curated else []
    merged: list[dict[str, object]] = concat_and_dedup_dicts(
        first=curated, second=fetched
    )
    result: PrefilterResult = prefilter_models(merged, max_params=max_params)
    print(
        f"{ts()} {len(curated)} curated + {len(fetched)} from CSVs -> "
        f"{len(merged)} unique -> {len(result.keep)} kept "
        f"({len(result.skipped)} over {max_params:,} params)",
        flush=True,
    )
    return result.keep


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_list_args(parser)
    parser.add_argument(
        "--generative-csv",
        type=Path,
        default=_REPO_ROOT / "resources" / "top_generative_models_DEPRECATED.csv",
        help="Generative catalog CSV (default: the resources/ *_DEPRECATED.csv).",
    )
    parser.add_argument(
        "--embedding-csv",
        type=Path,
        default=_REPO_ROOT / "resources" / "top_embedding_models_DEPRECATED.csv",
        help="Embedding catalog CSV (default: the resources/ *_DEPRECATED.csv).",
    )
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Leave out the curated models; keep only the CSV rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON model list.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = _parse_args(argv)
    rows: list[dict[str, object]] = build_from_csvs(
        args.generative_csv,
        args.embedding_csv,
        args.top_k_generative,
        args.top_k_embedding,
        args.max_params,
        include_curated=not args.no_curated,
    )
    save_model_list(args.output, rows, list_settings(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
