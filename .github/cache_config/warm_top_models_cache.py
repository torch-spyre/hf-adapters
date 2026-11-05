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
"""One-off warm-up of the shared HF cache with the weekly scan's model lists.

Builds the same list the weekly scan shards are cut from and downloads each
model into ``HF_HOME`` like ``manage_cache.py`` does. Per model type (generative
and embedding), that list is the union of:

1. the by-name models: the curated file plus every checkpoint in
   ``tests/model_registry.py`` (``model_fetcher.by_name_ids``);
2. the top ``--top-k-downloads`` by downloads over the last 30 days;
3. the top ``--top-k-trending`` by the Hub's trending score;

minus models over the scan's parameter limit and, unless ``--all-models``, models
no adapter resolves for. The scan's workers never call ``from_pretrained`` on
those, so their weights would never be read. The weekly scan only deletes weights
it downloaded itself, so models warmed here stay cached between scans.

The network is treated gently, on purpose:

* one model at a time, and one file at a time inside it (``max_workers=1``);
* a fixed pause after every model that touched the network;
* transient errors retry with exponential backoff, and HTTP 429 waits longer;
* after several consecutive failures the run cools down, and gives up if the
  failures continue after repeated cool-downs.

Run it with ``HF_HUB_DISABLE_XET=1`` (the workflow sets this), so each file
arrives over a single plain HTTP stream instead of Xet's parallel chunk fetches.

Safe to stop and dispatch again:

* A model counts as cached only when every file the warmer would download is on
  disk, including every shard its weight index lists. Anything less is
  downloaded again, and ``snapshot_download`` fetches only what is missing.
* ``--max-hours`` is a hard budget. Each download runs in a child process
  (``hf_snapshot_worker.py``) that is killed when the budget runs out, and no
  retry wait or cool-down is started that would outlast it.
* ``--min-free-gb`` is a floor on free space. A model is downloaded only if its
  real file sizes from the Hub (or, failing that, its per-dtype safetensors
  metadata) fit above the floor; a model that cannot be sized is skipped.

Fetching the list is itself a burst of Hub requests: about two per candidate
model, for several thousand candidates at the default top-K. Two flags keep
it down:

* ``--catalog-workers`` (default 4; the weekly scan uses
  ``DEFAULT_CATALOG_MAX_WORKERS``, 8) caps how many of those requests run at
  once, the adapter check included;
* ``--model-list-file`` reuses a list the size estimator saved
  (``--save-model-list``), so an estimate followed by a warm-up fetches once.

Usage::

    # Estimate the size and save the list
    python .github/cache_config/estimate_top_models_cache_size.py \\
        --save-model-list model_list.json

    # Warm the cache from that list, with no second fetch
    python .github/cache_config/warm_top_models_cache.py \\
        --model-list-file model_list.json \\
        --sleep-seconds 15 --min-free-gb 500 --max-hours 70
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# The project root must be importable before the tests/ and utils/ imports below:
# this file runs as a plain script from .github/cache_config/, whose own directory
# (for hf_snapshot_worker) is already on sys.path.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from hf_snapshot_worker import HTTP_TOO_MANY_REQUESTS, RESULT_PREFIX  # noqa: E402
from huggingface_hub import HfApi, constants, snapshot_download  # noqa: E402
from huggingface_hub.hf_api import ModelInfo  # noqa: E402
from huggingface_hub.utils import filter_repo_objects  # noqa: E402

from tests.spyre.weekly_generation import model_fetcher  # noqa: E402
from tests.spyre.weekly_generation.failure_categories import (  # noqa: E402
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    PrefilterResult,
    prefilter_models,
)
from tests.spyre.weekly_generation.model_type import ModelType  # noqa: E402
from utils.hf_model_catalog import (  # noqa: E402
    DEFAULT_CATALOG_MAX_WORKERS,
    Ranking,
)
from utils.utilities import concat_and_dedup_dicts, ts  # noqa: E402

# Formats transformers / sentence-transformers never load on this path. Skipping
# them avoids pulling the same weights two or three times from repos that ship
# ONNX, GGUF, Flax, TF, CoreML or original checkpoint copies.
_ALWAYS_IGNORE: tuple[str, ...] = (
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
    "*.mlmodel",
    "onnx/*",
    "openvino/*",
    "coreml/*",
    "original/*",
)

# A cached snapshot with none of these files has no weights yet. Repos the fetch
# step touched hold only the config.json that AutoConfig downloaded.
_WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")

# Sharded checkpoints list their shards in one of these; every listed shard must
# be on disk for the snapshot to load.
_WEIGHT_INDEX_FILES: tuple[str, ...] = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

# Bytes per element for the safetensors dtypes the Hub reports.
_DTYPE_BYTES: dict[str, float] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "C64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F6_E2M3": 0.75,
    "F6_E3M2": 0.75,
    "F4": 0.5,
}
_UNKNOWN_DTYPE_BYTES: float = 2

# Per ranking and model type, the weekly scan's defaults (generate_weekly_shards.py).
_DEFAULT_TOP_K: int = 1_000

# Threads for the list fetch's per-model Hub checks. The weekly scan uses
# DEFAULT_CATALOG_MAX_WORKERS (utils/hf_model_catalog.py); this one-off job has no
# deadline to meet.
_DEFAULT_CATALOG_WORKERS: int = 4

# Bump when the saved model-list layout changes. 2: the #753 selection (two
# rankings + model registry) and the adapter check.
_MODEL_LIST_FORMAT_VERSION: int = 2

_WORKER: Path = Path(__file__).resolve().parent / "hf_snapshot_worker.py"
_RATE_LIMIT_PAUSE_SECONDS: float = 600.0
_MAX_BACKOFF_SECONDS: float = 600.0


@dataclass
class ModelResult:
    """One line of the JSONL report."""

    index: int
    model_id: str
    status: str
    """cached | downloaded | failed | skipped_disk | skipped_unsized | interrupted"""
    downloads: int | None
    parameters: int | None
    estimated_bytes: int | None = None
    size_method: str | None = None
    """exact | safetensors"""
    bytes_on_disk: int | None = None
    seconds: float | None = None
    attempts: int = 0
    error: str | None = None


class AbortRunError(Exception):
    """Stop the whole run: the disk is full, or the Hub keeps failing."""


def human_bytes(n: float) -> str:
    """Format a byte count for logs, e.g. 1536 -> '1.5KB' (binary units)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    raise AssertionError("unreachable: the TB branch always returns")


def add_model_list_args(parser: argparse.ArgumentParser) -> None:
    """The flags that choose the model list, shared with the size estimator."""
    parser.add_argument(
        "--top-k-downloads",
        type=int,
        default=_DEFAULT_TOP_K,
        help=(
            "Per model type, the top N by downloads over the last 30 days "
            f"(default: {_DEFAULT_TOP_K}; 0 skips this ranking)."
        ),
    )
    parser.add_argument(
        "--top-k-trending",
        type=int,
        default=_DEFAULT_TOP_K,
        help=(
            "Per model type, the top N by the Hub's trending score "
            f"(default: {_DEFAULT_TOP_K}; 0 skips this ranking)."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Keep models no adapter resolves for. By default they are dropped: "
            "the weekly scan never loads their weights."
        ),
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=MAX_NUMBER_PARAMS,
        help=(
            "Skip models above this parameter count; the weekly scan never "
            f"tests them (default: {MAX_NUMBER_PARAMS:,})."
        ),
    )
    parser.add_argument(
        "--catalog-workers",
        type=int,
        default=_DEFAULT_CATALOG_WORKERS,
        help=(
            "Parallel Hub requests while fetching the list "
            f"(default: {_DEFAULT_CATALOG_WORKERS}; the weekly scan uses "
            f"{DEFAULT_CATALOG_MAX_WORKERS})."
        ),
    )
    parser.add_argument(
        "--model-list-file",
        type=Path,
        default=None,
        help=(
            "JSON model list to reuse instead of fetching from the Hub. The size "
            "estimator writes one with --save-model-list; the --top-k-*, "
            "--max-params and --all-models flags must match the ones it was "
            "built with."
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_list_args(parser)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=15.0,
        help="Pause after every model that touched the network (default: 15).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=500.0,
        help=(
            "Stop when free space on the HF_HOME volume drops below this, and "
            "skip any model whose size would cross it (default: 500)."
        ),
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=70.0,
        help=(
            "Hard wall-clock budget; a download still running when it ends is "
            "stopped (default: 70)."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Download attempts per model on transient errors (default: 4).",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("warm_cache_report.jsonl"),
        help="Append one JSON line per model here (default: warm_cache_report.jsonl).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Building the model list
# ---------------------------------------------------------------------------


def _mode_rows(
    model_type: ModelType, top_k_downloads: int, top_k_trending: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """*model_type*'s by-name rows and its ranked rows, as ``fetch_and_filter`` merges them.

    Calls the same fetchers as ``model_fetcher.fetch`` and ``load_by_name`` but
    keeps ``model_info``, whose ``siblings`` and ``safetensors`` the warmer reads
    (``ignore_patterns_for``, ``safetensors_bytes``). Ranked rows already in the
    by-name set, or in the downloads ranking for a trending row, are dropped.
    """
    mode: str = str(model_type)
    ids: list[str] = model_fetcher.by_name_ids(model_type)
    by_name: list[dict[str, object]] = [
        {**row, "mode": mode, "curated": True}
        for row in (
            model_fetcher.all_metadata_fetchers[model_type](model_ids=ids)
            if ids
            else []
        )
    ]
    ranked: list[dict[str, object]] = []
    rankings: tuple[tuple[Ranking, int], ...] = (
        ("downloads", top_k_downloads),
        ("trending_score", top_k_trending),
    )
    for ranking, top_k in rankings:
        if top_k <= 0:
            continue
        ranked = concat_and_dedup_dicts(
            first=ranked,
            second=[
                {**row, "mode": mode, "curated": False}
                for row in model_fetcher.all_fetchers[model_type](
                    limit=top_k, ranking=ranking
                )
            ],
        )
    print(
        f"{ts()} {mode}: {len(by_name)} by-name + {len(ranked)} ranked "
        f"(downloads {top_k_downloads}, trending {top_k_trending})",
        flush=True,
    )
    return by_name, ranked


def _adapter_verdict(model_id: str) -> str | None:
    """None when an adapter resolves for *model_id*, else why not.

    Same resolver the scan's worker calls before it loads anything
    (``resolve_adapter_module_for_test``), minus the pytest conftest it lives in.
    ``AutoConfig`` reads the config.json the list fetch already cached.
    """
    from hf_adapters.auto_spyre_model import resolve_adapter_module
    from tests.model_registry import REMOTE_CODE_PATHS

    try:
        resolve_adapter_module(
            model_id, trust_remote_code=model_id in REMOTE_CODE_PATHS
        )
    except Exception as e:  # noqa: BLE001 -- any failure means the worker skips it
        return type(e).__name__
    return None


def drop_models_without_adapter(
    rows: list[dict[str, object]], workers: int
) -> list[dict[str, object]]:
    """*rows* an adapter resolves for, in order; the scan never loads the others."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts: list[str | None] = list(
            pool.map(_adapter_verdict, [str(row["model_id"]) for row in rows])
        )
    dropped: dict[str, int] = {}
    for verdict in verdicts:
        if verdict is not None:
            dropped[verdict] = dropped.get(verdict, 0) + 1
    kept: list[dict[str, object]] = [
        row for row, verdict in zip(rows, verdicts, strict=True) if verdict is None
    ]
    print(
        f"{ts()} adapter check: {len(kept)} kept, {len(rows) - len(kept)} dropped "
        f"{dict(sorted(dropped.items()))}",
        flush=True,
    )
    return kept


def build_model_list(
    top_k_downloads: int,
    top_k_trending: int,
    max_params: int,
    adapters_only: bool = True,
    workers: int = _DEFAULT_CATALOG_WORKERS,
) -> list[dict[str, object]]:
    """The weekly scan's selection for both model types, as one download queue.

    By-name models of both types come first, then every ranked row by downloads,
    so a run that stops early has warmed the most-used models. Each row gets a
    ``mode`` ("generative" or "embedding"); a model in both types keeps its
    first row. Then the scan's parameter limit and, with *adapters_only*, the
    adapter check.
    """
    by_name: list[dict[str, object]] = []
    ranked: list[dict[str, object]] = []
    for model_type in ModelType:
        mode_by_name, mode_ranked = _mode_rows(
            model_type, top_k_downloads, top_k_trending
        )
        by_name.extend(mode_by_name)
        ranked.extend(mode_ranked)
    ranked.sort(key=lambda row: as_int(row.get("downloads")) or 0, reverse=True)

    merged: list[dict[str, object]] = concat_and_dedup_dicts(
        first=by_name, second=ranked
    )
    result: PrefilterResult = prefilter_models(merged, max_params=max_params)
    print(
        f"{ts()} {len(by_name)} by-name + {len(ranked)} ranked -> {len(merged)} "
        f"unique -> {len(result.keep)} within {max_params:,} params "
        f"({len(result.skipped)} over)",
        flush=True,
    )
    if not adapters_only:
        return result.keep
    return drop_models_without_adapter(result.keep, workers)


def _row_to_json(row: dict[str, object]) -> dict[str, object]:
    """A catalog row without its live ModelInfo, keeping the two fields used here.

    ``siblings`` (the repo's file names) chooses the weight format and checks a
    snapshot for completeness; ``safetensors`` (per-dtype parameter counts)
    sizes the download. Nothing else on ModelInfo is read by these scripts.
    """
    out: dict[str, object] = {k: v for k, v in row.items() if k != "model_info"}
    info: object = row.get("model_info")
    if isinstance(info, ModelInfo):
        out["siblings"] = (
            [s.rfilename for s in info.siblings] if info.siblings is not None else None
        )
        out["safetensors_parameters"] = (
            dict(info.safetensors.parameters) if info.safetensors else None
        )
    return out


def _row_from_json(data: dict[str, object]) -> dict[str, object]:
    """Inverse of ``_row_to_json``: rebuild a ModelInfo holding the saved fields."""
    row: dict[str, object] = {
        k: v for k, v in data.items() if k not in ("siblings", "safetensors_parameters")
    }
    siblings: object = data.get("siblings")
    by_dtype: object = data.get("safetensors_parameters")
    kwargs: dict[str, object] = {"id": str(data["model_id"])}
    if isinstance(siblings, list):
        kwargs["siblings"] = [{"rfilename": str(name)} for name in siblings]
    if isinstance(by_dtype, dict) and by_dtype:
        counts: dict[str, int] = {str(k): int(v) for k, v in by_dtype.items()}
        kwargs["safetensors"] = {"parameters": counts, "total": sum(counts.values())}
    row["model_info"] = ModelInfo(**kwargs)
    return row


def list_settings(args: argparse.Namespace) -> dict[str, int | bool]:
    """The flags that decide which models are on the list."""
    return {
        "top_k_downloads": args.top_k_downloads,
        "top_k_trending": args.top_k_trending,
        "max_params": args.max_params,
        "adapters_only": not args.all_models,
    }


def save_model_list(
    path: Path, rows: list[dict[str, object]], settings: dict[str, int | bool]
) -> None:
    """Write *rows* as JSON, written to a temporary file and renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": _MODEL_LIST_FORMAT_VERSION,
        "settings": settings,
        "models": [_row_to_json(row) for row in rows],
    }
    tmp: Path = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    print(f"{ts()} Saved {len(rows):,} models to {path}", flush=True)


def load_model_list(
    path: Path, settings: dict[str, int | bool]
) -> list[dict[str, object]]:
    """Read a list saved by ``save_model_list`` and check it was built the same way.

    A list built with other --top-k-*, --max-params or --all-models values holds different
    models than asked for, so it is refused instead of silently reused.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: not a saved model list")
    if payload.get("format_version") != _MODEL_LIST_FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version {payload.get('format_version')!r}, "
            f"expected {_MODEL_LIST_FORMAT_VERSION}"
        )
    saved: object = payload.get("settings")
    if saved != settings:
        raise ValueError(
            f"{path} was built with {saved}, but this run asks for {settings}; "
            "pass the same flags, or fetch a new list"
        )
    models: object = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{path}: 'models' is not a list")
    rows: list[dict[str, object]] = [
        _row_from_json(m) for m in models if isinstance(m, dict)
    ]
    print(f"{ts()} Loaded {len(rows):,} models from {path}", flush=True)
    return rows


def get_model_list(args: argparse.Namespace) -> list[dict[str, object]]:
    """The model list: from ``--model-list-file`` if given, else fetched gently.

    Fetching sets ``HF_CATALOG_MAX_WORKERS`` for the shared catalog code, so its
    per-model Hub checks run ``--catalog-workers`` at a time instead of the
    scan's ``DEFAULT_CATALOG_MAX_WORKERS``; the adapter check uses as many.
    """
    if args.catalog_workers < 1:
        raise ValueError("--catalog-workers must be >= 1")
    if args.model_list_file is not None:
        return load_model_list(args.model_list_file, list_settings(args))
    os.environ["HF_CATALOG_MAX_WORKERS"] = str(args.catalog_workers)
    print(
        f"{ts()} Fetching the model list with {args.catalog_workers} parallel "
        "Hub request(s)",
        flush=True,
    )
    return build_model_list(
        args.top_k_downloads,
        args.top_k_trending,
        args.max_params,
        adapters_only=not args.all_models,
        workers=args.catalog_workers,
    )


def as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def ignore_patterns_for(model_info: ModelInfo | None) -> list[str]:
    """Download safetensors when the repo has them, and ``.bin`` only when it doesn't.

    ``manage_cache.py`` always ignores ``*.bin``, which leaves older bin-only
    repos (common among embedders) without weights. When the file list is
    unknown, ``.bin`` is kept rather than risk caching a repo with no weights.
    """
    patterns: list[str] = list(_ALWAYS_IGNORE)
    siblings = model_info.siblings if model_info is not None else None
    files: list[str] = [s.rfilename for s in siblings or []]
    if any(name.endswith(".safetensors") for name in files):
        patterns.append("*.bin")
    return patterns


def selected_files(model_info: ModelInfo | None, ignore: list[str]) -> list[str]:
    """Repo files the warmer downloads, per the catalog's file list (may be empty)."""
    siblings = model_info.siblings if model_info is not None else None
    names: list[str] = [s.rfilename for s in siblings or []]
    return list(filter_repo_objects(items=names, ignore_patterns=ignore))


# ---------------------------------------------------------------------------
# Sizing (shared with estimate_top_models_cache_size.py)
# ---------------------------------------------------------------------------


def safetensors_bytes(model_info: ModelInfo | None) -> int | None:
    """Weight bytes from the safetensors per-dtype parameter counts, if any."""
    if model_info is None or not model_info.safetensors:
        return None
    by_dtype: dict[str, int] = model_info.safetensors.parameters or {}
    if not by_dtype:
        return None
    return int(
        sum(
            count * _DTYPE_BYTES.get(dtype.upper(), _UNKNOWN_DTYPE_BYTES)
            for dtype, count in by_dtype.items()
        )
    )


def exact_bytes(api: HfApi, repo_id: str, ignore_patterns: list[str]) -> int | None:
    """Total size of the files the warmer would download, from the Hub."""
    try:
        info: ModelInfo = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"    {repo_id}: file sizes unavailable ({type(e).__name__})")
        return None
    sizes: dict[str, int] = {
        s.rfilename: (s.lfs.size if s.lfs is not None else s.size) or 0
        for s in info.siblings or []
    }
    kept: list[str] = list(
        filter_repo_objects(items=list(sizes), ignore_patterns=ignore_patterns)
    )
    return sum(sizes[name] for name in kept)


def download_size(
    api: HfApi | None,
    repo_id: str,
    ignore_patterns: list[str],
    model_info: ModelInfo | None,
) -> tuple[int | None, str | None]:
    """Bytes a download will add, and how they were found; (None, None) if unknown.

    Real file sizes first. The per-dtype safetensors counts are the fallback:
    they price fp32 weights at 4 bytes, but miss non-weight files. A parameter
    count alone is not trusted, since it says nothing about the dtype.
    """
    if api is not None:
        size: int | None = exact_bytes(api, repo_id, ignore_patterns)
        if size is not None:
            return size, "exact"
    size = safetensors_bytes(model_info)
    if size is not None:
        return size, "safetensors"
    return None, None


# ---------------------------------------------------------------------------
# Cache and disk checks (no network)
# ---------------------------------------------------------------------------


def _indexed_shards(snapshot: Path, ignore: list[str]) -> list[str]:
    """Every shard the snapshot's weight indexes list, minus ignored formats.

    The ``.bin`` index ships with repos whose ``.bin`` shards the warmer skips
    in favour of safetensors, so its shards go through the ignore patterns too.
    An unreadable index makes the snapshot incomplete.
    """
    shards: set[str] = set()
    for index_name in _WEIGHT_INDEX_FILES:
        index: Path = snapshot / index_name
        if not index.is_file():
            continue
        data: object = json.loads(index.read_text(encoding="utf-8"))
        weight_map: object = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_name} has no weight_map")
        shards.update(str(name) for name in weight_map.values())
    return list(filter_repo_objects(items=sorted(shards), ignore_patterns=ignore))


def missing_files(
    snapshot: Path, model_info: ModelInfo | None, ignore: list[str]
) -> list[str]:
    """Files the snapshot still needs: catalog files, indexed shards, any weight.

    ``is_file`` follows the snapshot's symlinks, so a link whose blob is missing
    counts as missing. The catalog's file list describes the Hub's current
    commit, so an older cached commit can show files as missing; that only
    costs a ``snapshot_download``, which brings it up to date.
    """
    required: set[str] = set(selected_files(model_info, ignore))
    try:
        required.update(_indexed_shards(snapshot, ignore))
    except (OSError, ValueError) as e:
        return [f"<weight index unreadable: {e}>"]
    missing: list[str] = sorted(n for n in required if not (snapshot / n).is_file())
    if missing:
        return missing
    has_weights: bool = any(
        p.name.endswith(_WEIGHT_SUFFIXES) and p.is_file() for p in snapshot.rglob("*")
    )
    return [] if has_weights else ["<no weight files>"]


def cached_snapshot(
    repo_id: str,
    ignore: list[str],
    model_info: ModelInfo | None,
    cache_dir: Path | None = None,
) -> Path | None:
    """The local snapshot if nothing the warmer downloads is missing; no network.

    ``local_files_only`` resolves ``refs/main`` and, when a tree listing is
    cached, already rejects missing files. A snapshot without one (such as the
    config-only snapshot the fetch step's AutoConfig call creates, or one cached
    by an older ``huggingface_hub``) is checked here instead.
    """
    try:
        path: Path = Path(
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=ignore,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        )
    except Exception:  # LocalEntryNotFoundError, IncompleteSnapshotError, ...
        return None
    try:
        return None if missing_files(path, model_info, ignore) else path
    except OSError:
        return None


def _snapshot_bytes(path: Path) -> int:
    """Size of the files a snapshot links to (its symlinks resolve to blobs)."""
    total: int = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _free_bytes(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(directory).free


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


@dataclass
class DownloadOutcome:
    path: Path | None
    attempts: int
    error: str | None
    out_of_time: bool = False


def _run_worker(repo_id: str, ignore: list[str], timeout: float) -> dict[str, object]:
    """One download attempt in a child process, killed after *timeout* seconds.

    Raises ``subprocess.TimeoutExpired`` once the child has been killed.
    """
    request: str = json.dumps({"repo_id": repo_id, "ignore_patterns": ignore})
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(_WORKER), request],
        stdout=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            outcome: object = json.loads(line[len(RESULT_PREFIX) :])
            if isinstance(outcome, dict):
                return outcome
    return {
        "ok": False,
        "error": f"download worker exited {proc.returncode} without a result",
        "status": None,
        "transient": True,
        "disk_full": False,
    }


def download_with_retries(
    repo_id: str,
    ignore: list[str],
    max_attempts: int,
    deadline: float,
    rate_limit_pause: float = _RATE_LIMIT_PAUSE_SECONDS,
) -> DownloadOutcome:
    """Download one repo, retrying transient errors, never past *deadline*.

    *deadline* is a ``time.monotonic()`` value. An attempt still running at the
    deadline is killed, and a retry wait that would end after it is not started;
    either way the outcome has ``out_of_time`` set.
    """
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        remaining: float = deadline - time.monotonic()
        if remaining <= 0:
            return DownloadOutcome(None, attempt - 1, last_error, out_of_time=True)
        try:
            outcome: dict[str, object] = _run_worker(repo_id, ignore, remaining)
        except subprocess.TimeoutExpired:
            return DownloadOutcome(
                None,
                attempt,
                "stopped at the --max-hours budget; the next run resumes it",
                out_of_time=True,
            )
        if outcome.get("ok"):
            return DownloadOutcome(Path(str(outcome["path"])), attempt, None)
        last_error = str(outcome.get("error"))
        if outcome.get("disk_full"):
            raise AbortRunError(f"disk full while downloading {repo_id}")
        if not outcome.get("transient"):
            return DownloadOutcome(None, attempt, last_error)
        if attempt == max_attempts:
            break
        wait: float = (
            rate_limit_pause
            if outcome.get("status") == HTTP_TOO_MANY_REQUESTS
            else min(_MAX_BACKOFF_SECONDS, 60.0 * 2 ** (attempt - 1))
        )
        if time.monotonic() + wait >= deadline:
            return DownloadOutcome(None, attempt, last_error, out_of_time=True)
        reason: str = (
            "rate limited (429)"
            if outcome.get("status") == HTTP_TOO_MANY_REQUESTS
            else "transient error"
        )
        print(
            f"    {reason} (attempt {attempt}/{max_attempts}): {last_error}; "
            f"retrying in {wait:.0f}s",
            flush=True,
        )
        time.sleep(wait)
    return DownloadOutcome(None, max_attempts, last_error)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@dataclass
class _Breaker:
    """Cools down after consecutive failures, and aborts if they keep coming."""

    failures_before_cooldown: int = 5
    cooldown_seconds: float = 1800.0
    max_cooldowns: int = 3
    consecutive_failures: int = 0
    consecutive_cooldowns: int = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_cooldowns = 0

    def record_failure(self) -> float:
        """Seconds to cool down before the next model (0 for none)."""
        self.consecutive_failures += 1
        if self.consecutive_failures < self.failures_before_cooldown:
            return 0.0
        self.consecutive_cooldowns += 1
        if self.consecutive_cooldowns > self.max_cooldowns:
            raise AbortRunError(
                f"{self.consecutive_failures} consecutive failures across "
                f"{self.max_cooldowns} cool-downs; the Hub or network looks unhealthy"
            )
        print(
            f"{ts()} {self.consecutive_failures} consecutive failures; cooling "
            f"down {self.cooldown_seconds:.0f}s "
            f"({self.consecutive_cooldowns}/{self.max_cooldowns})",
            flush=True,
        )
        self.consecutive_failures = 0
        return self.cooldown_seconds


def _write_summary(counts: dict[str, int], total_bytes: int, stop_reason: str) -> None:
    lines: list[str] = [
        f"{ts()} Done: {stop_reason}",
        *(f"    {status:<15} {n:>6,}" for status, n in sorted(counts.items())),
        f"    {'bytes':<15} {human_bytes(total_bytes):>6}",
    ]
    print("\n".join(lines), flush=True)
    summary_path: str | None = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        rows: str = "\n".join(f"| {s} | {n:,} |" for s, n in sorted(counts.items()))
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"## Warm top-models cache\n\n{stop_reason}\n\n| status | models |\n"
                f"|---|---|\n{rows}\n| bytes | {human_bytes(total_bytes)} |\n"
            )


def _budget_stop(done: int, total: int) -> str:
    return (
        f"--max-hours budget reached after {done:,}/{total:,} models; "
        "dispatch again to continue"
    )


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = _parse_args(argv)
    # The budget covers the whole run, the list fetch included.
    deadline: float = time.monotonic() + args.max_hours * 3600
    token: str | None = os.environ.get("HF_TOKEN") or None
    if token is None:
        print("⚠️ HF_TOKEN is not set; anonymous requests get a lower rate limit.")
    if not constants.HF_HUB_DISABLE_XET:
        print("⚠️ HF_HUB_DISABLE_XET is not set; Xet fetches file chunks in parallel.")

    cache_dir: Path = Path(constants.HF_HUB_CACHE)
    min_free: int = int(args.min_free_gb * 1024**3)
    print(
        f"{ts()} cache={cache_dir} free={human_bytes(_free_bytes(cache_dir))} "
        f"min_free={human_bytes(min_free)} sleep={args.sleep_seconds:.0f}s "
        f"max_hours={args.max_hours}",
        flush=True,
    )

    models: list[dict[str, object]] = get_model_list(args)
    api: HfApi = HfApi(token=token)
    counts: dict[str, int] = {}
    total_bytes: int = 0
    breaker: _Breaker = _Breaker()
    stop_reason: str = f"processed all {len(models):,} models"
    exit_code: int = 0

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_file, "a", encoding="utf-8") as report:
        for index, row in enumerate(models, start=1):
            if time.monotonic() >= deadline:
                stop_reason = _budget_stop(index - 1, len(models))
                break

            repo_id: str = str(row["model_id"])
            info: object = row.get("model_info")
            model_info: ModelInfo | None = info if isinstance(info, ModelInfo) else None
            result: ModelResult = ModelResult(
                index=index,
                model_id=repo_id,
                status="",
                downloads=as_int(row.get("downloads")),
                parameters=as_int(row.get("parameters")),
            )
            ignore: list[str] = ignore_patterns_for(model_info)
            prefix: str = f"{ts()} [{index:,}/{len(models):,}] {repo_id}"
            touched_network: bool = False

            try:
                free: int = _free_bytes(cache_dir)
                if cached_snapshot(repo_id, ignore, model_info) is not None:
                    result.status = "cached"
                elif free < min_free:
                    raise AbortRunError(
                        f"free space {human_bytes(free)} is below "
                        f"--min-free-gb ({human_bytes(min_free)})"
                    )
                else:
                    touched_network = True
                    size, method = download_size(api, repo_id, ignore, model_info)
                    result.estimated_bytes, result.size_method = size, method
                    if size is None:
                        result.status = "skipped_unsized"
                    elif free - size < min_free:
                        result.status = "skipped_disk"
                    else:
                        print(
                            f"{prefix}: downloading {human_bytes(size)} ({method})",
                            flush=True,
                        )
                        started: float = time.monotonic()
                        outcome: DownloadOutcome = download_with_retries(
                            repo_id, ignore, args.max_attempts, deadline
                        )
                        result.seconds = round(time.monotonic() - started, 1)
                        result.attempts = outcome.attempts
                        result.error = outcome.error
                        if outcome.path is not None:
                            result.status = "downloaded"
                            result.bytes_on_disk = _snapshot_bytes(outcome.path)
                            total_bytes += result.bytes_on_disk
                            breaker.record_success()
                        elif outcome.out_of_time:
                            result.status = "interrupted"
                        else:
                            result.status = "failed"
            except AbortRunError as e:
                stop_reason = f"aborted at {repo_id}: {e}"
                exit_code = 1
                break

            counts[result.status] = counts.get(result.status, 0) + 1
            report.write(json.dumps(asdict(result)) + "\n")
            report.flush()
            if result.status == "downloaded" and result.bytes_on_disk is not None:
                print(
                    f"{prefix}: downloaded {human_bytes(result.bytes_on_disk)} "
                    f"in {result.seconds}s",
                    flush=True,
                )
            else:
                print(f"{prefix}: {result.status}", flush=True)
            if result.error and result.status != "downloaded":
                print(f"    {result.error}", flush=True)

            if result.status == "interrupted":
                stop_reason = _budget_stop(index - 1, len(models))
                break
            pause: float = args.sleep_seconds if touched_network else 0.0
            if result.status == "failed":
                try:
                    pause = max(pause, breaker.record_failure())
                except AbortRunError as e:
                    stop_reason = f"aborted after {repo_id}: {e}"
                    exit_code = 1
                    break
            if pause > 0:
                if time.monotonic() + pause >= deadline:
                    stop_reason = _budget_stop(index, len(models))
                    break
                time.sleep(pause)

    _write_summary(counts, total_bytes, stop_reason)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
