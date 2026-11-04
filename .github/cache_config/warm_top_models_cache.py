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

Builds the same list the weekly scan shards are cut from — the top-K generative
and embedding models (``utils/fetch_top_*_models.py``) plus the curated lists,
minus models over the scan's parameter limit — and downloads each one into
``HF_HOME`` like ``manage_cache.py`` does. The weekly scan only deletes weights
it downloaded itself, so models warmed here stay cached between scans.

The network is treated gently, on purpose:

* one model at a time, and one file at a time inside it (``max_workers=1``);
* a fixed pause after every model that touched the network;
* transient errors retry with exponential backoff, and HTTP 429 waits longer;
* after several consecutive failures the run cools down, and gives up if the
  failures continue after repeated cool-downs.

Run it with ``HF_HUB_DISABLE_XET=1`` (the workflow sets this), so each file
arrives over a single plain HTTP stream instead of Xet's parallel chunk fetches.

Models already in the cache are skipped without a network call, so the run can
be stopped and dispatched again: it carries on where it stopped. A
``--max-hours`` budget makes it stop cleanly before the CI job timeout, and a
``--min-free-gb`` floor keeps it from filling the shared volume.

Usage::

    # Estimate the download size first (see estimate_top_models_cache_size.py)
    python .github/cache_config/estimate_top_models_cache_size.py

    python .github/cache_config/warm_top_models_cache.py \\
        --top-k-generative 10000 --top-k-embedding 10000 \\
        --sleep-seconds 15 --min-free-gb 500 --max-hours 70
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

# The project root must be importable before the tests/ and utils/ imports below:
# this file runs as a plain script from .github/cache_config/.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402
from huggingface_hub import constants, snapshot_download  # noqa: E402
from huggingface_hub.errors import (  # noqa: E402
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import ModelInfo  # noqa: E402

from tests.spyre.weekly_generation.failure_categories import (  # noqa: E402
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    PrefilterResult,
    prefilter_models,
)
from utils.fetch_top_embedding_models import (  # noqa: E402
    fetch_curated_embedding_models_metadata,
    fetch_top_embedding_models,
)
from utils.fetch_top_generative_models import (  # noqa: E402
    fetch_curated_generative_models_metadata,
    fetch_top_generative_models,
)
from utils.hf_model_catalog import (  # noqa: E402
    load_curated_embedding_models,
    load_curated_generative_models,
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

# Rough on-disk size per parameter (bf16/fp16), used only for the disk-space guard
# and the dry-run total. Unsized models count as 0 bytes.
_BYTES_PER_PARAM: int = 2

_DEFAULT_TOP_K: int = 10_000

_HTTP_TOO_MANY_REQUESTS: int = 429
_HTTP_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class ModelResult:
    """One line of the JSONL report."""

    index: int
    model_id: str
    status: str
    """cached | downloaded | failed | skipped_disk | planned"""
    downloads: int | None
    parameters: int | None
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
        "--top-k-generative",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"How many top generative models to fetch (default: {_DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--top-k-embedding",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"How many top embedding models to fetch (default: {_DEFAULT_TOP_K}).",
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
            "skip any model whose estimated size would cross it (default: 500)."
        ),
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=70.0,
        help="Stop cleanly after this much wall-clock time (default: 70).",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and filter the list and report the estimated size; download nothing.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Building the model list
# ---------------------------------------------------------------------------


def build_model_list(
    top_k_generative: int, top_k_embedding: int, max_params: int
) -> list[dict[str, object]]:
    """Curated models first, then the fetched top-K of both types by downloads.

    Each row gets a ``mode`` ("generative" or "embedding"). A model in both
    lists keeps its first row. The fetchers leave ``model_info`` on each row;
    its ``siblings`` decide which weight format to download (see
    ``ignore_patterns_for``).
    """
    sources: list[
        tuple[
            str,
            Callable[[], list[str]],
            Callable[..., list[dict[str, object]]],
            Callable[..., list[dict[str, object]]],
            int,
        ]
    ] = [
        (
            "generative",
            load_curated_generative_models,
            fetch_curated_generative_models_metadata,
            fetch_top_generative_models,
            top_k_generative,
        ),
        (
            "embedding",
            load_curated_embedding_models,
            fetch_curated_embedding_models_metadata,
            fetch_top_embedding_models,
            top_k_embedding,
        ),
    ]
    curated: list[dict[str, object]] = []
    fetched: list[dict[str, object]] = []
    for mode, load_ids, fetch_metadata, fetch_top, top_k in sources:
        model_ids: list[str] = load_ids()
        mode_rows: list[dict[str, object]] = (
            fetch_metadata(model_ids=model_ids) if model_ids else []
        )
        mode_rows = [{**row, "mode": mode, "curated": True} for row in mode_rows]
        curated.extend(mode_rows)
        if top_k > 0:
            fetched.extend(
                {**row, "mode": mode, "curated": False}
                for row in fetch_top(limit=top_k)
            )
    fetched.sort(key=lambda row: as_int(row.get("downloads")) or 0, reverse=True)

    merged: list[dict[str, object]] = concat_and_dedup_dicts(
        first=curated, second=fetched
    )
    result: PrefilterResult = prefilter_models(merged, max_params=max_params)
    print(
        f"{ts()} {len(curated)} curated + {len(fetched)} fetched -> {len(merged)} "
        f"unique -> {len(result.keep)} kept ({len(result.skipped)} over "
        f"{max_params:,} params)",
        flush=True,
    )
    return result.keep


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


# ---------------------------------------------------------------------------
# Cache and disk checks (no network)
# ---------------------------------------------------------------------------


def _cached_snapshot(repo_id: str, ignore_patterns: list[str]) -> Path | None:
    """The local snapshot if it is complete and holds weights, without network.

    ``local_files_only`` returns the snapshot that ``refs/main`` points at and
    raises if the cached tree listing shows a file missing. A snapshot with no
    tree listing, such as one the fetch step's AutoConfig call created, is
    accepted only if it holds at least one weight file.
    """
    try:
        path: Path = Path(
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        )
    except Exception:  # LocalEntryNotFoundError, IncompleteSnapshotError, ...
        return None
    try:
        has_weights: bool = any(
            p.name.endswith(_WEIGHT_SUFFIXES) for p in path.rglob("*")
        )
    except OSError:
        return None
    return path if has_weights else None


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


def _http_status(error: Exception) -> int | None:
    if isinstance(error, HfHubHTTPError) and error.response is not None:
        return error.response.status_code
    return None


def _is_transient(error: Exception) -> bool:
    """True for errors a later attempt can fix: rate limits, 5xx, network drops."""
    # snapshot_download raises LocalEntryNotFoundError when the Hub is unreachable.
    # Checked first: it subclasses EntryNotFoundError, which is permanent.
    if isinstance(error, LocalEntryNotFoundError):
        return True
    if isinstance(
        error, (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError)
    ):
        return False  # also covers GatedRepoError (a RepositoryNotFoundError)
    if isinstance(error, HfHubHTTPError):
        status: int | None = _http_status(error)
        return status is None or status in _HTTP_TRANSIENT_STATUSES
    return isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))


def _download_with_retries(
    repo_id: str,
    ignore_patterns: list[str],
    token: str | None,
    max_attempts: int,
    rate_limit_pause: float,
) -> tuple[Path | None, int, str | None]:
    """Download one repo. Returns (snapshot path or None, attempts, last error)."""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            path: Path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    ignore_patterns=ignore_patterns,
                    max_workers=1,
                )
            )
            return path, attempt, None
        except Exception as e:
            # HfHubHTTPError and LocalEntryNotFoundError are OSErrors too, so the
            # disk-full check goes by errno, not by exception type.
            if isinstance(e, OSError) and e.errno == errno.ENOSPC:
                raise AbortRunError(f"disk full while downloading {repo_id}") from e
            if not _is_transient(e):
                return None, attempt, f"{type(e).__name__}: {e}"
            last_error = f"{type(e).__name__}: {e}"
            if _http_status(e) == _HTTP_TOO_MANY_REQUESTS:
                print(
                    f"    rate limited (429); pausing {rate_limit_pause:.0f}s",
                    flush=True,
                )
                time.sleep(rate_limit_pause)
                continue
        if attempt < max_attempts:
            backoff: float = min(600.0, 60.0 * 2 ** (attempt - 1))
            print(
                f"    transient error (attempt {attempt}/{max_attempts}): "
                f"{last_error}; retrying in {backoff:.0f}s",
                flush=True,
            )
            time.sleep(backoff)
    return None, max_attempts, last_error


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

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures < self.failures_before_cooldown:
            return
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
        time.sleep(self.cooldown_seconds)
        self.consecutive_failures = 0


def _write_summary(counts: dict[str, int], total_bytes: int, stop_reason: str) -> None:
    lines: list[str] = [
        f"{ts()} Done: {stop_reason}",
        *(f"    {status:<13} {n:>6,}" for status, n in sorted(counts.items())),
        f"    {'bytes':<13} {human_bytes(total_bytes):>6}",
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


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = _parse_args(argv)
    token: str | None = os.environ.get("HF_TOKEN") or None
    if token is None:
        print("⚠️ HF_TOKEN is not set; anonymous requests get a lower rate limit.")
    if not constants.HF_HUB_DISABLE_XET:
        print("⚠️ HF_HUB_DISABLE_XET is not set; Xet fetches file chunks in parallel.")

    cache_dir: Path = Path(constants.HF_HUB_CACHE)
    min_free: int = int(args.min_free_gb * 1024**3)
    deadline: float = time.monotonic() + args.max_hours * 3600
    print(
        f"{ts()} cache={cache_dir} free={human_bytes(_free_bytes(cache_dir))} "
        f"min_free={human_bytes(min_free)} sleep={args.sleep_seconds:.0f}s "
        f"max_hours={args.max_hours} dry_run={args.dry_run}",
        flush=True,
    )

    models: list[dict[str, object]] = build_model_list(
        args.top_k_generative, args.top_k_embedding, args.max_params
    )
    counts: dict[str, int] = {}
    total_bytes: int = 0
    breaker: _Breaker = _Breaker()
    stop_reason: str = f"processed all {len(models):,} models"
    exit_code: int = 0

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_file, "a", encoding="utf-8") as report:
        for index, row in enumerate(models, start=1):
            if time.monotonic() >= deadline:
                stop_reason = (
                    f"--max-hours budget reached after {index - 1:,}/{len(models):,} "
                    "models; dispatch again to continue"
                )
                break

            repo_id: str = str(row["model_id"])
            info: object = row.get("model_info")
            model_info: ModelInfo | None = info if isinstance(info, ModelInfo) else None
            params: int | None = as_int(row.get("parameters"))
            result: ModelResult = ModelResult(
                index=index,
                model_id=repo_id,
                status="",
                downloads=as_int(row.get("downloads")),
                parameters=params,
            )
            ignore: list[str] = ignore_patterns_for(model_info)
            prefix: str = f"{ts()} [{index:,}/{len(models):,}] {repo_id}"

            try:
                cached: Path | None = _cached_snapshot(repo_id, ignore)
                estimate: int = (params or 0) * _BYTES_PER_PARAM
                # A dry run downloads nothing, so subtract what it has planned so far.
                free: int = _free_bytes(cache_dir) - (
                    total_bytes if args.dry_run else 0
                )
                if cached is not None:
                    result.status = "cached"
                elif free < min_free:
                    raise AbortRunError(
                        f"free space {human_bytes(free)} is below "
                        f"--min-free-gb ({human_bytes(min_free)})"
                    )
                elif free - estimate < min_free:
                    result.status = "skipped_disk"
                elif args.dry_run:
                    result.status = "planned"
                    result.bytes_on_disk = estimate
                    total_bytes += estimate
                else:
                    print(
                        f"{prefix}: downloading (~{human_bytes(estimate)})", flush=True
                    )
                    started: float = time.monotonic()
                    path, attempts, error = _download_with_retries(
                        repo_id,
                        ignore,
                        token,
                        args.max_attempts,
                        rate_limit_pause=600.0,
                    )
                    result.seconds = round(time.monotonic() - started, 1)
                    result.attempts = attempts
                    if path is not None:
                        result.status = "downloaded"
                        result.bytes_on_disk = _snapshot_bytes(path)
                        total_bytes += result.bytes_on_disk
                        breaker.record_success()
                    else:
                        result.status = "failed"
                        result.error = error
            except AbortRunError as e:
                stop_reason = f"aborted at {repo_id}: {e}"
                exit_code = 1
                break

            counts[result.status] = counts.get(result.status, 0) + 1
            report.write(json.dumps(asdict(result)) + "\n")
            report.flush()
            if result.status != "downloaded" or index % 25 == 0:
                print(f"{prefix}: {result.status}", flush=True)
            elif result.bytes_on_disk is not None:
                print(
                    f"{prefix}: downloaded {human_bytes(result.bytes_on_disk)} "
                    f"in {result.seconds}s",
                    flush=True,
                )
            if result.status == "failed":
                print(f"    {result.error}", flush=True)

            if result.status == "failed":
                try:
                    breaker.record_failure()
                except AbortRunError as e:
                    stop_reason = f"aborted after {repo_id}: {e}"
                    exit_code = 1
                    break
            if result.status in ("downloaded", "failed"):
                time.sleep(args.sleep_seconds)

    _write_summary(counts, total_bytes, stop_reason)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
