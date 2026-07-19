"""Weekly Spyre evaluation suite.

Evaluates the top-k generative or embedding models against Spyre hardware.
Run directly with a required ``--mode`` argument::

    python tests/spyre/weekly_generation/weekly_test.py --mode generative [--top-k 200]
    python tests/spyre/weekly_generation/weekly_test.py --mode embedding  [--top-k 200]

``--mode generative`` fetches top causal-LM models and runs the CPU-load +
token-compare pipeline.  ``--mode embedding`` fetches top embedding models
and runs the CPU-load + cosine-compare pipeline.

Additional flags:

* ``--top-k N``        Number of top models to fetch by download count (default: 200).
* ``--write-to-csv F`` Write results to a CSV file instead of ClickHouse.
"""

import argparse
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback as _traceback
from asyncio import Queue
from datetime import date, datetime
from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError

from tests.spyre.weekly_generation.result_sink import (
    EmbeddingGenerativeMode,
)


def _ts() -> str:
    """Local-time timestamp prefix, e.g. '[2026-07-17 14:32:05]'."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


logging.getLogger("transformers").setLevel(logging.ERROR)

MAX_NUMBER_PARAMS = 60_000_000_000

FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER = "not-implemented-adapter"
FAILURE_CATEGORY_MODEL_TOO_LARGE = "model_too_large"
FAILURE_CATEGORY_CPU_LOAD_FAILED = "cpu_load_failed"
FAILURE_CATEGORY_QUANTIZED_MODEL = "quantized_model"
FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION = "test_execution_exception"
FAILURE_CATEGORY_VERIFICATION_FAILED = "verification_failed"
FAILURE_CATEGORY_WORKER_CRASHED = "worker_crashed"


def _classify_failure(err: str, default: str) -> str:
    """Bucket a raw error/traceback string into a failure_category.

    Runtime signal for quantized checkpoints — bitsandbytes / AWQ / GPTQ error
    text almost always contains 'quantiz', and 'optimum' catches the
    optimum-quanto / optimum-neuron loaders. Anything unrecognised falls
    through to *default* (usually the surrounding context's fallback:
    cpu_load_failed at load time, test_execution_exception at eval time).
    """
    if not err:
        return default
    lowered: str = err.lower()
    if "quantiz" in lowered or "optimum" in lowered:
        return FAILURE_CATEGORY_QUANTIZED_MODEL
    return default


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Weight-file suffixes. A repo with at least one of these cached "has weights";
# a repo with only config/tokenizer files does not, so its later-downloaded
# weights are eligible for deletion.
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
)

# How many models one spawned child processes before exiting. Higher values
# amortize per-child spawn + import + kernel-teardown cost (~15 s currently)
# across more work. Lower values reduce the blast radius when the Spyre
# driver/state gets into a bad shape mid-batch.
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS: int = 2
EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS: int = 90


def _repos_with_weights():
    """Set of repo_ids that already have >=1 weight file cached at startup."""
    from huggingface_hub import scan_cache_dir

    have = set()
    try:
        cache = scan_cache_dir()
    except Exception:
        return have
    for repo in cache.repos:
        for rev in repo.revisions:
            if any(fobj.file_name.endswith(_WEIGHT_SUFFIXES) for fobj in rev.files):
                have.add(repo.repo_id)
                break
    return have


def _get_adapter_dates() -> dict[str, str | None]:
    """Map adapter module name (e.g. 'hf_qwen3') -> ISO date it was first added.

    Derived from the git add-date of each hf_adapters/hf_*.py file.
    """
    dates: dict[str, str | None] = {}
    adapter_dir: Path = _REPO_ROOT / "hf_adapters"
    for f in sorted(adapter_dir.glob("hf_*.py")):
        module_name: str = f.stem
        try:
            out = subprocess.run(
                [
                    "git",
                    "log",
                    "--diff-filter=A",
                    "--follow",
                    "--format=%aI",
                    "-1",
                    "--",
                    str(f.relative_to(_REPO_ROOT)),
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            iso: list[str] = out.stdout.strip().splitlines()
            dates[module_name] = iso[-1][:10] if iso else None
        except OSError:
            dates[module_name] = None
    return dates


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["embedding", "generative"],
        help=(
            "Which model class to evaluate: 'embedding' runs the "
            "embedding load + cosine-compare pipeline; 'generative' runs the "
            "causal-LM load + token-compare pipeline."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of top models to fetch (by downloads).",
    )
    parser.add_argument(
        "--write-to-csv",
        type=Path,
        default=None,
        metavar="RESULTS_CSV",
        help=(
            "Write evaluation results to this CSV file instead of inserting "
            "into ClickHouse. No DB connection is made when this flag is set."
        ),
    )
    return parser.parse_args(argv)


def _load_on_cpu(
    model_path: str, mode: EmbeddingGenerativeMode
) -> tuple[bool, str | None]:
    """Try to load *model_path* on CPU. Returns ``(loaded, error_message)``.

    ``error_message`` is ``None`` on success. On failure it carries a
    ``"ExcType: message\\n<tail traceback>"`` string that the caller can
    stash into the row's ``error`` field. Transient HF 5xx propagate — the
    driver retries at a higher level.
    """
    import hf_adapters.hf_common as _hf_common
    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.auto_spyre_model import AutoSpyreModel
    from tests.conftest import get_dtype_for_cpu

    _orig_device = _hf_common.DEVICE  # save
    _hf_common.DEVICE = "cpu"  # patch
    try:
        dtype = get_dtype_for_cpu(model_path)
        model = None
        match mode:
            case "embedding":
                model = AutoSpyreModel.from_pretrained(model_path, dtype=dtype)
            case "generative":
                model = AutoSpyreModelForCausalLM.from_pretrained(
                    model_path, dtype=dtype
                )

        return model is not None, None
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code >= 500:
            raise
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_load_on_cpu exception - {e}")
        return False, err
    except Exception as e:
        err = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_load_on_cpu exception - {e}")
        return False, err
    finally:
        _hf_common.DEVICE = _orig_device  # restore


def eval_generative(model_id: str, adapter) -> dict:
    from tests.spyre.test_e2e_smoke_spyre import run_smoke_test
    from tests.spyre.test_e2e_token_compare_spyre import token_compare_spyre

    load_on_cpu = False
    run_smoke_status = False
    mismatches = True
    result = {"error": "", "failure_category": None}

    """Load and compare token outputs for one generative model. Returns a metrics dict."""
    try:
        if adapter is not None:
            load_on_cpu, load_error = _load_on_cpu(
                model_path=model_id, mode=EmbeddingGenerativeMode.GENERATIVE
            )
            if load_error and not result["error"]:
                result["error"] = load_error
            if load_on_cpu:
                run_smoke_status = (
                    run_smoke_test(model_path=model_id)["status"] == "PASS"
                )
                mismatches, _ = token_compare_spyre(model_id)

    except Exception as e:
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        result["error"] = err
        result["failure_category"] = _classify_failure(
            err, FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION
        )
    finally:
        result["correct"] = run_smoke_status and not mismatches
        result["load"] = load_on_cpu
        if result["failure_category"] is None and load_on_cpu and not result["correct"]:
            result["failure_category"] = FAILURE_CATEGORY_VERIFICATION_FAILED
        return result


def eval_embedding(model_id: str, adapter) -> dict:
    from tests.spyre.test_e2e_embed_compare_spyre import embed_compare_spyre

    load_on_cpu = False
    mismatches = True
    result = {"error": "", "failure_category": None}

    """Load and compare embeddings for one model. Returns a metrics dict."""
    try:
        if adapter is not None:
            # First we check that it is loadable on cpu:
            load_on_cpu, load_error = _load_on_cpu(
                model_path=model_id, mode=EmbeddingGenerativeMode.EMBEDDING
            )
            if load_error and not result["error"]:
                result["error"] = load_error

            if load_on_cpu:
                mismatches, _ = embed_compare_spyre(model_id)
    except Exception as e:
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        result["error"] = err
        result["failure_category"] = _classify_failure(
            err, FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION
        )
    finally:
        result["correct"] = not mismatches
        result["load"] = load_on_cpu
        if result["failure_category"] is None and load_on_cpu and not result["correct"]:
            result["failure_category"] = FAILURE_CATEGORY_VERIFICATION_FAILED
        return result


def _process_batch(
    batch: list[dict],
    adapter_dates: dict[str, str | None],
    result_queue: Queue,
    mode: EmbeddingGenerativeMode,
    snapshot_date: date,
) -> None:
    """Worker target: evaluate up to ``NUMBER_OF_MODEL_PER_PROCESS`` models
    in a single spawned child.

    Amortizes the per-child fixed cost (spawn + module imports + kernel
    teardown on exit) across N models. Puts a ``list[dict]`` on the queue —
    one full result dict per row, in the same order as *batch*. If a single
    model errors, its ``error`` field is populated and the loop continues to
    the next model; the child does NOT abort.

    Each returned dict has the same shape ``main`` expects for a rec plus an
    ``error`` field (str or None):

        {
            "model_name":       ...,
            "config_class":     ...,
            "adapter_name":     ...,
            "added_date":       ...,   # ISO 8601 str or None
            "snapshot_date":    ...,   # date object
            "verified_on_cpu":  bool,
            "verified_on_gpu":  False,
            "verified_on_spyre": bool,
            "num_downloads":    int,
            "family":           str,
            "architecture":     str,
            "parameters_number": int,
            "error":            None or str,
            "failure_category": None or str,
        }
    """
    import time as _t

    _child_entered: float = _t.monotonic()
    print(
        f"{_ts()}       child[{os.getpid()}] entered _process_batch with {len(batch)} model(s)",
        flush=True,
    )

    from tests.conftest import resolve_adapter_module_for_test

    results: list[dict] = []
    for row in batch:
        model_path: str = str(row["model_id"])
        rec: dict = {
            "model_name": model_path,
            "config_class": row.get("config_class"),
            "adapter_name": "",
            "added_date": None,
            "snapshot_date": snapshot_date,
            "verified_on_cpu": False,
            "verified_on_gpu": False,
            "verified_on_spyre": False,
            "num_downloads": int(row.get("downloads") or 0),
            "family": str(row.get("model_type") or ""),
            "architecture": str(row.get("architectures") or ""),
            "parameters_number": int(row.get("parameters") or 0),
            "error": None,
            "failure_category": None,
        }
        try:
            # Reject models too large to bring up on Spyre before evaluating.
            params = row.get("parameters")
            if params not in (None, "") and int(params) > MAX_NUMBER_PARAMS:
                rec["failure_category"] = FAILURE_CATEGORY_MODEL_TOO_LARGE
                raise Exception(
                    f"Model {model_path} has {int(params):,} parameters, "
                    f"exceeding the 60B limit for Spyre bring-up."
                )

            try:
                adapter_module = resolve_adapter_module_for_test(model_path)
            except Exception:
                rec["failure_category"] = FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
                raise
            adapter_name: str = os.path.splitext(
                os.path.basename(adapter_module.__file__)
            )[0]
            rec["adapter_name"] = adapter_name
            rec["added_date"] = adapter_dates.get(adapter_name)

            match mode:
                case EmbeddingGenerativeMode.EMBEDDING:
                    eval_fn = eval_embedding
                case EmbeddingGenerativeMode.GENERATIVE:
                    eval_fn = eval_generative
            metrics = eval_fn(model_path, adapter_module)
            rec["verified_on_cpu"] = bool(metrics.get("load", False))
            rec["verified_on_spyre"] = bool(metrics.get("correct", False))
            rec["error"] = metrics.get("error") or None
            rec["failure_category"] = metrics.get("failure_category") or None
            if not rec["verified_on_cpu"] and rec["failure_category"] is None:
                rec["failure_category"] = _classify_failure(
                    rec["error"] or "", FAILURE_CATEGORY_CPU_LOAD_FAILED
                )
        except Exception as e:
            # Skip the error/traceback for shallow failure categories where the
            # failure_category itself is fully self-describing.
            if rec["failure_category"] not in (
                FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
                FAILURE_CATEGORY_MODEL_TOO_LARGE,
            ):
                rec["error"] = (
                    f"{type(e).__name__}: {e}\n"
                    f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
                )
            if rec["failure_category"] is None:
                rec["failure_category"] = FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION
        results.append(rec)
        print(
            f"{_ts()}       child[{os.getpid()}] finished model "
            f"{len(results)}/{len(batch)}: {model_path!r}  "
            f"(verified_on_cpu={rec['verified_on_cpu']}, "
            f"verified_on_spyre={rec['verified_on_spyre']}, "
            f"failure_category={rec['failure_category']}, "
            f"error={rec['error']})",
            flush=True,
        )

    result_queue.put(results)
    print(
        f"{_ts()}       child[{os.getpid()}] done in "
        f"{_t.monotonic() - _child_entered:.2f}s ({len(results)} results)",
        flush=True,
    )

    # Skip Python's graceful shutdown: no atexit handlers, no thread
    # finalization, no torch/torch_spyre destructors walking the tensor graph
    # that the kernel is about to reclaim in bulk anyway. Closing the Spyre
    # device FD on _exit(2) triggers the driver's own release path (VFIO
    # unmap-all + IOMMU teardown), which is what actually returns the
    # accelerator memory. Prior measurements: leaving Python's graceful
    # shutdown in place cost ~30 s per child; running gc.collect() here on
    # top of that added another ~20 s.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _repo_cache_dir(repo_id: str) -> Path:
    """Return the local cache folder for a model repo without scanning the cache.

    HF layout: <HF_HUB_CACHE>/models--<org>--<name>
    e.g. 'BAAI/bge-m3' -> '<cache>/models--BAAI--bge-m3'
    """
    from huggingface_hub import constants

    folder_name = "models--" + repo_id.replace("/", "--")
    return Path(constants.HF_HUB_CACHE) / folder_name


def _delete_repo_weights(repo_id_list: list[str]) -> int:
    """Delete cached weight files (and their blobs) for a list of repos. Keep configs.

    Returns bytes freed. Navigates directly to each repo's cache folder using
    the known HF layout — avoids the expensive scan_cache_dir() call entirely.
    Only touches files whose name ends in a weight suffix; resolves each
    snapshot symlink to its blob and unlinks both.
    """
    if not repo_id_list:
        return 0
    freed = 0
    for repo_id in repo_id_list:
        repo_dir = _repo_cache_dir(repo_id)
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snap in snapshots_dir.rglob("*"):
            if not snap.name.endswith(_WEIGHT_SUFFIXES):
                continue
            try:
                blob = snap.resolve()
                if blob.exists():
                    freed += blob.stat().st_size
                    blob.unlink()
                if snap.is_symlink() or snap.exists():
                    snap.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"    warn: could not delete {snap}: {e}")
    return freed


def _cleanup_batch_weights(
    batch_paths: list[str],
    had_weights_map: dict[str, bool],
    total_freed: int,
) -> int:
    """Delete downloaded weights for paths that were not pre-cached at startup.

    Returns the updated total_freed byte count.
    """
    to_delete: list[str] = [
        path for path in batch_paths if not had_weights_map.get(path, False)
    ]
    freed = _delete_repo_weights(to_delete)
    total_freed += freed
    if freed:
        print(f"    freed {_human_bytes(freed)} (total {_human_bytes(total_freed)})")
    return total_freed


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _chunk_into_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    """Split *rows* into consecutive sub-lists of length *batch_size* (the
    last batch may be shorter).
    """
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def main(argv: list[str] | None = None) -> None:
    from tests.spyre.weekly_generation.result_sink import (
        ClickHouseResultSink,
        CsvResultSink,
        ResultSink,
    )
    from utils.fetch_top_embedding_models import fetch_top_embedding_models
    from utils.fetch_top_generative_models import fetch_top_generative_models

    args = _parse_args(argv)
    preexisting: set = _repos_with_weights()
    total_freed: int = 0
    snapshot_date = date.today()
    if args.mode == "generative":
        mode = EmbeddingGenerativeMode.GENERATIVE
        number_of_model_per_process = GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS
        to_process_list = fetch_top_generative_models(limit=args.top_k)
    elif args.mode == "embedding":
        mode = EmbeddingGenerativeMode.EMBEDDING
        number_of_model_per_process = EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS
        to_process_list = fetch_top_embedding_models(limit=args.top_k)
    else:
        raise Exception(f"Unknown mode: {args.mode}")
    adapter_dates: dict[str, str | None] = _get_adapter_dates()

    sink: ResultSink
    if args.write_to_csv:
        sink = CsvResultSink(path=args.write_to_csv, today=snapshot_date)
        print(
            f"CSV mode: results will be appended to '{args.write_to_csv}' (no DB access).\n"
        )
    else:
        sink = ClickHouseResultSink(today=snapshot_date, embedding_generative=mode)
        print("DB mode: results will be appended to the DB.\n")
    print("argsv = {}\n".format(args))
    total = len(to_process_list)
    processed = 0
    overall_start = time.monotonic()

    # Early-stop guard runs in the parent (fast: dict lookup for CSV,
    # single SELECT for CH), so we drop already-recent models BEFORE
    # batching. That keeps batch sizes uniform relative to real work.
    prefiltered: list[dict] = []
    early_skipped: int = 0
    print(f"{_ts()} Will process {len(to_process_list)} models in total.")
    for row in to_process_list:
        model_path: str = str(row["model_id"])
        if not sink.should_insert_row(model_path):
            early_skipped += 1
            print(
                f"{_ts()}     sink: '{model_path}' skipped early — "
                f"recent snapshot exists within the "
                f"{sink.__class__.__name__} skip window"
            )
            continue
        prefiltered.append(row)
    if early_skipped:
        print(
            f"\n{_ts()} Early-skip: {early_skipped}/{total} models already have a "
            f"recent snapshot; {len(prefiltered)} left to evaluate.\n"
        )

    batches: list[list[dict]] = _chunk_into_batches(
        prefiltered, number_of_model_per_process
    )
    total_batches: int = len(batches)

    ctx = multiprocessing.get_context("spawn")

    # Initialised here so the finally block can clean up the in-flight batch
    # even when KeyboardInterrupt fires mid-batch.
    batch_paths: list[str] = []
    had_weights_map: dict[str, bool] = {}

    try:
        for batch_idx, batch in enumerate(batches, start=1):
            batch_start = time.monotonic()
            batch_paths = [str(r["model_id"]) for r in batch]
            print(
                f"\n{_ts()} [batch {batch_idx}/{total_batches}] {len(batch)} model(s) "
                f"(overall elapsed: {batch_start - overall_start:.0f}s)"
            )
            for path in batch_paths:
                print(f"{_ts()}     - {path}")

            # Track which weights existed BEFORE this batch ran, so we can
            # decide per-model whether to delete after.
            had_weights_map = {path: path in preexisting for path in batch_paths}

            result_queue = ctx.SimpleQueue()
            proc = ctx.Process(
                target=_process_batch,
                args=(
                    batch,
                    adapter_dates,
                    result_queue,
                    mode,
                    snapshot_date,
                ),
            )
            proc.start()

            proc.join()

            # Drain the queue. SimpleQueue.get() blocks if empty, so probe
            # first; the child has already exited so a non-empty queue
            # returns instantly, and an empty one signals a crash.
            if result_queue.empty():
                print(
                    f"{_ts()}     batch: worker exited code {proc.exitcode} "
                    f"and returned no results — marking all {len(batch)} "
                    f"models as failed"
                )
                worker_results: list[dict] = [
                    {
                        "model_name": path,
                        "config_class": row.get("config_class"),
                        "adapter_name": "",
                        "added_date": None,
                        "snapshot_date": snapshot_date,
                        "verified_on_cpu": False,
                        "verified_on_gpu": False,
                        "verified_on_spyre": False,
                        "num_downloads": int(row.get("downloads") or 0),
                        "family": str(row.get("model_type") or ""),
                        "architecture": str(row.get("architectures") or ""),
                        "parameters_number": int(row.get("parameters") or 0),
                        "error": f"worker died (exitcode={proc.exitcode})",
                        "failure_category": FAILURE_CATEGORY_WORKER_CRASHED,
                    }
                    for row, path in zip(batch, batch_paths)
                ]
            else:
                worker_results = result_queue.get()

            # Write each result to the sink and clean up cache per model.
            for rec in worker_results:
                model_path = str(rec.get("model_name") or "")
                processed += 1

                if rec.get("error"):
                    print(f"{_ts()}     [{model_path}] error: {rec['error']}")

                # Coerce added_date from ISO string (as the worker wrote it)
                # to a date object for the sink.
                added_iso = rec.get("added_date")
                if isinstance(added_iso, str):
                    try:
                        rec["added_date"] = date.fromisoformat(added_iso)
                    except ValueError:
                        rec["added_date"] = None

                if sink.add_entry(
                    model_name=str(rec["model_name"]),
                    config_class=str(rec["config_class"]),
                    adapter_name=str(rec["adapter_name"]),
                    added_date=rec["added_date"],
                    snapshot_date=rec["snapshot_date"],
                    verified_on_cpu=bool(rec["verified_on_cpu"]),
                    verified_on_gpu=bool(rec["verified_on_gpu"]),
                    verified_on_spyre=bool(rec["verified_on_spyre"]),
                    num_downloads=int(rec["num_downloads"]),
                    family=str(rec["family"]),
                    architecture=str(rec["architecture"]),
                    parameters_number=int(rec["parameters_number"]),
                    failure_category=(
                        None
                        if rec.get("failure_category") is None
                        else str(rec["failure_category"])
                    ),
                    error=(None if rec.get("error") is None else str(rec["error"])),
                ):
                    print(
                        f"{_ts()}     sink: row written for '{model_path}' "
                        f"(verified_on_cpu={rec.get('verified_on_cpu')}, "
                        f"verified_on_spyre={rec.get('verified_on_spyre')})"
                    )
                else:
                    print(
                        f"{_ts()}     sink: row skipped for '{model_path}' (guard rejected)"
                    )

            # Cache cleanup: delete weights downloaded during this batch,
            # regardless of whether the worker processed each model.
            total_freed = _cleanup_batch_weights(
                batch_paths, had_weights_map, total_freed
            )

            batch_elapsed = time.monotonic() - batch_start
            print(
                f"{_ts()}     batch {batch_idx}/{total_batches} done: "
                f"{len(worker_results)} model(s) in {batch_elapsed:.1f}s  "
                f"(per-model avg: {batch_elapsed / max(1, len(worker_results)):.1f}s)"
            )

            # Durability boundary: flush accumulated rows now so a hard parent
            # crash before the next batch loses at most this batch, not the
            # whole run. No-op for sinks that write per-row (CSV).
            sink.flush()
    except KeyboardInterrupt:
        print(f"\n{_ts()} Interrupted — results so far are saved; rerun to resume.")
    finally:
        # Clean up weights for the in-flight batch if interrupted mid-run.
        _ = _cleanup_batch_weights(batch_paths, had_weights_map, total_freed)

        sink.close()
        if args.write_to_csv:
            print(
                f"\n{_ts()} CSV: '{args.write_to_csv}' closed ({processed} rows processed)."
            )

        overall_elapsed = time.monotonic() - overall_start
        mins, secs = divmod(int(overall_elapsed), 60)
        print(
            f"\n{'='*60}\n"
            f"{_ts()} Processed {processed}/{total} models  |  "
            f"Total time: {mins}m {secs:02d}s\n"
            f"{'='*60}"
        )


if __name__ == "__main__":
    print(f"{_ts()} Starting weekly generation...", flush=True)
    main()
