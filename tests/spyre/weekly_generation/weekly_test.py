"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_generation/weekly_test.py --top-k 200
"""

import argparse
import multiprocessing
import os
import random
import subprocess
import sys
import time
from asyncio import Queue
from datetime import date
from pathlib import Path

from tests.spyre.weekly_generation.result_sink import EmbeddingGenerativeMode

MAX_NUMBER_PARAMS = 60_000_000_000

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hf_adapters import AutoSpyreModelForCausalLM  # noqa: E402
from tests.conftest import get_dtype_for_cpu  # noqa: E402
from tests.spyre.test_e2e_embed_compare_spyre import embed_compare_spyre  # noqa: E402
from tests.spyre.test_e2e_smoke_spyre import run_smoke_test  # noqa: E402
from tests.spyre.test_e2e_token_compare_spyre import token_compare_spyre  # noqa: E402
from tests.spyre.weekly_generation.result_sink import (  # noqa: E402
    ClickHouseResultSink,
    CsvResultSink,
    ResultSink,
)
from utils.fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402
from utils.fetch_top_generative_models import fetch_top_generative_models  # noqa: E402

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
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS: int = 50
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

    Derived from the git add-date of each hf_adapters/hf_*.py file. Computed
    once per run in the parent and passed to workers, rather than re-running
    ~200 git subprocess calls (one per model).
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
        except Exception:
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
    parser.add_argument(
        "--random-run",
        action="store_true",
        default=False,
        help=(
            "Replace Spyre evaluation calls with random boolean stubs "
            "(_temp_random_bool). Useful for dry-runs without Spyre hardware."
        ),
    )
    return parser.parse_args(argv)


def _temp_random_bool() -> bool:
    return random.choice([True, False])


def _load_on_cpu(model_path: str, mode: EmbeddingGenerativeMode) -> bool:
    import hf_adapters.hf_common as _hf_common
    from hf_adapters.auto_spyre_model import AutoSpyreModel

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

        return model is not None
    except Exception as e:
        print(f"_load_embedding_on_cpu exception - {e}")
        return False
    finally:
        _hf_common.DEVICE = _orig_device  # restore


def eval_generative(model_id: str, adapter, random_run: bool = False) -> dict:
    load_on_cpu = False
    run_smoke_status = False
    mismatches = True

    """Load and compare token outputs for one generative model. Returns a metrics dict."""
    try:
        if adapter is not None:
            model_on_cpu = _load_on_cpu(
                model_path=model_id, mode=EmbeddingGenerativeMode.GENERATIVE
            )
            load_on_cpu = model_on_cpu is not None

            if load_on_cpu:
                if random_run:
                    run_smoke_status = _temp_random_bool()
                    mismatches = _temp_random_bool()
                else:
                    run_smoke_status = (
                        run_smoke_test(model_path=model_id)["status"] == "PASS"
                    )
                    mismatches, _ = token_compare_spyre(model_id)

    except Exception as e:
        print(f"eval_generative exception - {e}")
    finally:
        return {
            "correct": run_smoke_status and not mismatches,
            "load": load_on_cpu,
        }


def eval_embedding(model_id: str, adapter, random_run: bool = False) -> dict:
    load_on_cpu = False
    mismatches = True

    """Load and compare embeddings for one model. Returns a metrics dict."""
    try:
        if adapter is not None:
            # First we check that it is loadable on cpu:
            model_on_cpu = _load_on_cpu(
                model_path=model_id, mode=EmbeddingGenerativeMode.EMBEDDING
            )
            load_on_cpu = model_on_cpu is not None
            del model_on_cpu

            if load_on_cpu:
                if random_run:
                    mismatches = _temp_random_bool()
                else:
                    mismatches, _ = embed_compare_spyre(model_id)
    except Exception as e:
        print(f"eval_embedding exception - {e}")
    finally:
        return {
            "correct": not mismatches,
            "load": load_on_cpu,
        }


def _process_batch(
    batch: list[dict],
    random_run: bool,
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
            "architecture":     ...,
            "adapter_name":     ...,
            "added_date":       ...,   # ISO 8601 str or None
            "snapshot_date":    ...,   # date object
            "verified_on_cpu":  bool,
            "verified_on_gpu":  False,
            "verified_on_spyre": bool,
            "num_downloads":    int,
            "error":            None or str,
        }
    """
    import time as _t

    _child_entered: float = _t.monotonic()
    print(
        f"      child[{os.getpid()}] entered _process_batch with {len(batch)} model(s)",
        flush=True,
    )

    import traceback as _traceback

    from tests.conftest import resolve_adapter_module_for_test

    results: list[dict] = []
    for row in batch:
        model_path: str = str(row["model_id"])
        rec: dict = {
            "model_name": model_path,
            "architecture": row.get("config_class"),
            "adapter_name": "",
            "added_date": None,
            "snapshot_date": snapshot_date,
            "verified_on_cpu": False,
            "verified_on_gpu": False,
            "verified_on_spyre": False,
            "num_downloads": int(row.get("downloads") or 0),
            "error": None,
        }
        try:
            # Reject models too large to bring up on Spyre before evaluating.
            params = row.get("parameters")
            if params not in (None, "") and int(params) > MAX_NUMBER_PARAMS:
                raise Exception(
                    f"Model {model_path} has {int(params):,} parameters, "
                    f"exceeding the 60B limit for Spyre bring-up."
                )

            adapter_module = resolve_adapter_module_for_test(model_path)
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
            metrics = eval_fn(model_path, adapter_module, random_run=random_run)
            rec["verified_on_cpu"] = bool(metrics.get("load", False))
            rec["verified_on_spyre"] = bool(metrics.get("correct", False))
        except Exception as e:
            rec["error"] = (
                f"{type(e).__name__}: {e}\n"
                f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
            )
        results.append(rec)
        print(
            f"      child[{os.getpid()}] finished model "
            f"{len(results)}/{len(batch)}: {model_path!r}  "
            f"(verified_on_cpu={rec['verified_on_cpu']}, "
            f"verified_on_spyre={rec['verified_on_spyre']}, "
            f"error={bool(rec['error'])})",
            flush=True,
        )

    result_queue.put(results)
    print(
        f"      child[{os.getpid()}] done in "
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


def _delete_repo_weights(repo_id):
    """Delete cached weight files (and their blobs) for a repo. Keep configs.

    Returns bytes freed. Only touches files under the HF cache whose name ends
    in a weight suffix; resolves each snapshot symlink to its blob and unlinks
    both. Never touches datasets-- repos.
    """
    from huggingface_hub import scan_cache_dir

    if repo_id is None:
        return 0
    freed = 0
    try:
        cache = scan_cache_dir()
    except Exception:
        return 0
    for repo in cache.repos:
        if repo.repo_id != repo_id or repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            for fobj in rev.files:
                if not fobj.file_name.endswith(_WEIGHT_SUFFIXES):
                    continue
                snap = Path(fobj.file_path)
                # Resolve the blob (snapshot files are symlinks into blobs/).
                try:
                    blob = snap.resolve()
                    if blob.exists():
                        freed += blob.stat().st_size
                        blob.unlink()
                    if snap.is_symlink() or snap.exists():
                        snap.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"    warn: could not delete {snap}: {e}")
    return freed


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

    total = len(to_process_list)
    processed = 0
    overall_start = time.monotonic()

    ctx = multiprocessing.get_context("spawn")

    # Early-stop guard runs in the parent (fast: dict lookup for CSV,
    # single SELECT for CH), so we drop already-recent models BEFORE
    # batching. That keeps batch sizes uniform relative to real work.
    prefiltered: list[dict] = []
    early_skipped: int = 0
    for row in to_process_list:
        model_path: str = str(row["model_id"])
        if not sink.should_insert_row(model_path):
            early_skipped += 1
            print(
                f"    sink: '{model_path}' skipped early — "
                f"recent snapshot exists within the "
                f"{sink.__class__.__name__} skip window"
            )
            continue
        prefiltered.append(row)
    if early_skipped:
        print(
            f"\nEarly-skip: {early_skipped}/{total} models already have a "
            f"recent snapshot; {len(prefiltered)} left to evaluate.\n"
        )

    batches: list[list[dict]] = _chunk_into_batches(
        prefiltered, number_of_model_per_process
    )
    total_batches: int = len(batches)

    try:
        for batch_idx, batch in enumerate(batches, start=1):
            batch_start = time.monotonic()
            batch_paths: list[str] = [str(r["model_id"]) for r in batch]
            print(
                f"\n[batch {batch_idx}/{total_batches}] {len(batch)} model(s) "
                f"(overall elapsed: {batch_start - overall_start:.0f}s)"
            )
            for path in batch_paths:
                print(f"    - {path}")

            # Track which weights existed BEFORE this batch ran, so we can
            # decide per-model whether to delete after.
            had_weights_map: dict[str, bool] = {
                path: path in preexisting for path in batch_paths
            }

            result_queue = ctx.SimpleQueue()
            proc = ctx.Process(
                target=_process_batch,
                args=(
                    batch,
                    args.random_run,
                    adapter_dates,
                    result_queue,
                    args.mode,
                    snapshot_date,
                ),
            )
            t_spawn_start = time.monotonic()
            proc.start()
            t_spawn = time.monotonic() - t_spawn_start
            print(
                f"      parent: proc.start() returned pid={proc.pid} "
                f"after {t_spawn*1000:.1f}ms",
                flush=True,
            )

            t_join_start: float = time.monotonic()
            proc.join()
            t_join = time.monotonic() - t_join_start
            print(
                f"      parent: proc.join() returned pid={proc.pid} "
                f"after {t_join:.2f}s",
                flush=True,
            )

            # Drain the queue. SimpleQueue.get() blocks if empty, so probe
            # first; the child has already exited so a non-empty queue
            # returns instantly, and an empty one signals a crash.
            if result_queue.empty():
                print(
                    f"    batch: worker exited code {proc.exitcode} "
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
                        "error": f"worker died (exitcode={proc.exitcode})",
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
                    print(f"    [{model_path}] error: {rec['error']}")

                # Coerce added_date from ISO string (as the worker wrote it)
                # to a date object for the sink.
                added_iso = rec.get("added_date")
                if isinstance(added_iso, str):
                    try:
                        rec["added_date"] = date.fromisoformat(added_iso)
                    except ValueError:
                        rec["added_date"] = None

                # Sink writes: don't store the `error` field.
                sink_rec: dict = {k: v for k, v in rec.items() if k != "error"}
                if sink.add_entry(sink_rec):
                    print(
                        f"    sink: row written for '{model_path}' "
                        f"(verified_on_cpu={rec.get('verified_on_cpu')}, "
                        f"verified_on_spyre={rec.get('verified_on_spyre')})"
                    )
                else:
                    print(f"    sink: row skipped for '{model_path}' (guard rejected)")

                # Cache cleanup: weights absent at start -> delete downloaded.
                if not had_weights_map.get(model_path, False):
                    freed = _delete_repo_weights(model_path)
                    total_freed += freed
                    if freed:
                        print(
                            f"    freed {_human_bytes(freed)} "
                            f"(total {_human_bytes(total_freed)})"
                        )

            batch_elapsed = time.monotonic() - batch_start
            print(
                f"    batch {batch_idx}/{total_batches} done: "
                f"{len(worker_results)} model(s) in {batch_elapsed:.1f}s  "
                f"(per-model avg: {batch_elapsed / max(1, len(worker_results)):.1f}s)"
            )
    except KeyboardInterrupt:
        print("\nInterrupted — results so far are saved; rerun to resume.")
    finally:
        sink.close()
        if args.write_to_csv:
            print(f"\nCSV: '{args.write_to_csv}' closed ({processed} rows processed).")

        overall_elapsed = time.monotonic() - overall_start
        mins, secs = divmod(int(overall_elapsed), 60)
        print(
            f"\n{'='*60}\n"
            f"Processed {processed}/{total} models  |  "
            f"Total time: {mins}m {secs:02d}s\n"
            f"{'='*60}"
        )


if __name__ == "__main__":
    main()
