"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_generation/weekly_test.py --top-k 200
"""

import argparse
import gc
import multiprocessing
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from test_e2e_smoke_spyre import run_smoke_test

from hf_adapters import AutoSpyreModelForCausalLM

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from tests.conftest import get_dtype_for_cpu  # noqa: E402
from tests.spyre.test_e2e_embed_compare_spyre import embed_compare_spyre  # noqa: E402
from tests.spyre.test_e2e_token_compare_spyre import token_compare_spyre  # noqa: E402
from tests.spyre.test_load_spyre import load_causal_lm, load_embedding  # noqa: E402
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


class EmbeddingGenerativeMode(str, Enum):
    EMBEDDING = "embedding"
    GENERATIVE = "generative"


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
    loads_on_spyre = False
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
                    loads_on_spyre = _temp_random_bool()
                else:
                    model_is_not_none, callables, _ = load_causal_lm(
                        model_path=model_id
                    )
                    loads_on_spyre = model_is_not_none and callables

                    run_smoke_status = (
                        run_smoke_test(model_path=model_id)["status"] == "PASS"
                    )
                    mismatches, _ = token_compare_spyre(model_id)

    except Exception as e:
        print(f"eval_generative exception - {e}")
    finally:
        return {
            "correct": loads_on_spyre and run_smoke_status and not mismatches,
            "load": load_on_cpu,
        }


def eval_embedding(model_id: str, adapter, random_run: bool = False) -> dict:
    load_on_cpu = False
    loads_on_spyre = False
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
                    loads_on_spyre = _temp_random_bool()
                    mismatches = _temp_random_bool()
                else:
                    loads_on_spyre, _ = load_embedding(model_id)
                    mismatches, _ = embed_compare_spyre(model_id)
    except Exception as e:
        print(f"eval_embedding exception - {e}")
    finally:
        return {
            "correct": loads_on_spyre and not mismatches,
            "load": load_on_cpu,
            "compare_spyre": not mismatches,
        }


def _process_row(
    model_path: str,
    random_run: bool,
    adapter_dates: dict[str, str | None],
    result_queue,
    mode: Literal["embedding", "generative"],
) -> None:
    """Worker target for a raw spawn Process — no Pool machinery in the parent.

    Puts its result onto *result_queue* (a multiprocessing.Queue from the
    spawn context) so the parent can retrieve it after join(). Queues use an
    anonymous OS pipe inherited by the child at fork/spawn time — there is no
    on-disk socket that a `/tmp` cleaner (or an OOM-killed helper) can pull
    out from under us, which is why we avoid SyncManager here.

    *adapter_dates* is precomputed by the parent (one git log per adapter
    file, not per model) and passed in as a plain dict.
    """
    import traceback

    result: dict = {
        "adapter_name": "",
        "added_date": None,
        "metrics": {"correct": False, "load": False, "compare_spyre": False},
        "error": None,
    }
    try:
        from tests.conftest import resolve_adapter_module_for_test

        adapter_module = resolve_adapter_module_for_test(model_path)
        adapter_name: str = os.path.splitext(os.path.basename(adapter_module.__file__))[
            0
        ]
        result["adapter_name"] = adapter_name
        result["added_date"] = adapter_dates.get(adapter_name)
        if mode == "generative":
            eval_fn = eval_generative
        elif mode == "embedding":
            eval_fn = eval_embedding
        else:
            raise Exception(f"Unknown mode: {mode}")
        result["metrics"] = eval_fn(model_path, adapter_module, random_run=random_run)
    except Exception as e:
        result["error"] = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
    finally:
        try:
            result_queue.put(result)
            gc.collect()
        except Exception:
            pass


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


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    preexisting: set = _repos_with_weights()
    total_freed: int = 0
    snapshot_date = date.today()
    if args.mode == "generative":
        to_process_list = fetch_top_generative_models(limit=args.top_k)
    elif args.mode == "embedding":
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
        sink = ClickHouseResultSink(today=snapshot_date)

    total = len(to_process_list)
    processed = 0
    overall_start = time.monotonic()

    ctx = multiprocessing.get_context("spawn")
    try:
        for row in to_process_list:
            model_path = str(row["model_id"])
            processed += 1
            model_start = time.monotonic()
            elapsed_overall = model_start - overall_start
            print(
                f"\n[{processed}/{total}] {model_path}  "
                f"(overall elapsed: {elapsed_overall:.0f}s)"
            )

            if not sink.should_insert_row(model_path):
                print(
                    f"    sink: '{model_path}' skipped early — "
                    f"CPU-verified snapshot exists within the last "
                    f"{sink.__class__.__name__} skip window"
                )
                continue

            csv_config_class = row.get("config_class")

            had_weights = model_path in preexisting

            rec = {
                "model_name": model_path,
                "architecture": csv_config_class,
                "adapter_name": "",
                "added_date": None,
                "snapshot_date": snapshot_date,
                "verified_on_cpu": False,
                "verified_on_gpu": False,
                "verified_on_spyre": False,
                "num_downloads": int(row.get("downloads") or 0),
            }

            try:
                # Reject models too large to bring up on Spyre before spawning.
                params = row.get("parameters")
                if params not in (None, "") and int(params) > 60_000_000_000:
                    print(
                        f"Model {model_path} has {int(params):,} parameters, "
                        f"exceeding the 60B limit for Spyre bring-up."
                    )
                    continue

                # Use a raw Process instead of Pool so that no pipes,
                # semaphores, or pool-supervisor state accumulate in the
                # parent over 100+ iterations. A ctx.Queue uses an anonymous
                # OS pipe inherited by the child, so — unlike SyncManager —
                # there is no on-disk unix socket in /tmp that a pod cleaner
                # or OOM-killed helper can yank out from under us.
                result_queue = ctx.Queue()
                proc = ctx.Process(
                    target=_process_row,
                    args=(
                        model_path,
                        args.random_run,
                        args.mode,
                        adapter_dates,
                        result_queue,
                        args.mode,
                    ),
                )
                proc.start()
                proc.join()

                try:
                    worker_result = result_queue.get_nowait()
                except Exception:
                    worker_result = {
                        "adapter_name": "",
                        "added_date": None,
                        "metrics": {
                            "correct": False,
                            "load": False,
                            "compare_spyre": False,
                        },
                        "error": (
                            f"Worker process exited with code {proc.exitcode} "
                            f"and returned no result"
                        ),
                    }
                finally:
                    result_queue.close()
                    result_queue.join_thread()

                rec["adapter_name"] = worker_result["adapter_name"]
                if worker_result["added_date"] is not None:
                    rec["added_date"] = date.fromisoformat(worker_result["added_date"])

                if worker_result["error"]:
                    raise Exception(worker_result["error"])

                metrics = worker_result["metrics"]
                rec["verified_on_cpu"] = metrics.get("load", False)
                rec["verified_on_spyre"] = metrics.get("correct", False)
                model_elapsed = time.monotonic() - model_start
                print(
                    f"    verified_on_cpu={rec['verified_on_cpu']}  "
                    f"correct={metrics.get('correct')}  "
                    f"model_time={model_elapsed:.1f}s"
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                rec["verified_on_cpu"] = False
                model_elapsed = time.monotonic() - model_start
                print(
                    f"    verified_on_cpu=False  "
                    f"error={type(e).__name__}: {e}  "
                    f"model_time={model_elapsed:.1f}s"
                )
                print("".join(traceback.format_exc().splitlines(keepends=True)[-6:]))

            if sink.add_entry(rec):
                print(f"    sink: row written for '{model_path}'")
            else:
                print(f"    sink: row skipped for '{model_path}' (guard rejected)")

            # Cache cleanup: weights absent at start -> delete downloaded weights.
            if not had_weights:
                freed = _delete_repo_weights(model_path)
                total_freed += freed
                if freed:
                    print(
                        f"    freed {_human_bytes(freed)} "
                        f"(total {_human_bytes(total_freed)})"
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
