"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_test.py --top-k 200
"""

import argparse
import csv
import multiprocessing
import multiprocessing.managers
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from tests.conftest import get_dtype_for_cpu  # noqa: E402
from tests.spyre.test_e2e_embed_compare_spyre import embed_compare_spyre  # noqa: E402
from tests.spyre.test_load_spyre import load_embedding  # noqa: E402
from tests.spyre.weekly_generation.create_model_spyre_table import (  # noqa: E402
    CREATE_TABLE_SQL,
    get_client,
    insert_model_row,
    table_exists,
)
from utils.fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402

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


def _adapter_add_dates() -> dict[str, str]:
    """Map adapter module name (e.g. 'hf_qwen3') -> ISO date it was first added.

    Derived from the git add-date of each hf_adapters/hf_*.py file.
    """
    dates = {}
    adapter_dir = _REPO_ROOT / "hf_adapters"
    for f in sorted(adapter_dir.glob("hf_*.py")):
        module_name = f.stem
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
            iso = out.stdout.strip().splitlines()
            dates[module_name] = iso[-1][:10] if iso else None
        except Exception:
            dates[module_name] = None
    return dates


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of top embedding models to fetch (by downloads).",
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
        "--boolean-run",
        action="store_true",
        default=False,
        help=(
            "Replace Spyre evaluation calls with random boolean stubs "
            "(_temp_boolean_random). Useful for dry-runs without Spyre hardware."
        ),
    )
    return parser.parse_args(argv)


def _temp_boolean_random() -> bool:
    return random.choice([True, False])


def _load_embedding_on_cpu(model_path: str) -> bool:
    import hf_adapters.hf_common as _hf_common
    from hf_adapters.auto_spyre_model import AutoSpyreModel

    _orig_device = _hf_common.DEVICE  # save
    _hf_common.DEVICE = "cpu"  # patch
    try:
        dtype = get_dtype_for_cpu(model_path)
        model = AutoSpyreModel.from_pretrained(model_path, dtype=dtype)
        return model is not None
    except Exception as e:
        print(f"_load_embedding_on_cpu exception - {e}")
        return False
    finally:
        _hf_common.DEVICE = _orig_device  # restore


def eval_embedding(model_id: str, adapter, boolean_run: bool = False) -> dict:
    load_on_cpu = False
    loads_on_spyre = False
    mismatches = True

    """Load and compare embeddings for one model. Returns a metrics dict."""
    try:
        if adapter is not None:
            # First we check that it is loadable on cpu:
            model_on_cpu = _load_embedding_on_cpu(model_path=model_id)
            load_on_cpu = model_on_cpu is not None
            del model_on_cpu

            if load_on_cpu:
                if boolean_run:
                    loads_on_spyre = _temp_boolean_random()
                    mismatches = _temp_boolean_random()
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


def _process_row(model_path: str, boolean_run: bool, return_dict) -> None:
    """Worker target for a raw spawn Process — no Pool machinery in the parent.

    Writes its result into *return_dict* (a multiprocessing.Manager dict) so
    the parent can retrieve it after join(). Using a raw Process means zero
    persistent file-descriptors, semaphores, or pool-supervisor threads
    accumulate in the parent across iterations.
    """
    import traceback

    result = {
        "adapter_name": "",
        "added_date": None,
        "metrics": {"correct": False, "load": False, "compare_spyre": False},
        "error": None,
    }
    try:
        from tests.conftest import resolve_adapter_module_for_test

        adapter_module = resolve_adapter_module_for_test(model_path)
        result["adapter_name"] = os.path.splitext(
            os.path.basename(adapter_module.__file__)
        )[0]

        # add_date resolved inside the worker to avoid pickling issues
        import subprocess as _sp
        from pathlib import Path

        adapter_file = Path(adapter_module.__file__)
        repo_root = adapter_file.parents[2]
        out = _sp.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--follow",
                "--format=%aI",
                "-1",
                "--",
                str(adapter_file.relative_to(repo_root)),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        iso = out.stdout.strip().splitlines()
        result["added_date"] = iso[-1][:10] if iso else None

        result["metrics"] = eval_embedding(
            model_path, adapter_module, boolean_run=boolean_run
        )
    except Exception as e:
        result["error"] = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
    finally:
        return_dict["result"] = result


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
    total_freed = 0
    snapshot_date = date.today()
    supported_list = fetch_top_embedding_models(limit=args.top_k)

    # ClickHouse setup — skipped when --write-to-csv is given.
    db_client = None
    csv_rows: list[dict] = []
    if args.write_to_csv:
        print(
            f"CSV mode: results will be written to '{args.write_to_csv}' (no DB access).\n"
        )
    else:
        db_client = get_client()
        if not table_exists(db_client):
            db_client.command(CREATE_TABLE_SQL)
            print("ClickHouse: table created.\n")
        else:
            print("ClickHouse: table already exists.\n")

    total = len(supported_list)
    processed = 0
    overall_start = time.monotonic()

    ctx = multiprocessing.get_context("spawn")
    try:
        for row in supported_list:
            model_path = str(row["model_id"])
            csv_config_class = row.get("config_class")
            print(f"csv_config_class: {csv_config_class}")
            processed += 1
            model_start = time.monotonic()
            elapsed_overall = model_start - overall_start
            print(
                f"\n[{processed}/{total}] {model_path}  "
                f"(overall elapsed: {elapsed_overall:.0f}s)"
            )

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
                    raise Exception(
                        f"Model {model_path} has {int(params):,} parameters, "
                    )

                # Use a raw Process instead of Pool so that no pipes, semaphores,
                # or pool-supervisor state accumulate in the parent over 100+
                # iterations. The Manager lives only for the duration of one row.
                with multiprocessing.managers.SyncManager() as manager:
                    return_dict = manager.dict()
                    proc = ctx.Process(
                        target=_process_row,
                        args=(model_path, args.boolean_run, return_dict),
                    )
                    proc.start()
                    proc.join()

                worker_result = return_dict.get(
                    "result",
                    {
                        "adapter_name": "",
                        "added_date": None,
                        "metrics": {
                            "correct": False,
                            "load": False,
                            "compare_spyre": False,
                        },
                        "error": f"Worker process exited with code {proc.exitcode} and returned no result",
                    },
                )

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

            if args.write_to_csv:
                csv_rows.append(rec)
                print(f"    csv: row queued for '{model_path}'")
            else:
                inserted = insert_model_row(
                    db_client,
                    model_name=rec["model_name"],
                    architecture=rec["architecture"],
                    adapter_name=rec["adapter_name"],
                    added_date=rec["added_date"],
                    snapshot_date=rec["snapshot_date"],
                    verified_on_cpu=rec["verified_on_cpu"],
                    verified_on_gpu=rec["verified_on_gpu"],
                    verified_on_spyre=rec["verified_on_spyre"],
                    num_downloads=rec["num_downloads"],
                )
                if inserted:
                    print(f"    db: row inserted for '{model_path}'")
                else:
                    print(f"    db: row skipped for '{model_path}' (guard rejected)")

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
        if args.write_to_csv and csv_rows:
            _FIELDNAMES = [
                "model_name",
                "architecture",
                "adapter_name",
                "added_date",
                "snapshot_date",
                "verified_on_cpu",
                "verified_on_gpu",
                "verified_on_spyre",
                "num_downloads",
            ]
            args.write_to_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.write_to_csv, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nCSV: {len(csv_rows)} rows written to '{args.write_to_csv}'")

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
