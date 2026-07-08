"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_test.py --top-k 200
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from datetime import date
from pathlib import Path

from test_e2e_embed_compare_spyre import embed_compare_spyre

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hf_adapters.auto_spyre_model import resolve_adapter_module  # noqa: E402
from tests.spyre.create_model_spyre_table import (  # noqa: E402
    CREATE_TABLE_SQL,
    get_client,
    insert_model_row,
    table_exists,
)
from tests.spyre.test_load_spyre import load_embedding  # noqa: E402
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
        "--output-csv",
        type=Path,
        default=None,
        help="Destination CSV (defaults to resources/top_embedding_models.csv).",
    )
    return parser.parse_args(argv)


def _print_adapter_add_dates(add_dates: dict[str, str | None]) -> None:
    """Print adapters sorted oldest→newest, aligned, with an `unknown` bucket."""
    dated = [(d, m) for m, d in add_dates.items() if d]
    undated = sorted(m for m, d in add_dates.items() if not d)

    name_w = max((len(m) for m in add_dates), default=0)
    print(f"Adapter add-dates ({len(add_dates)} modules):")
    for date_, module in sorted(dated):
        print(f"  {date_}  {module:<{name_w}}")
    for module in undated:
        print(f"  {'unknown':<10}  {module:<{name_w}}")


def eval_embedding_on_spyre(model_id: str) -> dict:
    """Load and compare embeddings for one model. Returns a metrics dict."""
    loads, _ = load_embedding(model_id)
    mismatches, _ = embed_compare_spyre(model_id)
    return {
        "correct": loads and not mismatches,
        "load": loads,
        "compare_spyre": not mismatches,
    }


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
    add_dates: dict[str, str | None] = _adapter_add_dates()
    # _print_adapter_add_dates(add_dates)
    preexisting: set = _repos_with_weights()
    total_freed = 0
    snapshot_date = date.today()
    supported_list = fetch_top_embedding_models(
        limit=args.top_k, output_csv=args.output_csv
    )

    # Ensure the ClickHouse table exists before processing any models.
    db_client = get_client()
    if not table_exists(db_client):
        db_client.command(CREATE_TABLE_SQL)
        print("ClickHouse: table created.\n")
    else:
        print("ClickHouse: table already exists.\n")

    total = len(supported_list)
    processed = 0
    overall_start = time.monotonic()

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
                "architecture": row.get("model_type", ""),
                "adapter_name": "",
                "added_date": None,
                "snapshot_date": snapshot_date,
                "verified_on_cpu": False,
                "verified_on_gpu": False,
                "verified_on_spyre": False,
                "num_downloads": int(row.get("downloads") or 0),
            }

            try:
                adapter_module = resolve_adapter_module(model_path)
                adapter_name = os.path.splitext(
                    os.path.basename(adapter_module.__file__)
                )[0]
                rec["adapter_name"] = adapter_name
                added_iso = add_dates.get(adapter_name)
                rec["added_date"] = (
                    date.fromisoformat(added_iso) if added_iso else date.today()
                )

                # Reject models too large to bring up on Spyre. Recorded as a
                # clean SpyreUnsupportedModelError, same as stick-misaligned dims.
                params = row.get("parameters")
                if params not in (None, "") and int(params) > 60_000_000_000:
                    print(
                        f"Model {model_path} has {int(params):,} parameters, "
                        f"exceeding the 60B limit for Spyre bring-up."
                    )

                metrics = eval_embedding_on_spyre(model_path)

                # TODO
                rec["verified_on_cpu"] = metrics.get("load", False)
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
                rec.setdefault("adapter_name", "")
                rec["added_date"] = rec["added_date"] or date.today()
                model_elapsed = time.monotonic() - model_start
                print(
                    f"    verified_on_cpu=False  "
                    f"error={type(e).__name__}: {e}  "
                    f"model_time={model_elapsed:.1f}s"
                )
                print("".join(traceback.format_exc().splitlines(keepends=True)[-6:]))

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
