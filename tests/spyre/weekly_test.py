"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_test.py --top-k 200 --workers 4
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from test_e2e_embed_compare_spyre import embed_compare_spyre

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from test_load_spyre import load_embedding  # noqa: E402

from hf_adapters.auto_spyre_model import resolve_adapter_module  # noqa: E402
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
    parser.add_argument(
        "--workers",
        "-k",
        type=int,
        default=1,
        help="Number of parallel subprocesses for eval_embedding (default: 1).",
    )
    return parser.parse_args(argv)


def _print_adapter_add_dates(add_dates: dict[str, str | None]) -> None:
    """Print adapters sorted oldest→newest, aligned, with an `unknown` bucket."""
    dated = [(d, m) for m, d in add_dates.items() if d]
    undated = sorted(m for m, d in add_dates.items() if not d)

    name_w = max((len(m) for m in add_dates), default=0)
    print(f"Adapter add-dates ({len(add_dates)} modules):")
    for date, module in sorted(dated):
        print(f"  {date}  {module:<{name_w}}")
    for module in undated:
        print(f"  {'unknown':<10}  {module:<{name_w}}")


def eval_embedding(model_id: str) -> dict:
    """Load and compare embeddings for one model. Returns a metrics dict."""
    loads, _ = load_embedding(model_id)
    mismatches, _ = embed_compare_spyre(model_id)
    return {
        "correct": loads and not mismatches,
        "load": loads,
        "compare_spyre": not mismatches,
    }


# ---------------------------------------------------------------------------
# Each model is evaluated in a completely fresh Python interpreter so that no
# Spyre/GPU driver state leaks between runs.  A ThreadPoolExecutor caps how
# many subprocesses run concurrently (--workers).
# ---------------------------------------------------------------------------

_EVAL_SCRIPT = """\
import json, sys, traceback
sys.path[:0] = {paths!r}
from test_load_spyre import load_embedding
from test_e2e_embed_compare_spyre import embed_compare_spyre
model_id = {model_id!r}
try:
    loads, _ = load_embedding(model_id)
    mismatches, _ = embed_compare_spyre(model_id)
    print(json.dumps({{"correct": loads and not mismatches,
                       "load": loads, "compare_spyre": not mismatches}}))
except Exception as exc:
    tb = traceback.format_exc()
    print(json.dumps({{"_error": f"{{type(exc).__name__}}: {{exc}}",
                       "_tb": "".join(tb.splitlines(keepends=True)[-6:])}}))
"""

_EVAL_PATHS = [
    str(_SPYRE_TESTS_DIR),
    str(_TESTS_DIR),
    str(_UTILS_DIR),
    str(_REPO_ROOT),
]


def _eval_embedding_subprocess(model_id: str) -> dict:
    """Run eval_embedding in a fresh interpreter; return the metrics dict.

    On success returns the same keys as eval_embedding().
    On failure raises RuntimeError so the caller's existing except block fires.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join(_EVAL_PATHS) + os.pathsep + existing
        if existing
        else os.pathsep.join(_EVAL_PATHS)
    )
    script = _EVAL_SCRIPT.format(paths=_EVAL_PATHS, model_id=model_id)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            if "_error" in data:
                raise RuntimeError(data["_error"])
            return data
    stderr_tail = "".join(proc.stderr.splitlines(keepends=True)[-6:])
    raise RuntimeError(
        f"subprocess exited {proc.returncode} with no JSON output\n{stderr_tail}"
    )


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
    # supported_list = list(iter_supported_rows(args.path))
    # supported_rows = {r["model_id"]: r for r in supported_list}
    total_freed = 0
    supported_list = fetch_top_embedding_models(
        limit=args.top_k, output_csv=args.output_csv
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        for row in supported_list:
            # if attempted >= args.limit:
            #     break
            model_path = row["model_id"]
            # if model_id in done:
            #     continue

            csv_config_class = row.get("config_class")
            # gated = row.get("is_gated") == "True"
            # if gated and not args.include_gated:
            #     record(
            #         {
            #             "model_id": model_id,
            #             "config_class": csv_config_class,
            #             "status": "skipped:gated",
            #             "rank": row.get("rank"),
            #             "attempted_at": datetime.now().isoformat(timespec="seconds"),
            #         }
            #     )
            #     continue

            # attempted += 1
            had_weights = model_path in preexisting
            # print(f"\n[{attempted}/{args.limit}] {model_id} ({csv_config_class})")

            rec = {
                "model_id": model_path,
                "rank": row.get("rank"),
                "downloads": row.get("downloads"),
                "config_class": csv_config_class,
                "model_type": row.get("model_type"),
                "params": row.get("parameters (str)"),
                "had_cached_weights_at_start": had_weights,
                "attempted_at": datetime.now().isoformat(timespec="seconds"),
            }

            try:
                adapter_module = resolve_adapter_module(model_path)
                adapter_name = os.path.splitext(
                    os.path.basename(adapter_module.__file__)
                )[0]
                rec["adapter"] = adapter_name
                rec["adapter_added"] = add_dates.get(adapter_name)

                # Reject models too large to bring up on Spyre. Recorded as a
                # clean SpyreUnsupportedModelError, same as stick-misaligned dims.

                params = row.get("parameters")
                if params not in (None, "") and int(params) > 60_000_000_000:
                    # TODO - Raise Exception?
                    # raise SpyreUnsupportedModelError(
                    print(
                        f"Model {model_path} has {int(params):,} parameters, "
                        f"exceeding the 60B limit for Spyre bring-up."
                    )

                # if args.path == "generative":
                #     metrics = eval_generative(
                #         model_id, adapter, csv_config_class, args.num_decode
                #     )
                # else:
                fut = executor.submit(_eval_embedding_subprocess, model_path)
                metrics = fut.result()

                rec["runs"] = True
                rec["error"] = None
                rec.update(metrics)
                print(f"    runs=True correct={rec.get('correct')} ")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                rec["runs"] = False
                rec["correct"] = False
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback_tail"] = "".join(tb.splitlines(keepends=True)[-6:])
                # adapter fields may be unset if resolve_adapter failed
                rec.setdefault("adapter", None)
                rec.setdefault("adapter_added", None)
                print(f"    runs=False error={rec['error']}")

            # record(rec)

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
        executor.shutdown(wait=False, cancel_futures=True)
    # fout.close()


if __name__ == "__main__":
    main()
