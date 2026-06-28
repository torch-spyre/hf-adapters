"""Weekly Spyre test suite.

The first step is *not* a pytest: it refreshes the top-k embedding model
catalog by invoking ``utils/fetch_top_embedding_models.py``. Subsequent
pytests in this file consume the resulting CSV.

Run directly to perform the fetch step::

    python tests/spyre/weekly_test.py --top-k 200
"""

import argparse
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from tests._helpers import resolve_adapter
from tests.spyre.test_load_spyre import load_embedding

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402

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
    for date, module in sorted(dated):
        print(f"  {date}  {module:<{name_w}}")
    for module in undated:
        print(f"  {'unknown':<10}  {module:<{name_w}}")


def eval_embedding(model_id, adapter, csv_config_class) -> bool:
    # Run the embedding tests

    result = load_embedding(model_id)

    return result


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    add_dates: dict[str, str | None] = _adapter_add_dates()
    # _print_adapter_add_dates(add_dates)
    preexisting: set = _repos_with_weights()
    # supported_list = list(iter_supported_rows(args.path))
    # supported_rows = {r["model_id"]: r for r in supported_list}

    supported_list = fetch_top_embedding_models(
        limit=args.top_k, output_csv=args.output_csv
    )
    supported_rows = {r["model_id"]: r for r in supported_list}
    print("\n".join(supported_rows))

    try:
        for row in supported_list:
            # if attempted >= args.limit:
            #     break
            model_id = row["model_id"]
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
            had_weights = model_id in preexisting
            # print(f"\n[{attempted}/{args.limit}] {model_id} ({csv_config_class})")

            rec = {
                "model_id": model_id,
                "rank": row.get("rank"),
                "downloads": row.get("downloads"),
                "config_class": csv_config_class,
                "model_type": row.get("model_type"),
                "params": row.get("parameters (str)"),
                "had_cached_weights_at_start": had_weights,
                "attempted_at": datetime.now().isoformat(timespec="seconds"),
            }

            try:
                adapter, adapter_name = resolve_adapter(model_id)
                rec["adapter"] = adapter_name
                rec["adapter_added"] = add_dates.get(adapter_name)

                # Reject models too large to bring up on Spyre. Recorded as a
                # clean SpyreUnsupportedModelError, same as stick-misaligned dims.
                from hf_adapters.hf_common import SpyreUnsupportedModelError

                params = row.get("parameters")
                if params not in (None, "") and int(params) > 60_000_000_000:
                    raise SpyreUnsupportedModelError(
                        f"Model {model_id} has {int(params):,} parameters, "
                        f"exceeding the 60B limit for Spyre bring-up."
                    )

                # if args.path == "generative":
                #     metrics = eval_generative(
                #         model_id, adapter, csv_config_class, args.num_decode
                #     )
                # else:
                metrics = eval_embedding(model_id, adapter, csv_config_class)

                rec["runs"] = True
                rec["error"] = None
                rec.update(metrics)
                print(
                    f"    runs=True correct={rec.get('correct')} "
                    f"spyre_nan={rec.get('spyre_nan')}"
                )
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
    #         if not had_weights:
    #             freed = delete_repo_weights(model_id)
    #             total_freed += freed
    #             if freed:
    #                 print(
    #                     f"    freed {human_bytes(freed)} "
    #                     f"(total {human_bytes(total_freed)})"
    #                 )
    # except KeyboardInterrupt:
    #     print("\nInterrupted — results so far are saved; rerun to resume.")
    # finally:
    #     fout.close()

#
# if __name__ == "__main__":
#     main()
