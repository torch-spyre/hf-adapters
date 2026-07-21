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
from datetime import date
from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError

from tests.spyre.weekly_generation.result_sink import (
    EmbeddingGenerativeMode,
)
from utils.utilities import ts

logging.getLogger("transformers").setLevel(logging.ERROR)

MAX_NUMBER_PARAMS = 60_000_000_000

FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER = "not-implemented-adapter"
FAILURE_CATEGORY_MODEL_TOO_LARGE = "model_too_large"
FAILURE_CATEGORY_CPU_LOAD_FAILED = "cpu_load_failed"
FAILURE_CATEGORY_CPU_GENERATE_FAILED = "cpu_generate_failed"
FAILURE_CATEGORY_QUANTIZED_MODEL = "quantized_model"
FAILURE_CATEGORY_HARDWARE_EXCEPTION = "hardware_exception"
FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION = "test_execution_exception"
FAILURE_CATEGORY_VERIFICATION_FAILED = "verification_failed"
FAILURE_CATEGORY_WORKER_CRASHED = "worker_crashed"
FAILURE_CATEGORY_WORKER_TIMEOUT = "worker_timeout"
FAILURE_CATEGORY_MOE = "moe"

# Hard wall-clock cap for a single worker process (in seconds). If a batch
# takes longer than this, the parent kills the child, marks the entire batch
# as failed with FAILURE_CATEGORY_WORKER_TIMEOUT, and moves on. Prevents a
# single hung model from stalling the whole run indefinitely.


class HardwareExceptionAbortError(RuntimeError):
    """Raised in main() when a batch reports a hardware_exception row.

    The Spyre accelerator is unreachable and no subsequent work in this
    process can succeed, so the run aborts. Bubbling this up to the
    ``__main__`` block means the script exits with a non-zero code —
    CI / GHA can alert on it, and a subsequent scheduled run picks the
    aborted rows up automatically via the sink's retry-on-hardware_exception
    skip rule.
    """


def _classify_failure(err: str, default: str) -> str:
    """Bucket a raw error/traceback string into a failure_category.

    Signals in order of specificity:

    * ``"Failed to open the IBM Spyre VFIO device"`` — the accelerator itself
      is unreachable (driver, permissions, another process holding it, …);
      the model under test is not to blame, so tag as hardware_exception.
    * ``"quantiz"`` / ``"optimum"`` — bitsandbytes / AWQ / GPTQ error text
      almost always contains ``quantiz``, and ``optimum`` catches the
      optimum-quanto / optimum-neuron loaders.

    Anything unrecognised falls through to *default* (usually the surrounding
    context's fallback: cpu_load_failed at load time, test_execution_exception
    at eval time).
    """
    if not err:
        return default
    if "Failed to open the IBM Spyre VFIO device" in err or "Replace card" in err:
        return FAILURE_CATEGORY_HARDWARE_EXCEPTION
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
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS: int = 4
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
        required=True,
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


def _cpu_generate(model_path: str) -> tuple[bool, str | None]:
    """Run a single-prompt HF ``generate()`` on CPU for *model_path*.

    Separate from ``_load_on_cpu``: load succeeding tells us the checkpoint
    is well-formed, generate succeeding tells us the forward pass runs
    end-to-end (catches lazy shape errors, tokenizer/config mismatches, and
    custom-code bugs that don't surface at ``from_pretrained`` time).

    Returns ``(ok, error_message)`` with the same convention as
    ``_load_on_cpu``.
    """
    import hf_adapters.hf_common as _hf_common
    from tests.cpu._generate_helpers import simple_generate

    _orig_device = _hf_common.DEVICE
    _hf_common.DEVICE = "cpu"
    try:
        simple_generate(model_path=model_path)
        return True, None
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code >= 500:
            raise
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_cpu_generate exception - {e}")
        return False, err
    except Exception as e:
        err = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_cpu_generate exception - {e}")
        return False, err
    finally:
        _hf_common.DEVICE = _orig_device


def eval_model(model_id: str, adapter, mode: EmbeddingGenerativeMode) -> dict:
    """Load *model_id* on CPU then run the mode's verification pipeline.

    Generative mode: CPU-load → CPU-generate (single-prompt HF forward pass)
    → Spyre smoke + token-compare. The intermediate CPU-generate step catches
    lazy shape errors, tokenizer/config mismatches, and custom-code bugs that
    don't surface at ``from_pretrained`` time; on failure the row is tagged
    ``cpu_generate_failed`` and the Spyre steps are skipped.

    Embedding mode: CPU-load → Spyre cosine-compare (no generate step —
    embedders don't have a ``.generate()`` method).

    Returns a metrics dict with keys ``load``, ``correct``, ``error``,
    ``failure_category``. ``correct`` is ``smoke_passed and not mismatches``
    — in embedding mode there is no smoke step, so ``smoke_passed`` is
    treated as True and the outcome reduces to ``not mismatches``.
    """
    load_on_cpu = False
    smoke_passed = mode == EmbeddingGenerativeMode.EMBEDDING
    mismatches = True
    result: dict = {"error": "", "failure_category": None}

    try:
        if adapter is not None:
            load_on_cpu, load_error = _load_on_cpu(model_path=model_id, mode=mode)
            if load_error and not result["error"]:
                result["error"] = load_error
            if load_on_cpu:
                if mode == EmbeddingGenerativeMode.GENERATIVE:
                    # Extra CPU-generate step — a load that succeeds but crashes
                    # here means the checkpoint is malformed in a way that only
                    # surfaces during forward. Stop before we waste Spyre time.
                    generate_ok, generate_error = _cpu_generate(model_path=model_id)
                    if not generate_ok:
                        if generate_error and not result["error"]:
                            result["error"] = generate_error
                        result["failure_category"] = _classify_failure(
                            generate_error or "",
                            FAILURE_CATEGORY_CPU_GENERATE_FAILED,
                        )
                    else:
                        from tests.spyre.test_e2e_smoke_spyre import run_smoke_test
                        from tests.spyre.test_e2e_token_compare_spyre import (
                            token_compare_spyre,
                        )

                        smoke_passed = (
                            run_smoke_test(model_path=model_id)["status"] == "PASS"
                        )
                        mismatches, _ = token_compare_spyre(model_id)
                else:
                    from tests.spyre.test_e2e_embed_compare_spyre import (
                        embed_compare_spyre,
                    )

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
        result["correct"] = smoke_passed and not mismatches
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
        f"{ts()}       child[{os.getpid()}] entered _process_batch with {len(batch)} model(s)",
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

            metrics = eval_model(model_path, adapter_module, mode)
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
            f"{ts()}       child[{os.getpid()}] finished model "
            f"{len(results)}/{len(batch)}: {model_path!r}  "
            f"(verified_on_cpu={rec['verified_on_cpu']}, "
            f"verified_on_spyre={rec['verified_on_spyre']}, "
            f"failure_category={rec['failure_category']}, "
            f"error={rec['error']})",
            flush=True,
        )
        # Bail out of the batch immediately on a hardware exception — the
        # Spyre device is unreachable, so every remaining model in this
        # batch would hit the same wall. The parent picks up the signal
        # from the returned results and aborts the outer loop.
        if rec["failure_category"] == FAILURE_CATEGORY_HARDWARE_EXCEPTION:
            print(
                f"{ts()}       child[{os.getpid()}] aborting batch — "
                f"hardware_exception detected; "
                f"{len(batch) - len(results)} model(s) not attempted",
                flush=True,
            )
            break

    result_queue.put(results)
    print(
        f"{ts()}       child[{os.getpid()}] done in "
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


def main(
    mode: EmbeddingGenerativeMode, write_to_csv: Path | str | None, top_k: int
) -> None:
    from tests.spyre.weekly_generation.result_sink import (
        ClickHouseResultSink,
        CsvResultSink,
        ResultSink,
    )
    from utils.fetch_top_embedding_models import fetch_top_embedding_models
    from utils.fetch_top_generative_models import fetch_top_generative_models
    from utils.hf_model_catalog import is_moe

    print(f"{ts()} Starting main.")
    preexisting: set = _repos_with_weights()
    total_freed: int = 0
    snapshot_date = date.today()
    if mode == EmbeddingGenerativeMode.GENERATIVE:
        number_of_model_per_process = GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS
        to_process_list = fetch_top_generative_models(limit=top_k)
    elif mode == EmbeddingGenerativeMode.EMBEDDING:
        number_of_model_per_process = EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS
        to_process_list = fetch_top_embedding_models(limit=top_k)
    else:
        raise Exception(f"Unknown mode: {mode}")
    adapter_dates: dict[str, str | None] = _get_adapter_dates()

    sink: ResultSink
    if write_to_csv:
        sink = CsvResultSink(path=write_to_csv, today=snapshot_date)
        print(
            f"CSV mode: results will be appended to '{write_to_csv}' (no DB access).\n"
        )
    else:
        sink = ClickHouseResultSink(today=snapshot_date, embedding_generative=mode)
        print("DB mode: results will be appended to the DB.\n")
    total = len(to_process_list)
    processed = 0
    overall_start = time.monotonic()

    # Early-stop guard runs in the parent (fast: dict lookup for CSV,
    # single SELECT for CH), so we drop already-recent models BEFORE
    # batching. That keeps batch sizes uniform relative to real work.
    prefiltered: list[dict] = []
    early_skipped: int = 0
    moe_skipped: int = 0
    too_large_skipped: int = 0
    unsupported_skipped: int = 0
    print(f"{ts()} Will process {len(to_process_list)} models in total.")
    for row in to_process_list:
        model_path = str(row["model_id"])
        if not sink.should_insert_row(model_path):
            early_skipped += 1
            print(
                f"{ts()}     sink: '{model_path}' skipped early — "
                f"recent snapshot exists within the "
                f"{sink.__class__.__name__} skip window"
            )
            continue
        # No adapter registered for this model's config class — same terminal
        # decision resolve_adapter_module_for_test would reach in the worker,
        # but reached here without spawning one. Uses the fetcher-computed
        # is_supported flag (True iff config_class is in the adapter mapping).

        if row.get("is_supported") is False:
            unsupported_skipped += 1
            sink.add_entry(
                model_name=model_path,
                config_class=str(row.get("config_class") or ""),
                adapter_name="",
                added_date=None,
                snapshot_date=snapshot_date,
                verified_on_cpu=False,
                verified_on_gpu=False,
                verified_on_spyre=False,
                num_downloads=int(row.get("downloads") or 0),
                family=str(row.get("model_type") or ""),
                architecture=str(row.get("architectures") or ""),
                parameters_number=int(row.get("parameters") or 0),
                failure_category=FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
                error=None,
            )
            print(
                f"{ts()}     sink: '{model_path}' skipped early — "
                f"no adapter for config_class={row.get('config_class')!r}"
            )
            continue
        # Reject models too large to bring up on Spyre BEFORE spawning a
        # worker. Same intent as the in-worker guard in _process_batch, but
        # catches everything the fetcher already sized so no worker time is
        # wasted. _process_batch keeps its own check as a defensive backstop
        # for rows where parameters were unknown at fetch time.
        params = row.get("parameters")
        if params not in (None, "") and int(params) > MAX_NUMBER_PARAMS:
            too_large_skipped += 1
            sink.add_entry(
                model_name=model_path,
                config_class=str(row.get("config_class") or ""),
                adapter_name="",
                added_date=None,
                snapshot_date=snapshot_date,
                verified_on_cpu=False,
                verified_on_gpu=False,
                verified_on_spyre=False,
                num_downloads=int(row.get("downloads") or 0),
                family=str(row.get("model_type") or ""),
                architecture=str(row.get("architectures") or ""),
                parameters_number=int(params),
                failure_category=FAILURE_CATEGORY_MODEL_TOO_LARGE,
                error=None,
            )
            print(
                f"{ts()}     sink: '{model_path}' skipped early — "
                f"{int(params):,} parameters exceeds the "
                f"{MAX_NUMBER_PARAMS:,} limit"
            )
            continue
        # MoE models aren't supported on Spyre yet — write the row up-front
        # with failure_category=moe and don't send it to the workers.
        model_info = row.get("model_info")
        if model_info is not None and is_moe(model_info):
            moe_skipped += 1
            sink.add_entry(
                model_name=model_path,
                config_class=str(row.get("config_class") or ""),
                adapter_name="",
                added_date=None,
                snapshot_date=snapshot_date,
                verified_on_cpu=False,
                verified_on_gpu=False,
                verified_on_spyre=False,
                num_downloads=int(row.get("downloads") or 0),
                family=str(row.get("model_type") or ""),
                architecture=str(row.get("architectures") or ""),
                parameters_number=int(row.get("parameters") or 0),
                failure_category=FAILURE_CATEGORY_MOE,
                error=None,
            )
            print(f"{ts()}     sink: '{model_path}' skipped early — MoE model")
            continue
        prefiltered.append(row)
    if early_skipped:
        print(
            f"\n{ts()} Early-skip: {early_skipped}/{total} models already have a "
            f"recent snapshot; {len(prefiltered)} left to evaluate.\n"
        )
    if unsupported_skipped:
        print(
            f"{ts()} Unsupported-skip: {unsupported_skipped}/{total} models have "
            f"no adapter for their config_class and were written directly to the "
            f"sink.\n"
        )
    if too_large_skipped:
        print(
            f"{ts()} Too-large-skip: {too_large_skipped}/{total} models exceed "
            f"the {MAX_NUMBER_PARAMS:,} parameter limit and were written directly "
            f"to the sink.\n"
        )
    if moe_skipped:
        print(
            f"{ts()} MoE-skip: {moe_skipped}/{total} models tagged as moe and "
            f"written directly to the sink.\n"
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
                f"\n{ts()} [batch {batch_idx}/{total_batches}] {len(batch)} model(s) "
                f"(overall elapsed: {batch_start - overall_start:.0f}s)"
            )
            for path in batch_paths:
                print(f"{ts()}     - {path}")

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
            timeout = 10 * 60 * number_of_model_per_process

            proc.join(timeout=timeout)

            # Timeout guard: if the child is still alive after the deadline,
            # kill it, drop any partial results, and synthesise timeout rows
            # so the outer loop can proceed to the next batch.
            timed_out: bool = proc.is_alive()
            if timed_out:
                print(
                    f"{ts()}     batch: worker exceeded "
                    f"{timeout}s timeout — terminating "
                    f"pid={proc.pid} and marking {len(batch)} model(s) as "
                    f"failed with {FAILURE_CATEGORY_WORKER_TIMEOUT}"
                )
                proc.terminate()
                proc.join(timeout=30)
                if proc.is_alive():
                    print(
                        f"{ts()}     batch: worker pid={proc.pid} did not "
                        f"exit after SIGTERM — sending SIGKILL"
                    )
                    proc.kill()
                    proc.join(timeout=30)

            # Drain the queue. SimpleQueue.get() blocks if empty, so probe
            # first; the child has already exited so a non-empty queue
            # returns instantly, and an empty one signals a crash.
            if timed_out:
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
                        "error": (f"worker exceeded " f"{timeout}s timeout"),
                        "failure_category": FAILURE_CATEGORY_WORKER_TIMEOUT,
                    }
                    for row, path in zip(batch, batch_paths)
                ]
            elif result_queue.empty():
                print(
                    f"{ts()}     batch: worker exited code {proc.exitcode} "
                    f"and returned no results — marking all {len(batch)} "
                    f"models as failed"
                )
                worker_results = [
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
                    print(f"{ts()}     [{model_path}] error: {rec['error']}")

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
                        f"{ts()}     sink: row written for '{model_path}' "
                        f"(verified_on_cpu={rec.get('verified_on_cpu')}, "
                        f"verified_on_spyre={rec.get('verified_on_spyre')}, "
                        f"failure_category={rec.get('failure_category')}, )"
                    )
                else:
                    print(
                        f"{ts()}     sink: row skipped for '{model_path}' (guard rejected)"
                    )

            # Cache cleanup: delete weights downloaded during this batch,
            # regardless of whether the worker processed each model.
            total_freed = _cleanup_batch_weights(
                batch_paths, had_weights_map, total_freed
            )

            batch_elapsed = time.monotonic() - batch_start
            print(
                f"{ts()}     batch {batch_idx}/{total_batches} done: "
                f"{len(worker_results)} model(s) in {batch_elapsed:.1f}s  "
                f"(per-model avg: {batch_elapsed / max(1, len(worker_results)):.1f}s)"
            )

            # Durability boundary: flush accumulated rows now so a hard parent
            # crash before the next batch loses at most this batch, not the
            # whole run. No-op for sinks that write per-row (CSV).
            sink.flush()

            # Abort the whole run when any row in this batch reports a
            # hardware exception — the Spyre accelerator is unreachable and
            # every subsequent batch would waste worker time hitting the
            # same wall. The `raise` unwinds through the `finally` clause
            # below (cache cleanup, sink close, summary), then propagates
            # out of main() so the script exits with a non-zero status.
            if any(
                rec.get("failure_category") == FAILURE_CATEGORY_HARDWARE_EXCEPTION
                for rec in worker_results
            ):
                remaining: int = total_batches - batch_idx
                print(
                    f"\n{ts()} Aborting run — hardware_exception detected in "
                    f"batch {batch_idx}/{total_batches}; skipping the remaining "
                    f"{remaining} batch(es). Rerun once the accelerator is "
                    f"available; those rows will be picked up automatically "
                    f"by the sink's retry-on-hardware_exception rule."
                )
                raise HardwareExceptionAbortError(
                    f"hardware_exception in batch {batch_idx}/{total_batches}"
                )
    except KeyboardInterrupt:
        print(f"\n{ts()} Interrupted — results so far are saved; rerun to resume.")
    finally:
        # Clean up weights for the in-flight batch if interrupted mid-run.
        _ = _cleanup_batch_weights(batch_paths, had_weights_map, total_freed)

        sink.close()
        if write_to_csv:
            print(
                f"\n{ts()} CSV: '{write_to_csv}' closed ({processed} rows processed)."
            )

        overall_elapsed = time.monotonic() - overall_start
        mins, secs = divmod(int(overall_elapsed), 60)
        print(
            f"\n{'='*60}\n"
            f"{ts()} Processed {processed}/{total} models  |  "
            f"Total time: {mins}m {secs:02d}s\n"
            f"{'='*60}"
        )


if __name__ == "__main__":
    print(f"{ts()} Starting weekly generation...", flush=True)
    args = _parse_args()
    try:
        main(
            mode=EmbeddingGenerativeMode(args.mode),
            write_to_csv=args.write_to_csv,
            top_k=args.top_k,
        )
    except HardwareExceptionAbortError as e:
        # Non-zero exit so CI / GHA scheduled runs can alert. main()'s
        # `finally` has already flushed the sink and cleaned the cache.
        print(f"{ts()} Exiting with status 1 ({e}).")
        sys.exit(1)
