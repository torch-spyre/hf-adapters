"""Weekly Spyre evaluation suite.

Two ways to say which models to evaluate — exactly one is required::

    # CI: evaluate a shard prepared upstream
    python tests/spyre/weekly_generation/weekly_test.py \\
        --mode generative --model-list-file shards/generative-x1-shard-000.json

    # Manual: fetch, filter and evaluate in one pass
    python tests/spyre/weekly_generation/weekly_test.py \\
        --mode embedding --fetch --top-k 200

Either way the same three pre-filters apply — no adapter for the config class,
too large for Spyre, MoE — and each dropped model gets a terminal row recording
why. With ``--model-list-file`` that happened upstream in
``.github/scripts/generate_weekly_shards.py``; with ``--fetch`` it happens here,
through the same ``fetch_and_filter``. So everything that reaches the evaluation
loop needs a Spyre card.

Filtering before the list is sharded is what keeps shard durations comparable:
the dropped models cluster by download count, so filtering per-shard used to
leave some CI jobs finishing in minutes and others running for hours.

``--mode`` is parsed into a ``ModelType``, which selects the per-process batch
size, the model class loaded in ``_load_on_cpu``, the verification pipeline in
``eval_model`` (token-compare for generative, cosine-compare for embedding), and
which ClickHouse table the sink reads and writes. With ``--fetch`` it also picks
the catalog to fetch.

Process model
-------------
Each batch is evaluated in a freshly spawned child (``weekly_sub_process``) that
exits when the batch ends, which is what actually returns the accelerator's
memory. The parent owns the sink for the whole run and does all the writing; the
child only returns plain dicts over a queue.

Flags:

* ``--mode {generative,embedding}``  Required. See above.
* ``--model-list-file F``            Evaluate this JSON list. Mutually exclusive
  with ``--fetch``; one of the two is required.
* ``--fetch``                        Fetch and filter here instead.
* ``--top-k N``                      With ``--fetch``: how many to fetch
  (default: 10000).
* ``--max-params N``                 With ``--fetch``: parameter ceiling.
* ``--write-to-csv F``               Record results in a new CSV instead of
  ClickHouse, for runs with no database access. Write-only: the file must not
  already exist, and nothing is read back.

Result rows
-----------
One row is recorded per model handled, so a run's row count never silently
disagrees with its input. Nothing is filtered at write time — a row reaching the
sink is one the caller decided to record, and dropping it there would leave a run
with fewer rows than the models it handled and no accounting for the difference.
Same-day duplicates are collapsed on merge by
``ReplacingMergeTree(snapshot_date)``.
"""

import argparse
import json
import logging
import multiprocessing
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION,
    FAILURE_CATEGORY_WORKER_CRASHED,
    FAILURE_CATEGORY_WORKER_TIMEOUT,
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import fetch_and_filter
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.sink.sink_factory import create_sink, csv_path_for
from tests.spyre.weekly_generation.weekly_sub_process import _process_batch
from utils.utilities import human_bytes, ts

logging.getLogger("transformers").setLevel(logging.ERROR)

# Per-model wall-clock allowance for a worker process, in seconds. A batch's cap
# is this times its model count; see the timeout guard in main(), which kills the
# child and marks the batch FAILURE_CATEGORY_WORKER_TIMEOUT once it is exceeded,
# so one hung model cannot stall the run indefinitely.
_WORKER_TIMEOUT_SECONDS_PER_MODEL: int = 10 * 60


class HardwareExceptionAbortError(RuntimeError):
    """Raised in main() when a batch reports a hardware_exception row.

    The Spyre accelerator is unreachable and no subsequent work in this
    process can succeed, so the run aborts. Bubbling this up to the
    ``__main__`` block means the script exits with a non-zero code, so CI / GHA
    can alert on it.

    The aborted models are not picked up automatically: every model handed to a
    run is evaluated, so the next scheduled scan re-evaluates the whole list
    rather than just these. Re-running only the hardware-exception rows is what
    ``ClickHouseResultSink.fetch_hw_failure_models`` is for, but nothing calls it
    yet — a recovery run has to be driven by hand until that is wired up.
    """


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
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS: int = 10
EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS: int = 90


def _repos_with_weights(repo_ids: list[str]) -> set[str]:
    """Subset of *repo_ids* that already have >=1 weight file cached at startup.

    Navigates directly to each repo's cache folder using the known HF layout —
    same trick as ``_delete_repo_weights`` — instead of calling
    ``scan_cache_dir()``, which walks and stats every blob under HF_HOME. On the
    Spyre pod that cache is a shared network mount holding the accumulated
    weights of the whole scan, and up to 32 shard jobs call this concurrently at
    startup, so the full walk cost hours of wall-clock per job.

    Only models in the current shard are ever looked up (see ``had_weights_map``
    in ``main``), so scoping the check to *repo_ids* loses nothing.
    """
    print(f"{ts()} Scanning the weight cache for {len(repo_ids)} model(s)…", flush=True)
    started: float = time.monotonic()
    have: set[str] = set()
    for repo_id in repo_ids:
        snapshots_dir = _repo_cache_dir(repo_id) / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        try:
            if any(p.name.endswith(_WEIGHT_SUFFIXES) for p in snapshots_dir.rglob("*")):
                have.add(repo_id)
        except OSError as e:
            print(f"    warn: could not scan {snapshots_dir}: {e}")
    print(
        f"{ts()} Weight-cache scan done in {time.monotonic() - started:.2f}s — "
        f"{len(have)}/{len(repo_ids)} model(s) already cached.",
        flush=True,
    )
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
        "--write-to-csv",
        type=Path,
        default=None,
        metavar="RESULTS_CSV",
        help=(
            "Write evaluation results to this CSV file instead of inserting "
            "into ClickHouse. No DB connection is made when this flag is set."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-list-file",
        type=Path,
        default=None,
        metavar="MODEL_LIST_JSON",
        help=(
            "Evaluate the models in this JSON file, as produced by "
            ".github/scripts/generate_weekly_shards.py. It applies the "
            "pre-filters and records a terminal row for each model it drops, so "
            "everything in the file is expected to need a Spyre card. This is "
            "how CI runs: the list is fetched and filtered once, then sharded "
            "across parallel jobs."
        ),
    )
    source.add_argument(
        "--fetch",
        action="store_true",
        help=(
            "Fetch the top --top-k models for --mode, apply the same "
            "pre-filters, record the terminal rows, and evaluate the survivors — "
            "all in one pass. For manual runs where there is no shard file."
        ),
    )
    fetch_opts = parser.add_argument_group("--fetch options")
    fetch_opts.add_argument(
        "--top-k",
        type=int,
        default=10_000,
        help=(
            "With --fetch: how many top models to fetch by downloads "
            "(default: 10000). Ignored with --model-list-file."
        ),
    )
    fetch_opts.add_argument(
        "--max-params",
        type=int,
        default=MAX_NUMBER_PARAMS,
        help=(
            "With --fetch: reject models above this parameter count "
            f"(default: {MAX_NUMBER_PARAMS:,}). Ignored with --model-list-file, "
            "whose producer applied its own limit."
        ),
    )
    parser.add_argument(
        "--snapshot-date",
        type=date.fromisoformat,
        required=True,
        metavar="YYYY-MM-DD",
        help=(
            "Date to record as the snapshot date for all rows written in this run. "
            "Use $(date -u +%Y-%m-%d) for today when invoking manually."
        ),
    )
    args = parser.parse_args(argv)
    # Reject the silent no-op of passing a fetch tuning flag without --fetch.
    if not args.fetch:
        for flag, dest in (("--top-k", "top_k"), ("--max-params", "max_params")):
            if getattr(args, dest) != parser.get_default(dest):
                parser.error(f"{flag} only applies with --fetch")
    return args


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
        print(f"    freed {human_bytes(freed)} (total {human_bytes(total_freed)})")
    return total_freed


def _chunk_into_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    """Split *rows* into consecutive sub-lists of length *batch_size* (the
    last batch may be shorter).
    """
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def main(
    model_type: ModelType,
    model_list_file: Path | None,
    write_to_csv: Path | None,
    fetch: bool,
    top_k: int,
    max_params: int,
    snapshot_date: date,
) -> None:
    """Evaluate a model list on Spyre, one spawned worker per batch.

    Exactly one of *model_list_file* (a shard prepared by
    ``generate_weekly_shards``) and *fetch* supplies the list; argparse enforces
    that. Both arrive pre-filtered, so every model reaching the batch loop is
    expected to need a card.

    Owns the sink for the whole run: it is created here, written to by both the
    ``--fetch`` pre-filter and the evaluation loop, flushed at each batch
    boundary, and closed once in the ``finally`` below. ``top_k``/``max_params``
    apply only under *fetch*.

    Raises:
        HardwareExceptionAbortError: a batch reported ``hardware_exception``, so
            the accelerator is unreachable and the remaining batches are skipped.
    """
    from tests.spyre.weekly_generation.sink.result_sink import ResultSink

    print(f"{ts()} Starting main.")
    print(f"{ts()} Snapshot date: {snapshot_date}")

    total_freed: int = 0

    models_per_process = {
        ModelType.GENERATIVE: GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS,
        ModelType.EMBEDDING: EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS,
    }

    adapter_dates: dict[str, str | None] = _get_adapter_dates()

    sink: ResultSink = create_sink(
        model_type=model_type,
        write_to_csv=write_to_csv,
    )

    # All six are read by the finally block, so they are bound before the try
    # opens — otherwise a failure while building the model list would raise
    # UnboundLocalError from the cleanup path and mask the real error.
    processed = 0
    total = 0
    overall_start = time.monotonic()
    batch_paths: list[str] = []
    had_weights_map: dict[str, bool] = {}
    preexisting: set[str] = set()

    # The try opens before the model list is built so that a failure while
    # fetching (a Hub outage, say) still closes the sink — under --fetch the
    # pre-filter has by then already written terminal rows worth keeping.
    try:
        if fetch:
            rows: list[dict] = fetch_and_filter(
                model_type=model_type,
                snapshot_date=snapshot_date,
                top_k=top_k,
                sink=sink,
                max_params=max_params,
            )
        else:
            assert model_list_file is not None  # argparse guarantees one of the two
            print(f"{ts()} Loading model list from '{model_list_file}'.")
            rows = json.loads(model_list_file.read_text())

        total = len(rows)
        print(f"{ts()} Will evaluate {total} model(s).")

        # Must run after *rows* is resolved — the cache check is scoped to this
        # run's models rather than walking the whole (network-mounted) cache.
        preexisting = _repos_with_weights([str(row["model_id"]) for row in rows])

        batch_size = models_per_process[model_type]
        batches: list[list[dict]] = _chunk_into_batches(
            rows=rows,
            batch_size=batch_size,
        )
        total_batches: int = len(batches)

        ctx = multiprocessing.get_context("spawn")

        for batch_idx, batch in enumerate(batches, start=1):
            batch_start = time.monotonic()
            batch_paths = [str(r["model_id"]) for r in batch]
            batch_curated = [r["curated"] for r in batch]
            print(
                f"\n{ts()} [batch {batch_idx}/{total_batches}] {len(batch)} model(s) "
                f"(overall elapsed: {batch_start - overall_start:.0f}s)"
            )
            for path, curated in zip(batch_paths, batch_curated):
                print(f"{ts()} - {path}" + ("\t(curated)" if curated else ""))

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
                    model_type,
                    snapshot_date,
                ),
            )
            proc.start()
            timeout = _WORKER_TIMEOUT_SECONDS_PER_MODEL * batch_size

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
                        "curated": bool(row["curated"]),
                        "num_downloads": int(row.get("downloads") or 0),
                        "family": str(row.get("model_type") or ""),
                        "architecture": str(row.get("architectures") or ""),
                        "parameters_number": int(row.get("parameters") or 0),
                        "error": f"worker exceeded {timeout}s timeout",
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
                        "curated": bool(row["curated"]),
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

                sink.add_entry(
                    model_name=str(rec["model_name"]),
                    config_class=str(rec["config_class"]),
                    adapter_name=str(rec["adapter_name"]),
                    added_date=rec["added_date"],
                    snapshot_date=rec["snapshot_date"],
                    verified_on_cpu=bool(rec["verified_on_cpu"]),
                    verified_on_gpu=bool(rec["verified_on_gpu"]),
                    verified_on_spyre=bool(rec["verified_on_spyre"]),
                    curated=bool(rec["curated"]),
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
                )
                print(
                    f"{ts()}     sink: row written for '{model_path}' "
                    f"(verified_on_cpu={rec.get('verified_on_cpu')}, "
                    f"verified_on_spyre={rec.get('verified_on_spyre')}, "
                    f"failure_category={rec.get('failure_category')}, "
                    f"curated={rec.get('curated')})"
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
            # Report the file actually written, not the bare --write-to-csv
            # argument — the factory suffixes it with the model type.
            written_to = csv_path_for(write_to_csv, model_type)
            print(f"\n{ts()} CSV: '{written_to}' closed ({processed} rows processed).")

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
            model_type=ModelType(args.mode),
            model_list_file=args.model_list_file,
            write_to_csv=args.write_to_csv,
            fetch=args.fetch,
            top_k=args.top_k,
            max_params=args.max_params,
            snapshot_date=args.snapshot_date,
        )
    except HardwareExceptionAbortError as e:
        # Non-zero exit so CI / GHA scheduled runs can alert. main()'s
        # `finally` has already flushed the sink and cleaned the cache.
        print(f"{ts()} Exiting with status 1 ({e}).")
        sys.exit(1)
