"""Weekly Spyre evaluation suite.

Two ways to say which models to evaluate — exactly one is required::

    # CI: evaluate a shard prepared upstream
    python tests/spyre/weekly_generation/weekly_test.py \\
        --mode generative --model-list-file shards/generative-x1-shard-000.json

    # Manual: fetch, filter and evaluate in one pass
    python tests/spyre/weekly_generation/weekly_test.py \\
        --mode embedding --fetch --top-k-downloads 200 --top-k-trending 200

Either way the same catalog (ranked top-K by downloads and by trending score,
plus the curated file and the model registry) and the same pre-filter (too
large for Spyre) apply, and each dropped model gets a terminal row recording
why. With ``--model-list-file`` that happened upstream in
``.github/scripts/generate_weekly_shards.py``; with ``--fetch`` it happens here,
through the same ``fetch_and_filter``. Adapter resolution and MoE classification
happen in the worker.

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
* ``--top-k-downloads N``            With ``--fetch``: top-N by downloads over
  the last 30 days (default: 1000; 0 skips it).
* ``--top-k-trending N``             With ``--fetch``: top-N by trending score
  (default: 1000; 0 skips it).
* ``--max-params N``                 With ``--fetch``: parameter ceiling.
* ``--write-to-csv F``               Record results in a new CSV instead of
  ClickHouse, for runs with no database access. Write-only: the file must not
  already exist, and nothing is read back.
* ``--deadline ISO8601``             Hard wall-clock stop (issue #745). No batch
  starts at or after it, and a batch still running when it passes is stopped
  without recording that batch. Workers return results only at batch boundaries,
  so completed models inside an interrupted batch must be retried too. Rows from
  earlier batches are kept; the run exits 0 with a warning and job summary.
  Exception: a GitHub Actions re-run attempt (``GITHUB_RUN_ATTEMPT`` > 1) that
  starts after the deadline scans nothing, so it exits 1 instead of going green.

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
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
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
from utils.utilities import ts

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


class DeadlineReachedError(RuntimeError):
    """Raised in main() when the ``--deadline`` passes (issue #745).

    The weekly scan must not hold Spyre cards into the US workweek. Rows written
    before the deadline are flushed by main()'s ``finally``; the models that were
    not reached get no row, so a re-run with the same ``--snapshot-date`` picks
    them up (rows already recorded for that date are skipped).
    """


class DeadlineBeforeStartupError(DeadlineReachedError):
    """The ``--deadline`` had already passed when main() started: nothing ran.

    Distinguished from a mid-scan stop so ``__main__`` can fail a re-run attempt
    that would otherwise exit 0 having scanned zero models.
    """


def _parse_deadline(value: str) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp (a trailing ``Z`` is accepted)."""
    parsed: datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"deadline {value!r} must include a timezone (e.g. 2026-11-09T14:00:00Z)"
        )
    return parsed


def _seconds_until(deadline: datetime) -> float:
    """Seconds from now until *deadline* (negative once it has passed)."""
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def _report_deadline_stop(error: DeadlineReachedError) -> None:
    """Make an expected deadline stop visible in the scan's own job summary."""
    print(f"::warning title=Weekly scan deadline::{error}")
    print(f"{ts()} Stopped at deadline ({error}).")
    summary_path: str | None = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a") as summary:
            summary.write(
                f"## Weekly scan stopped at its deadline\n\n{error}\n\n"
                "Previously flushed batches are preserved. "
                "The interrupted batch and remaining models have no new verdict.\n"
            )


def _deadline_stop_exit_code(error: DeadlineReachedError) -> int:
    """Exit status for a deadline stop; reports it either way.

    A deadline stop during the scheduled run is expected and exits 0. A re-run
    attempt (``GITHUB_RUN_ATTEMPT`` > 1, e.g. "Re-run failed jobs") reuses the
    original run's deadline; if that had already passed, the shard scanned
    nothing, and exiting 0 would turn the re-run green. Fail it instead.
    """
    _report_deadline_stop(error)
    run_attempt: int = int(os.environ.get("GITHUB_RUN_ATTEMPT") or 1)
    if isinstance(error, DeadlineBeforeStartupError) and run_attempt > 1:
        print(
            f"::error title=Weekly scan deadline::Re-run attempt {run_attempt} "
            "started after the weekly deadline; nothing was scanned. Start a "
            "workflow_dispatch run (no deadline) to fill in these models."
        )
        return 1
    return 0


def _stop_worker(proc: multiprocessing.process.BaseProcess) -> None:
    """Terminate a still-running batch worker, escalating to SIGKILL if needed."""
    proc.terminate()
    proc.join(timeout=30)
    if proc.is_alive():
        print(
            f"{ts()}     batch: worker pid={proc.pid} did not "
            f"exit after SIGTERM — sending SIGKILL"
        )
        proc.kill()
        proc.join(timeout=30)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPYRE_TESTS_DIR = _REPO_ROOT / "tests" / "spyre"
_TESTS_DIR = _REPO_ROOT / "tests"
_UTILS_DIR = _REPO_ROOT / "utils"
for _p in (_SPYRE_TESTS_DIR, _TESTS_DIR, _UTILS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# How many models one spawned child processes before exiting. Higher values
# amortize per-child spawn + import + kernel-teardown cost (~15 s currently)
# across more work. Lower values reduce the blast radius when the Spyre
# driver/state gets into a bad shape mid-batch.
GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS: int = 10
EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS: int = 90


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
            "Fetch the --top-k-downloads / --top-k-trending catalog (plus the "
            "curated and registry models) for --mode, apply the same "
            "pre-filters, record the terminal rows, and evaluate the survivors — "
            "all in one pass. For manual runs where there is no shard file."
        ),
    )
    fetch_opts = parser.add_argument_group("--fetch options")
    fetch_opts.add_argument(
        "--top-k-downloads",
        type=int,
        default=1000,
        help=(
            "With --fetch: top models by downloads over the last 30 days, "
            "counted after the keep() filter (default: 1000; 0 skips it)."
        ),
    )
    fetch_opts.add_argument(
        "--top-k-trending",
        type=int,
        default=1000,
        help=(
            "With --fetch: top models by the Hub's trending score, counted "
            "after the keep() filter (default: 1000; 0 skips it)."
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
    parser.add_argument(
        "--deadline",
        type=_parse_deadline,
        default=None,
        metavar="ISO8601",
        help=(
            "Hard wall-clock stop, e.g. 2026-11-09T14:00:00Z. No batch starts "
            "after it, and a batch still running when it passes is stopped "
            "without recording its models. Default: no deadline."
        ),
    )
    args = parser.parse_args(argv)
    # Reject the silent no-op of passing a fetch tuning flag without --fetch.
    if not args.fetch:
        for flag, dest in (
            ("--top-k-downloads", "top_k_downloads"),
            ("--top-k-trending", "top_k_trending"),
            ("--max-params", "max_params"),
        ):
            if getattr(args, dest) != parser.get_default(dest):
                parser.error(f"{flag} only applies with --fetch")
    return args


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
    top_k_downloads: int,
    top_k_trending: int,
    max_params: int,
    snapshot_date: date,
    deadline: datetime | None = None,
) -> None:
    """Evaluate a model list on Spyre, one spawned worker per batch.

    Exactly one of *model_list_file* (a shard prepared by
    ``generate_weekly_shards``) and *fetch* supplies the list; argparse enforces
    that. Both arrive pre-filtered, so every model reaching the batch loop is
    expected to need a card.

    Owns the sink for the whole run: it is created here, written to by both the
    ``--fetch`` pre-filter and the evaluation loop, flushed at each batch
    boundary, and closed once in the ``finally`` below. ``top_k_downloads``,
    ``top_k_trending`` and ``max_params`` apply only under *fetch*.

    Raises:
        HardwareExceptionAbortError: a batch reported ``hardware_exception``, so
            the accelerator is unreachable and the remaining batches are skipped.
        DeadlineReachedError: *deadline* passed before every batch finished.
    """
    from tests.spyre.weekly_generation.sink.result_sink import ResultSink

    print(f"{ts()} Starting main.")
    print(f"{ts()} Snapshot date: {snapshot_date}")
    if deadline is not None:
        print(f"{ts()} Deadline: {deadline.isoformat()}")
        if _seconds_until(deadline) <= 0:
            raise DeadlineBeforeStartupError(
                f"deadline {deadline.isoformat()} reached before shard startup"
            )

    models_per_process = {
        ModelType.GENERATIVE: GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS,
        ModelType.EMBEDDING: EMBEDDING_NUMBER_OF_MODEL_PER_PROCESS,
    }

    adapter_dates: dict[str, str | None] = _get_adapter_dates()

    sink: ResultSink = create_sink(
        model_type=model_type,
        write_to_csv=write_to_csv,
        model_list_file=model_list_file,
    )

    # All three are read by the finally block, so they are bound before the try
    # opens — otherwise a failure while building the model list would raise
    # UnboundLocalError from the summary and mask the real error.
    processed = 0
    total = 0
    overall_start = time.monotonic()

    # The try opens before the model list is built so that a failure while
    # fetching (a Hub outage, say) still closes the sink — under --fetch the
    # pre-filter has by then already written terminal rows worth keeping.
    try:
        if fetch:
            rows: list[dict] = fetch_and_filter(
                model_type=model_type,
                snapshot_date=snapshot_date,
                top_k_downloads=top_k_downloads,
                top_k_trending=top_k_trending,
                sink=sink,
                max_params=max_params,
            )
        else:
            assert model_list_file is not None  # argparse guarantees one of the two
            print(f"{ts()} Loading model list from '{model_list_file}'.")
            rows = json.loads(model_list_file.read_text())

        already_recorded: set[str] = sink.get_models_at_snapshot_date(
            snapshot_date=snapshot_date
        )
        if already_recorded:
            before = len(rows)
            rows = [r for r in rows if r["model_id"] not in already_recorded]
            print(
                f"{ts()} Skipping {before - len(rows)} model(s) already recorded "
                f"for {snapshot_date} ({len(rows)} remaining)."
            )

        total = len(rows)
        print(f"{ts()} Will evaluate {total} model(s).")

        batch_size = models_per_process[model_type]
        batches: list[list[dict]] = _chunk_into_batches(
            rows=rows,
            batch_size=batch_size,
        )
        total_batches: int = len(batches)

        ctx = multiprocessing.get_context("spawn")

        for batch_idx, batch in enumerate(batches, start=1):
            if deadline is not None and _seconds_until(deadline) <= 0:
                raise DeadlineReachedError(
                    f"deadline {deadline.isoformat()} reached before batch "
                    f"{batch_idx}/{total_batches}; skipped "
                    f"{total_batches - batch_idx + 1} batch(es)"
                )
            batch_start = time.monotonic()
            batch_paths = [str(r["model_id"]) for r in batch]
            batch_curated = [r["curated"] for r in batch]
            print(
                f"\n{ts()} [batch {batch_idx}/{total_batches}] {len(batch)} model(s) "
                f"(overall elapsed: {batch_start - overall_start:.0f}s)"
            )
            for path, curated in zip(batch_paths, batch_curated):
                print(f"{ts()} - {path}" + ("\t(curated)" if curated else ""))

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

            # Never wait past the deadline: if it arrives first, stop the
            # worker and record nothing for this batch. Its complete results
            # are unavailable; a timeout/crash row would be a false verdict.
            # Decide which limit bounds the join *before* joining, and branch on
            # that decision afterwards. Re-reading the wall clock after the join
            # could disagree with it (e.g. an NTP step backwards) and misfile a
            # deadline stop as worker_timeout — a permanent false verdict, since
            # a re-run skips every model that already has a row for the date.
            seconds_left: float | None = (
                None if deadline is None else max(0.0, _seconds_until(deadline))
            )
            deadline_bounds_join: bool = (
                seconds_left is not None and seconds_left <= timeout
            )
            join_timeout: float = (
                seconds_left
                if seconds_left is not None and deadline_bounds_join
                else float(timeout)
            )
            proc.join(timeout=join_timeout)

            if proc.is_alive() and deadline is not None and deadline_bounds_join:
                print(
                    f"{ts()}     batch: deadline {deadline.isoformat()} reached — "
                    f"terminating pid={proc.pid}; no rows recorded for its "
                    f"{len(batch)} model(s)"
                )
                _stop_worker(proc)
                raise DeadlineReachedError(
                    f"deadline {deadline.isoformat()} reached during batch "
                    f"{batch_idx}/{total_batches}; skipped "
                    f"{total_batches - batch_idx + 1} batch(es)"
                )

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
                _stop_worker(proc)

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

            # Weights stay cached: the shared HF_HOME keeps every scanned model warm.
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
            # below (sink close, summary), then propagates
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
            top_k_downloads=args.top_k_downloads,
            top_k_trending=args.top_k_trending,
            max_params=args.max_params,
            snapshot_date=args.snapshot_date,
            deadline=args.deadline,
        )
    except DeadlineReachedError as e:
        # Expected for the weekly scan (issue #745): exit 0 so the shard is not
        # reported as a failure — unless this is a re-run attempt that started
        # after the deadline and scanned nothing. Any sink opened by main() has
        # been closed.
        exit_code: int = _deadline_stop_exit_code(e)
        if exit_code:
            sys.exit(exit_code)
    except HardwareExceptionAbortError as e:
        # Non-zero exit so CI / GHA scheduled runs can alert. main()'s
        # `finally` has already flushed the sink and cleaned the cache.
        print(f"{ts()} Exiting with status 1 ({e}).")
        sys.exit(1)
