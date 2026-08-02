"""Build a model list for ``weekly_test.py`` to evaluate — the manual-run path.

``weekly_test.py`` requires ``--model-list-file`` and no longer fetches anything
itself. In CI that file comes from ``.github/scripts/generate_weekly_shards.py``,
which also splits it across parallel jobs. This script is the equivalent for a
single manual run on the pod: fetch the top-K list, apply the same pre-filters,
record the terminal verdicts, and write out what is left.

    # DB-backed (the normal case — honours the 10-day skip window)
    python tests/spyre/weekly_generation/prepare_weekly_model_list.py \
        --mode embedding --top-k 200 --output /tmp/embedding.json
    python tests/spyre/weekly_generation/weekly_test.py \
        --mode embedding --model-list-file /tmp/embedding.json

    # No database at all (see the --no-db warning)
    python tests/spyre/weekly_generation/prepare_weekly_model_list.py \
        --mode embedding --top-k 50 --output /tmp/e.json --no-db

Whichever destination this script writes its terminal rows to, point
``weekly_test.py`` at the same one, or the two halves of a run end up split
across a CSV and the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.spyre.weekly_generation.failure_categories import (  # noqa: E402
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import (  # noqa: E402
    prefilter_models,
)
from utils.utilities import ts  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["embedding", "generative"],
        required=True,
        help=(
            "Which catalog to fetch, and which ClickHouse table to consult and "
            "write to. Must match the --mode passed to weekly_test.py."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10_000,
        help="Number of top models to fetch by downloads (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="MODEL_LIST_JSON",
        help="Where to write the model list for weekly_test.py --model-list-file.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=MAX_NUMBER_PARAMS,
        help=(
            "Reject models above this parameter count "
            f"(default: {MAX_NUMBER_PARAMS:,})."
        ),
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--write-to-csv",
        type=Path,
        default=None,
        metavar="RESULTS_CSV",
        help=(
            "Record the terminal rows in this CSV instead of ClickHouse, and use "
            "it as the skip-window source. Pass the same file to weekly_test.py "
            "--write-to-csv so the run's rows all land together."
        ),
    )
    destination.add_argument(
        "--no-db",
        action="store_true",
        help=(
            "Do not read or write any result store. The skip window is NOT "
            "applied, so the output includes models that were scanned recently, "
            "and the terminal verdicts are discarded rather than recorded. For "
            "inspecting what a fetch returns — not for a production scan."
        ),
    )
    return parser.parse_args(argv)


def _fetch_rows(mode: str, top_k: int) -> list[dict]:
    """Fetch the top-*top_k* catalog for *mode*, JSON-ready.

    ``model_info`` is popped here rather than later: ``prefilter_models`` hands
    back the same dict objects it was given, so a surviving (non-serializable)
    ModelInfo would break the json.dumps below.
    """
    if mode == "generative":
        from utils.fetch_top_generative_models import fetch_top_generative_models

        rows: list[dict] = fetch_top_generative_models(limit=top_k)
    else:
        from utils.fetch_top_embedding_models import fetch_top_embedding_models

        rows = fetch_top_embedding_models(limit=top_k)
    for row in rows:
        row.pop("model_info", None)
    return rows


def _make_sink(args: argparse.Namespace, snapshot_date: date):
    """Build the sink that supplies the skip window and takes the terminal rows.

    Returns None under ``--no-db``.

    The two backends differ deliberately in ``dedup_guard``:

    * ClickHouse — guard OFF. ``prefilter_models`` has already consulted the
      identical skip set through ``should_insert_row``, so a second check at
      write time could only discard a row we chose to write.
    * CSV — guard ON. The file is its own skip-window index and may hold rows
      from arbitrary earlier runs at arbitrary dates, so re-running against the
      same CSV should behave like re-running against the database rather than
      appending duplicates.
    """
    if args.no_db:
        # Imported below rather than at module scope so --no-db works on a host
        # with no clickhouse_connect installed (result_sink pulls it in
        # transitively via clickhouse_db).
        return None

    from tests.spyre.weekly_generation.result_sink import (
        ClickHouseResultSink,
        CsvResultSink,
        EmbeddingGenerativeMode,
    )

    if args.write_to_csv:
        return CsvResultSink(path=args.write_to_csv, today=snapshot_date)
    return ClickHouseResultSink(
        embedding_generative=EmbeddingGenerativeMode(args.mode),
        today=snapshot_date,
        dedup_guard=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    snapshot_date = date.today()

    if args.no_db:
        print(
            f"{ts()} WARNING: --no-db — the skip window is NOT applied and no "
            f"terminal rows are recorded. '{args.output}' will include models "
            f"already scanned recently, and feeding it to weekly_test.py means "
            f"re-evaluating them.",
            flush=True,
        )

    rows = _fetch_rows(args.mode, args.top_k)
    print(f"{ts()} Fetched {len(rows)} {args.mode} model(s).")

    sink = _make_sink(args, snapshot_date)
    should_scan = sink.should_insert_row if sink is not None else (lambda _: True)

    result = prefilter_models(rows, should_scan=should_scan, max_params=args.max_params)

    written = 0
    if sink is not None:
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        with sink:
            written = write_skipped_rows(
                sink, result.skipped, snapshot_date=snapshot_date
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.keep))

    print(
        f"\n{ts()} {len(rows)} fetched -> {len(result.keep)} to evaluate "
        f"({len(result.window_skipped)} within the skip window, "
        f"{written} terminal row(s) recorded)\n"
        f"{ts()} Breakdown: {result.counts}\n"
        f"{ts()} Wrote '{args.output}'. Run:\n"
        f"    python tests/spyre/weekly_generation/weekly_test.py "
        f"--mode {args.mode} --model-list-file {args.output}"
    )


if __name__ == "__main__":
    main()
