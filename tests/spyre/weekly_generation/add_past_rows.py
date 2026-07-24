"""Read and display rows from a historical CSV file.

Deduplicates by ``model_name``: when the same model_name appears in multiple
rows, only the row with the most recent ``snapshot_date`` is kept. Ties on
``snapshot_date`` are broken by keeping whichever row was seen last in the
input.

Usage::

    python tests/spyre/weekly_generation/add_past_rows.py <csv_file> <mode>

Arguments:

* ``csv_file``  Path to a CSV file whose columns match ``TABLE_COLUMNS``
                (see ``clickhouse_db.py``).
* ``mode``      Either ``embedding`` or ``generative``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_file",
        metavar="csv_file",
        help="Path to the CSV file to read.",
    )
    return parser.parse_args(argv)


def _load_latest_by_model_name(csv_path: str) -> dict[str, dict[str, str]]:
    """Load *csv_path* and return one row per unique ``model_name``.

    When the same ``model_name`` appears more than once, the returned row is
    the one with the greatest ``snapshot_date`` — compared as ISO 8601
    strings, which sort lexicographically the same way they sort as dates.
    Rows with a blank ``model_name`` are silently skipped. Rows with a blank
    ``snapshot_date`` sort as ``""`` and therefore lose any comparison, which
    is the intended behavior.
    """
    latest: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model_name: str = (row.get("model_name") or "").strip()
            if not model_name:
                continue
            snap: str = (row.get("snapshot_date") or "").strip()
            prev = latest.get(model_name)
            if prev is None or snap >= (prev.get("snapshot_date") or "").strip():
                latest[model_name] = row
    return latest


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print(f"Reading '{args.csv_file}'\n", flush=True)
    latest: dict[str, dict[str, str]] = _load_latest_by_model_name(args.csv_file)

    print(f"\nDone — {len(latest)} unique model_name(s).")

    # Weekly-window walk starting 7 days before the earliest adapter add-date.
    # `added_date` is the git add-date of the model's adapter file, so this is
    # roughly "the first date on which any of the tracked adapters existed,
    # minus one week". Some rows have an empty added_date (models with no
    # adapter yet); skip them to avoid pulling the earliest to "".
    added_dates: list[date] = []
    for row in latest.values():
        s: str = (row.get("added_date") or "").strip()
        if not s:
            continue
        try:
            added_dates.append(date.fromisoformat(s))
        except ValueError:
            # Non-ISO date in the CSV — skip rather than crash.
            continue

    if not added_dates:
        print("\nNo usable added_date found — skipping weekly-window walk.")
        return

    earliest: date = min(added_dates)
    starting_time: date = earliest - timedelta(days=7)
    today: date = date.today()

    print(
        f"\nEarliest added_date: {earliest.isoformat()}   "
        f"starting_time (earliest - 7d): {starting_time.isoformat()}   "
        f"today: {today.isoformat()}"
    )
    # Output goes next to the input CSV, with a "_enriched" suffix. E.g.
    # results_embedding_20072026_1535_deduplicate.csv →
    #   results_embedding_20072026_1535_deduplicate_enriched.csv
    input_path: Path = Path(args.csv_file)
    if input_path.suffix.lower() == ".csv":
        output_path: Path = input_path.with_name(input_path.stem + "_enriched.csv")
    else:
        # Fallback for input paths without a .csv extension — just append.
        output_path = input_path.with_name(input_path.name + "_enriched.csv")

    # Preserve the input's column order so downstream consumers (sink, DB)
    # see identical schema. Every projected row we write has the same keys.
    with input_path.open(newline="", encoding="utf-8") as fh:
        fieldnames: list[str] = list(csv.DictReader(fh).fieldnames or [])

    print(f"\nWriting projected rows to '{output_path}' …", flush=True)
    total_written: int = 0
    # Track (model_name, snapshot_date) pairs already emitted by the weekly
    # walk so we don't re-emit an original row that happens to land on one
    # of the projected weekly dates.
    written_keys: set[tuple[str, str]] = set()
    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()

        print("Weekly window dates (starting_time → today, step 7 days):")
        d: date = starting_time
        while d <= today:
            print(f"  {d.isoformat()}")
            for row in latest.values():
                # Shallow copy — the row is dict[str, str] and string values
                # are immutable, so deepcopy would be wasted work.
                new_row: dict[str, str] = row.copy()
                # If the adapter DID NOT exist yet on `d`, blank out every
                # field that only exists because of the adapter+run: the
                # adapter attribution, the verification flags, the failure
                # category (which historically would have been
                # "not-implemented-adapter") and any error message from a
                # later run.
                added: str = (row.get("added_date") or "").strip()
                if row.get("adapter_name") and added:
                    try:
                        row_added: date | None = date.fromisoformat(added)
                    except ValueError:
                        # Non-ISO date in the CSV — treat as "unknown, assume
                        # not yet added" so we don't accidentally credit an
                        # adapter that may not have existed.
                        row_added = None
                    # Strictly-greater: the adapter DID exist on its own add-date.
                    if row_added is None or row_added > d:
                        new_row["adapter_name"] = ""
                        new_row["added_date"] = ""
                        new_row["verified_on_cpu"] = "False"
                        new_row["verified_on_spyre"] = "False"
                        new_row["failure_category"] = "not-implemented-adapter"
                        new_row["error"] = ""

                new_row["snapshot_date"] = d.isoformat()

                writer.writerow({k: new_row.get(k, "") for k in fieldnames})
                total_written += 1
                model_name: str = (new_row.get("model_name") or "").strip()
                written_keys.add((model_name, new_row["snapshot_date"]))
            d += timedelta(days=7)

        # Append the original (post-dedup) rows, skipping any whose
        # (model_name, snapshot_date) was already produced by the weekly walk
        # above. This preserves the source rows' snapshot_date and other
        # fields verbatim — no blanking, no date rewrite.
        original_written: int = 0
        for row in latest.values():
            model_name = (row.get("model_name") or "").strip()
            snap: str = (row.get("snapshot_date") or "").strip()
            if (model_name, snap) in written_keys:
                continue
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            written_keys.add((model_name, snap))
            original_written += 1
            total_written += 1

    print(
        f"\nWrote {total_written} row(s) to '{output_path}' "
        f"({total_written - original_written} projected, "
        f"{original_written} original)."
    )


if __name__ == "__main__":
    main()
