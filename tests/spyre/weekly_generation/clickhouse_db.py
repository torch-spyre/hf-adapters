"""ClickHouse client factory for the model_spyre_support tables.

The table shape itself lives in ``table_schema`` — a leaf module with no
dependencies — and is re-exported here so existing callers keep working. Import
from ``table_schema`` directly when you only need the column list or DDL:
importing *this* module pulls in ``clickhouse_connect`` and reads ``.env`` off
disk, neither of which a consumer of a column tuple should have to pay for.

Credentials are loaded from a .env file at the repo root, then fall back to
environment variables already set in the shell. Copy .env.example → .env and fill
in the values before running.

  CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASS, CLICKHOUSE_DB
"""

import os
from datetime import date
from pathlib import Path

import clickhouse_connect
from dotenv import load_dotenv

# Re-exported for backwards compatibility; table_schema is the source of truth.
from tests.spyre.weekly_generation.table_schema import (
    DATABASE,
    EMBEDDING_CREATE_TABLE_SQL,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_CREATE_TABLE_SQL,
    GENERATIVE_TABLE_NAME,
    TABLE_COLUMNS,
)

__all__ = [
    "DATABASE",
    "EMBEDDING_CREATE_TABLE_SQL",
    "EMBEDDING_TABLE_NAME",
    "GENERATIVE_CREATE_TABLE_SQL",
    "GENERATIVE_TABLE_NAME",
    "TABLE_COLUMNS",
    "get_client",
    "table_exists",
]

# Locate the repo root (.env lives three directories above this module:
#   repo_root/tests/spyre/weekly_generation/clickhouse_db.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", ".."),
        secure=True,
    )


def table_exists(client, table_name: str) -> bool:
    result = client.query(
        "SELECT count() FROM system.tables "
        "WHERE database = {db:String} AND name = {tbl:String}",
        parameters={"db": DATABASE, "tbl": table_name},
    )
    return result.result_rows[0][0] > 0


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def _parse_date(value: str) -> date:
    """Parse a date string in either ISO (YYYY-MM-DD) or US (MM/DD/YYYY) format."""
    v = value.strip()
    if "/" in v:
        return date(*(int(x) for x in reversed(v.split("/"))))
    return date.fromisoformat(v)


_NULL_SENTINELS = {"", "\\n", "\\N", "null", "none", "na", "n/a"}


def _parse_nullable_date(value: str | None) -> date | None:
    v = (value or "").strip()
    return _parse_date(v) if v.lower() not in _NULL_SENTINELS else None


def _parse_nullable_str(value: str | None) -> str | None:
    v = (value or "").strip()
    return v if v.lower() not in _NULL_SENTINELS else None


def _parse_int(value: str | None) -> int:
    """Parse an integer, tolerating scientific notation and NULL sentinels.

    ``int(s)`` rejects any string containing ``.`` or ``e``, but some CSV
    exporters (spreadsheets in particular) render large UInt64 values in
    scientific form. Route through ``float`` first so those parse. Empty
    input and ClickHouse's ``\\N`` NULL sentinel map to 0, since the target
    columns (num_downloads, parameters_number) are non-nullable UInt64.
    """
    v = (value or "").strip()
    if not v or v.lower() in _NULL_SENTINELS:
        return 0
    try:
        return int(v)
    except ValueError:
        return int(float(v))


def import_csv(sink, csv_path: str, flush_every: int = 5000) -> tuple[int, int]:
    """Read *csv_path* and insert rows into *sink*, respecting its dedup guard.

    Uses ``sink.add_entry()`` so ``should_insert_row`` is applied for every row.
    Flushes the sink every *flush_every* buffered rows so a large CSV does not
    accumulate into a single multi-hundred-thousand-row INSERT that overruns the
    HTTP write timeout. Returns an ``(inserted, skipped)`` tuple.
    """
    import csv

    inserted = skipped = malformed = 0
    buffered_since_flush: int = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            snapshot_raw: str = (row.get("snapshot_date") or "").strip()
            model_name: str = (row.get("model_name") or "").strip()
            if not snapshot_raw or not model_name:
                malformed += 1
                print(
                    f"    warn: skipping CSV row {reader.line_num} "
                    f"(model_name={model_name!r}, snapshot_date={snapshot_raw!r}): "
                    "missing required field"
                )
                continue
            try:
                snapshot_date_val: date = _parse_date(snapshot_raw)
            except ValueError:
                malformed += 1
                print(
                    f"    warn: skipping CSV row {reader.line_num} "
                    f"(model_name={model_name!r}): "
                    f"invalid snapshot_date={snapshot_raw!r}"
                )
                continue
            written = sink.add_entry(
                model_name=model_name,
                config_class=(row.get("config_class") or "").strip(),
                adapter_name=(row.get("adapter_name") or "").strip(),
                added_date=_parse_nullable_date(row.get("added_date")),
                snapshot_date=snapshot_date_val,
                verified_on_cpu=_parse_bool(row.get("verified_on_cpu") or ""),
                verified_on_gpu=_parse_bool(row.get("verified_on_gpu") or ""),
                verified_on_spyre=_parse_bool(row.get("verified_on_spyre") or ""),
                curated=_parse_bool(row.get("curated") or ""),
                num_downloads=_parse_int(row.get("num_downloads")),
                family=(row.get("family") or "").strip(),
                architecture=(row.get("architecture") or "").strip(),
                parameters_number=_parse_int(row.get("parameters_number")),
                failure_category=_parse_nullable_str(row.get("failure_category")),
                error=_parse_nullable_str(row.get("error")),
            )
            if written:
                inserted += 1
                buffered_since_flush += 1
                if buffered_since_flush >= flush_every:
                    sink.flush()
                    buffered_since_flush = 0
            else:
                skipped += 1

    if malformed:
        print(f"    import_csv: {malformed} malformed row(s) skipped.")
    return inserted, skipped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClickHouse table management utility.")

    add_csv_group = parser.add_argument_group("import CSV")
    add_csv_group.add_argument(
        "--add_csv", metavar="CSV_FILE", help="CSV file to import into the table."
    )
    add_csv_group.add_argument(
        "--table_name", metavar="TABLE_NAME", help="Target table for --add_csv."
    )
    add_csv_group.add_argument(
        "--flush_every",
        metavar="N",
        type=int,
        default=5000,
        help=(
            "Flush the sink every N inserted rows (default 5000). Lower this if "
            "the server times out on the bulk INSERT; raise it for fewer round trips."
        ),
    )

    args = parser.parse_args()

    if args.add_csv or args.table_name:
        if not args.add_csv or not args.table_name:
            parser.error("--add_csv and --table_name must be used together.")
        csv_file = args.add_csv
        table = args.table_name
        # Lazy import to avoid a circular dependency (clickhouse_sink imports from this module).
        from tests.spyre.weekly_generation.model_type import ModelType
        from tests.spyre.weekly_generation.sink.clickhouse_sink import (
            ClickHouseResultSink,
        )

        if table == EMBEDDING_TABLE_NAME:
            mode = ModelType.EMBEDDING
        elif table == GENERATIVE_TABLE_NAME:
            mode = ModelType.GENERATIVE
        else:
            parser.error(
                f"Unknown table '{table}'. Expected one of: "
                f"{EMBEDDING_TABLE_NAME}, {GENERATIVE_TABLE_NAME}."
            )
        with ClickHouseResultSink(mode) as sink:
            inserted, skipped = import_csv(sink, csv_file, flush_every=args.flush_every)
        print(
            f"Inserted {inserted} row(s) into '{DATABASE}.{table}' ({skipped} skipped by dedup guard)."
        )
    else:
        parser.print_help()
