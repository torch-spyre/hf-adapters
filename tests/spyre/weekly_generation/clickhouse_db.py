#!/usr/bin/env python3
"""
Creates or drops the model_spyre_support table in ClickHouse.

Columns:
  - model_name         (String)   – unique model identifier
  - config_class       (String)   – model config_class (e.g. BertConfig, Qwen3Config)
  - adapter_name       (String)   – adapter name (e.g. hf_bert, hf_gemma4)
  - added_date         (Date?)    – date the adapter was added to the git repo (optional - None if not existing)
  - snapshot_date      (Date)     – date this weekly snapshot was taken
  - verified_on_cpu    (Bool)     – passes on CPU
  - verified_on_gpu    (Bool)     – passes on GPU
  - verified_on_spyre  (Bool)     – passes on Spyre
  - num_downloads      (UInt64)   – number of downloads
  - family             (String)   – model family reported by the catalog
  - architecture       (String)   – model architecture reported by the catalog
  - parameters_number  (UInt64)   – number of model parameters
  - failure_category   (String?)  – classification for failures (optional - None if model passed)
  - error              (String?)  – error message if the model failed (optional - None if model passed)

Credentials are loaded from a .env file at the repo root (two levels above this
script), then fall back to environment variables already set in the shell.
Copy .env.example → .env and fill in the values before running.

  CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASS, CLICKHOUSE_DB

"""

import os
from datetime import date
from pathlib import Path

import clickhouse_connect
from dotenv import load_dotenv

# Locate the repo root (.env lives two directories above this script:
#   repo_root/.github/scripts/clickhouse_db.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


EMBEDDING_TABLE_NAME = "embedding_model_spyre_support"
GENERATIVE_TABLE_NAME = "generative_model_spyre_support"
DATABASE = "spyre"

# Single source of truth for the Python-facing column list. Order matches the
# CREATE TABLE DDL below and the positional values used by insert_model_row —
# keep the three in sync when adding columns.
TABLE_COLUMNS: tuple[str, ...] = (
    "model_name",
    "config_class",
    "adapter_name",
    "added_date",
    "snapshot_date",
    "verified_on_cpu",
    "verified_on_gpu",
    "verified_on_spyre",
    "num_downloads",
    "family",
    "architecture",
    "parameters_number",
    "failure_category",
    "error",
)


def _make_create_table_sql(table_name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {DATABASE}.{table_name}
(
    model_name        String,
    config_class      String,
    adapter_name      String,
    added_date        Nullable(Date),
    snapshot_date     Date,
    verified_on_cpu   Bool,
    verified_on_gpu   Bool,
    verified_on_spyre Bool,
    num_downloads     UInt64,
    family            String,
    architecture      String,
    parameters_number UInt64,
    failure_category  Nullable(String),
    error             Nullable(String)
)
ENGINE = ReplacingMergeTree(snapshot_date)
ORDER BY (model_name, snapshot_date)
"""


# The column names and order below MUST match ``TABLE_COLUMNS`` above.
# When adding/removing/renaming a column, update both in the same change.
CREATE_TABLE_SQL = _make_create_table_sql(EMBEDDING_TABLE_NAME)
GENERATIVE_CREATE_TABLE_SQL = _make_create_table_sql(GENERATIVE_TABLE_NAME)


def table_exists(client, table_name: str) -> bool:
    result = client.query(
        "SELECT count() FROM system.tables "
        "WHERE database = {db:String} AND name = {tbl:String}",
        parameters={"db": DATABASE, "tbl": table_name},
    )
    return result.result_rows[0][0] > 0


def print_table(client, table_name: str) -> None:
    result = client.query(
        "SELECT name, type FROM system.columns "
        "WHERE database = {db:String} AND table = {tbl:String} "
        "ORDER BY position",
        parameters={"db": DATABASE, "tbl": table_name},
    )
    print(f"Table '{DATABASE}.{table_name}' already exists with columns:")
    for col_name, col_type in result.result_rows:
        print(f"  {col_name:<25} {col_type}")


def insert_model_row(
    client,
    *,
    table_name: str,
    model_name: str,
    config_class: str,
    adapter_name: str,
    added_date: date | None,
    snapshot_date: date,
    verified_on_cpu: bool,
    verified_on_gpu: bool,
    verified_on_spyre: bool,
    num_downloads: int,
    family: str,
    architecture: str,
    parameters_number: int,
    failure_category: str | None,
    error: str | None,
) -> bool:
    """Insert a single row into the given table.

    The caller is responsible for any duplicate-suppression guard (e.g.
    ``ResultSink.should_insert_row``). This function always writes.
    """
    client.insert(
        table_name,
        [
            [
                model_name,
                config_class,
                adapter_name,
                added_date,
                snapshot_date,
                verified_on_cpu,
                verified_on_gpu,
                verified_on_spyre,
                num_downloads,
                family,
                architecture,
                parameters_number,
                failure_category,
                error,
            ]
        ],
        column_names=list(TABLE_COLUMNS),
    )
    return True


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def _parse_nullable_date(value: str) -> date | None:
    v = value.strip()
    return date.fromisoformat(v) if v else None


def _parse_nullable_str(value: str) -> str | None:
    v = value.strip()
    return v if v else None


def import_csv(sink, csv_path: str) -> tuple[int, int]:
    """Read *csv_path* and insert rows into *sink*, respecting its dedup guard.

    Uses ``sink.add_entry()`` so ``should_insert_row`` is applied for every row.
    Returns a ``(inserted, skipped)`` tuple.
    """
    import csv

    inserted = skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            written = sink.add_entry(
                model_name=row["model_name"].strip(),
                config_class=row["config_class"].strip(),
                adapter_name=row["adapter_name"].strip(),
                added_date=_parse_nullable_date(row["added_date"]),
                snapshot_date=date.fromisoformat(row["snapshot_date"].strip()),
                verified_on_cpu=_parse_bool(row["verified_on_cpu"]),
                verified_on_gpu=_parse_bool(row["verified_on_gpu"]),
                verified_on_spyre=_parse_bool(row["verified_on_spyre"]),
                num_downloads=int(row["num_downloads"].strip()),
                family=row["family"].strip(),
                architecture=row["architecture"].strip(),
                parameters_number=int(row["parameters_number"].strip()),
                failure_category=_parse_nullable_str(row["failure_category"]),
                error=_parse_nullable_str(row.get("error", "")),
            )
            if written:
                inserted += 1
            else:
                skipped += 1
    return inserted, skipped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClickHouse table management utility.")
    parser.add_argument(
        "--drop",
        metavar="TABLE_NAME",
        help="Drop the specified table after confirmation.",
    )

    add_csv_group = parser.add_argument_group("import CSV")
    add_csv_group.add_argument(
        "--add_csv", metavar="CSV_FILE", help="CSV file to import into the table."
    )
    add_csv_group.add_argument(
        "--table_name", metavar="TABLE_NAME", help="Target table for --add_csv."
    )

    args = parser.parse_args()

    if args.add_csv or args.table_name:
        if not args.add_csv or not args.table_name:
            parser.error("--add_csv and --table_name must be used together.")
        csv_file = args.add_csv
        table = args.table_name
        # Lazy import to avoid a circular dependency (result_sink imports from this module).
        from tests.spyre.weekly_generation.result_sink import (
            ClickHouseResultSink,
            EmbeddingGenerativeMode,
        )

        if table == EMBEDDING_TABLE_NAME:
            mode = EmbeddingGenerativeMode.EMBEDDING
        elif table == GENERATIVE_TABLE_NAME:
            mode = EmbeddingGenerativeMode.GENERATIVE
        else:
            parser.error(
                f"Unknown table '{table}'. Expected one of: "
                f"{EMBEDDING_TABLE_NAME}, {GENERATIVE_TABLE_NAME}."
            )
        with ClickHouseResultSink(mode) as sink:
            inserted, skipped = import_csv(sink, csv_file)
        print(
            f"Inserted {inserted} row(s) into '{DATABASE}.{table}' ({skipped} skipped by dedup guard)."
        )
    elif args.drop:
        table = args.drop
        answer = (
            input(f"Are you sure you want to drop table '{DATABASE}.{table}'? [y/N] ")
            .strip()
            .lower()
        )
        if answer == "y":
            client = get_client()
            if not table_exists(client, table):
                print(f"Table '{DATABASE}.{table}' does not exist.")
            else:
                client.command(f"DROP TABLE {DATABASE}.{table}")
                print(f"Table '{DATABASE}.{table}' dropped.")
        else:
            print("Aborted.")
    else:
        parser.print_help()
