#!/usr/bin/env python3
"""
Creates or drops the model_spyre_support table in ClickHouse.

Columns:
  - model_name         (String)  – unique model identifier
  - architecture       (String)  – model architecture (e.g. llama, mistral)
  - adapter_name       (String)  – adapter name (e.g. LoRA config name)
  - added_date         (Date?)   – date the adapter was added to the git repo (optional)
  - snapshot_date      (Date)    – date this weekly snapshot was taken
  - verified_on_cpu    (Bool)    – passes on CPU
  - verified_on_gpu    (Bool)    – passes on GPU
  - verified_on_spyre  (Bool)    – passes on Spyre
  - num_downloads      (UInt64)  – number of downloads

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
#   repo_root/.github/scripts/create_model_spyre_table.py)
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
    "architecture",
    "adapter_name",
    "added_date",
    "snapshot_date",
    "verified_on_cpu",
    "verified_on_gpu",
    "verified_on_spyre",
    "num_downloads",
)


def _make_create_table_sql(table_name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {DATABASE}.{table_name}
(
    model_name        String,
    architecture      String,
    adapter_name      String,
    added_date        Nullable(Date),
    snapshot_date     Date,
    verified_on_cpu   Bool,
    verified_on_gpu   Bool,
    verified_on_spyre Bool,
    num_downloads     UInt64
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
    architecture: str,
    adapter_name: str,
    added_date: date | None,
    snapshot_date: date,
    verified_on_cpu: bool,
    verified_on_gpu: bool,
    verified_on_spyre: bool,
    num_downloads: int,
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
                architecture,
                adapter_name,
                added_date,
                snapshot_date,
                verified_on_cpu,
                verified_on_gpu,
                verified_on_spyre,
                num_downloads,
            ]
        ],
        column_names=list(TABLE_COLUMNS),
    )
    return True
