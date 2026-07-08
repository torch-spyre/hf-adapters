#!/usr/bin/env python3
"""
Creates or drops the model_spyre_support table in ClickHouse.

Columns:
  - model_name         (String)  – unique model identifier
  - architecture       (String)  – model architecture (e.g. llama, mistral)
  - adapter_name       (String)  – adapter name (e.g. LoRA config name)
  - added_date         (Date)    – date the adapter was added to the git repo
  - snapshot_date      (Date)    – date this weekly snapshot was taken
  - verified_on_cpu    (Bool)    – passes on CPU
  - verified_on_gpu    (Bool)    – passes on GPU
  - verified_on_spyre  (Bool)    – passes on Spyre
  - num_downloads      (UInt64)  – number of downloads

Credentials are loaded from a .env file at the repo root (two levels above this
script), then fall back to environment variables already set in the shell.
Copy .env.example → .env and fill in the values before running.

  CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASS, CLICKHOUSE_DB

Usage:
    # check / create table
    python3 .github/scripts/create_model_spyre_table.py

    # drop table (asks for confirmation)
    python3 .github/scripts/create_model_spyre_table.py --drop

    # drop table without confirmation prompt
    python3 .github/scripts/create_model_spyre_table.py --drop --yes

    # insert a single row
    python3 .github/scripts/create_model_spyre_table.py --insert \\
        --model-name  "meta-llama/Llama-3.2-1B" \\
        --architecture llama \\
        --adapter-name my_lora \\
        --added-date   2025-01-15 \\
        --snapshot-date 2025-07-01 \\
        --verified-on-cpu \\
        --num-downloads 42000
"""

import argparse
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
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


TABLE_NAME = "embedding_model_spyre_support"
DATABASE = "spyre"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DATABASE}.{TABLE_NAME}
(
    model_name        String,
    architecture      String,
    adapter_name      String,
    added_date        Date,
    snapshot_date     Date,
    verified_on_cpu   Bool,
    verified_on_gpu   Bool,
    verified_on_spyre Bool,
    num_downloads     UInt64
)
ENGINE = ReplacingMergeTree(snapshot_date)
ORDER BY (model_name, snapshot_date)
"""


def table_exists(client) -> bool:
    result = client.query(
        "SELECT count() FROM system.tables "
        "WHERE database = {db:String} AND name = {tbl:String}",
        parameters={"db": DATABASE, "tbl": TABLE_NAME},
    )
    return result.result_rows[0][0] > 0


def print_table(client) -> None:
    result = client.query(
        "SELECT name, type FROM system.columns "
        "WHERE database = {db:String} AND table = {tbl:String} "
        "ORDER BY position",
        parameters={"db": DATABASE, "tbl": TABLE_NAME},
    )
    print(f"Table '{DATABASE}.{TABLE_NAME}' already exists with columns:")
    for col_name, col_type in result.result_rows:
        print(f"  {col_name:<25} {col_type}")


def should_insert_row(
    client,
    *,
    model_name: str,
    snapshot_date: date,
    verified_on_cpu: bool,
) -> bool:
    """
    Return True when the row may be inserted.

    If the model already exists in the table and the most-recent snapshot_date
    for that model is within 3 days of *snapshot_date*, the row is only allowed
    when *verified_on_cpu* is True.
    """
    result = client.query(
        "SELECT max(snapshot_date) FROM {db:Identifier}.{tbl:Identifier} "
        "WHERE model_name = {model:String}",
        parameters={"db": DATABASE, "tbl": TABLE_NAME, "model": model_name},
    )
    existing_snapshot = result.result_rows[0][0]  # None when no rows match

    if existing_snapshot is None:
        return True  # model not yet in the table — always insert

    delta = abs((snapshot_date - existing_snapshot).days)
    if delta < 3:
        return verified_on_cpu  # too close: only insert when CPU-verified

    return True


def insert_model_row(
    client,
    *,
    model_name: str,
    architecture: str,
    adapter_name: str,
    added_date: date,
    snapshot_date: date,
    verified_on_cpu: bool = False,
    verified_on_gpu: bool = False,
    verified_on_spyre: bool = False,
    num_downloads: int = 0,
) -> bool:
    """Insert a single row into model_spyre_support.

    Calls should_insert_row first; returns False (and skips the write) when
    the guard rejects the row, True when the row was actually inserted.
    """
    if not should_insert_row(
        client,
        model_name=model_name,
        snapshot_date=snapshot_date,
        verified_on_cpu=verified_on_cpu,
    ):
        return False

    client.insert(
        TABLE_NAME,
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
        column_names=[
            "model_name",
            "architecture",
            "adapter_name",
            "added_date",
            "snapshot_date",
            "verified_on_cpu",
            "verified_on_gpu",
            "verified_on_spyre",
            "num_downloads",
        ],
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Manage the model_spyre_support table in ClickHouse."
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the table instead of creating it.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt when dropping.",
    )
    parser.add_argument(
        "--insert",
        action="store_true",
        help="Insert a single row into the table.",
    )
    # --insert fields
    parser.add_argument("--model-name", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--adapter-name", default="")
    parser.add_argument(
        "--added-date", default=str(date.today()), help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--snapshot-date", default=str(date.today()), help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument("--verified-on-cpu", action="store_true")
    parser.add_argument("--verified-on-gpu", action="store_true")
    parser.add_argument("--verified-on-spyre", action="store_true")
    parser.add_argument("--num-downloads", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("Connected.\n")

    if args.drop:
        if not table_exists(client):
            print(f"Table '{DATABASE}.{TABLE_NAME}' does not exist — nothing to drop.")
            return
        if not args.yes:
            confirm = input(
                f"Are you sure you want to drop '{DATABASE}.{TABLE_NAME}'? "
                "All data will be lost. [y/N] "
            )
            if confirm.strip().lower() != "y":
                print("Aborted.")
                return
        client.command(f"DROP TABLE IF EXISTS {DATABASE}.{TABLE_NAME}")
        print(f"Table '{DATABASE}.{TABLE_NAME}' dropped successfully.")
    elif args.insert:
        if not args.model_name:
            parser.error("--insert requires --model-name")
        inserted = insert_model_row(
            client,
            model_name=args.model_name,
            architecture=args.architecture,
            adapter_name=args.adapter_name,
            added_date=date.fromisoformat(args.added_date),
            snapshot_date=date.fromisoformat(args.snapshot_date),
            verified_on_cpu=args.verified_on_cpu,
            verified_on_gpu=args.verified_on_gpu,
            verified_on_spyre=args.verified_on_spyre,
            num_downloads=args.num_downloads,
        )
        if inserted:
            print(
                f"Inserted row for model '{args.model_name}' into "
                f"'{DATABASE}.{TABLE_NAME}'."
            )
        else:
            print(
                f"Skipped: model '{args.model_name}' already has a snapshot within "
                f"3 days of {args.snapshot_date} and verified_on_cpu is False."
            )
    else:
        if table_exists(client):
            print_table(client)
        else:
            client.command(CREATE_TABLE_SQL)
            print(
                f"Table '{DATABASE}.{TABLE_NAME}' did not exist — created successfully."
            )


if __name__ == "__main__":
    main()
