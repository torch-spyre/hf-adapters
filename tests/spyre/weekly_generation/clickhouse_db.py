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
