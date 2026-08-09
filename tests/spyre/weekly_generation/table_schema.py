"""Table shape for the model_spyre_support tables. Pure data, no dependencies.

Deliberately a leaf module: nothing here imports ``clickhouse_connect``,
``dotenv``, or anything else outside the standard library, and importing it has no
side effects. That is the point — ``csv_sink`` needs ``TABLE_COLUMNS`` so its
output is column-for-column identical to the live table, but a ``--write-to-csv``
run should not need the ClickHouse driver installed or read a ``.env`` file off
disk to get a tuple of column names. ``clickhouse_db`` is where the client and
credentials live, and it re-exports everything below for its existing callers.

The DDL and ``TABLE_COLUMNS`` are kept in this one file, adjacent, because they
must agree: the column list is what ``ClickHouseResultSink`` passes as
``column_names`` to a positional bulk insert, so a mismatch against the CREATE
TABLE body silently writes values into the wrong columns. ``test_table_schema.py``
asserts the two match by parsing the DDL, so the requirement is enforced rather
than merely commented.

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
  - curated            (Bool)     – model came from a hand-maintained curated list
                                    rather than the top-K download ranking
"""

from __future__ import annotations

EMBEDDING_TABLE_NAME = "embedding_model_spyre_support"
GENERATIVE_TABLE_NAME = "generative_model_spyre_support"
DATABASE = "spyre"

# Single source of truth for the Python-facing column list. Order matches the
# CREATE TABLE DDL below and the positional values the sinks write — enforced by
# test_table_schema.py rather than left to reviewers.
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
    "curated",
)


def _make_create_table_sql(table_name: str) -> str:
    """DDL for one snapshot table. Column order must match ``TABLE_COLUMNS``.

    ``ReplacingMergeTree(snapshot_date)`` is what makes a duplicate row a
    tolerable failure mode: same-day rows for one model collapse on merge, which
    is why the weekly scan prefers writing a possible duplicate over dropping a
    result it was asked to produce.
    """
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
    error             Nullable(String),
    curated           Bool DEFAULT false
)
ENGINE = ReplacingMergeTree(snapshot_date)
ORDER BY (model_name, snapshot_date)
"""


EMBEDDING_CREATE_TABLE_SQL = _make_create_table_sql(EMBEDDING_TABLE_NAME)
GENERATIVE_CREATE_TABLE_SQL = _make_create_table_sql(GENERATIVE_TABLE_NAME)
