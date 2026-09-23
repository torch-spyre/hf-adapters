#!/usr/bin/env python3
"""
ingest_module_tests.py
-----------------------
Reads the JSON produced by parse_module_test_logs.py and batch-inserts
the rows into two ClickHouse tables:

  module_test_suites   – one row per model per GHA run (summary counts)
  module_test_variants – one row per individual module test
                         (module, test_label, status, shapes/strides/dtypes,
                          prefill and decode inputs, cpu_fallback_ops)

Table lifecycle (every run):
  1. CREATE TABLE IF NOT EXISTS  module_test_suites   (idempotent)
  2. CREATE TABLE IF NOT EXISTS  module_test_variants (idempotent)
  3. INSERT all suite rows   (ReplacingMergeTree deduplicates on suite_id)
  4. INSERT all variant rows (ReplacingMergeTree deduplicates on variant_id)

Re-ingesting the same GHA run is safe: suite_id / variant_id are
deterministic SHA-256 digests of the run + identifier fields.

Usage:
    python3 ingest_module_tests.py \\
        --json-file module_tests_88094510086.json \\
        --workflow  "run-tests"                    \\
        --branch    "main"                         \\
        --sha       "abc123..."                    \\
        --run-id    "88094510086"

Environment variables (required):
    CLICKHOUSE_HOST   CLICKHOUSE_PORT   CLICKHOUSE_USER
    CLICKHOUSE_PASS   CLICKHOUSE_DB
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect

# ─────────────────────────────────────────────────────────────────────────────
# ClickHouse DDL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_SUITES = """
CREATE TABLE IF NOT EXISTS module_test_suites
(
    -- Unique primary key: SHA-256(gha_run_id || suite_name)
    suite_id        FixedString(64),

    -- Provenance
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite identity
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',
    github_ref      String DEFAULT '',
    runner_name     String DEFAULT '',
    runner_node     String DEFAULT '',

    -- Summary counts
    total_tests           UInt32 DEFAULT 0,
    spyre_enabled_count   UInt32 DEFAULT 0,
    not_implemented_count UInt32 DEFAULT 0,
    cpu_fallback_count    UInt32 DEFAULT 0,
    spyre_failed_count    UInt32 DEFAULT 0,

    -- Timestamps
    test_date       Date,
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY suite_id
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

_CREATE_VARIANTS = """
CREATE TABLE IF NOT EXISTS module_test_variants
(
    -- Unique primary key: SHA-256(gha_run_id || suite_name || test_name || variant_seq)
    variant_id      FixedString(64),

    -- Foreign key → module_test_suites
    suite_id        FixedString(64),

    -- Provenance
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite identity
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',

    -- Test identity
    module          LowCardinality(String),   -- GraniteAttention, GraniteMLP, …
    test_label      LowCardinality(String),   -- test_forward | test_eager_vs_compile | test_layout_stride | test_with_cpu
    test_name       String,
    test_file       LowCardinality(String) DEFAULT '',
    classification  LowCardinality(String),   -- spyre_enabled | not_implemented | cpu_fallback | spyre_failed
    status          LowCardinality(String),   -- XPASS | XFAIL | FAILED
    dtype           LowCardinality(String) DEFAULT '',
    duration_s      Float32 DEFAULT 0,

    -- CPU-fallback ops (serialised JSON array of aten op name strings)
    has_cpu_fallback  UInt8 DEFAULT 0,
    cpu_fallback_ops  String DEFAULT '[]',

    -- Prefill inputs: shapes, strides, dtypes (serialised JSON)
    prefill_arg_shapes    String DEFAULT '{}',
    prefill_arg_strides   String DEFAULT '{}',
    prefill_arg_dtypes    String DEFAULT '{}',
    prefill_kwarg_shapes  String DEFAULT '{}',
    prefill_kwarg_strides String DEFAULT '{}',
    prefill_kwarg_dtypes  String DEFAULT '{}',

    -- Decode inputs: shapes, strides, dtypes (serialised JSON)
    decode_arg_shapes    String DEFAULT '{}',
    decode_arg_strides   String DEFAULT '{}',
    decode_arg_dtypes    String DEFAULT '{}',
    decode_kwarg_shapes  String DEFAULT '{}',
    decode_kwarg_strides String DEFAULT '{}',
    decode_kwarg_dtypes  String DEFAULT '{}',

    -- Timestamps
    test_date       Date,
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY variant_id
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

# ─────────────────────────────────────────────────────────────────────────────
# Column name lists (must match DDL order exactly)
# ─────────────────────────────────────────────────────────────────────────────

SUITE_COLS = [
    "suite_id",
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "github_ref",
    "runner_name",
    "runner_node",
    "total_tests",
    "spyre_enabled_count",
    "not_implemented_count",
    "cpu_fallback_count",
    "spyre_failed_count",
    "test_date",
    "triggered_at",
    "ingested_at",
]

VARIANT_COLS = [
    "variant_id",
    "suite_id",
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "module",
    "test_label",
    "test_name",
    "test_file",
    "classification",
    "status",
    "dtype",
    "duration_s",
    "has_cpu_fallback",
    "cpu_fallback_ops",
    "prefill_arg_shapes",
    "prefill_arg_strides",
    "prefill_arg_dtypes",
    "prefill_kwarg_shapes",
    "prefill_kwarg_strides",
    "prefill_kwarg_dtypes",
    "decode_arg_shapes",
    "decode_arg_strides",
    "decode_arg_dtypes",
    "decode_kwarg_shapes",
    "decode_kwarg_strides",
    "decode_kwarg_dtypes",
    "test_date",
    "triggered_at",
    "ingested_at",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
        verify=False,
    )


def _s(v, default: str = "") -> str:
    return str(v).strip() if v is not None else default


def _i(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _jstr(v) -> str:
    """Serialise to JSON string; empty/None → '{}'."""
    if not v:
        return "{}"
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _jlist(v) -> str:
    """Serialise a list to JSON string; empty/None → '[]'."""
    if not v:
        return "[]"
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


def _make_id(*parts) -> str:
    raw = "\x00".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_date(date_str: str):
    """Parse YYYY-MM-DD string → Python date object (or today)."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).date()


def _flatten_inputs(inputs_dict: dict, variant_label: str) -> dict:
    """
    Extract arg_shapes/strides/dtypes and kwarg_shapes/strides/dtypes
    from inputs[variant_label] and return them as JSON-serialised strings.

    Returns a dict with keys:
      arg_shapes, arg_strides, arg_dtypes,
      kwarg_shapes, kwarg_strides, kwarg_dtypes
    """
    vdata = inputs_dict.get(variant_label, {})
    args = vdata.get(
        "args", {}
    )  # {"arg_0": {"shape": [...], "stride": [...], "dtype": "..."}}
    kwargs = vdata.get("kwargs", {})  # {"hidden_states": {"shape": [...], ...}, ...}

    # args → { "arg_0": [1,128,4096], ... }
    arg_shapes = {k: v.get("shape", []) for k, v in args.items()}
    arg_strides = {k: v.get("stride", []) for k, v in args.items()}
    arg_dtypes = {k: v.get("dtype", "") for k, v in args.items()}

    # kwargs → { "hidden_states": [1,128,4096], "position_embeddings": [[1,128,128], [1,128,128]], ... }
    kwarg_shapes: dict = {}
    kwarg_strides: dict = {}
    kwarg_dtypes: dict = {}
    for kname, kval in kwargs.items():
        if isinstance(kval, list):
            # list of tensors (e.g. position_embeddings)
            kwarg_shapes[kname] = [t.get("shape", []) for t in kval]
            kwarg_strides[kname] = [t.get("stride", []) for t in kval]
            kwarg_dtypes[kname] = [t.get("dtype", "") for t in kval]
        else:
            kwarg_shapes[kname] = kval.get("shape", [])
            kwarg_strides[kname] = kval.get("stride", [])
            kwarg_dtypes[kname] = kval.get("dtype", "")

    return {
        "arg_shapes": _jstr(arg_shapes),
        "arg_strides": _jstr(arg_strides),
        "arg_dtypes": _jstr(arg_dtypes),
        "kwarg_shapes": _jstr(kwarg_shapes),
        "kwarg_strides": _jstr(kwarg_strides),
        "kwarg_dtypes": _jstr(kwarg_dtypes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def ensure_tables(client) -> None:
    print("[info] Ensuring tables exist ...")
    client.command(_CREATE_SUITES)
    print("[info]   module_test_suites   — ok")
    client.command(_CREATE_VARIANTS)
    print("[info]   module_test_variants — ok\n")


# ─────────────────────────────────────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────────────────────────────────────


def build_suite_row(rec: dict, args, gha_run_id: int, now: datetime) -> list:
    suite_name = _s(rec.get("suite_name"))
    summary = rec.get("summary", {})
    test_date = _parse_date(rec.get("test_date", ""))
    suite_id = _make_id(gha_run_id, suite_name)

    return [
        suite_id,
        gha_run_id,
        _s(rec.get("run_id") or args.run_id),
        _s(args.workflow),
        _s(args.branch),
        _s(args.sha)[:40],
        suite_name,
        _s(rec.get("model_name")),
        _s(rec.get("yaml_file")),
        _s(rec.get("github_ref")),
        _s(rec.get("runner_name")),
        _s(rec.get("runner_node")),
        _i(summary.get("total_tests")),
        _i(summary.get("spyre_enabled_count")),
        _i(summary.get("not_implemented_count")),
        _i(summary.get("cpu_fallback_count")),
        _i(summary.get("spyre_failed_count")),
        test_date,
        now,  # triggered_at
        now,  # ingested_at
    ]


def build_variant_rows(rec: dict, args, gha_run_id: int, now: datetime) -> list[list]:
    suite_name = _s(rec.get("suite_name"))
    model_name = _s(rec.get("model_name"))
    yaml_file = _s(rec.get("yaml_file"))
    run_id = _s(rec.get("run_id") or args.run_id)
    test_date = _parse_date(rec.get("test_date", ""))
    suite_id = _make_id(gha_run_id, suite_name)

    rows: list[list] = []
    seq = [0]

    def _row(t: dict, classification: str) -> list:
        test_name = _s(t.get("test_name"))
        seq[0] += 1
        variant_id = _make_id(gha_run_id, suite_name, test_name, seq[0])

        inputs = t.get("inputs", {})
        pre = _flatten_inputs(inputs, "prefill")
        dec = _flatten_inputs(inputs, "decode")

        return [
            variant_id,
            suite_id,
            gha_run_id,
            run_id,
            _s(args.workflow),
            _s(args.branch),
            _s(args.sha)[:40],
            suite_name,
            model_name,
            yaml_file,
            _s(t.get("module")),
            _s(t.get("test_label")),
            test_name,
            _s(t.get("test_file")),
            classification,
            _s(t.get("status")),
            _s(t.get("dtype")),
            _f(t.get("duration_s")),
            int(bool(t.get("has_cpu_fallback"))),
            _jlist(t.get("cpu_fallback_ops", [])),
            # prefill
            pre["arg_shapes"],
            pre["arg_strides"],
            pre["arg_dtypes"],
            pre["kwarg_shapes"],
            pre["kwarg_strides"],
            pre["kwarg_dtypes"],
            # decode
            dec["arg_shapes"],
            dec["arg_strides"],
            dec["arg_dtypes"],
            dec["kwarg_shapes"],
            dec["kwarg_strides"],
            dec["kwarg_dtypes"],
            test_date,
            now,  # triggered_at
            now,  # ingested_at
        ]

    ops = rec.get("operations", {})
    for t in ops.get("spyre_enabled", []):
        rows.append(_row(t, "spyre_enabled"))
    for t in ops.get("not_implemented", []):
        rows.append(_row(t, "not_implemented"))
    for t in ops.get("cpu_fallback", []):
        rows.append(_row(t, "cpu_fallback"))
    for t in ops.get("spyre_failed", []):
        rows.append(_row(t, "spyre_failed"))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest module-test JSON → ClickHouse (module_test_suites + module_test_variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--json-file", required=True, help="Path to JSON from parse_module_test_logs.py"
    )
    ap.add_argument("--workflow", default="run-tests", help="GHA workflow name")
    ap.add_argument("--branch", default="", help="Git branch name")
    ap.add_argument("--sha", default="", help="Git commit SHA")
    ap.add_argument("--run-id", default="", help="GHA run ID (numeric string)")
    args = ap.parse_args()

    # ── Load JSON ──────────────────────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as fh:
        records = json.load(fh)

    records = [r for r in records if r.get("suite_name", "").strip()]
    if not records:
        print("[info] No valid records — nothing to ingest.")
        sys.exit(0)

    print(f"[info] Loaded {len(records)} suite record(s) from {json_path.name}")

    # ── Connect ────────────────────────────────────────────────────────────────
    print(
        f"[info] Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.\n")

    ensure_tables(client)

    gha_run_id = _i(args.run_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Build rows ──────────────────────────────────────────────────────────────
    all_suite_rows: list[list] = []
    all_variant_rows: list[list] = []

    for rec in records:
        suite_name = _s(rec.get("suite_name"))
        model_name = _s(rec.get("model_name"))
        summary = rec.get("summary", {})

        try:
            all_suite_rows.append(build_suite_row(rec, args, gha_run_id, now))
            print(
                f"  [suite]    {suite_name!r}  model={model_name}  "
                f"total={summary.get('total_tests',0)}  "
                f"xpass={summary.get('spyre_enabled_count',0)}  "
                f"xfail={summary.get('not_implemented_count',0)}  "
                f"fallback={summary.get('cpu_fallback_count',0)}"
            )
        except Exception as exc:
            print(f"  [suite err] {suite_name!r}: {exc}", file=sys.stderr)

        try:
            vrows = build_variant_rows(rec, args, gha_run_id, now)
            all_variant_rows.extend(vrows)
            print(f"  [variants] {suite_name!r}: {len(vrows)} rows built")
        except Exception as exc:
            print(f"  [variants err] {suite_name!r}: {exc}", file=sys.stderr)

    # ── Batch insert ────────────────────────────────────────────────────────────
    print(f"\n[info] Inserting {len(all_suite_rows)} suite rows ...")
    if all_suite_rows:
        client.insert("module_test_suites", all_suite_rows, column_names=SUITE_COLS)
        print(f"[info]   module_test_suites   — {len(all_suite_rows)} rows inserted")

    print(f"[info] Inserting {len(all_variant_rows)} variant rows ...")
    if all_variant_rows:
        client.insert(
            "module_test_variants", all_variant_rows, column_names=VARIANT_COLS
        )
        print(f"[info]   module_test_variants — {len(all_variant_rows)} rows inserted")

    # ── Verify ─────────────────────────────────────────────────────────────────
    n_s = client.query("SELECT count() FROM module_test_suites").result_rows[0][0]
    n_v = client.query("SELECT count() FROM module_test_variants").result_rows[0][0]

    print("\n[info] ── Ingest complete ──────────────────────────────────────────────")
    print(f"[info]   module_test_suites   : {n_s} rows")
    print(f"[info]   module_test_variants : {n_v} rows")
    print(f"[info]   gha_run_id           : {gha_run_id}")
    print(f"[info]   workflow             : {args.workflow}")
    print(f"[info]   branch               : {args.branch}")
    print(f"[info]   sha                  : {args.sha[:12]}")


if __name__ == "__main__":
    main()
