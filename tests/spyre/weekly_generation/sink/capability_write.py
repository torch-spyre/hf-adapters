"""The schema-v2 write path for the weekly scan: capabilities + capability_runs.

Kept apart from ``clickhouse_sink`` because it is a different GENERATION of the schema,
not a second table: v1's ``{embedding,generative}_model_spyre_support`` encode the model
type in the table NAME and carry one row per model with three ``verified_on_*`` booleans,
while v2 has one table pair for every product and one row per (model, backend). The v1
writer stays untouched and authoritative until v1 is retired.

WHAT THIS FIXES. ``verified_on_cpu/gpu/spyre`` are three columns that cannot be grouped,
filtered or joined as one axis, so "which models work on spyre but not cpu" needs three
predicates over a wide row rather than one GROUP BY. v2 makes the backend a VALUE, the same
way ``benchmark_runs.backend`` already does, so a capability verdict and a performance
measurement of the same model segment alike.

A re-scan is also kept rather than overwritten: v1 is
``ReplacingMergeTree(snapshot_date) ORDER BY (model_name, snapshot_date)``, so scanning the
same model twice on one day collapses to the last write and the earlier verdict is gone.
``capability_runs`` keys on ``run_id``, so both survive and are attributable.

IDENTITY. ``run_id`` is derived, never minted, from the GHA run coordinate -- so all shards
of one weekly scan share it with nothing threaded between them, and "this scan" is a
``run_id`` filter rather than a ``snapshot_date`` heuristic. The scan fans out over up to 25
shards per tier, each its own process with its own client, so each stamps ``props['shard']``
and the dedup guard scopes on it: keyed on the run alone, the first shard to flush would make
every other shard look already-ingested and its verdicts would be dropped with no error.

INSTALLED PER JOB, NOT DECLARED IN ``pyproject.toml``. ``build-hf-adapters`` runs
``uv sync --frozen``, so a manifest entry would need a lock entry and would pin the library to
a locked commit -- a fix there would then need an hf-adapters re-lock to take effect. The
workflow installs it as a step instead (see ``CH_INGEST_LIB``), which is also how
spyre-inference's ingest gets it. The import is therefore INSIDE ``write()`` and the caller
treats an ImportError like any other v2 failure: the scan logs it and keeps its v1 rows.
"""

from __future__ import annotations

import os
from typing import Any

# model_support scans HuggingFace Hub checkpoints, not a build of ours, so these rows join no
# artifact -- unlike model_ops, whose subject IS something we built.
TEST_TYPE = "model_support"
COMPONENT = "hf-adapters"

# Every tier runs on x86_64: the x1/x2/x4 split routes by PARAMETER COUNT, not by hardware
# (see generate_weekly_shards._tier_for), so the tier is not an arch and must not be hashed as
# one -- doing so would mint three run_ids for one scan and invent an arch that does not exist.
ARCH = "x86_64"

# The three v1 booleans, as the backend vocabulary benchmark_runs already uses.
_BACKENDS = (
    ("verified_on_cpu", "cpu"),
    ("verified_on_gpu", "gpu"),
    ("verified_on_spyre", "spyre"),
)


def _shard_of(model_list_file: str | None) -> str:
    """The shard discriminator: the shard file's stem.

    It already encodes mode, tier and index (``generative-x1-shard-000``), so it identifies
    this writer among the run's shards without a second flag to thread. Empty for a
    ``--fetch`` run, which is a single unsharded process.
    """
    if not model_list_file:
        return ""
    return os.path.basename(str(model_list_file)).removesuffix(".json")


def capability_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One v2 result per (model, backend) from the v1 per-model rows.

    UNROLLS the three ``verified_on_*`` booleans, which is a real row-count change: ~195k v1
    rows become ~580k. That is the point -- a backend that is a value can be grouped and
    joined, and the three-column shape cannot.

    ``fail_reason`` carries v1's ``failure_category`` (a closed 13-value vocabulary worth
    grouping by) and is stamped only on a FAILING backend: the category describes why the
    model did not work, so copying it onto a backend that passed would make a successful
    verdict look explained by a failure.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        model_name = str(r.get("model_name") or "")
        if not model_name:
            continue
        # The Hub catalog facts describe the SUBJECT, not one backend's verdict, so they ride
        # on every row rather than being split across them; a reader filtering to one backend
        # still sees the model's size and family.
        props = {
            k: str(v)
            for k, v in (
                ("snapshot_date", r.get("snapshot_date")),
                ("added_date", r.get("added_date")),
                ("config_class", r.get("config_class")),
                ("num_downloads", r.get("num_downloads")),
                ("family", r.get("family")),
                ("architecture", r.get("architecture")),
                ("parameters_number", r.get("parameters_number")),
                ("curated", r.get("curated")),
                ("error", r.get("error")),
            )
            if v not in (None, "")
        }
        # adapter_name is the capability asked of the model, and is empty on a row that
        # failed before an adapter was ever selected (worker timeout/crash). Fall back to a
        # literal so the identity is still derivable -- capabilities.name is CHECKed non-empty,
        # so an empty one would be refused and the failure would go unrecorded entirely.
        adapter = str(r.get("adapter_name") or "").strip() or "unknown"
        failure_category = str(r.get("failure_category") or "")
        for field, backend in _BACKENDS:
            passed = bool(r.get(field))
            out.append(
                {
                    "subject": model_name,
                    "name": adapter,
                    "status": "passed" if passed else "failed",
                    "backend": backend,
                    "fail_reason": "" if passed else failure_category,
                    "props": props,
                }
            )
    return out


def write(rows: list[dict[str, Any]], model_list_file: str | None = None) -> int:
    """Dual-write the buffered v1 rows as v2 capability rows. Returns rows written.

    A no-op returning 0 when the v2 database is unset or its tables are absent, so a
    deployment that has not created them yet is not an error. Every failure is swallowed by
    the caller: a v2 problem must never cost the v1 rows this run exists to produce.
    """
    if not rows:
        return 0

    from spyre_clickhouse_ingest import (
        capabilities_already_ingested,
        get_client,
        insert_capabilities,
        run_id_of,
        tables_present,
        target_database,
    )
    from spyre_clickhouse_ingest.schema import CAPABILITIES, CAPABILITY_RUNS

    db = target_database()
    if not db:
        return 0

    gha_run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not gha_run_id:
        print("  v2: no GITHUB_RUN_ID -- run_id is not derivable, skipping v2 write.")
        return 0
    run_id = run_id_of("gha", gha_run_id, ARCH, TEST_TYPE)
    if not run_id:
        return 0

    client = get_client()
    if not tables_present(client, db, (CAPABILITIES, CAPABILITY_RUNS)):
        print(f"  v2: {db} has no capability tables -- skipping v2 write.")
        return 0

    shard = _shard_of(model_list_file)
    if capabilities_already_ingested(client, db, run_id, COMPONENT, TEST_TYPE, shard):
        print(
            f"  v2: run {run_id} shard {shard or '<none>'} already ingested -- skipping."
        )
        return 0

    results = capability_results(rows)
    n = insert_capabilities(
        client, db, COMPONENT, run_id, TEST_TYPE, results, arch=ARCH, shard=shard
    )
    print(
        f"  v2: wrote {n} capability row(s) to {db} (run {run_id}, shard {shard or '<none>'})."
    )
    return n


def rows_from_pending(
    pending: list[list[Any]], columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Re-key the sink's positional buffer by column name.

    The sink buffers positional lists because that is what a bulk insert takes; this pairs
    them with the one column list rather than re-deriving the order, so the two cannot drift.
    """
    return [dict(zip(columns, row)) for row in pending]
