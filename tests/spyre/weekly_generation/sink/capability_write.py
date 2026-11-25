"""The weekly scan's write path: schema-v2 ``capabilities`` + ``capability_runs``.

One row per (model, backend) with the backend as a VALUE, so "which models work on spyre but
not cpu" is one GROUP BY. The scan's v1 ``{embedding,generative}_model_spyre_support`` tables
are no longer written from CI. Rows keep the Hub catalog facts (model type, snapshot date,
downloads, family, ...) as props, and a re-scan is kept rather than overwritten: rows key on
``run_id``, so both survive and are attributable.

IDENTITY. ``run_id`` is derived, never minted, from the GHA run coordinate -- so all shards
of one weekly scan share it with nothing threaded between them, and "this scan" is a
``run_id`` filter rather than a ``snapshot_date`` heuristic. The scan fans out over up to 25
shards per tier, each its own process with its own client, so each stamps ``props['shard']``
and the dedup guard scopes on it: keyed on the run alone, the first shard to flush would make
every other shard look already-ingested and its verdicts would be dropped with no error.

DEDUP. A flush may be retried (and a job re-run) under the same run_id and shard, so ``write``
drops the models of a batch that this run and shard already recorded. A whole-shard guard would
instead drop every batch after a shard's first.

INSTALLED PER JOB, NOT DECLARED IN ``pyproject.toml``. ``build-hf-adapters`` runs
``uv sync --frozen``, so a manifest entry would need a lock entry and would pin the library to
a locked commit -- a fix there would then need an hf-adapters re-lock to take effect. The
workflow installs it as a step instead (see ``CH_INGEST_LIB``), which is also how
spyre-inference's ingest gets it. The import is therefore INSIDE the functions that need it.
"""

from __future__ import annotations

import os
from typing import Any

# The subjects are Hub checkpoints, but the spyre verdicts depend on the image that ran them:
# artifact_link.py ties each scan to it with one artifact_results leg.
TEST_TYPE = "model_support"
COMPONENT = "hf-adapters"

# Every tier runs on x86_64: the x1/x2/x4 split routes by PARAMETER COUNT, not by hardware
# (see generate_weekly_shards._tier_for), so the tier is not an arch and must not be hashed as
# one -- doing so would mint three run_ids for one scan and invent an arch that does not exist.
ARCH = "x86_64"

# The v1 booleans the scan actually establishes, as benchmark_runs' backend vocabulary. No gpu:
# the scan has no GPU leg (verified_on_gpu is always False), so a gpu verdict would read as a
# failure on every model rather than as "not tested".
_BACKENDS = (
    ("verified_on_cpu", "cpu"),
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
    """One v2 result per (model, backend) from the per-model rows.

    UNROLLS the tested ``verified_on_*`` booleans (see ``_BACKENDS``): a backend that is a
    value can be grouped and joined, and the three-column shape cannot.

    ``fail_reason`` carries the row's ``failure_category`` (a closed 13-value vocabulary worth
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
                ("model_type", r.get("model_type")),
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
        adapter = str(r.get("adapter_name") or "").strip()
        if not adapter:
            # A full VLM and its tower can both time out before adapter resolution.
            # Preserve distinct capability identities even for these early failures.
            model_type = str(r.get("model_type") or "")
            adapter = (
                f"unknown_{model_type}"
                if model_type in ("vision", "vision_tower")
                else "unknown"
            )
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


_SUBJECTS_RECORDED_SQL = (
    "SELECT DISTINCT c.subject FROM {db:Identifier}.capability_runs AS r "
    "INNER JOIN {db:Identifier}.capabilities AS c ON c.capability_id = r.capability_id "
    "WHERE r.component = {component:String} AND r.test_type = {tt:String} "
)


_SUBJECT_CHUNK = 500


def recorded_in_run(
    client: Any, db: str, run_id: str, shard: str, subjects: list[str]
) -> set[str]:
    """The *subjects* this run and shard already wrote.

    Queried in chunks: the prefilter flushes every filtered model at once, and a bound array
    travels in the request URL.
    """
    found: set[str] = set()
    for i in range(0, len(subjects), _SUBJECT_CHUNK):
        result = client.query(
            _SUBJECTS_RECORDED_SQL
            + "AND r.run_id = {run_id:UUID} AND r.props['shard'] = {shard:String} "
            "AND c.subject IN {subjects:Array(String)}",
            parameters={
                "db": db,
                "component": COMPONENT,
                "tt": TEST_TYPE,
                "run_id": run_id,
                "shard": shard,
                "subjects": subjects[i : i + _SUBJECT_CHUNK],
            },
        )
        found.update(row[0] for row in result.result_rows)
    return found


def recorded_for_snapshot(
    client: Any,
    db: str,
    snapshot_date: str,
    model_type: str,
    fail_reason: str = "",
) -> set[str]:
    """Models with a verdict for a snapshot date and model type.

    Rows written before ``model_type`` became a prop carry none, so a missing one matches.
    With *fail_reason*, only models whose LATEST spyre verdict failed with it: one that
    failed once and passed on a later run is not returned.
    """
    sql = (
        _SUBJECTS_RECORDED_SQL
        + "AND r.props['snapshot_date'] = {snapshot_date:String} "
        "AND r.props['model_type'] IN ({model_type:String}, '')"
    )
    parameters = {
        "db": db,
        "component": COMPONENT,
        "tt": TEST_TYPE,
        "snapshot_date": snapshot_date,
        "model_type": model_type,
    }
    if fail_reason:
        sql += (
            " AND r.backend = 'spyre' GROUP BY c.subject"
            " HAVING argMax(r.fail_reason, r.ts) = {fail_reason:String}"
        )
        parameters["fail_reason"] = fail_reason
    return {row[0] for row in client.query(sql, parameters=parameters).result_rows}


def write(
    rows: list[dict[str, Any]],
    client: Any,
    db: str,
    model_list_file: str | None = None,
) -> int:
    """Write the buffered rows as v2 capability rows. Returns rows written.

    The scan's only write path, so a problem raises rather than being swallowed: a run that
    cannot record its verdicts must fail. Models this run and shard already wrote are skipped.
    """
    if not rows:
        return 0

    from spyre_clickhouse_ingest import insert_capabilities, run_id_of

    gha_run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not gha_run_id:
        raise RuntimeError(
            "GITHUB_RUN_ID is unset, so the v2 run_id cannot be derived; "
            "outside CI use --write-to-csv."
        )
    run_id = run_id_of("gha", gha_run_id, ARCH, TEST_TYPE)
    if not run_id:
        raise RuntimeError(f"no v2 run_id derivable from GITHUB_RUN_ID={gha_run_id!r}")

    shard = _shard_of(model_list_file)
    models = sorted({str(r.get("model_name") or "") for r in rows} - {""})
    seen = recorded_in_run(client, db, run_id, shard, models) if models else set()
    fresh = [r for r in rows if str(r.get("model_name") or "") not in seen]
    if len(fresh) < len(rows):
        print(
            f"  v2: skipped {len(rows) - len(fresh)} model(s) this run and shard already recorded."
        )
    if not fresh:
        print(
            f"  v2: run {run_id} shard {shard or '<none>'} already recorded this batch -- skipping."
        )
        return 0

    n = insert_capabilities(
        client,
        db,
        COMPONENT,
        run_id,
        TEST_TYPE,
        capability_results(fresh),
        arch=ARCH,
        shard=shard,
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
