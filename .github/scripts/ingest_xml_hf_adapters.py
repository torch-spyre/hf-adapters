#!/usr/bin/env python3
"""
Parses hf-adapters' pytest JUnit XML into hf_test_runs / hf_test_cases /
hf_run_properties.

Usage (called by the GHA workflow):
    python3 ingest_xml_hf_adapters.py \
        --xml-dir xml_artifacts \
        --workflow "test_pull_request" \
        --branch   "main" \
        --sha      "abcdef1234..." \
        --run-id   "3f2b9c1e-...-uuid" \
        --gha-run-id "12345678" \
        --triggered-at "2026-04-25T14:20:45Z" \
        --pr-number 2271 \
        --platform "x86_64"
"""

import argparse
import os
import platform as _platform
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree
from spyre_clickhouse_ingest import (
    extract_properties,
    get_client,
    insert_v2,
    promote_xpass,
    v2_already_ingested,
    v2_component,
    v2_database,
    v2_run_id_for,
    v2_source_and_external_run_id,
    v2_tables_present,
)
from spyre_clickhouse_ingest.junit import _runner_run_id, _threaded_run_id

# ---------------------------------------------------------------------------
# hf_test_runs / hf_test_cases / hf_run_properties are provisioned out of
# band; this script never creates them. If they're missing, ingest is a
# silent no-op rather than a failure.
# ---------------------------------------------------------------------------


def tables_exist(client, db: str) -> bool:
    return bool(client.command(f"EXISTS TABLE {db}.hf_test_runs"))


# ---------------------------------------------------------------------------
# TEST-RESULT XML parsing
# ---------------------------------------------------------------------------


def classify_testcase(tc_el):
    """Return (status, fail_message, skip_message) for one <testcase>."""
    failure_el = tc_el.find("failure")
    error_el = tc_el.find("error")
    skipped_el = tc_el.find("skipped")

    if error_el is not None:
        msg = (error_el.get("message", "") + "\n" + (error_el.text or "")).strip()
        return "error", msg, ""

    if failure_el is not None:
        ftype = (failure_el.get("type") or "").lower()
        msg = (failure_el.get("message", "") + "\n" + (failure_el.text or "")).strip()
        if "xfail" in ftype:
            return "xpass", msg, ""
        return "failed", msg, ""

    if skipped_el is not None:
        stype = (skipped_el.get("type") or "").lower()
        msg = (skipped_el.get("message") or skipped_el.text or "").strip()
        if "xfail" in stype:
            return "xfail", "", msg
        return "skipped", "", msg

    return "passed", "", ""


def parse_test_xml(xml_path: Path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suites = root.findall(".//testsuite")
    if not suites:
        print(f"  [warn] No <testsuite> found in {xml_path.name}", file=sys.stderr)
        return None, []

    suite = suites[0]
    suite_attrs = suite.attrib

    ts_str = suite_attrs.get("timestamp", "")
    try:
        triggered_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        triggered_at = datetime.now(timezone.utc)

    raw_cases = []
    for tc in suite.findall(".//testcase"):
        status, fail_msg, skip_msg = classify_testcase(tc)
        properties = extract_properties(tc)
        raw_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "classname": tc.get("classname", ""),
                "name": tc.get("name", ""),
                "status": status,
                "duration_s": float(tc.get("time", 0) or 0),
                "fail_message": fail_msg,
                "skip_message": skip_msg,
                "properties": properties,
                "_is_bare": (status == "passed"),
                "triggered_at": triggered_at,
            }
        )

    promote_xpass(raw_cases, suite_attrs)

    counts = Counter(c["status"] for c in raw_cases)
    run = {
        "suite_name": suite_attrs.get("name", xml_path.stem),
        "filename": xml_path.name,
        "triggered_at": triggered_at,
        "total_tests": len(raw_cases),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "xfail": counts.get("xfail", 0),
        "errors": counts.get("error", 0),
        "xpass": counts.get("xpass", 0),
        "duration_s": float(suite_attrs.get("time", 0) or 0),
    }
    return run, raw_cases


# ---------------------------------------------------------------------------
# ClickHouse insertion
# ---------------------------------------------------------------------------


def insert_run(client, run_id: str, run: dict, args):
    client.insert(
        "hf_test_runs",
        [
            [
                run_id,
                args.workflow,
                run["suite_name"],
                run["filename"],
                args.branch,
                (args.sha or "").ljust(40)[:40],
                int(args.pr_number) if args.pr_number.strip() else 0,
                _runner_run_id(args, run_id),
                run["triggered_at"].replace(tzinfo=None),
                run["total_tests"],
                run["passed"],
                run["failed"],
                run["skipped"],
                run["xfail"],
                run["errors"],
                run["xpass"],
                run["duration_s"],
                args.platform,
                args.trigger_type or "unknown",
                args.img_digest,
            ]
        ],
        column_names=[
            "run_id",
            "workflow",
            "suite_name",
            "filename",
            "branch",
            "commit_sha",
            "pr_number",
            "runner_run_id",
            "triggered_at",
            "total_tests",
            "passed",
            "failed",
            "skipped",
            "xfail",
            "errors",
            "xpass",
            "duration_s",
            "platform",
            "test_type",
            "img_digest",
        ],
    )


def insert_cases(client, run_id: str, cases: list[dict], workflow: str = ""):
    if not cases:
        return
    client.insert(
        "hf_test_cases",
        [
            [
                run_id,
                c["case_id"],
                c["classname"],
                c["name"],
                c["status"],
                c["duration_s"],
                c["skip_message"][:8192],
                c["fail_message"][:8192],
                c["triggered_at"].replace(tzinfo=None),
                workflow,
            ]
            for c in cases
        ],
        column_names=[
            "run_id",
            "case_id",
            "classname",
            "name",
            "status",
            "duration_s",
            "skip_message",
            "fail_message",
            "triggered_at",
            "workflow",
        ],
    )


def insert_properties(client, run_id: str, cases: list[dict]):
    rows = [
        {
            "run_id": run_id,
            "case_id": c["case_id"],
            "prop_name": pname,
            "prop_value": pvalue,
            "triggered_at": c["triggered_at"],
        }
        for c in cases
        for pname, pvalue in c["properties"]
    ]
    if rows:
        client.insert(
            "hf_run_properties",
            [
                [
                    r["run_id"],
                    r["case_id"],
                    r["prop_name"],
                    r["prop_value"],
                    r["triggered_at"].replace(tzinfo=None),
                ]
                for r in rows
            ],
            column_names=[
                "run_id",
                "case_id",
                "prop_name",
                "prop_value",
                "triggered_at",
            ],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ── SCHEMA v2: test_cases + test_case_runs ─────────────────────────────────
#
# ADDITIVE. Everything above still writes the v1 tables exactly as before; this
# path writes the two v2 tables alongside them and is skipped entirely if they do
# not exist, so the script is safe to deploy before the v2 migration lands.
#
# Both ids are DERIVED, never minted. Four writers (this script, the two sibling
# product ingests, and the orchestrator's pushToClickhouse.pushArtifactResult)
# compute them independently with no threading contract -- which is the only thing
# that makes the tables joinable: v1 minted four unrelated identity schemes and
# artifact_results.run_id consequently joined test_runs.run_id in 2 of 1,266 rows.
#
# BYTE-EXACTNESS IS THE CONTRACT. Disagree about the namespace, the separator, the
# field order or the normalisation and you mint a different uuid for the same row --
# and an orphaned row is indistinguishable from "no tests ran", so the failure is
# silent. The reference implementation, its rule list and the golden values every
# port must reproduce live in spyre-frameworks pipelines/lib/run_identity.py and
# pipelines/lib/test_run_identity.py. Keep this block in sync with it.
# ---------------------------------------------------------------------------

# The product this script ingests for. Replaces v1's hf_/si_ table-name prefixes: one
# v2 table pair serves all three products, discriminated by this column. It is also a
# test_case_id hash input, so it cannot drift from the identity it is stamped on.
# The product this script ingests for by DEFAULT. A default, not a constant: a test cell may
# run ANOTHER component's suite through this script, and hardcoding the owner stamps those rows
# with the wrong component. Because component is a test_case_id hash input, that does not merely
# mislabel -- the same test reconciles to a DIFFERENT identity depending on whose script ran it,
# splitting one suite across two components. --component lets the caller name the suite's real
# owner; product-test already knows it (config.yaml's PRODUCT).
V2_COMPONENT_DEFAULT = "hf-adapters"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--component",
        default="",
        help="Component to stamp on v2 rows. Defaults to this repo's own product; set it "
        "when a cell runs ANOTHER component's suite through this script, so the rows (and "
        "the test_case_id they hash into) name the suite's real owner.",
    )
    parser.add_argument("--gha-run-id", default="")
    parser.add_argument("--triggered-at", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument(
        "--jenkins-run-key",
        default="",
        help="This leg's own Jenkins externalizable id, e.g. 'Spyre/component-build#417'. "
        "Hashed into the schema-v2 run_id, which is how the orchestrator's "
        "artifact_results row and these per-case rows join without threading a uuid.",
    )
    parser.add_argument(
        "--trigger-type",
        default="",
        help="Suite tier that produced this run, e.g. smoke | core | full | trunk | weekly | nightly",
    )
    parser.add_argument(
        "--platform",
        default=_platform.machine() or "",
        help="Hardware platform the SUITE ran on, e.g. x86_64 | s390x | ppc64le. Folded into "
        "the run_id hash, so a wrong value mints an id that joins to nothing: pass it "
        "explicitly whenever the ingest does not run on the test host (the workflow_run "
        "ingest does not -- it is pinned to x86_64). The default is the INGEST host's arch, "
        "correct only for a direct run. An empty value makes v2_run_id refuse to derive an "
        "id, so every row lands unjoinable.",
    )
    parser.add_argument(
        "--img-digest",
        default="",
        help="Digest of the runner image the suite ran against, if known",
    )
    # Which schema generation to write. Defaults to v1 ONLY, so an un-updated caller behaves
    # exactly as before -- this script runs from a BAKED image, so old and new images coexist
    # until every product image is rebuilt.
    #
    # v1 is not a permanent home: v2 replaces these tables outright. run_properties becomes
    # test_cases.tags, and the per-product test_runs/test_cases collapse into the shared
    # test_cases + test_case_runs keyed by component -- which is why the spyre-frameworks v2
    # DDL deliberately does not define them. No data is ported: v1 rows cannot produce a v2
    # run_id, since the hash inputs were never recorded per row. New data only.
    parser.add_argument(
        "--schema",
        choices=["v1", "v2", "both"],
        default=os.environ.get("INGEST_SCHEMA", "v1"),
        help="Which schema generation to write: v1 (default, the legacy per-product tables), "
        "v2 (the shared replacement tables only), or both (the migration window). Also "
        "settable via INGEST_SCHEMA so a workflow can set it once for every leg.",
    )
    args = parser.parse_args()
    # Resolved once, so the two paths cannot drift into disagreeing about what was asked for.
    args.write_v1 = args.schema in ("v1", "both")
    args.write_v2 = args.schema in ("v2", "both")
    print(f"  schema={args.schema} (v1={args.write_v1} v2={args.write_v2})")

    if args.xml_file:
        xml_root = Path(args.xml_file).parent
        xml_files = [Path(args.xml_file)]
    elif args.xml_dir:
        xml_root = Path(args.xml_dir)
        xml_files = sorted(xml_root.rglob("*.xml"))
    else:
        print("Error: provide --xml-dir or --xml-file")
        sys.exit(1)

    if not xml_files:
        print("No XML files found — nothing to ingest.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    # One client, both generations: v2 is reached by QUALIFYING every statement with this
    # database name (see v2_database). "" means v2 is not configured.
    v2db = v2_database() if args.write_v2 else ""
    if args.write_v2 and not v2db:
        print(
            "  WARN --schema asked for v2 but CLICKHOUSE_DB_V2 is unset — v2 rows skipped",
            file=sys.stderr,
        )
    client.command("SELECT 1")
    print("Connected.\n")

    db = os.environ.get("CLICKHOUSE_DB", "spyre")
    # Gated on write_v1: this checks a V1 table, so under --schema v2 an absent v1 schema
    # is expected, not a reason to abort before the v2 write runs.
    if args.write_v1 and not tables_exist(client, db):
        print(
            f"{db}.hf_test_runs does not exist — nothing to ingest into. Silent no-op."
        )
        sys.exit(0)

    total_cases = 0

    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")

        run, cases = parse_test_xml(xml_path)
        if run is None:
            continue

        # Different suites can independently produce a JUnit XML with the
        # same basename (e.g. GitHub Actions strips the model-key
        # subdirectory when an artifact is a single file), so the path
        # relative to the XML root -- not the bare basename -- is what makes
        # `filename` actually unique for both dedup and storage.
        run["filename"] = str(xml_path.relative_to(xml_root))

        # One run_id per TEST RUN, not per XML file: the dispatching orchestrator generates
        # a uuid and threads it down as --run-id, stamping the SAME value on
        # artifact_results, so the two tables join. `filename` stays the per-file
        # discriminator among the rows that share it. Falls back to a fresh uuid4 when
        # --run-id is absent or not a uuid (standalone / GHA-only run): the rows are still
        # valid, just not linked to an artifact.
        run_id = _threaded_run_id(args) or str(uuid.uuid4())

        # Dedup on (run_id, filename): a re-ingest of the SAME test run must be idempotent,
        # but two distinct runs must never collapse. runner_run_id mirrors run_id for a Jenkins/standalone leg, so it's only an independent signal for a GHA numeric id.
        runner_run_id = _runner_run_id(args, run_id)
        # v1-table reads, so gated on v1 being written. v2 dedups against its own
        # table via v2_already_ingested(run_id, component).
        if args.write_v1:
            existing = client.query(
                "SELECT count() FROM hf_test_runs "
                "WHERE run_id = {run_id:String} AND filename = {filename:String}",
                parameters={"run_id": run_id, "filename": run["filename"]},
            )
            if (
                existing.result_rows[0][0] == 0
                and runner_run_id
                and runner_run_id != run_id
            ):
                # A GHA re-ingest mints a fresh uuid4, so fall back to the numeric
                # run id to keep that path idempotent.
                existing = client.query(
                    "SELECT count() FROM hf_test_runs WHERE "
                    "runner_run_id = {runner_run_id:String} AND filename = {filename:String}",
                    parameters={
                        "runner_run_id": runner_run_id,
                        "filename": run["filename"],
                    },
                )
            if existing.result_rows[0][0] > 0:
                print(f"  Already ingested — skipping {run['filename']}")
                continue

        print(
            f"  run_id={run_id}  tests={run['total_tests']}  "
            f"passed={run['passed']}  failed={run['failed']}  "
            f"xpass={run['xpass']}  xfail={run['xfail']}  skipped={run['skipped']}"
        )

        if args.write_v1:
            insert_run(client, run_id, run, args)
            insert_cases(client, run_id, cases, workflow=args.workflow)
            insert_properties(client, run_id, cases)
        # v2 tables, alongside v1. Failure here must never cost a v1 row: v1 is still
        # authoritative, so the experimental write is contained rather than allowed to
        # abort the loop and drop every remaining file's v1 insert.
        try:
            if v2db and v2_tables_present(client, v2db):
                _v2_source, _v2_ext = v2_source_and_external_run_id(args, run_id)
                _v2_tier = (getattr(args, "trigger_type", "") or "").strip()
                _v2_arch = (args.platform or run.get("platform") or "").strip()
                _v2_run_id = v2_run_id_for(args, run_id, _v2_arch, _v2_tier)
                if not _v2_run_id:
                    # Loud, because a blank run_id means these cases reach v2 unjoinable
                    # to any artifact -- and that reads downstream as "no tests ran".
                    print(
                        f"  [warn] v2 skipped: run_id not derivable "
                        f"(source={_v2_source} ext={_v2_ext!r} arch={_v2_arch!r} "
                        f"tier={_v2_tier!r}); --trigger-type is the field usually missing",
                        file=sys.stderr,
                    )
                elif v2_already_ingested(
                    client,
                    v2db,
                    _v2_run_id,
                    v2_component(args, V2_COMPONENT_DEFAULT),
                    xml_path.name,
                ):
                    print(f"  v2: already ingested run_id={_v2_run_id} — skipping")
                else:
                    _n = insert_v2(
                        client,
                        v2db,
                        v2_component(args, V2_COMPONENT_DEFAULT),
                        _v2_run_id,
                        cases,
                        xml_path.name,
                    )
                    print(f"  v2: {_n} test_case_runs under run_id={_v2_run_id}")
        except Exception as _v2_err:
            print(
                f"  [warn] v2 write failed, v1 unaffected: {_v2_err!r}", file=sys.stderr
            )

        total_cases += len(cases)
        if args.write_v1:
            print(
                f"  Inserted {len(cases)} test cases + "
                f"{sum(len(c['properties']) for c in cases)} properties"
            )

    print(f"\nDone. {len(xml_files)} file(s) processed.")
    print(f"  Test cases ingested:  {total_cases}")


if __name__ == "__main__":
    main()
