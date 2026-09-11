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
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect
from lxml import etree

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


def extract_properties(tc_el):
    props = []
    props_el = tc_el.find("properties")
    if props_el is None:
        return props
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name:
            props.append((name, value))
    return props


def promote_xpass(raw_cases, suite_attrs):
    """Mirror torch-spyre's ingest_xml.py: pytest's plain <failures> count
    lumps strict and non-strict xfail-passed cases together with real
    failures, so bare-passed cases must be promoted to "xpass" to reconcile
    the suite-level failure count."""
    failures = int(suite_attrs.get("failures", 0))
    true_fail_raw = sum(1 for c in raw_cases if c["status"] in ("failed", "error"))
    strict_xpass_raw = sum(1 for c in raw_cases if c["status"] == "xpass")
    non_strict = max(0, failures - true_fail_raw - strict_xpass_raw)

    promoted = 0
    for c in raw_cases:
        if promoted >= non_strict:
            break
        if c["_is_bare"]:
            c["status"] = "xpass"
            promoted += 1


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


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


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


def _runner_run_id(args, run_id: str) -> str:
    """This leg's own run id: --gha-run-id when GHA-dispatched, else the same uuid as run_id."""
    raw = (getattr(args, "gha_run_id", "") or "").strip()
    if raw:
        try:
            int(raw)
            return raw
        except (ValueError, TypeError):
            pass
    return run_id


def _threaded_run_id(args) -> str:
    """--run-id when it is a real UUID, else "" so the caller mints one.

    Only a well-formed uuid is honoured: the column is a UUID join key, so any
    other value (a build number, a GHA run id) must be ignored, not stored.
    """
    raw = (getattr(args, "run_id", "") or "").strip()
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gha-run-id", default="")
    parser.add_argument("--triggered-at", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument(
        "--trigger-type",
        default="",
        help="Suite tier that produced this run, e.g. smoke | core | full | trunk | weekly | nightly",
    )
    parser.add_argument(
        "--platform",
        default="",
        help="Hardware platform the suite ran on, e.g. x86_64 | s390x | ppc64le",
    )
    parser.add_argument(
        "--img-digest",
        default="",
        help="Digest of the runner image the suite ran against, if known",
    )
    args = parser.parse_args()

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
    client.command("SELECT 1")
    print("Connected.\n")

    db = os.environ.get("CLICKHOUSE_DB", "spyre")
    if not tables_exist(client, db):
        print(
            f"{db}.hf_test_runs does not exist — nothing to ingest into. "
            "Silent no-op."
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

        insert_run(client, run_id, run, args)
        insert_cases(client, run_id, cases, workflow=args.workflow)
        insert_properties(client, run_id, cases)

        total_cases += len(cases)
        print(
            f"  Inserted {len(cases)} test cases + "
            f"{sum(len(c['properties']) for c in cases)} properties"
        )

    print(f"\nDone. {len(xml_files)} file(s) processed.")
    print(f"  Test cases ingested:  {total_cases}")


if __name__ == "__main__":
    main()
