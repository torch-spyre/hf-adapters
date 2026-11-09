#!/usr/bin/env python3
"""
Downloads the raw job log of every "Spyre model-module tests" leg of a GHA run
into --out-dir, for push-test-results-to-clickhouse.yaml's ingest-module-logs
job, and writes `job_count=<n>` to $GITHUB_OUTPUT.

Each leg corresponds to one config/model YAML. Logs are saved as
<job_id>_<job name with "/" and " " replaced by "_">.txt, which
parse_module_test_logs.py matches. A job whose log can't be fetched is skipped
with a warning.

Usage (called by the GHA workflow, with GH_TOKEN set for `gh`):
    python3 download_module_test_logs.py --run-id 12345678 --out-dir raw_logs
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

JOB_NAME_PREFIX = "Spyre model-module tests"


def list_jobs(repo: str, run_id: str) -> list[dict]:
    """All "Spyre model-module tests" jobs of the run, across every page."""
    stream = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--jq",
            ".jobs[] | {id: .id, name: .name}",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    jobs = [json.loads(line) for line in stream.splitlines() if line.strip()]
    return [j for j in jobs if j["name"].startswith(JOB_NAME_PREFIX)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True, help="GHA run to read")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo the run belongs to (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--out-dir", default="raw_logs", type=Path)
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")

    jobs = list_jobs(args.repo, args.run_id)
    print(f"Found {len(jobs)} model-module test job(s)")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"job_count={len(jobs)}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        safe = job["name"].replace("/", "_").replace(" ", "_")
        out_path = args.out_dir / f"{job['id']}_{safe}.txt"
        result = subprocess.run(
            ["gh", "api", f"/repos/{args.repo}/actions/jobs/{job['id']}/logs"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            out_path.write_bytes(result.stdout)
            print(f"  saved: {out_path.name} ({len(result.stdout)} bytes)")
        else:
            print(
                f"  WARN: no log for job {job['id']} ({job['name']}): "
                f"{result.stderr.decode()[:200]}"
            )


if __name__ == "__main__":
    main()
