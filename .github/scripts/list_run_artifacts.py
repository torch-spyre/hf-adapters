#!/usr/bin/env python3
"""
Lists every artifact of a GHA run as a JSON array of {"id", "name"} objects,
for push-test-results-to-clickhouse.yaml, and writes it to $GITHUB_OUTPUT as
`artifacts=<json>` for download_xml_artifacts.py.

`--paginate` is required: the artifacts endpoint defaults to 30 per page, and a
run with more jobs than that silently lost every artifact past the first page,
with no error -- rows just never showed up in ClickHouse. `--paginate` applies
`--jq` per page and concatenates the filtered results across pages (`gh api`
rejects `--slurp` combined with `--jq`), so the per-artifact objects come out as
a JSON stream, not a single array; this script slurps that stream into one.

Usage (called by the GHA workflow, with GH_TOKEN set for `gh`):
    python3 list_run_artifacts.py --run-id 12345678
"""

import argparse
import json
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True, help="GHA run to list")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo the run belongs to (default: $GITHUB_REPOSITORY)",
    )
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")

    stream = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{args.repo}/actions/runs/{args.run_id}/artifacts?per_page=100",
            "--jq",
            ".artifacts[] | {id: .id, name: .name}",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    artifacts = [json.loads(line) for line in stream.splitlines() if line.strip()]

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"artifacts={json.dumps(artifacts)}\n")
    print("Found artifacts:")
    for a in artifacts:
        print(" •", a["name"])


if __name__ == "__main__":
    main()
