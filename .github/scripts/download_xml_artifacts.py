#!/usr/bin/env python3
"""
Downloads and unzips every artifact of a GHA run into per-artifact
subdirectories of --out-dir, for push-test-results-to-clickhouse.yaml.

Each artifact is fetched with `gh api` and unzipped, retried whole up to
--attempts times (a truncated zip only fails at unzip). An artifact that still
fails is skipped with a ::warning:: so one bad artifact doesn't drop the whole
run's results; the script exits non-zero only if every artifact failed.

Usage (called by the GHA workflow, with GH_TOKEN set for `gh`):
    python3 download_xml_artifacts.py \
        --artifacts '[{"id": 123, "name": "smoke-Qwen3-0.6B"}, ...]' \
        --repo torch-spyre/hf-adapters \
        --out-dir xml_artifacts
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def fetch(repo: str, artifact: dict, zip_path: Path, dest: Path, attempts: int) -> bool:
    """Download and unzip one artifact; retried whole, since a truncated zip only fails at unzip."""
    for attempt in range(1, attempts + 1):
        shutil.rmtree(dest, ignore_errors=True)
        for cmd in (
            ["gh", "api", f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
            ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)],
        ):
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                err = (
                    (result.stderr or result.stdout[-500:])
                    .decode(errors="replace")
                    .strip()
                )
                print(
                    f"[retry] {artifact['name']}: {cmd[0]} attempt {attempt}/{attempts} "
                    f"exited {result.returncode}: {err}"
                )
                break
            if cmd[0] == "gh":
                zip_path.write_bytes(result.stdout)
        else:
            return True
        if attempt < attempts:
            time.sleep(2**attempt)
    shutil.rmtree(dest, ignore_errors=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--artifacts",
        default=os.environ.get("ARTIFACTS", "[]"),
        help='JSON array of {"id", "name"} objects (default: $ARTIFACTS)',
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo the artifacts belong to (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--out-dir", default="xml_artifacts", type=Path)
    parser.add_argument("--attempts", default=4, type=int)
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")

    artifacts = json.loads(args.artifacts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for a in artifacts:
        print(f"Downloading: {a['name']} (id={a['id']})")
        # Per-artifact subdirectory: two suites can both produce e.g. "Qwen3-0.6B.xml", and
        # ingest_xml_hf_adapters.py globs --xml-dir recursively, keyed on the relative path.
        dest = args.out_dir / a["name"]
        if not fetch(
            args.repo, a, args.out_dir / f"{a['name']}.zip", dest, args.attempts
        ):
            print(
                f"::warning::Skipping artifact {a['name']} (id={a['id']}): "
                f"download/unzip failed {args.attempts} times; its results are not ingested."
            )
            failed.append(a["name"])
    print(f"Fetched {len(artifacts) - len(failed)}/{len(artifacts)} artifact(s).")
    if artifacts and len(failed) == len(artifacts):
        print("::error::No artifact could be downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
