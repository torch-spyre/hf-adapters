"""Seal an lm-eval run and its attachments for offline quality ingestion."""

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def create_bundle(run_dir, out, *, artifact, arch, component="hf-adapters", run_url=""):
    """Validate and seal a completed run, without network access or database writes.

    The optional ingestion package owns schema validation, hashes, and archive
    layout. Artifact identity and architecture must describe the tested build.
    """
    try:
        from spyre_clickhouse_ingest import offline
    except ImportError as error:
        raise ValueError(
            "Bundle creation requires spyre-clickhouse-ingest; install "
            "'git+https://github.com/torch-spyre/torch-spyre@main"
            "#subdirectory=extensions/clickhouse-ingest'"
        ) from error

    source, destination = Path(run_dir).resolve(), Path(out).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output already exists: {destination}")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Bundle output must be outside the source run directory")
    if not artifact.startswith(("id:", "image:")):
        raise ValueError(
            "--artifact needs id:<UUID> or image:<repository>@sha256:<digest>"
        )

    # Copy first: validation and checksums describe the snapshot that is sealed.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".lm-eval-bundle-", dir=destination.parent
    ) as tmp:
        root = Path(tmp) / "bundle"
        (root / "results").mkdir(parents=True)
        shutil.copy2(source / "benchmarks.json", root / "results" / "benchmarks.json")
        if (source / "attachments").exists():
            shutil.copytree(source / "attachments", root / "attachments")
        report_path = root / "attachments" / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        meta = {
            "schema_version": offline.SCHEMA_VERSION,
            "kind": "benchmarks",
            "test_type": "quality",
            "component": component,
            "arch": arch,
            "artifact": artifact,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Historical runs without timestamps remain undated: sealing is not
        # execution, and the connected uploader is not the runner.
        for field, value in (
            ("started_at", report.get("created_at")),
            ("ended_at", report.get("updated_at")),
            ("run_url", run_url),
        ):
            if value:
                meta[field] = value
        try:
            meta = offline.normalized(meta)
            meta["files"] = [
                {"path": name, "sha256": offline.sha256(root / name)}
                for name in offline.bundle_files(root)
            ]
            (root / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n")
            offline.check(root)
            # Validate identities as well as field types. Ingesting an old
            # payload must not silently sum prompt and instruction counts.
            from spyre_clickhouse_ingest.bundle import generic_entries
            from spyre_clickhouse_ingest.writer import BenchmarkWriter, DryRunClient

            rows = offline.benchmark_entries(root / "results" / "benchmarks.json")
            preview = DryRunClient()
            count = BenchmarkWriter.insert(
                preview,
                "spyre_v2",
                component,
                "00000000-0000-0000-0000-000000000000",
                generic_entries(rows),
            )
            if count != len(rows):
                raise ValueError(
                    "Benchmark identities collide: the ingest writer would merge rows "
                    "and sum their iterations. Regenerate benchmarks.json with the "
                    "current dashboard_results exporter."
                )
            if destination.name.endswith((".tgz", ".tar.gz")):
                archive = Path(tmp) / "bundle.tgz"
                offline.pack(root, archive)
                # Exclusive publication also protects against concurrent writers.
                os.link(archive, destination)
            else:
                root.rename(destination)
        except offline.BundleError as error:
            raise ValueError(str(error)) from error
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory or .tgz/.tar.gz file"
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Tested artifact: id:<UUID> or image:<repo>@sha256:<digest>",
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Tested architecture, e.g. x86_64; never inferred from the uploader",
    )
    parser.add_argument("--component", default="hf-adapters")
    parser.add_argument("--run-url", default="", help="Original CI run URL, if known")
    args = parser.parse_args(argv)
    try:
        print(create_bundle(**vars(args)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
