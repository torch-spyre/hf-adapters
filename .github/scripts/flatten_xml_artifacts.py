#!/usr/bin/env python3
"""
Copies every *.xml under --src-dir into --dest-dir as a flat listing, for
push-test-results-to-clickhouse.yaml.

The shared ingest-xml-to-clickhouse action's ingest_xml.py globs --xml-dir with
Path.glob("*.xml") -- NON-recursive -- while download_xml_artifacts.py nests
files one level deep in per-artifact subdirectories. A non-recursive glob
against that nesting finds nothing, so each file is copied up with its relative
path joined by "__" (e.g. smoke-Qwen3/Qwen3-0.6B.xml ->
smoke-Qwen3__Qwen3-0.6B.xml), preserving the same-model-different-suite
uniqueness the nesting existed for.

Usage (called by the GHA workflow):
    python3 flatten_xml_artifacts.py --src-dir xml_artifacts --dest-dir xml_artifacts_flat
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src-dir", default="xml_artifacts", type=Path)
    parser.add_argument("--dest-dir", default="xml_artifacts_flat", type=Path)
    args = parser.parse_args()

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    for xml_path in sorted(args.src_dir.rglob("*.xml")):
        flat_name = "__".join(xml_path.relative_to(args.src_dir).parts)
        shutil.copy2(xml_path, args.dest_dir / flat_name)


if __name__ == "__main__":
    main()
