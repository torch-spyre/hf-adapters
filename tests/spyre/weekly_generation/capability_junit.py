"""A model-support scan's ``--write-to-csv`` output as JUnit XML with ``capability.*`` properties.

The scan's verdicts then reach spyre_v2 the way every suite's do: the CI's own JUnit ingest
writes the capability rows under the leg's run_id, and the leg's artifact_results row ties them
to the image it tested. One case per (model, backend), as ``capability_write`` unrolls them.

    python -m tests.spyre.weekly_generation.capability_junit --out <junit.xml> <csv>...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from tests.spyre.weekly_generation.sink.capability_write import (
    TEST_TYPE,
    capability_results,
)

_BOOLS = ("verified_on_cpu", "verified_on_gpu", "verified_on_spyre", "curated")


def read_rows(paths: list[Path]) -> list[dict[str, str | bool]]:
    """The CSV rows, with the sink's ``True``/``False`` text back as booleans, each carrying the
    model type its file is named for (``csv_path_for``'s ``-<model_type>`` suffix)."""
    rows: list[dict[str, str | bool]] = []
    for path in paths:
        model_type = path.stem.rsplit("-", 1)[-1]
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rec = {k: (v == "True") if k in _BOOLS else v for k, v in row.items()}
                rows.append({**rec, "model_type": model_type})
    return rows


def junit(rows: list[dict]) -> ET.ElementTree:
    # capability_results drops unknown columns, so the model type rides alongside each result.
    results = [
        {**res, "model_type": row["model_type"]}
        for row in rows
        for res in capability_results([row])
    ]
    suite = ET.Element(
        "testsuite",
        name=TEST_TYPE,
        tests=str(len(results)),
        failures=str(sum(r["status"] == "failed" for r in results)),
    )
    for r in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=TEST_TYPE,
            name=f"{r['subject']}[{r['name']}-{r['backend']}]",
            time="0",
        )
        props = ET.SubElement(case, "properties")
        fields = [
            ("test_type", TEST_TYPE),
            ("subject", r["subject"]),
            ("name", r["name"]),
            ("backend", r["backend"]),
        ] + [
            (f"prop.{k}", v)
            for k, v in {**r["props"], "model_type": r["model_type"]}.items()
        ]
        for key, value in fields:
            ET.SubElement(props, "property", name=f"capability.{key}", value=str(value))
        if r["status"] == "failed":
            ET.SubElement(case, "failure", message=r["fail_reason"] or "failed")
    return ET.ElementTree(suite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("csv", type=Path, nargs="+")
    args = parser.parse_args(argv)
    paths = [p for p in args.csv if p.exists()]
    if not paths:
        print(f"capability_junit: none of {args.csv} exists", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    junit(read_rows(paths)).write(args.out, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
