"""capability_junit: a model-support scan's CSV as capability-declaring JUnit cases."""

from __future__ import annotations

import csv
from xml.etree import ElementTree as ET

import pytest

from tests.spyre.weekly_generation import capability_junit
from tests.spyre.weekly_generation.table_schema import TABLE_COLUMNS


def _csv(tmp_path, rows):
    path = tmp_path / "model-support-generative.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TABLE_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in TABLE_COLUMNS})
    return path


ROW = {
    "model_name": "org/model-a",
    "adapter_name": "hf_llama",
    "verified_on_cpu": True,
    "verified_on_gpu": False,
    "verified_on_spyre": False,
    "failure_category": "spyre_compile",
    "curated": False,
    "parameters_number": 1000,
}


def _cases(tmp_path, rows):
    out = tmp_path / "junit.xml"
    assert capability_junit.main(["--out", str(out), str(_csv(tmp_path, rows))]) == 0
    return ET.parse(out).getroot().findall("testcase")


def test_one_case_per_backend_declaring_its_capability(tmp_path):
    cpu, spyre = _cases(tmp_path, [ROW])
    props = {
        p.get("name"): p.get("value") for p in spyre.find("properties").iter("property")
    }
    assert props["capability.test_type"] == "model_support"
    assert (props["capability.subject"], props["capability.name"]) == (
        "org/model-a",
        "hf_llama",
    )
    assert props["capability.backend"] == "spyre"
    assert props["capability.prop.parameters_number"] == "1000"
    assert props["capability.prop.model_type"] == "generative"
    # The CSV's "False" text is a failed backend, not a truthy string.
    assert spyre.find("failure").get("message") == "spyre_compile"
    assert cpu.find("failure") is None


def test_case_names_are_unique_per_model_and_backend(tmp_path):
    names = [c.get("name") for c in _cases(tmp_path, [ROW, {**ROW, "model_name": "b"}])]
    assert len(names) == len(set(names)) == 4


def test_no_csv_is_an_error(tmp_path):
    assert capability_junit.main(["--out", str(tmp_path / "o.xml"), "missing.csv"]) == 1


def test_the_shared_ingest_reads_each_case_as_a_capability_verdict(tmp_path):
    writer = pytest.importorskip("spyre_clickhouse_ingest.writer")
    for case in _cases(tmp_path, [ROW]):
        props = [(p.get("name"), p.get("value")) for p in case.iter("property")]
        decl, problem = writer.capability_declaration({"properties": props})
        assert problem == "" and decl["test_type"] == "model_support"
