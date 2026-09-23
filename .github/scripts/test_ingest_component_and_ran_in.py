"""Pins the two properties that let this ingest serve a shared suite.

`--component` so a cell running ANOTHER product's suite through this script attributes the
rows (and the test_case_id they hash into) to the suite's real owner; `ran_in` on EXECUTED
rows because it is the base case a reuse copy preserves.
"""

import importlib.util
import sys
import types
from pathlib import Path

INGEST = Path(__file__).resolve().parent / "ingest_xml_hf_adapters.py"
RID = "2731e944-b152-48a2-9747-3d28f3f10bbe"


def _load():
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    spec = importlib.util.spec_from_file_location("ingest_hf", INGEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    def __init__(self):
        self.rows = {}

    def insert(self, table, rows, column_names=None, **kw):
        self.rows.setdefault(table, []).extend(
            [dict(zip(column_names, r)) for r in rows] if column_names else list(rows)
        )

    def query(self, *a, **k):
        class _R:
            result_rows = []

        return _R()


def test_component_defaults_to_this_repos_product():
    assert (lambda m: m.v2_component(_Args(component=""), m.V2_COMPONENT_DEFAULT))(
        _load()
    ) == "hf-adapters"


def test_component_honours_an_explicit_override():
    assert (
        lambda m: m.v2_component(_Args(component="torch-spyre"), m.V2_COMPONENT_DEFAULT)
    )(_load()) == "torch-spyre"


def test_component_treats_blank_as_absent():
    assert (lambda m: m.v2_component(_Args(component="   "), m.V2_COMPONENT_DEFAULT))(
        _load()
    ) == "hf-adapters"


def test_component_survives_a_caller_that_passes_no_flag():
    # An older caller's Namespace has no `component` attribute; falling back keeps the ingest
    # working while callers are updated, rather than raising.
    assert (lambda m: m.v2_component(_Args(), m.V2_COMPONENT_DEFAULT))(
        _load()
    ) == "hf-adapters"


def test_component_changes_test_case_identity():
    # Why a wrong stamp is not merely a mislabel: component is a test_case_id hash input.
    # Imported from the library, which the ingest now uses rather than a local copy.
    from spyre_clickhouse_ingest import v2_test_case_id

    a = v2_test_case_id("hf-adapters", "T", "test_x", [])
    b = v2_test_case_id("torch-spyre", "T", "test_x", [])
    assert a and b and a != b


def _one_run_row(source_file="junit-token.xml"):
    m, client = _load(), _FakeClient()
    cases = [
        {
            "classname": "T",
            "name": "test_x",
            "status": "passed",
            "duration_s": 0.1,
            "tags": ["testtype__integration"],
            "fail_message": "",
        }
    ]
    m.insert_v2(client, "spyre_v2", "hf-adapters", RID, cases, source_file)
    return [r for r in client.rows.get("test_case_runs", []) if "props" in r]


def test_executed_rows_carry_ran_in_naming_this_run():
    rows = _one_run_row()
    assert rows and all(r["props"]["ran_in"] == RID for r in rows)


def test_ran_in_does_not_displace_source_file():
    # source_file is the per-file dedup key; ran_in is added beside it, not instead of it.
    rows = _one_run_row()
    assert all(r["props"]["source_file"] == "junit-token.xml" for r in rows)


def test_ran_in_is_present_even_without_a_source_file():
    rows = _one_run_row(source_file="")
    assert rows and all(r["props"]["ran_in"] == RID for r in rows)
    assert all("source_file" not in r["props"] for r in rows)


def test_ingest_uses_the_shared_library_not_a_local_copy():
    # The point of extensions/clickhouse-ingest is that ONE definition runs. A local copy that
    # merely agrees today passes every value-based test while drifting silently, so assert
    # object identity: editing the library must change what this ingest executes.
    import spyre_clickhouse_ingest as lib

    module = _load()
    for name in (
        "extract_properties",
        "get_client",
        "insert_v2",
        "promote_xpass",
        "v2_already_ingested",
        "v2_component",
        "v2_database",
        "v2_run_id_for",
        "v2_source_and_external_run_id",
        "v2_tables_present",
    ):
        assert getattr(module, name) is getattr(lib, name), name


def test_component_default_is_hf_not_the_library_default():
    # v2_component takes the default as a PARAMETER precisely so this repo stamps itself.
    # `component` hashes into test_case_id, so falling back to the library's own default would
    # mint torch-spyre identities for hf rows -- a wrong identity, not a mislabel.
    import spyre_clickhouse_ingest as lib

    module = _load()
    assert module.V2_COMPONENT_DEFAULT == "hf-adapters"
    args = _Args(component="")
    assert lib.v2_component(args, module.V2_COMPONENT_DEFAULT) == "hf-adapters"
    assert lib.v2_component(args) != "hf-adapters"
