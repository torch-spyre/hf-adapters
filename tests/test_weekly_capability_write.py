"""Unit tests for the weekly scan's schema-v2 capability rows.

Run with ``pytest --noconftest``, like ``test_weekly_prefilter.py``: ``capability_results`` is
pure and ``write`` runs against a fake ingest library and client, so no torch, database
driver or ingest library is needed.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests.spyre.weekly_generation.sink.capability_write import (
    capability_results,
    recorded_for_snapshot,
    write,
)


def _row(**overrides):
    row = {
        "model_name": "org/model",
        "adapter_name": "causal_lm",
        "verified_on_cpu": True,
        "verified_on_gpu": False,
        "verified_on_spyre": False,
        "failure_category": "not-implemented-adapter",
    }
    row.update(overrides)
    return row


def test_no_gpu_verdict():
    backends = {r["backend"] for r in capability_results([_row()])}
    assert backends == {"cpu", "spyre"}


def test_status_and_fail_reason_per_backend():
    by_backend = {r["backend"]: r for r in capability_results([_row()])}
    assert by_backend["cpu"]["status"] == "passed"
    assert by_backend["cpu"]["fail_reason"] == ""
    assert by_backend["spyre"]["status"] == "failed"
    assert by_backend["spyre"]["fail_reason"] == "not-implemented-adapter"


def test_prefilter_skip_fails_tested_backends_only():
    rows = capability_results(
        [
            _row(
                adapter_name="",
                verified_on_cpu=False,
                failure_category="model_too_large",
            )
        ]
    )
    assert [(r["backend"], r["status"], r["name"]) for r in rows] == [
        ("cpu", "failed", "unknown"),
        ("spyre", "failed", "unknown"),
    ]


def test_model_type_rides_as_a_prop():
    rows = capability_results([_row(model_type="embedding")])
    assert {r["props"]["model_type"] for r in rows} == {"embedding"}


class _Client:
    """Answers the per-run dedup query with the models already recorded."""

    def __init__(self, recorded=()):
        self.recorded = list(recorded)
        self.params: list[dict] = []
        self.sqls: list[str] = []

    def query(self, sql, parameters):
        self.params.append(parameters)
        self.sqls.append(sql)
        return types.SimpleNamespace(result_rows=[(m,) for m in self.recorded])


@pytest.fixture
def inserted(monkeypatch):
    calls: list[dict] = []
    lib = types.ModuleType("spyre_clickhouse_ingest")
    lib.run_id_of = lambda *parts: "run-" + "-".join(parts)  # type: ignore[attr-defined]

    def insert(client, db, component, run_id, test_type, results, **kw):
        calls.append({"db": db, "run_id": run_id, "results": results, **kw})
        return len(results)

    lib.insert_capabilities = insert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spyre_clickhouse_ingest", lib)
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    return calls


def test_write_skips_only_the_models_this_shard_already_recorded(inserted):
    client = _Client(recorded=["org/a"])
    n = write(
        [_row(model_name="org/a"), _row(model_name="org/b")],
        client,
        "spyre_v2",
        model_list_file="shards/generative-x1-shard-003.json",
    )
    assert n == 2
    assert {r["subject"] for r in inserted[0]["results"]} == {"org/b"}
    assert inserted[0]["shard"] == "generative-x1-shard-003"
    assert client.params[0]["shard"] == "generative-x1-shard-003"


def test_a_later_batch_of_the_same_shard_is_written(inserted):
    client = _Client()
    shard = "shards/generative-x1-shard-003.json"
    write([_row(model_name="org/a")], client, "spyre_v2", model_list_file=shard)
    write([_row(model_name="org/b")], client, "spyre_v2", model_list_file=shard)
    assert len(inserted) == 2


def test_a_fully_recorded_batch_writes_nothing(inserted):
    assert write([_row()], _Client(recorded=["org/model"]), "spyre_v2") == 0
    assert inserted == []


def test_write_without_a_run_id_raises(inserted, monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID")
    with pytest.raises(RuntimeError, match="GITHUB_RUN_ID"):
        write([_row()], _Client(), "spyre_v2")


def test_recorded_for_snapshot_scopes_on_date_and_type():
    client = _Client(recorded=["org/a"])
    got = recorded_for_snapshot(client, "spyre_v2", "2026-11-21", "generative")
    assert got == {"org/a"}
    assert client.params[0]["snapshot_date"] == "2026-11-21"
    assert client.params[0]["model_type"] == "generative"


def _sink(monkeypatch, *, db="spyre_v2", present=True, fail_insert=False):
    from tests.spyre.weekly_generation.model_type import ModelType
    from tests.spyre.weekly_generation.sink.clickhouse_sink import ClickHouseResultSink

    lib = sys.modules["spyre_clickhouse_ingest"]
    lib.get_client = lambda: _Client()  # type: ignore[attr-defined]
    lib.target_database = lambda: db  # type: ignore[attr-defined]
    lib.tables_present = lambda *a, **k: present  # type: ignore[attr-defined]
    schema = types.ModuleType("spyre_clickhouse_ingest.schema")
    schema.CAPABILITIES = "capabilities"  # type: ignore[attr-defined]
    schema.CAPABILITY_RUNS = "capability_runs"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spyre_clickhouse_ingest.schema", schema)
    if fail_insert:

        def boom(*a, **k):
            raise ConnectionError("down")

        lib.insert_capabilities = boom  # type: ignore[attr-defined]
    return ClickHouseResultSink(
        ModelType.EMBEDDING, "shards/embedding-x1-shard-000.json"
    )


def _buffer(sink, name):
    sink._insert_entry(
        model_name=name, config_class="C", adapter_name="a", added_date=None,
        snapshot_date=__import__("datetime").date(2026, 11, 21),
        verified_on_cpu=True, verified_on_gpu=False, verified_on_spyre=True,
        curated=False, num_downloads=1, family="f", architecture="A",
        parameters_number=1, failure_category=None, error=None,
    )  # fmt: skip


def test_sink_flush_writes_v2_rows_tagged_with_the_model_type(inserted, monkeypatch):
    sink = _sink(monkeypatch)
    _buffer(sink, "org/a")
    sink.flush()
    props = inserted[0]["results"][0]["props"]
    assert props["model_type"] == "embedding" and props["snapshot_date"] == "2026-11-21"
    assert sink._pending == []


def test_sink_keeps_rows_when_the_write_fails(inserted, monkeypatch):
    sink = _sink(monkeypatch, fail_insert=True)
    _buffer(sink, "org/a")
    with pytest.raises(ConnectionError):
        sink.flush()
    assert len(sink._pending) == 1


def test_sink_refuses_an_unset_database_or_missing_tables(inserted, monkeypatch):
    with pytest.raises(RuntimeError, match="CLICKHOUSE_DB_V2"):
        _sink(monkeypatch, db="")
    with pytest.raises(RuntimeError, match="capability tables"):
        _sink(monkeypatch, present=False)


def test_the_dedup_query_is_chunked(inserted):
    client = _Client()
    rows = [_row(model_name=f"org/m{i}") for i in range(1201)]
    assert write(rows, client, "spyre_v2") == 2402
    assert [len(p["subjects"]) for p in client.params] == [500, 500, 201]


def test_a_missing_model_type_matches_the_skip_set():
    client = _Client()
    recorded_for_snapshot(client, "spyre_v2", "2026-11-21", "generative")
    # Rows written before model_type was a prop still count as recorded.
    assert "IN ({model_type:String}, '')" in client.sqls[0]


def test_hardware_failures_are_judged_on_the_latest_spyre_verdict():
    client = _Client(recorded=["org/a"])
    got = recorded_for_snapshot(
        client, "spyre_v2", "2026-11-21", "generative", fail_reason="hardware_exception"
    )
    assert got == {"org/a"}
    sql = client.sqls[0]
    assert "r.backend = 'spyre'" in sql
    assert "HAVING argMax(r.fail_reason, r.ts) = {fail_reason:String}" in sql
    assert client.params[0]["fail_reason"] == "hardware_exception"


def test_a_partial_drop_is_logged(inserted, capsys):
    write([_row(model_name="org/a"), _row(model_name="org/b")], _Client(["org/a"]), "d")
    assert "skipped 1 model(s)" in capsys.readouterr().out


def test_a_failing_close_does_not_replace_the_body_error():
    from tests.spyre.weekly_generation.sink.result_sink import ResultSink

    class _Sink(ResultSink):
        def _insert_entry(self, **_k):
            pass

        def get_models_at_snapshot_date(self, *, snapshot_date):
            return set()

        def close(self):
            raise ConnectionError("close failed")

    with pytest.raises(ValueError, match="body"):
        with _Sink():
            raise ValueError("body")
    with pytest.raises(ConnectionError):
        with _Sink():
            pass
