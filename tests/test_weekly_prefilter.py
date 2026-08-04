"""Unit tests for the weekly-scan pre-filter.

Lives at the ``tests/`` root rather than under ``tests/spyre/`` (Spyre hardware)
or ``tests/cpu/`` (torch fixtures), following ``test_adapter_coverage.py``: run
with ``pytest --noconftest`` so the torch-importing root conftest is bypassed and
only ``pytest`` itself is needed. See test_pull_request.yaml's adapter-coverage
job for the same pattern.

``prefilter_models`` takes a ``ResultSink`` but only ever calls
``should_insert_row`` on it, so these tests drive it with ``_StubSink`` — a
duck-typed stand-in with no storage behind it. That is not just convenience:
the concrete sinks import ``clickhouse_connect`` transitively, and the stub is
what keeps the pre-filter tests runnable on an interpreter with no database
driver. Tests that do need a real sink are gated on ``requires_sink`` and import
one inside the test body.

A fake is likewise the only way to test the ``add_entry`` guard at all:
``CsvResultSink`` is write-only by design and reports nothing as blocking, so it
cannot exercise a rejection. See ``_fake_sink_cls``.
"""

from __future__ import annotations

import importlib.util
import inspect
from datetime import date
from pathlib import Path

import pytest

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_MOE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
    MAX_NUMBER_PARAMS,
)
from tests.spyre.weekly_generation.model_prefilter import prefilter_models
from tests.spyre.weekly_generation.model_type import ModelType

# The concrete sinks pull in clickhouse_connect transitively; the pre-filter and
# its stub sink must not. Gating at class level (rather than importing
# CsvResultSink at module scope) keeps the pure tests runnable with no DB driver.
requires_sink = pytest.mark.skipif(
    importlib.util.find_spec("clickhouse_connect") is None,
    reason="clickhouse_connect not installed (the ClickHouse sink imports it)",
)

# Subprocess tests below need the repo root as cwd to import from tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class _StubSink:
    """The slice of ``ResultSink`` that ``prefilter_models`` actually uses.

    Deliberately not a ``ResultSink`` subclass: importing the ABC would drag in
    nothing here, but constructing one requires implementing ``_insert_entry``,
    and duck-typing keeps this file's import graph free of ``result_sink``
    entirely. ``prefilter_models`` only calls ``should_insert_row``, and this
    records the calls so tests can assert it was consulted per model.
    """

    def __init__(self, blocking: set[str] | None = None) -> None:
        self._blocking = blocking or set()
        self.asked: list[str] = []

    def should_insert_row(self, model_name: str) -> bool:
        self.asked.append(model_name)
        return model_name not in self._blocking


def _filter(
    rows: list[dict],
    *,
    blocking: set[str] | None = None,
    max_params: int = MAX_NUMBER_PARAMS,
):
    """Run the pre-filter over *rows* with a stub sink.

    *blocking* is the set of model_ids the sink reports as already-scanned
    (i.e. ``should_insert_row`` returns False for them).
    """
    return prefilter_models(
        rows,
        sink=_StubSink(blocking),  # type: ignore[arg-type]  # duck-typed, see class
        max_params=max_params,
    )


def _row(model_id: str, **overrides: object) -> dict:
    """A catalog row that passes every filter unless *overrides* say otherwise."""
    row: dict = {
        "model_id": model_id,
        "downloads": 100,
        "parameters": 1_000_000_000,
        "is_supported": True,
        "is_moe": False,
        "config_class": "LlamaConfig",
        "model_type": "llama",
        "architectures": "LlamaForCausalLM",
    }
    row.update(overrides)
    return row


class TestFilterBranches:
    def test_clean_row_is_kept(self) -> None:
        result = _filter([_row("org/ok")])
        assert [r["model_id"] for r in result.keep] == ["org/ok"]
        assert result.skipped == []
        assert result.window_skipped == []

    def test_unsupported_config_class(self) -> None:
        result = _filter([_row("org/unsup", is_supported=False)])
        assert result.keep == []
        assert len(result.skipped) == 1
        assert result.skipped[0].failure_category == (
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
        )

    def test_missing_is_supported_is_not_treated_as_unsupported(self) -> None:
        """The check is ``is False``: unknown support status still gets evaluated.

        A missing key or None means the fetcher could not determine the config
        class, which is not the same as knowing there is no adapter for it.
        """
        assert [
            r["model_id"] for r in _filter([_row("org/a", is_supported=None)]).keep
        ] == ["org/a"]
        no_key = _row("org/b")
        del no_key["is_supported"]
        assert [r["model_id"] for r in _filter([no_key]).keep] == ["org/b"]

    def test_too_large(self) -> None:
        result = _filter([_row("org/huge", parameters=999)], max_params=100)
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MODEL_TOO_LARGE

    def test_exactly_at_the_limit_is_kept(self) -> None:
        """The guard is ``>`` not ``>=`` — a model exactly at the cap is fine."""
        result = _filter([_row("org/edge", parameters=100)], max_params=100)
        assert [r["model_id"] for r in result.keep] == ["org/edge"]

    def test_moe(self) -> None:
        result = _filter([_row("org/moe", is_moe=True)])
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MOE

    def test_skip_window(self) -> None:
        result = _filter([_row("org/recent")], blocking={"org/recent"})
        assert result.keep == []
        assert result.skipped == [], "window skips must not produce a row to write"
        assert [r["model_id"] for r in result.window_skipped] == ["org/recent"]


class TestSinkInteraction:
    """The sink is consulted for the verdict and never written to here.

    Recording the terminal rows is ``write_skipped_rows``' job, called separately
    by ``fetch_and_filter``, so ``prefilter_models`` stays pure.
    """

    def test_every_model_is_checked_against_the_sink(self) -> None:
        sink = _StubSink()
        rows = [_row("org/a"), _row("org/b"), _row("org/c")]
        prefilter_models(rows, sink=sink, max_params=MAX_NUMBER_PARAMS)  # type: ignore[arg-type]
        assert sink.asked == ["org/a", "org/b", "org/c"]

    def test_prefilter_does_not_write_to_the_sink(self) -> None:
        """A stub with no ``add_entry`` at all must get through untouched."""
        sink = _StubSink()
        rows = [_row("org/unsup", is_supported=False), _row("org/moe", is_moe=True)]
        result = prefilter_models(rows, sink=sink, max_params=MAX_NUMBER_PARAMS)  # type: ignore[arg-type]
        assert len(result.skipped) == 2
        assert not hasattr(sink, "written")


class TestPrecedence:
    def test_window_skip_wins_over_terminal_categories(self) -> None:
        """A model that is BOTH unsupported and recently recorded gets no new row.

        Order matters here: if the terminal checks ran first, every weekly run
        would append another not-implemented-adapter row for the same model.
        """
        rows = [
            _row("org/unsup-and-recent", is_supported=False),
            _row("org/moe-and-recent", is_moe=True),
            _row("org/huge-and-recent", parameters=10**15),
        ]
        result = _filter(rows, blocking={r["model_id"] for r in rows})
        assert result.skipped == []
        assert len(result.window_skipped) == 3

    def test_unsupported_wins_over_moe(self) -> None:
        """Both apply; the reported category is the first check that fires."""
        result = _filter([_row("org/both", is_supported=False, is_moe=True)])
        assert result.skipped[0].failure_category == (
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
        )


class TestParameterCoercion:
    @pytest.mark.parametrize("value", [None, "", "not-a-number", object()])
    def test_unknown_size_is_kept_for_the_in_worker_backstop(
        self, value: object
    ) -> None:
        """Unsizable rows must NOT be treated as zero-parameter or as too large.

        There is no worker-side size check to fall back on, so such a model is
        judged by whether it actually loads. That matches the behaviour before
        this filter moved upstream: the parent's guard skipped unsizable rows
        too, and no in-worker check ever existed despite a comment claiming one.
        """
        result = _filter([_row("org/unknown", parameters=value)], max_params=100)
        assert [r["model_id"] for r in result.keep] == ["org/unknown"]

    def test_numeric_string_is_compared_as_a_number(self) -> None:
        """A stringified count must not be compared lexicographically."""
        result = _filter([_row("org/str", parameters="999")], max_params=100)
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MODEL_TOO_LARGE

    def test_zero_parameters_is_kept(self) -> None:
        result = _filter([_row("org/zero", parameters=0)])
        assert [r["model_id"] for r in result.keep] == ["org/zero"]


class TestOrderAndTallies:
    def test_keep_preserves_input_order(self) -> None:
        """The tier router and shard chunker rely on downloads-descending order."""
        rows = [_row(f"org/m{i}", downloads=1000 - i) for i in range(20)]
        rows[3]["is_moe"] = True
        rows[11]["is_supported"] = False
        result = _filter(rows, blocking={"org/m7"})

        kept = [r["model_id"] for r in result.keep]
        assert kept == sorted(kept, key=lambda m: -(1000 - int(m.split("m")[1])))
        assert "org/m3" not in kept and "org/m11" not in kept
        assert "org/m7" not in kept
        assert len(kept) == 17

    def test_counts_reconcile_with_the_input(self) -> None:
        rows = [
            _row("org/a"),
            _row("org/b", is_supported=False),
            _row("org/c", is_moe=True),
            _row("org/d", parameters=10**15),
            _row("org/e"),
        ]
        result = _filter(rows, blocking={"org/e"})
        counts = result.counts
        assert counts["keep"] == 1
        assert counts["window_skipped"] == 1
        assert counts[FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER] == 1
        assert counts[FAILURE_CATEGORY_MOE] == 1
        assert counts[FAILURE_CATEGORY_MODEL_TOO_LARGE] == 1
        assert sum(counts.values()) == len(rows)

    def test_empty_input(self) -> None:
        result = _filter([])
        assert result.keep == []
        assert result.counts == {"keep": 0, "window_skipped": 0}

    def test_rows_are_returned_by_identity(self) -> None:
        """The same dict objects come back, so a caller's pop() still applies."""
        row = _row("org/same")
        result = _filter([row])
        assert result.keep[0] is row


class TestModelType:
    """ModelType must format as its bare value, not as ``ModelType.GENERATIVE``.

    It reaches shard filenames, the GHA matrix, and operator-facing log lines. A
    plain ``(str, Enum)`` inherits ``Enum.__str__`` and renders the qualified
    name, which is how shard files were once written as
    ``ModelType.GENERATIVE-x1-shard-000.json``.
    """

    @pytest.mark.parametrize("member", list(ModelType))
    def test_str_and_format_are_the_value(self, member: ModelType) -> None:
        assert str(member) == member.value
        assert f"{member}" == member.value
        assert f"{member}-x1-shard-000.json" == f"{member.value}-x1-shard-000.json"

    @pytest.mark.parametrize("member", list(ModelType))
    def test_json_serializable_as_the_value(self, member: ModelType) -> None:
        """The GHA matrix is emitted as JSON and read back by the workflow."""
        import json

        assert json.dumps({"mode": member}) == f'{{"mode": "{member.value}"}}'

    def test_round_trips_from_the_cli_string(self) -> None:
        """--mode / --model-type pass the raw string through ModelType(...)."""
        assert ModelType("generative") is ModelType.GENERATIVE
        assert ModelType("embedding") is ModelType.EMBEDDING

    def test_values_match_the_cli_choices(self) -> None:
        assert {m.value for m in ModelType} == {"generative", "embedding"}


@requires_sink
class TestWriteSkippedRows:
    def test_writes_one_row_per_skipped_model(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        rows = [
            _row("org/unsup", is_supported=False),
            _row("org/moe", is_moe=True),
        ]
        result = _filter(rows)
        today = date.today()

        with CsvResultSink(path=csv_path, today=today) as sink:
            written = write_skipped_rows(
                sink, result.skipped, snapshot_date=today, verbose=False
            )

        assert written == 2
        text = csv_path.read_text()
        assert "org/unsup" in text and "org/moe" in text
        assert FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER in text
        assert FAILURE_CATEGORY_MOE in text

    def test_field_mapping_matches_the_replaced_add_entry_calls(self, tmp_path) -> None:
        """Pin the 14 column values the three deleted branches used to write."""
        import csv as _csv

        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        today = date.today()
        result = _filter(
            [
                _row(
                    "org/unsup",
                    is_supported=False,
                    downloads=42,
                    parameters=7,
                    model_type="mistral",
                    architectures="MistralForCausalLM",
                    config_class="MistralConfig",
                )
            ]
        )
        with CsvResultSink(path=csv_path, today=today) as sink:
            write_skipped_rows(sink, result.skipped, snapshot_date=today, verbose=False)

        written_row = next(iter(_csv.DictReader(csv_path.open())))
        assert written_row["model_name"] == "org/unsup"
        assert written_row["config_class"] == "MistralConfig"
        assert written_row["adapter_name"] == ""
        assert written_row["added_date"] == ""
        assert written_row["snapshot_date"] == str(today)
        assert written_row["verified_on_cpu"] == "False"
        assert written_row["verified_on_gpu"] == "False"
        assert written_row["verified_on_spyre"] == "False"
        assert written_row["num_downloads"] == "42"
        assert written_row["family"] == "mistral"
        assert written_row["architecture"] == "MistralForCausalLM"
        assert written_row["parameters_number"] == "7"
        assert written_row["failure_category"] == (
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
        )
        assert written_row["error"] == ""

    def test_empty_skipped_list_writes_nothing(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        with CsvResultSink(path=csv_path) as sink:
            assert write_skipped_rows(sink, [], snapshot_date=date.today()) == 0


@requires_sink
class TestFetchAndFilter:
    """The shared entry point: fetch, filter, record verdicts, return the rest.

    Both producers go through this, so the ownership contract it keeps matters:
    it must write the terminal rows and must NOT close the sink, because
    ``weekly_test.main`` hands in the sink it uses for the whole run and keeps
    writing evaluation results to it afterwards.
    """

    @pytest.fixture
    def _stub_fetcher(self, monkeypatch):
        """Replace the Hub fetchers so no network call happens."""
        import tests.spyre.weekly_generation.model_fetcher as model_fetcher

        rows = [
            _row("org/keep"),
            _row("org/moe", is_moe=True),
            _row("org/recent"),
        ]

        def _fake(limit: int, **_kw) -> list[dict]:
            # Fresh dicts per call, and one carrying the non-serializable
            # model_info the real fetcher attaches, so the pop is exercised.
            out = [dict(r) for r in rows]
            out[0]["model_info"] = object()
            return out[:limit]

        monkeypatch.setattr(
            model_fetcher,
            "all_fetchers",
            {ModelType.GENERATIVE: _fake, ModelType.EMBEDDING: _fake},
        )
        return rows

    def test_keeps_survivors_records_verdicts_and_leaves_sink_open(
        self, tmp_path, _stub_fetcher
    ) -> None:
        import csv as _csv

        from tests.spyre.weekly_generation.model_prefilter import fetch_and_filter
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        path = tmp_path / "out.csv"
        today = date.today()
        sink = CsvResultSink(path=path, today=today)

        kept = fetch_and_filter(
            model_type=ModelType.GENERATIVE,
            snapshot_date=today,
            top_k=10,
            sink=sink,
            max_params=MAX_NUMBER_PARAMS,
        )

        # The MoE model is dropped and recorded; the other two survive.
        assert [r["model_id"] for r in kept] == ["org/keep", "org/recent"]

        # The sink must still be writable — main() writes every evaluation
        # result through this same object after fetch_and_filter returns.
        with sink:
            sink.add_entry(
                model_name="org/keep",
                config_class="LlamaConfig",
                adapter_name="hf_llama",
                added_date=None,
                snapshot_date=today,
                verified_on_cpu=True,
                verified_on_gpu=False,
                verified_on_spyre=True,
                num_downloads=100,
                family="llama",
                architecture="LlamaForCausalLM",
                parameters_number=1,
                failure_category=None,
                error=None,
            )

        written = list(_csv.DictReader(path.open()))
        assert [r["model_name"] for r in written] == ["org/moe", "org/keep"]
        assert written[0]["failure_category"] == FAILURE_CATEGORY_MOE

    def test_drops_the_unserializable_model_info_field(
        self, tmp_path, _stub_fetcher
    ) -> None:
        """Shards are JSON-dumped, and ModelInfo would break that dump."""
        import json

        from tests.spyre.weekly_generation.model_prefilter import fetch_and_filter
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        with CsvResultSink(path=tmp_path / "o.csv") as sink:
            kept = fetch_and_filter(
                model_type=ModelType.GENERATIVE,
                snapshot_date=date.today(),
                top_k=10,
                sink=sink,
                max_params=MAX_NUMBER_PARAMS,
            )
        assert all("model_info" not in r for r in kept)
        json.dumps(kept)  # must not raise

    def test_window_skipped_models_produce_no_row(
        self, tmp_path, _stub_fetcher
    ) -> None:
        """A blocked model is neither evaluated nor re-recorded."""
        import csv as _csv

        from tests.spyre.weekly_generation.model_prefilter import fetch_and_filter
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        path = tmp_path / "o.csv"

        class _BlockingCsvSink(CsvResultSink):
            def should_insert_row(self, model_name: str) -> bool:
                return model_name != "org/recent"

        with _BlockingCsvSink(path=path) as sink:
            kept = fetch_and_filter(
                model_type=ModelType.GENERATIVE,
                snapshot_date=date.today(),
                top_k=10,
                sink=sink,
                max_params=MAX_NUMBER_PARAMS,
            )

        assert [r["model_id"] for r in kept] == ["org/keep"]
        # Only the MoE verdict is written — nothing for org/recent.
        assert [r["model_name"] for r in _csv.DictReader(path.open())] == ["org/moe"]

    def test_max_params_is_honoured(self, tmp_path, _stub_fetcher) -> None:
        from tests.spyre.weekly_generation.model_prefilter import fetch_and_filter
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        with CsvResultSink(path=tmp_path / "o.csv") as sink:
            kept = fetch_and_filter(
                model_type=ModelType.GENERATIVE,
                snapshot_date=date.today(),
                top_k=10,
                sink=sink,
                max_params=1,  # every stub row is 1e9 params
            )
        assert kept == []


def _fake_sink_cls():
    """Build a minimal concrete ResultSink subclass, importing the base lazily.

    The guard lives entirely in the base class's ``add_entry``, so testing it
    needs a sink, not a *storage backend*. ``CsvResultSink`` used to serve here,
    but it is now write-only by design and reports nothing as blocking, so it can
    no longer exercise a rejection. This fake can — and defining it inside a
    function keeps ``result_sink`` out of this module's import-time dependencies.
    """
    from tests.spyre.weekly_generation.sink.result_sink import ResultSink

    class _FakeSink(ResultSink):
        def __init__(self, blocking: set[str] | None = None, **kwargs) -> None:
            super().__init__(**kwargs)
            self._blocking = blocking or set()
            self.written: list[str] = []

        def should_insert_row(self, model_name):
            return model_name not in self._blocking

        def _insert_entry(self, *, model_name, **_rest) -> None:
            self.written.append(model_name)

    return _FakeSink


def _add(sink, name: str) -> bool:
    return sink.add_entry(
        model_name=name,
        config_class="LlamaConfig",
        adapter_name="hf_llama",
        added_date=None,
        snapshot_date=date.today(),
        verified_on_cpu=True,
        verified_on_gpu=False,
        verified_on_spyre=True,
        num_downloads=1,
        family="llama",
        architecture="LlamaForCausalLM",
        parameters_number=1,
        failure_category=None,
        error=None,
    )


@requires_sink
class TestAddEntryAlwaysWrites:
    """add_entry records every row it is handed.

    The skip-window rule is applied once upstream while the model list is built,
    so a row reaching add_entry is one the caller already decided to record.
    Re-applying the rule here would silently drop results a run was asked to
    produce, leaving its row count lower than its input with no accounting.
    """

    @pytest.fixture(autouse=True)
    def _bind(self):
        self.FakeSink = _fake_sink_cls()

    def test_writes_even_when_should_insert_row_says_blocked(self) -> None:
        sink = self.FakeSink(blocking={"org/blocked"})
        _add(sink, "org/blocked")
        assert sink.written == ["org/blocked"]

    def test_writes_an_unblocked_model(self) -> None:
        sink = self.FakeSink()
        _add(sink, "org/fresh")
        assert sink.written == ["org/fresh"]

    def test_repeated_writes_of_the_same_model_all_land(self) -> None:
        """One row per call, so a run's count matches the models it handled."""
        sink = self.FakeSink()
        _add(sink, "org/dup")
        _add(sink, "org/dup")
        assert sink.written == ["org/dup", "org/dup"]

    def test_should_insert_row_still_reports_blocks(self) -> None:
        """The producers call it directly to build the filtered model list."""
        sink = self.FakeSink(blocking={"org/blocked"})
        assert sink.should_insert_row("org/blocked") is False
        assert sink.should_insert_row("org/other") is True

    def test_empty_model_name_is_still_rejected(self) -> None:
        sink = self.FakeSink()
        with pytest.raises(ValueError, match="model_name"):
            _add(sink, "   ")


@requires_sink
class TestSinkConstructors:
    """Pin each sink's constructor signature and where it is imported from.

    The sinks moved out of ``result_sink`` into the ``sink`` package, and the
    ClickHouse sink's first parameter was renamed from ``embedding_generative``
    to ``model_type`` (now a ``ModelType``, not a parallel enum). Both are
    load-bearing for ``sink_factory``, which passes them by keyword.
    """

    @pytest.mark.parametrize(
        "module_path, cls_name, expected_positional",
        [
            (
                "tests.spyre.weekly_generation.sink.csv_sink",
                "CsvResultSink",
                ["path", "today"],
            ),
            (
                "tests.spyre.weekly_generation.sink.clickhouse_sink",
                "ClickHouseResultSink",
                ["model_type", "today"],
            ),
        ],
    )
    def test_constructor_signatures(
        self, module_path: str, cls_name: str, expected_positional: list[str]
    ) -> None:
        """No stray keyword-only params, and the positional order is unchanged."""
        import importlib

        module = importlib.import_module(module_path)
        params = inspect.signature(getattr(module, cls_name).__init__).parameters
        positional = [
            name
            for name, p in params.items()
            if name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == expected_positional
        assert [p for p in params.values() if p.kind is p.KEYWORD_ONLY] == []

    def test_sinks_are_no_longer_exported_from_result_sink(self) -> None:
        """result_sink holds only the ABC now; the factory is the way in."""
        import tests.spyre.weekly_generation.sink.result_sink as rs

        assert not hasattr(rs, "CsvResultSink")
        assert not hasattr(rs, "ClickHouseResultSink")

    def test_both_sinks_implement_the_abc(self) -> None:
        from tests.spyre.weekly_generation.sink.clickhouse_sink import (
            ClickHouseResultSink,
        )
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
        from tests.spyre.weekly_generation.sink.result_sink import ResultSink

        assert issubclass(CsvResultSink, ResultSink)
        assert issubclass(ClickHouseResultSink, ResultSink)


@requires_sink
class TestSinkFactory:
    """``create_sink`` picks the backend and, for CSV, the per-model-type path."""

    def test_write_to_csv_yields_a_csv_sink_with_a_suffixed_path(
        self, tmp_path
    ) -> None:
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
        from tests.spyre.weekly_generation.sink.sink_factory import create_sink

        base = tmp_path / "verdicts.csv"
        with create_sink(
            model_type=ModelType.EMBEDDING,
            snapshot_date=date.today(),
            write_to_csv=base,
        ) as sink:
            assert isinstance(sink, CsvResultSink)

        # One file per model type, since a single run can cover both — and named
        # from the enum's *value*, not "verdicts-ModelType.EMBEDDING.csv".
        assert (tmp_path / "verdicts-embedding.csv").exists()
        assert not base.exists()

    def test_csv_path_for_matches_the_file_the_sink_creates(self, tmp_path) -> None:
        """The helper weekly_test logs from must agree with what got written.

        ``main`` reports its output path via ``csv_path_for`` while the sink was
        built by ``create_sink``; if the two computed the name separately they
        could drift, and the run would name a file that does not exist.
        """
        from tests.spyre.weekly_generation.sink.sink_factory import (
            create_sink,
            csv_path_for,
        )

        base = tmp_path / "results.csv"
        expected = csv_path_for(base, ModelType.GENERATIVE)
        with create_sink(
            model_type=ModelType.GENERATIVE,
            snapshot_date=date.today(),
            write_to_csv=base,
        ):
            pass
        assert expected.exists()
        assert expected.name == "results-generative.csv"

    def test_each_model_type_gets_its_own_file(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.sink.sink_factory import create_sink

        base = tmp_path / "v.csv"
        for model_type in ModelType:
            with create_sink(
                model_type=model_type,
                snapshot_date=date.today(),
                write_to_csv=base,
            ):
                pass
        assert {p.name for p in tmp_path.iterdir()} == {
            "v-generative.csv",
            "v-embedding.csv",
        }

    def test_no_csv_path_builds_the_clickhouse_sink(self, monkeypatch) -> None:
        """Without --write-to-csv the factory must reach for ClickHouse.

        The constructor is stubbed out: instantiating the real one would connect
        and read the skip set, and what is under test is the branch, not the
        driver. Patched on ``clickhouse_sink`` rather than on the factory,
        because the factory imports it lazily inside the function body — see
        ``test_csv_branch_needs_no_clickhouse_driver`` for why.
        """
        import tests.spyre.weekly_generation.sink.clickhouse_sink as ch_module
        from tests.spyre.weekly_generation.sink.sink_factory import create_sink

        seen: dict = {}

        class _FakeClickHouseSink:
            def __init__(self, *, model_type, today) -> None:
                seen["model_type"] = model_type
                seen["today"] = today

        monkeypatch.setattr(ch_module, "ClickHouseResultSink", _FakeClickHouseSink)
        today = date(2026, 8, 4)
        sink = create_sink(
            model_type=ModelType.GENERATIVE, snapshot_date=today, write_to_csv=None
        )
        assert isinstance(sink, _FakeClickHouseSink)
        assert seen == {"model_type": ModelType.GENERATIVE, "today": today}

    def test_csv_branch_does_not_connect_to_clickhouse(
        self, tmp_path, monkeypatch
    ) -> None:
        """--write-to-csv must not open a DB connection.

        ``get_client`` is the only thing that reaches the network, so binding it
        to a raiser is the check that matters: the CSV branch must complete
        without it.
        """
        import tests.spyre.weekly_generation.clickhouse_db as db
        from tests.spyre.weekly_generation.sink.sink_factory import create_sink

        def _boom(*_a, **_k):
            raise AssertionError("--write-to-csv must not connect to ClickHouse")

        monkeypatch.setattr(db, "get_client", _boom)

        with create_sink(
            model_type=ModelType.GENERATIVE,
            snapshot_date=date.today(),
            write_to_csv=tmp_path / "v.csv",
        ) as sink:
            assert sink.should_insert_row("org/x") is True
        assert (tmp_path / "v-generative.csv").exists()

    def test_csv_branch_runs_with_no_clickhouse_driver_installed(
        self, tmp_path
    ) -> None:
        """--write-to-csv must work on a host with neither the driver nor dotenv.

        That is the whole point of the flag, and it is easy to lose: taking
        ``TABLE_COLUMNS`` from ``clickhouse_db`` instead of ``table_schema``, or
        hoisting the ClickHouseResultSink import to ``sink_factory``'s module
        scope, would each reintroduce the dependency.

        Runs in a subprocess with both modules blocked at the finder level.
        In-process patching cannot test this — ``sink_factory`` and its imports
        are already in ``sys.modules`` by then, so blocking them later proves
        nothing.
        """
        import subprocess
        import sys
        import textwrap

        program = textwrap.dedent("""
            import sys

            BLOCKED = ("clickhouse_connect", "dotenv")

            class _Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in BLOCKED:
                        raise ModuleNotFoundError(f"No module named {name!r}")
                    return None

            sys.meta_path.insert(0, _Blocker())

            from datetime import date
            from pathlib import Path

            from tests.spyre.weekly_generation.model_type import ModelType
            from tests.spyre.weekly_generation.sink.sink_factory import create_sink

            with create_sink(
                model_type=ModelType.GENERATIVE,
                snapshot_date=date.today(),
                write_to_csv=Path(sys.argv[1]) / "v.csv",
            ) as sink:
                sink.add_entry(
                    model_name="org/x", config_class="LlamaConfig",
                    adapter_name="hf_llama", added_date=None,
                    snapshot_date=date.today(), verified_on_cpu=True,
                    verified_on_gpu=False, verified_on_spyre=True,
                    num_downloads=1, family="llama",
                    architecture="LlamaForCausalLM", parameters_number=1,
                    failure_category=None, error=None,
                )
            print("OK")
            """)
        proc = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert (
            proc.returncode == 0
        ), f"CSV branch failed without the driver installed:\n{proc.stderr}"
        assert "OK" in proc.stdout
        written = (tmp_path / "v-generative.csv").read_text()
        assert "model_name" in written and "org/x" in written


@requires_sink
class TestCsvSinkIsWriteOnly:
    """The CSV sink writes one run to a new file and never reads one back."""

    def test_refuses_a_non_empty_existing_file(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        existing = tmp_path / "already.csv"
        existing.write_text("model_name\norg/x\n")
        with pytest.raises(FileExistsError, match="already exists"):
            CsvResultSink(path=existing)

    def test_accepts_an_empty_existing_file(self, tmp_path) -> None:
        """A zero-byte file is not a previous run's results."""
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        empty = tmp_path / "empty.csv"
        empty.touch()
        sink = CsvResultSink(path=empty)
        sink.close()
        assert "model_name" in empty.read_text()

    def test_creates_parent_directories(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        sink = CsvResultSink(path=tmp_path / "a" / "b" / "out.csv")
        sink.close()
        assert (tmp_path / "a" / "b" / "out.csv").exists()

    def test_reports_nothing_as_blocking(self, tmp_path) -> None:
        """So --fetch against a CSV evaluates every model that clears the rest."""
        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        sink = CsvResultSink(path=tmp_path / "o.csv")
        _add(sink, "org/x")
        assert sink.should_insert_row("org/x") is True
        sink.close()

    def test_writes_every_row_including_repeats(self, tmp_path) -> None:
        import csv as _csv

        from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink

        path = tmp_path / "o.csv"
        sink = CsvResultSink(path=path)
        _add(sink, "org/x")
        _add(sink, "org/x")
        sink.close()
        assert len(list(_csv.DictReader(path.open()))) == 2
