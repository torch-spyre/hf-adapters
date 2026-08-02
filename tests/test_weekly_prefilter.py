"""Unit tests for the weekly-scan pre-filter.

Lives at the ``tests/`` root rather than under ``tests/spyre/`` (Spyre hardware)
or ``tests/cpu/`` (torch fixtures), following ``test_adapter_coverage.py``: run
with ``pytest --noconftest`` so the torch-importing root conftest is bypassed and
only ``pytest`` itself is needed. See test_pull_request.yaml's adapter-coverage
job for the same pattern.

The pre-filter tests are pure dict-in/partition-out and need no third-party
package at all. The sink-backed tests additionally need ``clickhouse_connect``,
because ``result_sink`` imports ``clickhouse_db`` at module scope — they skip
cleanly when it is absent, so the pure tests still run on a bare interpreter.
That split is exactly what the ``failure_categories`` leaf module buys, and the
skipping keeps it honest.

The dedup-guard tests use a fake in-memory sink rather than ``CsvResultSink``:
the guard belongs to the base class, and the CSV sink is write-only by design, so
it reports nothing as blocking and cannot exercise a rejection.
"""

from __future__ import annotations

import importlib.util
import inspect
from datetime import date

import pytest

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_MOE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
)
from tests.spyre.weekly_generation.model_prefilter import prefilter_models

# The sink pulls in clickhouse_connect transitively; the pre-filter must not.
# Gating at class level (rather than importing CsvResultSink at module scope)
# keeps the pure tests runnable on an interpreter with no DB driver.
requires_sink = pytest.mark.skipif(
    importlib.util.find_spec("clickhouse_connect") is None,
    reason="clickhouse_connect not installed (result_sink imports it at module scope)",
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
        result = prefilter_models([_row("org/ok")], should_scan=lambda _: True)
        assert [r["model_id"] for r in result.keep] == ["org/ok"]
        assert result.skipped == []
        assert result.window_skipped == []

    def test_unsupported_config_class(self) -> None:
        result = prefilter_models(
            [_row("org/unsup", is_supported=False)], should_scan=lambda _: True
        )
        assert result.keep == []
        assert len(result.skipped) == 1
        assert result.skipped[0].failure_category == (
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
        )

    def test_too_large(self) -> None:
        result = prefilter_models(
            [_row("org/huge", parameters=999)],
            should_scan=lambda _: True,
            max_params=100,
        )
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MODEL_TOO_LARGE

    def test_exactly_at_the_limit_is_kept(self) -> None:
        """The guard is ``>`` not ``>=`` — a model exactly at the cap is fine."""
        result = prefilter_models(
            [_row("org/edge", parameters=100)],
            should_scan=lambda _: True,
            max_params=100,
        )
        assert [r["model_id"] for r in result.keep] == ["org/edge"]

    def test_moe(self) -> None:
        result = prefilter_models(
            [_row("org/moe", is_moe=True)], should_scan=lambda _: True
        )
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MOE

    def test_skip_window(self) -> None:
        result = prefilter_models([_row("org/recent")], should_scan=lambda _: False)
        assert result.keep == []
        assert result.skipped == [], "window skips must not produce a row to write"
        assert [r["model_id"] for r in result.window_skipped] == ["org/recent"]


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
        result = prefilter_models(rows, should_scan=lambda _: False)
        assert result.skipped == []
        assert len(result.window_skipped) == 3

    def test_unsupported_wins_over_moe(self) -> None:
        """Both apply; the reported category is the first check that fires."""
        result = prefilter_models(
            [_row("org/both", is_supported=False, is_moe=True)],
            should_scan=lambda _: True,
        )
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
        result = prefilter_models(
            [_row("org/unknown", parameters=value)],
            should_scan=lambda _: True,
            max_params=100,
        )
        assert [r["model_id"] for r in result.keep] == ["org/unknown"]

    def test_numeric_string_is_compared_as_a_number(self) -> None:
        """A stringified count must not be compared lexicographically."""
        result = prefilter_models(
            [_row("org/str", parameters="999")],
            should_scan=lambda _: True,
            max_params=100,
        )
        assert result.keep == []
        assert result.skipped[0].failure_category == FAILURE_CATEGORY_MODEL_TOO_LARGE

    def test_zero_parameters_is_kept(self) -> None:
        result = prefilter_models(
            [_row("org/zero", parameters=0)], should_scan=lambda _: True
        )
        assert [r["model_id"] for r in result.keep] == ["org/zero"]


class TestOrderAndTallies:
    def test_keep_preserves_input_order(self) -> None:
        """The tier router and shard chunker rely on downloads-descending order."""
        rows = [_row(f"org/m{i}", downloads=1000 - i) for i in range(20)]
        rows[3]["is_moe"] = True
        rows[11]["is_supported"] = False
        result = prefilter_models(rows, should_scan=lambda m: m != "org/m7")

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
        result = prefilter_models(rows, should_scan=lambda m: m != "org/e")
        counts = result.counts
        assert counts["keep"] == 1
        assert counts["window_skipped"] == 1
        assert counts[FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER] == 1
        assert counts[FAILURE_CATEGORY_MOE] == 1
        assert counts[FAILURE_CATEGORY_MODEL_TOO_LARGE] == 1
        assert sum(counts.values()) == len(rows)

    def test_empty_input(self) -> None:
        result = prefilter_models([], should_scan=lambda _: True)
        assert result.keep == []
        assert result.counts == {"keep": 0, "window_skipped": 0}

    def test_rows_are_returned_by_identity(self) -> None:
        """The same dict objects come back, so a caller's pop() still applies."""
        row = _row("org/same")
        result = prefilter_models([row], should_scan=lambda _: True)
        assert result.keep[0] is row


@requires_sink
class TestWriteSkippedRows:
    def test_writes_one_row_per_skipped_model(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.result_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        rows = [
            _row("org/unsup", is_supported=False),
            _row("org/moe", is_moe=True),
        ]
        result = prefilter_models(rows, should_scan=lambda _: True)
        today = date.today()

        with CsvResultSink(path=csv_path, today=today, dedup_guard=False) as sink:
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

        from tests.spyre.weekly_generation.result_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        today = date.today()
        result = prefilter_models(
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
            ],
            should_scan=lambda _: True,
        )
        with CsvResultSink(path=csv_path, today=today, dedup_guard=False) as sink:
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
        from tests.spyre.weekly_generation.result_sink import CsvResultSink
        from tests.spyre.weekly_generation.skip_writer import write_skipped_rows

        csv_path = tmp_path / "out.csv"
        with CsvResultSink(path=csv_path, dedup_guard=False) as sink:
            assert write_skipped_rows(sink, [], snapshot_date=date.today()) == 0


def _fake_sink_cls():
    """Build a minimal concrete ResultSink subclass, importing the base lazily.

    The guard lives entirely in the base class's ``add_entry``, so testing it
    needs a sink, not a *storage backend*. ``CsvResultSink`` used to serve here,
    but it is now write-only by design and reports nothing as blocking, so it can
    no longer exercise a rejection at all. This fake can — and defining it inside
    a function keeps ``result_sink`` (and therefore ``clickhouse_connect``) out of
    this module's import-time dependencies.
    """
    from tests.spyre.weekly_generation.result_sink import ResultSink

    class _FakeSink(ResultSink):
        def __init__(self, blocking: set[str] | None = None, **kwargs) -> None:
            super().__init__(**kwargs)
            self._blocking = blocking or set()
            self.written: list[str] = []

        def get_recent_blocking_entries(self, model_name):
            return [{"model_name": model_name}] if model_name in self._blocking else []

        def get_all_models(self):
            return []

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
class TestDedupGuardFlag:
    """The guard must stay opt-out, and stay keyword-only.

    ``clickhouse_db.py`` constructs ``ClickHouseResultSink(mode)`` positionally,
    so a new positional parameter there would silently misbind to
    ``embedding_generative``. None of this needs a DB connection.
    """

    @pytest.fixture(autouse=True)
    def _bind(self):
        self.FakeSink = _fake_sink_cls()

    def test_guard_on_rejects_a_blocked_model(self) -> None:
        sink = self.FakeSink(blocking={"org/blocked"}, dedup_guard=True)
        assert _add(sink, "org/blocked") is False
        assert sink.written == []

    def test_guard_off_writes_a_blocked_model_anyway(self) -> None:
        """The weekly pipeline's behaviour: the decision was made upstream."""
        sink = self.FakeSink(blocking={"org/blocked"}, dedup_guard=False)
        assert _add(sink, "org/blocked") is True
        assert sink.written == ["org/blocked"]

    def test_guard_on_allows_an_unblocked_model(self) -> None:
        sink = self.FakeSink(dedup_guard=True)
        assert _add(sink, "org/fresh") is True
        assert sink.written == ["org/fresh"]

    def test_guard_defaults_to_on(self) -> None:
        """clickhouse_db.import_csv depends on the default, so it must not drift."""
        assert self.FakeSink()._dedup_guard is True

    def test_should_insert_row_still_works_with_the_guard_off(self) -> None:
        """Producers call it directly — turning the guard off must not break it."""
        sink = self.FakeSink(blocking={"org/blocked"}, dedup_guard=False)
        assert sink.should_insert_row("org/blocked") is False
        assert sink.should_insert_row("org/other") is True

    @pytest.mark.parametrize(
        "cls_name, expected_positional, expected_default",
        [
            ("CsvResultSink", ["path", "today"], False),
            ("ClickHouseResultSink", ["embedding_generative", "today"], True),
        ],
    )
    def test_dedup_guard_is_keyword_only(
        self, cls_name: str, expected_positional: list[str], expected_default: bool
    ) -> None:
        import tests.spyre.weekly_generation.result_sink as rs

        params = inspect.signature(getattr(rs, cls_name).__init__).parameters
        positional = [
            name
            for name, p in params.items()
            if name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == expected_positional, (
            "positional parameters changed — clickhouse_db.py constructs "
            "ClickHouseResultSink(mode) positionally and would misbind"
        )
        assert params["dedup_guard"].kind is inspect.Parameter.KEYWORD_ONLY
        # CsvResultSink defaults to False: a fresh output file cannot hold a
        # blocking row, so the guard would be pure overhead there.
        assert params["dedup_guard"].default is expected_default


@requires_sink
class TestCsvSinkIsWriteOnly:
    """The CSV sink writes one run to a new file and never reads one back."""

    def test_refuses_a_non_empty_existing_file(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.result_sink import CsvResultSink

        existing = tmp_path / "already.csv"
        existing.write_text("model_name\norg/x\n")
        with pytest.raises(FileExistsError, match="already exists"):
            CsvResultSink(path=existing)

    def test_accepts_an_empty_existing_file(self, tmp_path) -> None:
        """A zero-byte file is not a previous run's results."""
        from tests.spyre.weekly_generation.result_sink import CsvResultSink

        empty = tmp_path / "empty.csv"
        empty.touch()
        sink = CsvResultSink(path=empty)
        sink.close()
        assert "model_name" in empty.read_text()

    def test_creates_parent_directories(self, tmp_path) -> None:
        from tests.spyre.weekly_generation.result_sink import CsvResultSink

        sink = CsvResultSink(path=tmp_path / "a" / "b" / "out.csv")
        sink.close()
        assert (tmp_path / "a" / "b" / "out.csv").exists()

    def test_reports_nothing_as_blocking(self, tmp_path) -> None:
        """So --fetch against a CSV evaluates every model that clears the rest."""
        from tests.spyre.weekly_generation.result_sink import CsvResultSink

        sink = CsvResultSink(path=tmp_path / "o.csv")
        _add(sink, "org/x")
        assert sink.should_insert_row("org/x") is True
        assert sink.get_recent_blocking_entries("org/x") == []
        assert sink.get_all_models() == []
        sink.close()

    def test_writes_every_row_including_repeats(self, tmp_path) -> None:
        import csv as _csv

        from tests.spyre.weekly_generation.result_sink import CsvResultSink

        path = tmp_path / "o.csv"
        sink = CsvResultSink(path=path)
        assert _add(sink, "org/x") is True
        assert _add(sink, "org/x") is True
        sink.close()
        assert len(list(_csv.DictReader(path.open()))) == 2
