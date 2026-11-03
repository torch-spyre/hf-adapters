"""Deadline orchestration tests; run with pytest --noconftest, without Spyre."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock
from urllib.request import Request

import pytest
import yaml

from tests.spyre.weekly_generation import weekly_test
from tests.spyre.weekly_generation.model_type import ModelType

_WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/push-to-clickhouse.yaml"
)


def _step_script(job: str, name: str) -> str:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    for step in workflow["jobs"][job]["steps"]:
        if step.get("name") == name:
            script: str = step["run"]
            return script
    raise AssertionError(f"No step named {name!r} in {job!r}")


class _Clock(dt.datetime):
    current: ClassVar[dt.datetime] = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    @classmethod
    def now(cls: type[_Clock], tz: dt.tzinfo | None = None) -> dt.datetime:
        return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)


def _compute_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created_at: str,
    *,
    now: str = "2026-01-01T00:00:00+00:00",
    event: str = "schedule",
) -> None:
    def metadata(request: Request, timeout: int) -> io.BytesIO:
        assert event == "schedule", "Manual scans must not query run metadata"
        assert (
            request.full_url
            == "https://api.github.com/repos/example/repo/actions/runs/123"
        )
        assert timeout == 30
        return io.BytesIO(json.dumps({"created_at": created_at}).encode())

    monkeypatch.setattr(_Clock, "current", dt.datetime.fromisoformat(now))
    monkeypatch.setattr(dt, "datetime", _Clock)
    monkeypatch.setattr("urllib.request.urlopen", metadata)
    for key, value in {
        "GITHUB_EVENT_NAME": event,
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_REPOSITORY": "example/repo",
        "GITHUB_RUN_ID": "123",
        "GH_TOKEN": "test-token",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)
    script = _step_script("generate-matrix", "Compute scan deadline")
    # Execute the actual inline Python, including its output and expiry checks.
    source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(source, str(_WORKFLOW), "exec"), {})


@pytest.mark.parametrize(
    ("created_at", "expected"),
    [
        ("2026-11-07T22:00:00Z", "2026-11-09T13:40:00+00:00"),
        ("2026-07-04T22:00:00Z", "2026-07-06T12:40:00+00:00"),
        ("2026-11-08T03:30:00Z", "2026-11-09T13:40:00+00:00"),
        ("2026-10-31T22:00:00Z", "2026-11-02T13:40:00+00:00"),
        ("2026-03-07T22:00:00Z", "2026-03-09T12:40:00+00:00"),
        ("2026-11-09T15:00:00Z", "2026-11-09T13:40:00+00:00"),
    ],
)
def test_deadline_leaves_cleanup_time_before_monday(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, created_at: str, expected: str
) -> None:
    _compute_deadline(tmp_path, monkeypatch, created_at)
    assert (tmp_path / "output").read_text() == f"deadline={expected}\n"
    assert "Fleet cutoff:" in (tmp_path / "summary").read_text()


@pytest.mark.parametrize(
    "now", ["2026-11-09T13:40:00+00:00", "2026-11-16T15:00:00+00:00"]
)
def test_expired_rerun_keeps_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: str
) -> None:
    with pytest.raises(SystemExit, match="already reached its deadline"):
        _compute_deadline(tmp_path, monkeypatch, "2026-11-07T22:00:00Z", now=now)
    assert (tmp_path / "output").read_text() == "deadline=2026-11-09T13:40:00+00:00\n"


def test_manual_scan_has_no_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _compute_deadline(tmp_path, monkeypatch, "unused", event="workflow_dispatch")
    assert (tmp_path / "output").read_text() == "deadline=\n"


@pytest.mark.parametrize("list_fails", [False, True])
def test_cancellation_selects_only_older_unfinished_scans(
    tmp_path: Path, list_fails: bool
) -> None:
    runs: list[dict[str, object]] = [
        {
            "databaseId": run_id,
            "status": status,
            "displayTitle": title,
            "createdAt": created,
        }
        for run_id, status, title, created in [
            (1, "in_progress", "Weekly model scan", "2026-11-07T22:00:00Z"),
            (2, "queued", "Weekly model scan", "2026-11-07T22:00:00Z"),
            (3, "completed", "Weekly model scan", "2026-11-07T22:00:00Z"),
            (4, "in_progress", "Weekly scan deadline", "2026-11-02T13:55:00Z"),
            (5, "in_progress", "Weekly model scan", "2026-11-14T22:00:00Z"),
            (123, "in_progress", "Weekly scan deadline", "2026-11-09T13:55:00Z"),
        ]
    ]
    # Fake only the network CLI. Execute the real shell and jq selection logic.
    fake_gh = """
    gh() {
      case "$1 $2" in
        'run view') echo '2026-11-09T13:55:00Z' ;;
        'run list')
          [[ "$*" == *'--event schedule'* ]] || return 2
          [[ "$LIST_FAILS" == 'false' ]] || return 1
          printf '%s' "$RUNS_JSON" ;;
        'run cancel') echo "$3" >> "$CANCEL_LOG" ;;
        *) return 2 ;;
      esac
    }
    """
    script = _step_script("cancel-weekly-scan", "Cancel unfinished scheduled scans")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", fake_gh + script],
        env={
            **os.environ,
            "GITHUB_RUN_ID": "123",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "example/repo",
            "RUNS_JSON": json.dumps(runs),
            "CANCEL_LOG": str(tmp_path / "cancelled"),
            "LIST_FAILS": str(list_fails).lower(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if list_fails:
        assert result.returncode != 0
        assert not (tmp_path / "cancelled").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "cancelled").read_text().splitlines() == ["1", "2"]


@pytest.fixture
def scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Mock, Mock]:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps(
            [
                {"model_id": "org/a", "curated": False},
                {"model_id": "org/b", "curated": False},
            ]
        )
    )
    sink = Mock()
    sink.get_models_at_snapshot_date.return_value = set()
    context = Mock()
    monkeypatch.setattr(weekly_test, "create_sink", Mock(return_value=sink))
    monkeypatch.setattr(weekly_test, "_get_adapter_dates", Mock(return_value={}))
    monkeypatch.setattr(
        weekly_test, "_cache_warmed_repos", Mock(return_value=frozenset())
    )
    monkeypatch.setattr(weekly_test, "_repos_with_weights", Mock(return_value=set()))
    monkeypatch.setattr(weekly_test, "_cleanup_batch_weights", Mock(return_value=0))
    monkeypatch.setattr(weekly_test, "GENERATIVE_NUMBER_OF_MODEL_PER_PROCESS", 1)
    monkeypatch.setattr(
        weekly_test.multiprocessing, "get_context", Mock(return_value=context)
    )
    return models, sink, context


def _run_scan(models: Path, deadline: dt.datetime | None) -> None:
    weekly_test.main(
        model_type=ModelType.GENERATIVE,
        model_list_file=models,
        write_to_csv=None,
        fetch=False,
        top_k=0,
        max_params=0,
        snapshot_date=dt.date(2026, 11, 7),
        deadline=deadline,
    )


def test_expired_shard_does_not_open_sink_or_start_worker(
    scan: tuple[Path, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    models, sink, context = scan
    monkeypatch.setattr(weekly_test, "_seconds_until", Mock(return_value=0))
    with pytest.raises(weekly_test.DeadlineReachedError, match="before shard startup"):
        _run_scan(models, dt.datetime.now(dt.timezone.utc))
    context.Process.assert_not_called()
    assert not sink.mock_calls


def test_mid_batch_deadline_terminates_worker_without_false_verdicts(
    scan: tuple[Path, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    models, sink, context = scan
    worker: Mock = context.Process.return_value
    worker.is_alive.side_effect = [True, False]
    monkeypatch.setattr(
        weekly_test, "_seconds_until", Mock(side_effect=[100, 100, 0, 0])
    )
    with pytest.raises(weekly_test.DeadlineReachedError, match="during batch 1/2"):
        _run_scan(models, dt.datetime.now(dt.timezone.utc))
    worker.terminate.assert_called_once()
    worker.kill.assert_not_called()
    sink.add_entry.assert_not_called()
    sink.close.assert_called_once()


@pytest.mark.parametrize("has_deadline", [True, False])
def test_completed_batches_are_flushed(
    scan: tuple[Path, Mock, Mock], monkeypatch: pytest.MonkeyPatch, has_deadline: bool
) -> None:
    models, sink, context = scan
    context.Process.return_value.is_alive.return_value = False
    context.SimpleQueue.return_value.empty.return_value = False
    context.SimpleQueue.return_value.get.return_value = [
        {
            "model_name": "org/a",
            "config_class": "LlamaConfig",
            "adapter_name": "hf_llama",
            "added_date": None,
            "snapshot_date": dt.date(2026, 11, 7),
            "verified_on_cpu": True,
            "verified_on_gpu": False,
            "verified_on_spyre": True,
            "curated": False,
            "num_downloads": 0,
            "family": "llama",
            "architecture": "LlamaForCausalLM",
            "parameters_number": 1,
        }
    ]
    if has_deadline:
        monkeypatch.setattr(
            weekly_test, "_seconds_until", Mock(side_effect=[100, 100, 100, 0])
        )
        with pytest.raises(weekly_test.DeadlineReachedError, match="before batch 2/2"):
            _run_scan(models, dt.datetime.now(dt.timezone.utc))
    else:
        _run_scan(models, None)
    expected_batches = 1 if has_deadline else 2
    assert sink.add_entry.call_count == expected_batches
    assert sink.flush.call_count == expected_batches
    sink.close.assert_called_once()


def test_unresponsive_worker_is_killed() -> None:
    worker = Mock()
    worker.is_alive.return_value = True
    weekly_test._stop_worker(worker)
    worker.terminate.assert_called_once()
    worker.kill.assert_called_once()


def test_deadline_stop_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    weekly_test._report_deadline_stop(
        weekly_test.DeadlineReachedError("stopped during batch 2/3")
    )
    assert "stopped during batch 2/3" in summary.read_text()
    assert "Previously flushed batches are preserved" in summary.read_text()


def test_naive_deadline_is_rejected() -> None:
    with pytest.raises(
        weekly_test.argparse.ArgumentTypeError, match="must include a timezone"
    ):
        weekly_test._parse_deadline("2026-11-09T14:00:00")
