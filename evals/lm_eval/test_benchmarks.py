"""CPU tests for suite reproducibility, native execution, and reference reports."""

import copy
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evals.lm_eval import run_benchmarks as runner


@pytest.fixture
def config(tmp_path):
    defaults = yaml.safe_load(
        (runner.ROOT / "evals/lm_eval/benchmarks.yaml").read_text()
    )["defaults"]
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "test-suite",
                "defaults": {
                    **defaults,
                    "limit": 2,
                    "max_length": 32,
                    "dtype": "float32",
                    "apply_chat_template": False,
                },
                "benchmarks": [{"id": "toy", "tasks": ["toy"]}],
            }
        )
    )
    return path


def raw_result(value=0.75):
    return {
        "results": {
            "toy": {"alias": "Toy", "acc,none": value, "acc_stderr,none": "N/A"}
        },
        "samples": {
            "toy": [
                {
                    "doc_id": 0,
                    "doc_hash": "doc",
                    "prompt_hash": "prompt",
                    "target_hash": "target",
                    "filter": "none",
                    "metrics": ["acc"],
                    "acc": value,
                },
            ]
        },
        "config": {"model_sha": "checkpoint123", "model_dtype": "torch.float32"},
        "versions": {"toy": 1},
        "higher_is_better": {"toy": {"acc": True}},
        "n-samples": {"toy": {"original": 10, "effective": 1}},
    }


def compact(config, value=0.75):
    spec = runner.load_suite(config)["benchmarks"][0]
    versions = {p: "1" for p in runner.PROTOCOL_PACKAGES}
    return runner.summarize_result(raw_result(value), spec, versions)


def test_defaults_and_per_benchmark_overrides(config):
    suite = runner.load_suite(config)
    spec = suite["benchmarks"][0]
    assert spec["limit"] == 2 and spec["max_length"] == 32
    assert spec["batch_size"] == 1 and spec["seed"] == 1234
    data = json.loads(json.dumps(yaml.safe_load(config.read_text())))
    data["benchmarks"][0].update(
        limit=3,
        generation={"max_gen_toks": 5, "do_sample": False},
    )
    config.write_text(json.dumps(data))  # JSON configurations use the same loader.
    updated = runner.load_suite(config)["benchmarks"][0]
    assert updated["limit"] == 3
    assert updated["generation"] == {"max_gen_toks": 5, "do_sample": False}


@pytest.mark.parametrize("field", sorted(runner.BENCHMARK_SETTINGS))
def test_missing_settings_have_no_script_fallback(config, field):
    data = yaml.safe_load(config.read_text())
    value = data["defaults"].pop(field)
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=f"toy: missing required settings: .*{field}"):
        runner.load_suite(config)
    # Providing the setting in the entry instead satisfies the requirement.
    data["benchmarks"][0][field] = value
    config.write_text(yaml.safe_dump(data))
    assert runner.load_suite(config)["benchmarks"][0][field] == value


def test_complete_entries_and_frozen_config_need_no_defaults(config):
    expected = runner.load_suite(config)
    data = yaml.safe_load(config.read_text())
    data["benchmarks"][0].update(data.pop("defaults"))
    config.write_text(yaml.safe_dump(data))
    assert runner.load_suite(config) == expected
    # The supervisor's frozen JSON contains only complete, resolved entries.
    config.write_text(json.dumps(expected))
    assert runner.load_suite(config) == expected


@pytest.mark.parametrize("field", ["max_gen_toks", "do_sample"])
def test_generation_settings_must_be_explicit(config, field):
    data = yaml.safe_load(config.read_text())
    generation = {"max_gen_toks": 5, "do_sample": False}
    del generation[field]
    data["benchmarks"][0]["generation"] = generation
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(
        ValueError, match=f"generation: missing required settings: .*{field}"
    ):
        runner.load_suite(config)


@pytest.mark.parametrize(
    "change",
    [
        {"limit": True},
        {"limit": 0},
        {"sampling": "first"},
        {"sampling": "uniform"},
        {"max_length": -1},
        {"num_fewshot": -1},
        {"tasks": ["mmlu_*"]},
        {"tasks": ["toy", "toy"]},
        {"max_lenght": 64},
        {"generation": {"max_gen_toks": 32, "do_sample": False}},
        {"generation": {"max_gen_toks": 5, "do_sample": True}},
        {"enabled": False},
        {"reason": "unused"},
        {"batch_size": True},
        {"seed": -1},
        {"timeout_seconds": 0},
        {"dtype": "invalid"},
        {"apply_chat_template": "false"},
        {"generation": []},
        {"generation": {"max_gen_toks": 5, "do_sample": False, "until": "stop"}},
        {"id": "../escape"},
        {"fewshot_as_multiturn": True},
    ],
)
def test_invalid_configs_fail_before_execution(config, change):
    data = yaml.safe_load(config.read_text())
    data["benchmarks"][0].update(change)
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        runner.load_suite(config)


def test_uniform_sampling_is_reproducible_and_independent(config):
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["limit"] = 50
    state = random.getstate()
    selected = runner.select_sample_indices("toy", 1000, spec)
    assert random.getstate() == state
    assert len(selected) == len(set(selected)) == 50
    assert selected == sorted(selected)
    assert all(0 <= index < 1000 for index in selected)
    assert selected != list(range(50))
    # Pin the specified uniform-without-replacement algorithm.
    expected_rng = random.Random(runner.digest([spec["seed"], "toy"]))
    assert selected == sorted(expected_rng.sample(range(1000), 50))
    assert selected != runner.select_sample_indices("other", 1000, spec)
    assert selected != runner.select_sample_indices(
        "toy", 1000, {**spec, "seed": spec["seed"] + 1}
    )
    # Other task calls and unrelated global random draws cannot perturb it.
    random.random()
    assert runner.select_sample_indices("toy", 1000, spec) == selected
    random.setstate(state)


def test_sampling_small_and_empty_datasets(config):
    spec = runner.load_suite(config)["benchmarks"][0]
    for size in (1, 2):
        assert runner.select_sample_indices("toy", size, spec) == list(range(size))
    with pytest.raises(ValueError, match="no evaluation documents"):
        runner.select_sample_indices("toy", 0, spec)


def test_matrix_and_native_imports_are_spyre_independent():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from evals.lm_eval import run_benchmarks as r; "
            "r.main(['--print-matrix']); "
            "assert 'torch' not in sys.modules; assert 'hf_adapters' not in sys.modules"
        ),
    ]
    process = subprocess.run(
        command, cwd=runner.ROOT, capture_output=True, text=True, check=True
    )
    matrix = json.loads(process.stdout)["include"]
    expected = yaml.safe_load(
        (runner.ROOT / "tests/model_lists/perf.yaml").read_text()
    )["nightly"]["models"]
    assert [row["model"] for row in matrix] == expected
    assert len({row["key"] for row in matrix}) == len(expected)
    assert runner.model_key("a/b") != runner.model_key("a--b")


def test_workflow_structure_and_shell_syntax():
    workflow = yaml.load(
        (runner.ROOT / ".github/workflows/lm-eval.yaml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"]["schedule"] == [{"cron": "0 3 * * *"}]
    assert "workflow_dispatch" in workflow["on"]
    jobs = workflow["jobs"]
    assert jobs["evaluate"]["needs"] == "plan"
    assert "spyre_pf_x1" in jobs["evaluate"]["runs-on"]
    assert jobs["evaluate"]["strategy"]["fail-fast"] == "false"
    assert (
        jobs["evaluate"]["strategy"]["matrix"]
        == "${{ fromJSON(needs.plan.outputs.matrix) }}"
    )
    for job in jobs.values():
        for step in job["steps"]:
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                assert step["if"] == "always()"


def test_worker_timeout_kills_and_reaps_process(config, tmp_path, monkeypatch):
    class TimedOutProcess:
        pid = 123
        args = ["worker"]
        reaped = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.reaped = True

    process = TimedOutProcess()
    signals = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    args = SimpleNamespace(config=str(config), backend="hf", device="cpu")
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["timeout_seconds"] = 0  # Expire immediately without a real long wait.
    result = runner.run_benchmark(args, "model", spec, tmp_path / "timed-out")
    assert result["status"] == "failed" and "Timed out" in result["error"]
    assert signals == [(process.pid, runner.signal.SIGKILL)] and process.reaped


@pytest.mark.parametrize("exit_code", [0, 1])
def test_worker_handoff_is_consumed(config, tmp_path, monkeypatch, exit_code):
    output = tmp_path / "worker"
    expected = compact(config)

    def launch(command, **kwargs):
        assert command[:4] == [
            sys.executable,
            "-u",
            "-m",
            "evals.lm_eval.run_benchmarks",
        ]
        assert kwargs["start_new_session"] is True
        runner.write_json(output / ".worker-result.json", expected)
        return SimpleNamespace(wait=lambda timeout: exit_code)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    args = SimpleNamespace(config=str(config), backend="hf", device="cpu")
    spec = runner.load_suite(config)["benchmarks"][0]
    result = runner.run_benchmark(args, "model", spec, output)
    assert result["metrics"] == expected["metrics"]
    assert result["exit_code"] == exit_code
    assert result["status"] == ("failed" if exit_code else "completed")
    assert not (output / ".worker-result.json").exists()
    assert sorted(p.name for p in output.iterdir()) == ["run.log"]


def test_worker_streams_before_exit_and_preserves_log(config, tmp_path, monkeypatch):
    output = tmp_path / "streamed-worker"
    acknowledged = tmp_path / "console-received-progress"

    class Console(io.StringIO):
        def write(self, text):
            size = super().write(text)
            if "progress\r" in self.getvalue():
                acknowledged.touch()
            return size

    console = Console()
    popen = subprocess.Popen
    expected = compact(config)
    script = f"""
import os, time
from pathlib import Path
os.write(1, b'progress\\r')
deadline = time.monotonic() + 5
while not Path({str(acknowledged)!r}).exists():
    if time.monotonic() > deadline:
        raise RuntimeError('Output was not streamed while running')
    time.sleep(0.01)
os.write(2, b'stderr\\n')
os.write(1, b'\\xe2')
time.sleep(0.2)
os.write(1, b'\\x82\\xac trailing output')
"""

    def launch(command, **kwargs):
        assert kwargs["stderr"] == subprocess.STDOUT
        runner.write_json(output / ".worker-result.json", expected)
        return popen([sys.executable, "-u", "-c", script], **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.sys, "stdout", console)
    args = SimpleNamespace(config=str(config), backend="hf", device="cpu")
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["timeout_seconds"] = 10
    result = runner.run_benchmark(args, "model", spec, output)
    assert result["status"] == "completed"
    text = "progress\rstderr\n€ trailing output"
    assert console.getvalue() == text
    assert (output / "run.log").read_bytes() == text.encode()


@pytest.mark.parametrize("silent", [True, False])
def test_streaming_worker_timeout_is_enforced(
    config, tmp_path, monkeypatch, capsys, silent
):
    popen = subprocess.Popen
    processes = []
    script = (
        "import time; "
        + ("" if silent else "print('partial', end='', flush=True); ")
        + "time.sleep(60)"
    )

    def launch(command, **kwargs):
        process = popen([sys.executable, "-u", "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    args = SimpleNamespace(config=str(config), backend="hf", device="cpu")
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["timeout_seconds"] = 1
    started = time.monotonic()
    result = runner.run_benchmark(args, "model", spec, tmp_path / "timeout")
    assert time.monotonic() - started < 5
    assert (
        result["status"] == "failed" and "Timed out after 1 seconds" in result["error"]
    )
    assert processes[0].poll() is not None
    assert capsys.readouterr().out == ("" if silent else "partial")


def test_result_console_includes_scores_and_reference_changes(config, capsys):
    current, baseline = compact(config, 0.6), compact(config, 0.8)
    runner.print_benchmark_result(runner.compare(current, baseline))
    text = capsys.readouterr().out
    assert "completed" in text
    assert "toy | acc,none: 0.6 (reference: 0.8; change: -25.00%)" in text
    runner.print_benchmark_result(runner.compare(current, compact(config, 0)))
    assert "N/A (zero reference)" in capsys.readouterr().out
    baseline["protocol"]["model_sha"] = "different"
    runner.print_benchmark_result(runner.compare(current, baseline))
    text = capsys.readouterr().out
    assert "incompatible_reference: model_sha" in text
    assert "toy | acc,none: 0.6" in text and "change:" not in text
    runner.print_benchmark_result({"status": "failed", "error": "bad load"})
    assert "failed: bad load" in capsys.readouterr().out


def test_streaming_interrupt_kills_worker_and_drains_log(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "run.log"
    log_path.write_bytes(b"last partial output")
    signals = []

    class InterruptedProcess:
        pid = 123
        reaped = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt
            self.reaped = True

    process = InterruptedProcess()
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with pytest.raises(KeyboardInterrupt):
        runner.wait_with_output(process, log_path, 10)
    assert signals == [(process.pid, runner.signal.SIGKILL)]
    assert process.reaped
    assert capsys.readouterr().out == "last partial output"


def test_percent_change_and_zero_reference(config):
    current, baseline = compact(config, 0.6), compact(config, 0.8)
    row = runner.compare(current, baseline)["metrics"][0]
    assert row["value"] == 0.6 and row["reference"] == 0.8
    assert row["percent_change"] == pytest.approx(-25)
    assert row["delta"] == pytest.approx(-0.2)
    row = runner.compare(current, compact(config, 0))["metrics"][0]
    assert row["comparison"] == "zero_reference"
    assert row["percent_change"] is None and row["delta"] == 0.6
    assert "comparison" not in current and "reference" not in current["metrics"][0]
    assert "comparison" not in baseline and "reference" not in baseline["metrics"][0]


@pytest.mark.parametrize(
    "field",
    [
        "settings",
        "sampling",
        "packages",
        "task_versions",
        "samples",
        "model_sha",
        "model_dtype",
    ],
)
def test_incompatible_protocols_never_get_percentages(config, field):
    current, baseline = compact(config), compact(config)
    baseline["protocol"][field] = "different"
    comparison = runner.compare(current, baseline)
    assert comparison["comparison"] == "incompatible_reference"
    assert comparison["differences"] == [field]
    assert "percent_change" not in comparison["metrics"][0]


@pytest.mark.parametrize("sampling", [None, "different_algorithm"])
def test_sampling_protocol_mismatch_is_not_compared(config, sampling):
    current, baseline = compact(config), compact(config)
    if sampling is None:
        del baseline["protocol"]["sampling"]
    else:
        baseline["protocol"]["sampling"] = sampling
    comparison = runner.compare(current, baseline)
    assert comparison["comparison"] == "incompatible_reference"
    assert comparison["differences"] == ["sampling"]


def test_missing_and_unknown_references(config):
    current = compact(config)
    assert runner.compare(current, None)["comparison"] == "missing_reference"
    assert runner.compare({"status": "failed"}, current) == {"status": "failed"}
    baseline = copy.deepcopy(current)
    baseline["metrics"] = []
    assert (
        runner.compare(current, baseline)["metrics"][0]["comparison"]
        == "missing_metric"
    )
    current["protocol"]["model_sha"] = None
    assert runner.compare(current, current)["comparison"] == "incompatible_reference"


def test_comparison_matches_task_and_metric_not_row_order(config):
    current, baseline = compact(config), compact(config)
    current["metrics"].extend(
        [
            {**current["metrics"][0], "task": "other", "value": 0.4},
            {**current["metrics"][0], "metric": "exact_match,none", "value": 0.1},
        ]
    )
    baseline["metrics"] = [
        {**current["metrics"][1], "value": 0.5},
        {**current["metrics"][0], "value": 0.6},
    ]
    rows = runner.compare(current, baseline)["metrics"]
    assert rows[0]["percent_change"] == pytest.approx(25)
    assert rows[1]["percent_change"] == pytest.approx(-20)
    assert rows[2]["comparison"] == "missing_metric"
    assert "percent_change" not in rows[2]


@pytest.mark.parametrize("value", [None, True, "0.8", float("nan"), float("inf")])
def test_invalid_reference_values_are_not_compared(config, value):
    current, baseline = compact(config), compact(config)
    baseline["metrics"][0]["value"] = value
    with pytest.raises(ValueError, match="Invalid reference metric"):
        runner.compare(current, baseline)


def test_sample_provenance_and_metrics_validation(config):
    spec = runner.load_suite(config)["benchmarks"][0]
    versions = {p: "1" for p in runner.PROTOCOL_PACKAGES}
    raw = raw_result()
    # Multiple extraction filters log the same document multiple times.
    raw["samples"]["toy"].append(copy.deepcopy(raw["samples"]["toy"][0]))
    result = runner.summarize_result(raw, spec, versions)
    assert result["protocol"]["samples"]["toy"]["count"] == 1
    assert result["protocol"]["samples"]["toy"]["doc_ids"] == [0]
    assert result["metrics"] == [
        {
            "task": "toy",
            "metric": "acc,none",
            "value": 0.75,
            "higher_is_better": True,
            "iterations": 1,
            "num_fewshot": 0,
        }
    ]
    raw["samples"]["toy"][0].pop("prompt_hash")
    with pytest.raises(ValueError, match="provenance"):
        runner.summarize_result(raw, spec, versions)
    with pytest.raises(ValueError, match="non-finite"):
        runner.summarize_result(raw_result(float("nan")), spec, versions)


def test_summary_uses_effective_fewshot_count(config):
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["num_fewshot"] = 5
    raw = raw_result()
    raw["n-shot"] = {"toy": 0}  # A task may explicitly force zero-shot.
    versions = {p: "1" for p in runner.PROTOCOL_PACKAGES}
    result = runner.summarize_result(raw, spec, versions)
    assert result["metrics"][0]["num_fewshot"] == 0


def test_supervisor_continues_and_exports_partial_native_reference(
    config, tmp_path, monkeypatch, capsys
):
    data = yaml.safe_load(config.read_text())
    data["benchmarks"] += [
        {"id": "broken", "tasks": ["broken"]},
        {"id": "after_failure", "tasks": ["toy"]},
    ]
    config.write_text(yaml.safe_dump(data))
    calls = []

    def worker(args, model, spec, output):
        assert Path(args.config).name == "suite.json"
        assert runner.load_suite(args.config)["name"] == "test-suite"
        if spec["id"] == "broken":
            assert "toy | acc,none: 0.75" in capsys.readouterr().out
        elif spec["id"] == "after_failure":
            assert "failed: compile failed" in capsys.readouterr().out
        calls.append(spec["id"])
        return (
            {"status": "failed", "error": "compile failed"}
            if spec["id"] == "broken"
            else compact(config)
        )

    monkeypatch.setattr(runner, "run_benchmark", worker)
    output, refs = tmp_path / "output", tmp_path / "refs"
    args = [
        "--config",
        str(config),
        "--backend",
        "hf",
        "--model",
        "org/model",
        "--output-dir",
        str(output),
        "--reference-dir",
        str(refs),
        "--write-references",
    ]
    assert runner.main(args) == 1
    assert calls == ["toy", "broken", "after_failure"]
    reference = json.loads(next(refs.glob("*.json")).read_text())
    assert set(reference["models"]["org/model"]) == {"toy", "after_failure"}
    assert "compile failed" in (output / "attachments" / "summary.md").read_text()
    report = json.loads((output / "attachments" / "report.json").read_text())
    payload = json.loads((output / "benchmarks.json").read_text())
    assert {entry["name"] for entry in payload["benchmarks"]} == {
        "toy",
        "after_failure",
    }
    assert all(entry["iterations"] == 1 for entry in payload["benchmarks"])
    assert {p.name for p in output.iterdir()} == {"benchmarks.json", "attachments"}
    # The reference is a subset of the report, not another serialization format.
    assert reference == {
        **report,
        "models": {
            "org/model": {
                k: v for k, v in report["models"]["org/model"].items() if k != "broken"
            }
        },
    }
    assert not list(output.rglob("summary.json"))
    with pytest.raises(SystemExit):
        runner.main(args)  # Existing references cannot be overwritten accidentally.
    assert calls == ["toy", "broken", "after_failure"]


def test_supervisor_reads_reference_and_writes_comparison(
    config, tmp_path, monkeypatch
):
    refs = tmp_path / "refs"
    refs.mkdir()
    reference = {
        "schema_version": runner.RESULT_SCHEMA_VERSION,
        "backend": "hf",
        "models": {"org/model": {"toy": compact(config, 0.8)}},
    }
    runner.write_json(refs / f"{runner.model_key('org/model')}.json", reference)
    monkeypatch.setattr(runner, "run_benchmark", lambda *a: compact(config, 0.6))
    output = tmp_path / "output"
    assert (
        runner.main(
            [
                "--config",
                str(config),
                "--model",
                "org/model",
                "--output-dir",
                str(output),
                "--reference-dir",
                str(refs),
            ]
        )
        == 0
    )
    assert "-25.00%" in (output / "attachments" / "summary.md").read_text()
    report = json.loads((output / "attachments" / "report.json").read_text())
    row = report["models"]["org/model"]["toy"]["metrics"][0]
    assert row["value"] == 0.6 and row["reference"] == 0.8
    assert row["percent_change"] == pytest.approx(-25)


def test_exported_reference_can_be_read_without_conversion(
    config, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "run_benchmark", lambda *a: compact(config, 0.8))
    common = [
        "--config",
        str(config),
        "--model",
        "org/model",
        "--reference-dir",
        str(tmp_path / "refs"),
    ]
    assert (
        runner.main(
            common
            + [
                "--backend",
                "hf",
                "--write-references",
                "--output-dir",
                str(tmp_path / "native"),
            ]
        )
        == 0
    )
    monkeypatch.setattr(runner, "run_benchmark", lambda *a: compact(config, 0.6))
    assert runner.main(common + ["--output-dir", str(tmp_path / "spyre")]) == 0
    report = json.loads(
        (tmp_path / "spyre" / "attachments" / "report.json").read_text()
    )
    assert report["schema_version"] == 1
    result = report["models"]["org/model"]["toy"]
    assert result["comparison"] == "compatible"
    assert result["metrics"][0]["percent_change"] == pytest.approx(-25)
    native = json.loads((tmp_path / "native" / "benchmarks.json").read_text())[
        "benchmarks"
    ][0]
    spyre = json.loads((tmp_path / "spyre" / "benchmarks.json").read_text())[
        "benchmarks"
    ][0]
    assert native["props"] == spyre["props"]
    assert native["backend"] == "cuda" and spyre["backend"] == "spyre"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 0},
        {"backend": "spyre"},
        {"models": {}},
    ],
)
def test_invalid_reference_metadata_is_rejected_before_execution(
    config, tmp_path, monkeypatch, capsys, change
):
    refs = tmp_path / "refs"
    refs.mkdir()
    runner.write_json(
        refs / f"{runner.model_key('org/model')}.json",
        {
            "schema_version": 1,
            "backend": "hf",
            "models": {"org/model": {"toy": compact(config)}},
            **change,
        },
    )

    def unexpected_worker(*args):
        pytest.fail("Invalid references must be rejected before starting workers")

    monkeypatch.setattr(runner, "run_benchmark", unexpected_worker)
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--config",
                str(config),
                "--model",
                "org/model",
                "--reference-dir",
                str(refs),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert "Invalid native reference" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("output_type", ["multiple_choice", "generate_until"])
@pytest.mark.parametrize("task_selection", ["toy", "toy_group"])
@pytest.mark.parametrize("model_type", ["gpt2", "mistral3"])
def test_native_worker_with_real_hflm_and_offline_task(
    config, tmp_path, monkeypatch, output_type, task_selection, model_type
):
    """Actual stock model inference and lm-eval aggregation, with no Hub access."""
    import datasets
    import lm_eval.tasks
    import torch
    from lm_eval.api.group import Group
    from lm_eval.api.task import ConfigurableTask
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    modules_before = set(sys.modules)
    torch.random.default_generator.manual_seed(42)
    model_dir = tmp_path / "model"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    if model_type == "mistral3":
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
        )

        model = Mistral3ForConditionalGeneration(
            Mistral3Config(
                text_config=Ministral3Config(
                    vocab_size=5,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=64,
                    bos_token_id=1,
                    eos_token_id=1,
                    pad_token_id=0,
                ),
                vision_config=PixtralVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    head_dim=8,
                    image_size=8,
                    patch_size=2,
                ),
                image_token_index=9,
                tie_word_embeddings=False,
            )
        )
    model.save_pretrained(model_dir)
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<eos>": 1, "hello": 2, "world": 3, "other": 4},
            unk_token="<pad>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
    ).save_pretrained(model_dir)

    class OfflineTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = datasets.DatasetDict(
                test=datasets.Dataset.from_dict(
                    {
                        "prompt": ["hello " * (i + 1) for i in range(12)],
                        "choices": [["world", "other"] for _ in range(12)],
                        "answer": [i % 2 for i in range(12)],
                        "target": ["world", "other"] * 6,
                    }
                )
            )

    task_config = dict(
        task="toy",
        dataset_path="unused",
        test_split="test",
        doc_to_text="prompt",
        output_type=output_type,
        metadata={"version": 1},
    )
    if output_type == "multiple_choice":
        task_config.update(
            doc_to_choice="choices",
            doc_to_target="answer",
            metric_list=[dict(metric="acc", aggregation="mean", higher_is_better=True)],
        )
    else:
        task_config.update(
            doc_to_target="target",
            generation_kwargs={"max_gen_toks": 3, "do_sample": False, "until": []},
            metric_list=[
                dict(metric="exact_match", aggregation="mean", higher_is_better=True)
            ],
        )
    load_calls, loaded_tasks, captured = [], {}, {}

    class OfflineTaskManager(lm_eval.tasks.TaskManager):
        def __init__(self):
            super().__init__(include_defaults=False)
            self._all_tasks = ["toy", "toy_group"]

        def _load_spec(self, name):
            load_calls.append(name)
            # Loading still occurs once, after the harness has seeded the RNGs.
            assert random.getstate() == random.Random(spec["seed"]).getstate()
            task = OfflineTask(config=copy.deepcopy(task_config))
            loaded_tasks["toy"] = task
            if name == "toy":
                return task
            group = Group("toy_group")
            group.add(task)
            sibling = OfflineTask(config={**task_config, "task": "toy_sibling"})
            loaded_tasks["toy_sibling"] = sibling
            group.add(sibling)
            return group

    evaluate = lm_eval.simple_evaluate

    def capture_evaluation(**kwargs):
        assert kwargs.get("limit") is None
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        native = kwargs["model"]
        assert native.backend == "causal"
        assert native.AUTO_MODEL_CLASS is (
            AutoModelForImageTextToText
            if model_type == "mistral3"
            else AutoModelForCausalLM
        )
        assert HFLM.AUTO_MODEL_CLASS is None  # No global monkey-patching.
        raw = evaluate(**kwargs)
        captured.update(raw)
        assert set(kwargs["samples"]) == set(loaded_tasks)
        return raw

    monkeypatch.setattr(lm_eval.tasks, "TaskManager", OfflineTaskManager)
    monkeypatch.setattr(lm_eval, "simple_evaluate", capture_evaluation)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    output = tmp_path / "worker"
    output.mkdir()
    args = SimpleNamespace(
        output_dir=str(output), backend="hf", device="cpu", model=[str(model_dir)]
    )
    spec = runner.load_suite(config)["benchmarks"][0]
    spec["tasks"] = [task_selection]
    if output_type == "generate_until":
        spec["generation"] = {"max_gen_toks": 3, "do_sample": False}
    assert runner.run_worker(args, spec) == 0
    assert load_calls == [task_selection]
    assert not any(
        name.startswith("hf_adapters") for name in set(sys.modules) - modules_before
    )
    result = json.loads((output / ".worker-result.json").read_text())
    assert result["protocol"]["sampling"] == "uniform_without_replacement_v1"
    for name, task in loaded_tasks.items():
        indices = runner.select_sample_indices(name, 12, spec)
        assert indices != [0, 1]
        manifest = result["protocol"]["samples"][name]
        assert manifest["count"] == 2 and manifest["doc_ids"] == indices
        assert result["sample_counts"][name] == {"original": 12, "effective": 2}
        # Verify non-prefix selection and original index/document alignment
        # through actual request building, scoring/generation, and aggregation.
        rows = captured["samples"][name]
        assert [row["doc_id"] for row in rows] == indices
        assert [row["doc"] for row in rows] == [task.eval_docs[i] for i in indices]
    assert result["protocol"]["model_dtype"] == "torch.float32"
    assert (output / "results.json").exists()
    assert "samples" not in json.loads((output / "results.json").read_text())
    samples = json.loads((output / "samples.json").read_text())
    for task, rows in samples.items():
        assert [row["doc_id"] for row in rows] == result["protocol"]["samples"][task][
            "doc_ids"
        ]
        assert all(
            "resps" in row and "doc" not in row and "arguments" not in row
            for row in rows
        )
    assert all(row["iterations"] == 2 for row in result["metrics"])
