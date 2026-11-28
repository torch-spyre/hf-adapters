"""Check summary values, units, and warmup boundaries using recorded output."""

import subprocess
import sys
from pathlib import Path

import pytest
from summarize_prefill import parse_setup_timings, render_summary

_SCRIPT = Path(__file__).with_name("summarize_prefill.py")
_MODEL = "ibm-granite/granite-4.1-8b"

# Relevant lines from the successful three-model workflow run 44323251512.
_REPORT = """
Granite Config: granite-4.1-8b_bs1_pl512_prefill_only_full_model
Metric                                     tsp          sendnn tsp/sendnn
wall_clock_ms.mean_ms                  177.841             N/A        N/A
spyre_ms.mean_ms                       167.362             N/A        N/A
kernel_ms.mean_ms                      166.849             N/A        N/A
memory_transfer_ms.mean_ms               0.513             N/A        N/A
runtime_launch_ms.mean_ms                9.004             N/A        N/A
compile_ms.mean_ms                       0.000             N/A        N/A
Granite Config: granite-4.1-8b_bs1_pl512_prefill_only_one_block
wall_clock_ms.mean_ms                   18.713             N/A        N/A
spyre_ms.mean_ms                        10.923             N/A        N/A
kernel_ms.mean_ms                       10.732             N/A        N/A
memory_transfer_ms.mean_ms               0.191             N/A        N/A
runtime_launch_ms.mean_ms               10.136             N/A        N/A
compile_ms.mean_ms                       0.000             N/A        N/A
"""
_LOG = """
Block: full_blocks
  Load time: 13.53s
Running 1 warmup iterations...
[SPYRE-PERF-SUITE] E2E timing information: 70894.86502902582ms
  Warmup 1/1 complete
Warmup complete
Running 5 benchmark iterations...
[SPYRE-PERF-SUITE] E2E timing information: 178.608ms
[SPYRE-PERF-SUITE] E2E timing information: 178.126ms
Block: one_block
  Load time: 13.38s
Running 1 warmup iterations...
[SPYRE-PERF-SUITE] E2E timing information: 11738.838664023206ms
  Warmup 1/1 complete
Warmup complete
Running 5 benchmark iterations...
[SPYRE-PERF-SUITE] E2E timing information: 20.349ms
[SPYRE-PERF-SUITE] E2E timing information: 18.849ms
"""


def test_summary_matches_recorded_timings() -> None:
    summary = render_summary(_MODEL, _REPORT, _LOG)
    assert summary.startswith("### Granite 4.1 8B\n")
    assert "All values are in **ms**" in summary
    assert summary[summary.index("| Measure") :].splitlines() == [
        "| Measure | Full model | One block |",
        "|---|---:|---:|",
        "| End-to-end wall clock | 177.841 | 18.713 |",
        "| Spyre device | 167.362 | 10.923 |",
        "| Kernel execution | 166.849 | 10.732 |",
        "| Memory transfer | 0.513 | 0.191 |",
        "| Runtime launch | 9.004 | 10.136 |",
        "| Compilation during measurement | 0.000 | 0.000 |",
        "| Model loading — excluded | 13,530 | 13,380 |",
        "| Warmup — excluded | 70,894.865 | 11,738.839 |",
    ]


@pytest.mark.parametrize(
    ("model", "name"),
    [
        ("mistralai/Ministral-3-14B-Instruct-2512", "Ministral 3 14B Instruct"),
        ("google/gemma-4-26B-A4B-it", "Gemma 4 26B-A4B"),
        ("org/new-model", "org/new-model"),
    ],
)
def test_other_model_report_names_and_titles(model: str, name: str) -> None:
    prefix = model.rsplit("/", 1)[-1]
    report = (
        _REPORT.replace(
            "granite-4.1-8b_bs1_pl512_prefill_only_", f"{prefix}_prefill-mode_"
        )
        .replace("full_model\n", "full_model_torch-spyre\n")
        .replace("one_block\n", "one_block_torch-spyre\n")
    )
    summary = render_summary(model, report, _LOG)
    assert summary.startswith(f"### {name}\n")
    assert "| End-to-end wall clock | 177.841 | 18.713 |" in summary


def test_warmup_total_excludes_benchmark_iterations() -> None:
    log = """
Block: full_blocks
Load time: 2.5s
Running 2 warmup iterations...
[SPYRE-PERF-SUITE] E2E timing information: 1000ms
[SPYRE-PERF-SUITE] E2E timing information: 200ms
Warmup complete
Running 5 benchmark iterations...
[SPYRE-PERF-SUITE] E2E timing information: 123ms
"""
    assert parse_setup_timings(log)["full_model"] == {
        "load_ms": 2500,
        "warmup_ms": 1200,
    }


def test_missing_metric_does_not_turn_into_zero() -> None:
    report = _REPORT.replace("compile_ms.mean_ms", "different_metric")
    with pytest.raises(ValueError, match="Missing timings.*compile_ms.mean_ms"):
        render_summary(_MODEL, report, _LOG)


def test_missing_setup_timing_does_not_turn_into_zero() -> None:
    with pytest.raises(ValueError, match="Missing timings.*load_ms"):
        render_summary(_MODEL, _REPORT, _LOG.replace("Load time:", "Other:"))


def test_incomplete_warmup_fails() -> None:
    log = _LOG.replace("Running 1 warmup", "Running 2 warmup")
    with pytest.raises(ValueError, match="Incomplete warmup"):
        render_summary(_MODEL, _REPORT, log)


@pytest.mark.parametrize("invalid", ["nan", "inf", "-1", "N/A"])
def test_invalid_report_values_fail(invalid: str) -> None:
    with pytest.raises(ValueError):
        render_summary(_MODEL, _REPORT.replace("177.841", invalid), _LOG)


def test_multiple_reports_are_not_silently_combined() -> None:
    with pytest.raises(ValueError, match="duplicate report value"):
        render_summary(_MODEL, _REPORT + _REPORT, _LOG)


def test_ansi_colors_are_ignored() -> None:
    report = _REPORT.replace("177.841", "\x1b[32m177.841\x1b[0m")
    assert render_summary(_MODEL, report, _LOG) == render_summary(_MODEL, _REPORT, _LOG)


def test_cli_prints_and_appends_the_same_summary(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(_REPORT, encoding="utf-8")
    log = tmp_path / "benchmark.log"
    log.write_text(_LOG, encoding="utf-8")
    summary_file = tmp_path / "summary.md"
    summary_file.write_text("Existing summary\n\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--model",
            _MODEL,
            "--report",
            str(report),
            "--log",
            str(log),
            "--summary-file",
            str(summary_file),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert summary_file.read_text(encoding="utf-8") == (
        "Existing summary\n\n" + result.stdout
    )
    assert "| Warmup — excluded | 70,894.865 | 11,738.839 |" in result.stdout
