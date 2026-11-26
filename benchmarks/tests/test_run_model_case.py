# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""run_model_case.py keeps one case's rows of a per-model or combined perf report."""

import json
import xml.etree.ElementTree as ET

import pytest
import run_model_case as rmc

SUITE = (
    '<?xml version="1.0"?><testsuites><testsuite name="s">{}</testsuite></testsuites>'
)


def tc(op, metric, **tags):
    tags = {"op": op, **tags}
    props = "".join(f'<property name="tag" value="{k}__{v}"/>' for k, v in tags.items())
    return f'<testcase name="perf_{op}_{metric}_ms" time="1.0"><properties>{props}</properties></testcase>'


def combined_report():
    rows = []
    for cfg in ("full_model", "one_block"):
        for mode in ("prefill_only", "decode_only"):
            rows.append(
                tc(
                    f"gpt2_{mode}-mode_{cfg}",
                    "wall_clock",
                    model="gpt2",
                    config=cfg,
                    mode=mode,
                )
            )
            rows.append(
                tc(
                    f"Qwen3-0.6B_{mode}-mode_{cfg}",
                    "wall_clock",
                    model="Qwen3-0.6B",
                    config=cfg,
                    mode=mode,
                )
            )
        rows.append(
            tc(f"granite_bs1_pl512_generate_{cfg}", "wall_clock", model="granite-4.1-8b", config=cfg, mode="generate")
        )  # fmt: skip
    return SUITE.format("".join(rows))


def run(tmp_path, name, config, mode, report=None):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "report.xml").write_text(report or combined_report())
    raw = tmp_path / "raw"
    code = rmc.main(
        [
            "--model", f"org/{name}", "--name", name, "--config", config, "--mode", mode,
            "--suite-dir", str(tmp_path / "suite"), "--raw-dir", str(raw),
        ]  # fmt: skip
    )
    return code, raw


def names(raw):
    out = []
    for f in sorted(raw.glob("*.xml")):
        out += [tc.get("name") for tc in ET.parse(f).getroot().iter("testcase")]
    return out


def test_a_case_keeps_only_its_own_model_config_and_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    code, raw = run(tmp_path, "gpt2", "one_block", "decode_only")
    assert code == 0
    assert names(raw) == ["perf_gpt2_decode_only-mode_one_block_wall_clock_ms"]


def test_the_name_match_is_case_insensitive_like_the_id_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    code, raw = run(tmp_path, "qwen3-0.6b", "full_model", "prefill_only")
    assert code == 0 and len(names(raw)) == 1


def test_granite_ops_fold_into_the_granite_name(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    code, raw = run(tmp_path, "granite", "one_block", "generate")
    assert code == 0 and "granite_bs1" in names(raw)[0]


def test_two_cases_never_share_a_testcase(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    seen = []
    for cfg in ("full_model", "one_block"):
        for mode in ("prefill_only", "decode_only"):
            for name in ("gpt2", "qwen3-0.6b"):
                _, raw = run(tmp_path, name, cfg, mode)
                seen += names(raw)
                for f in raw.glob("*.xml"):
                    f.unlink()
    assert len(seen) == len(set(seen)) == 8


def test_a_case_with_no_row_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    code, _ = run(tmp_path, "gpt2", "full_model", "generate")
    assert code == 3


def test_missing_reuse_reports_are_an_error_not_a_silent_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        rmc.main(
            ["--model", "org/gpt2", "--name", "gpt2", "--config", "full_model", "--mode", "generate",
             "--suite-dir", str(tmp_path / "s"), "--raw-dir", str(tmp_path / "r")]
        )  # fmt: skip


def test_the_suite_runs_once_per_model_through_the_configured_command(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BENCH_PERF_RESULTS", raising=False)
    report = tmp_path / "fake-report.xml"
    report.write_text(combined_report())
    monkeypatch.setenv("BENCH_SUITE_CMD", f"cp {report} {{suite_dir}}/report.xml")
    args = ["--model", "org/gpt2", "--name", "gpt2", "--config", "full_model", "--mode", "prefill_only",
            "--suite-dir", str(tmp_path / "suite"), "--raw-dir", str(tmp_path / "raw")]  # fmt: skip
    assert rmc.main(args) == 0
    monkeypatch.setenv(
        "BENCH_SUITE_CMD", "false"
    )  # a second case reuses the report, the suite is not run again
    assert rmc.main(args) == 0


def test_the_wall_clock_row_becomes_the_engine_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PERF_RESULTS", str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "report.xml").write_text(
        combined_report().replace('time="1.0"', 'time="2500.0"')
    )
    args = ["--model", "org/gpt2", "--name", "gpt2", "--config", "full_model", "--mode", "prefill_only",
            "--suite-dir", str(tmp_path / "s"), "--raw-dir", str(tmp_path / "raw"), "--repeat-dir", str(tmp_path / "rep")]  # fmt: skip
    assert rmc.main(args) == 0
    sample = json.loads((tmp_path / "rep/metrics.json").read_text())
    assert sample == {"samples": {"kernel.total_duration": [2.5]}}


def test_a_half_written_suite_directory_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_PERF_RESULTS", raising=False)
    half = tmp_path / "suite" / "org_gpt2"
    half.mkdir(parents=True)
    (half / "report.xml").write_text(combined_report())  # no completion marker
    with pytest.raises(SystemExit):
        rmc.main(
            ["--model", "org/gpt2", "--name", "gpt2", "--config", "full_model", "--mode", "generate",
             "--suite-dir", str(tmp_path / "suite"), "--raw-dir", str(tmp_path / "raw")]
        )  # fmt: skip
