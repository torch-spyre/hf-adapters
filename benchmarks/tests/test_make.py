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

"""The make wrappers: BENCH_MODE, and a side-by-side run over a synthetic perf report with the real engine."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CENTRAL = {
    "project.yaml": """apiVersion: bench/v1alpha1
kind: Project
metadata: {name: central, owner: test}
spec:
  packs: [{name: spyre, version: "0.2"}]
  visibility: internal
""",
    "validity-policy.yaml": """apiVersion: bench/v1alpha1
kind: ValidityPolicy
metadata: {name: ci-default, owner: test}
spec:
  purposes:
    pr:
      floor: shared_env
      requires: {qualification: any, clean: any, min_repeats: 1, max_cv_pct: 10, discard_warmup: 1, placement: optional}
      baseline: false
      reference: false
""",
    "node-classes.yaml": """apiVersion: bench/v1alpha1
kind: NodeClass
metadata: {name: ci-x86_64}
spec:
  arch: x86_64
  match: {labels: {example.com/pool: test}}
  host_profile: {governor: performance, smt: "on"}
  extensions: {spyre: {device_function: pf}}
""",
    "environments.yaml": """apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: ci-x86_64-1card}
spec:
  devices:
    requests: [{name: cards, device_class: spyre/pf, count: 1}]
  host: {node_class: ci-x86_64, arch: x86_64}
  labels: {device.count: 1, node.class: ci-x86_64}
  policy: {validity_floor: shared_env}
""",
}


def make(*args, env=None):
    full = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        **(env or {}),
    }
    return subprocess.run(
        ["make", "-C", str(ROOT), "PYTHON=" + sys.executable, *args],
        capture_output=True,
        text=True,
        env=full,
    )


def test_mode_off_does_nothing(tmp_path):
    out = make(
        "bench-perf", "BENCH_MODE=off", f"RESULTS_DIR={tmp_path}/r", "MODEL_PATH=gpt2"
    )
    assert out.returncode == 0 and not (tmp_path / "r-bench").exists()


def test_an_unknown_mode_is_refused():
    assert make("bench-perf", "BENCH_MODE=both", "MODEL_PATH=gpt2").returncode == 2


def test_the_tests_target_calls_bench_perf_only_when_the_mode_is_not_off():
    text = (ROOT / "Makefile").read_text()
    assert '[ "$(BENCH_MODE)" != off ]; then $(MAKE) bench-perf' in text
    assert '[ "$(BENCH_MODE)" != replace ]; then $(MAKE) perf-tests' in text


def synthetic_report(model="gpt2"):
    def tc(op, tags, extra=""):
        props = "".join(
            f'<property name="tag" value="{k}__{v}"/>' for k, v in tags.items()
        )
        return f'<testcase {extra}name="{op}" time="0.5"><properties>{props}</properties></testcase>'

    bench, kern = [], []
    for cfg in ("full_model", "one_block"):
        for mode in ("prefill_only", "decode_only", "generate"):
            tags = {"model": model, "config": cfg, "mode": mode}
            op = f"{model}_{mode}-mode_{cfg}"
            bench.append(tc(f"perf_{op}_wall_clock_ms", {"op": op, **tags}))
            for k in ("aten::copy_", "Total"):
                kern.append(
                    tc(
                        "k",
                        {
                            "op": f"{model}_{mode}-mode_{cfg}",
                            "kernel": k,
                            "metric": "spyre_kernel_ms",
                            **tags,
                        },
                        'classname="kernel_benchmark" ',
                    )
                )
    suite = '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">{}</testsuite></testsuites>'
    return suite.format("".join(bench)), suite.format("".join(kern))


@pytest.fixture
def engine():
    pytest.importorskip("spyre_bench")
    pytest.importorskip("spyre_clickhouse_ingest")
    pytest.importorskip("lxml")


def test_side_by_side_reads_the_legacy_reports_and_writes_beside_them(tmp_path, engine):
    results = tmp_path / "perf-results"
    results.mkdir()
    bench, kern = synthetic_report()
    (results / "report.xml").write_text(bench)
    (results / "spyre_kernel_report.xml").write_text(kern)
    legacy = {p.name: p.read_text() for p in results.iterdir()}
    central = tmp_path / "central"
    central.mkdir()
    for name, text in CENTRAL.items():
        (central / name).write_text(text)

    out = make(
        "bench-perf", "BENCH_MODE=side-by-side", "MODEL_PATH=gpt2", f"RESULTS_DIR={results}", f"BENCH_CENTRAL={central}",
        "BENCH_ENVIRONMENT=ci-x86_64-1card", "BENCH_PURPOSE=pr", "BENCH_ARCH=x86_64", "BENCH_REQUIRED=1",
    )  # fmt: skip
    assert out.returncode == 0, out.stdout + out.stderr
    assert {
        p.name: p.read_text() for p in results.iterdir()
    } == legacy  # the legacy results are untouched
    run = json.loads((tmp_path / "perf-results-bench/run.json").read_text())
    assert [c["status"] for c in run["cases"]] == ["ok"] * 6

    collected = tmp_path / "collected"
    done = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/tools/collect_ingest_xml.py"), str(tmp_path / "perf-results-bench"), str(collected)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert (
        done.returncode == 0 and "count=12" in done.stdout
    )  # one report and one kernel file per case
    from lxml import etree
    from spyre_clickhouse_ingest import results as ingest

    names = []
    for xml in sorted(collected.glob("*.xml")):
        root = etree.parse(str(xml)).getroot()
        parse = (
            ingest.parse_kernel_xml
            if ingest.is_kernel_benchmark_xml(root)
            else ingest.parse_benchmark_xml
        )
        _, recs = parse(xml, "t", "", "x86_64")
        names += [
            (r["operation_name"], r["config_name"], r["run_mode"], r.get("kernel_name"))
            for r in recs
        ]
    assert (
        len(names) == len(set(names)) == 6 * 3
    )  # a report row, a kernel and a total per case, none twice


def test_replace_mode_runs_the_suite_through_the_engine(tmp_path, engine):
    central = tmp_path / "central"
    central.mkdir()
    for name, text in CENTRAL.items():
        (central / name).write_text(text)
    bench, kern = synthetic_report()
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "report.xml").write_text(bench)
    (fake / "spyre_kernel_report.xml").write_text(kern)
    out = make(
        "bench-perf", "BENCH_MODE=replace", "MODEL_PATH=gpt2", f"RESULTS_DIR={tmp_path}/r", f"BENCH_CENTRAL={central}",
        "BENCH_ENVIRONMENT=ci-x86_64-1card", "BENCH_PURPOSE=pr", "BENCH_ARCH=x86_64", "BENCH_REQUIRED=1",
        f"BENCH_SUITE_CMD=cp {fake}/report.xml {fake}/spyre_kernel_report.xml {{suite_dir}}", "BENCH_PASS_ENV=BENCH_SUITE_CMD",
    )  # fmt: skip
    assert out.returncode == 0, out.stdout + out.stderr
    run = json.loads((tmp_path / "r-bench/run.json").read_text())
    assert [c["status"] for c in run["cases"]] == ["ok"] * 6
    assert shutil.which("make")


def run_tests_target(tmp_path, mode, suite_cmd="false"):
    return make(
        "-s", "tests", "TEST_TYPE=perf", "MODEL_PATH=gpt2", f"RESULTS_DIR={tmp_path}/r", f"BENCH_MODE={mode}",
        "PERF_SUITE=false", f"BENCH_SUITE_CMD={suite_cmd}", "BENCH_REQUIRED=1",
    )  # fmt: skip


def test_mode_off_runs_the_legacy_path_only_and_keeps_its_failure(tmp_path):
    out = run_tests_target(tmp_path, "off")
    assert (
        out.returncode != 0
        and "spyre-perf-suite is not installed" in out.stderr + out.stdout
    )
    assert (
        "bench-perf" not in out.stdout + out.stderr
        and not (tmp_path / "r-bench").exists()
    )


def test_side_by_side_never_hides_the_failure_of_the_legacy_path(tmp_path):
    out = run_tests_target(tmp_path, "side-by-side")
    assert out.returncode != 0
    assert "spyre-perf-suite is not installed" in out.stdout + out.stderr


def test_replace_fails_when_the_engine_path_fails(tmp_path):
    out = run_tests_target(tmp_path, "replace")
    assert out.returncode != 0 and "bench-perf failed" in out.stderr
    assert (
        "spyre-perf-suite is not installed" not in out.stdout + out.stderr
    )  # the legacy path did not run


def test_bench_perf_needs_a_results_dir():
    assert (
        make("bench-perf", "BENCH_MODE=side-by-side", "MODEL_PATH=gpt2").returncode == 2
    )


def test_a_missing_engine_fails_loudly_with_the_install_command():
    out = make("bench-validate", "BENCH=no-such-spyre-bench")
    assert out.returncode == 2
    assert "Install spyre-bench" in out.stderr and "uv pip install" in out.stderr


def test_a_missing_engine_is_a_notice_only_when_optional():
    out = make("bench-validate", "BENCH=no-such-spyre-bench", "BENCH_OPTIONAL=1")
    assert out.returncode == 0 and "skipped (BENCH_OPTIONAL=1)" in out.stderr


def test_a_bare_validate_and_plan_place_the_cases(tmp_path, engine):
    out = make(
        "bench-validate",
        "bench-plan",
        f"BENCH_RESULTS_DIR={tmp_path}/b",
        "BENCH_SELECTION=smoke",
        "BENCH_ARCH=x86_64",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert (
        "no_environment" not in out.stdout and (tmp_path / "b/plan/plan.json").is_file()
    )
