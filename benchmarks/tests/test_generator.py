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

"""The generated model definitions follow perf.yaml and declare the ingest's identity, run command and cost."""

import generate_bench_definitions as gen
import yaml
from bench_model_cases import cases_for


def test_generated_files_are_current():
    want = gen.build()
    have = {
        str(p.relative_to(gen.OUT_DIR)): p.read_text()
        for p in (gen.OUT_DIR / gen.SUITE).glob("*.yaml")
    }
    have["selections.yaml"] = (gen.OUT_DIR / "selections.yaml").read_text()
    assert have == want


def test_every_perf_model_has_a_benchmark_with_six_cases():
    perf = yaml.safe_load(gen.PERF_YAML.read_text())
    models = gen._models(perf, "nightly") + gen._models(perf, "weekly")
    cases, missing = cases_for(models)
    assert not missing
    assert len(cases) == 6 * len(set(models))
    assert len(set(cases)) == len(cases)


def test_a_model_without_a_definition_is_reported_not_guessed():
    cases, missing = cases_for(["org/not-in-perf-yaml"])
    assert cases == [] and missing == ["org/not-in-perf-yaml"]


def test_every_benchmark_declares_identity_command_and_cost_but_no_cadence():
    for path in (gen.OUT_DIR / gen.SUITE).glob("*.yaml"):
        doc = yaml.safe_load(path.read_text())
        spec = doc["spec"]
        assert (
            spec["identity"]["mode"] == "ingest"
            and spec["identity"]["profile"] == "perf-kernel"
        )
        assert spec["identity"]["name"] == doc["metadata"]["name"]
        assert [d["key"] for d in spec["identity"]["discriminators"]] == [
            "record_type", "config_name", "input_shapes", "run_mode", "kernel_name", "is_total",
        ]  # fmt: skip
        assert spec["est_hours"] > 0
        assert (
            "--name" in spec["run"]["command"]
            and doc["metadata"]["name"] in spec["run"]["command"]
        )
        assert "schedule" not in spec and not (doc["metadata"].get("annotations") or {})


def test_the_smoke_selection_names_a_declared_case():
    smoke = yaml.safe_load_all((gen.OUT_DIR / "selections.yaml").read_text())
    smoke = next(d for d in smoke if d["metadata"]["name"] == "smoke")
    assert smoke["spec"]["cases"][0].startswith("gpt2/")
