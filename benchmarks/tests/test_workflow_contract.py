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

"""bench-on-demand.yml declares every input the Jenkins orchestrator dispatches it with.

GitHub rejects a whole `workflow_dispatch` with HTTP 422 when the payload names an input the
workflow does not declare, and the orchestrator sends a fixed set -- so dropping or renaming one of
these makes the nightly and weekly bench legs fail to start, with nothing in this repository to show
why. The names below are the *declared* side (the values) of the bench cell's `config.gha.inputs`
map in spyre-frameworks `pipelines/components/hf-adapters/config.yaml`; keep them in step with it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/bench-on-demand.yml"

ORCHESTRATOR_INPUTS = {
    "ref",
    "repository",
    "runner_label",
    "prebaked_image",
    "run_id",
    "test_type",
    "purpose",
    "selection",
    "clickhouse_env",
    "triggering_repo",
    "triggering_sha",
}

# Every Selection the `selection` input is allowed to name must exist in the catalog.
SELECTIONS = {"smoke", "hf-models-nightly", "hf-models-weekly", "model-correctness"}


def workflow():
    document = yaml.safe_load(WORKFLOW.read_text())
    # `on` is the YAML 1.1 boolean true, so safe_load gives it either spelling.
    triggers = document.get("on") or document.get(True)
    return document, (triggers["workflow_dispatch"] or {}).get("inputs") or {}


def test_every_orchestrator_input_is_declared():
    _, inputs = workflow()
    missing = sorted(ORCHESTRATOR_INPUTS - set(inputs))
    assert not missing, (
        f"bench-on-demand.yml does not declare {missing}; the orchestrator's dispatch would 422 "
        "and the nightly bench leg would never start"
    )


def test_the_dispatch_inputs_stay_within_the_github_limit():
    _, inputs = workflow()
    assert (
        len(inputs) <= 25
    ), f"workflow_dispatch allows 25 inputs, this declares {len(inputs)}"


def test_the_selection_input_names_a_declared_selection():
    _, inputs = workflow()
    assert inputs["selection"]["default"] in SELECTIONS


def test_every_allowed_selection_exists_in_the_catalog():
    declared = set()
    for path in (ROOT / "benchmarks").rglob("*.yaml"):
        for doc in yaml.safe_load_all(path.read_text()):
            if isinstance(doc, dict) and doc.get("kind") == "Selection":
                declared.add(doc["metadata"]["name"])
    assert (
        SELECTIONS <= declared
    ), f"missing Selections: {sorted(SELECTIONS - declared)}"


def test_the_build_action_is_given_the_clone_directory():
    """build-hf-adapters wipes and re-clones $CLONED_HF_ADAPTERS_DIR, and the bench steps cd to it.

    No input carries it, and the runner's shell init runs those steps under `set -u`, so a workflow
    that calls the action without setting it fails on the action's first command.
    """
    document, _ = workflow()
    env = document.get("env") or {}
    assert env.get("CLONED_HF_ADAPTERS_DIR") == "${{ github.workspace }}"


def test_rows_are_routed_by_clickhouse_env_rather_than_pinned_to_dev():
    """A leg that ignores clickhouse_env files every row in dev, whatever the folder asked for."""
    text = WORKFLOW.read_text()
    assert "BENCH_CLICKHOUSE_ENV" in text
    # The schema-v2 inputs of the shared ingest action, not the retired v1 ones.
    assert "clickhouse-v2-host" in text and "clickhouse-db-v2" in text
    assert "clickhouse-pass:" not in text and "clickhouse-user:" not in text
