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

"""run_eval_case.py scores evals/ samples with an injected generator and records what the engine and ingest read."""

import json

import check_identity_ids as chk
import pytest
import run_eval_case as rec

pytest.importorskip("spyre_bench")


def stub(correct_every):
    """A generator that answers right on every Nth question."""
    state = {"i": 0}

    def generate(prompt):
        i = state["i"]
        state["i"] += 1
        return f"#### {state['truth'][i]}" if i % correct_every == 0 else "#### -1"

    return generate, state


def run(tmp_path, task, truth, every):
    gen, state = stub(every)
    state["truth"] = truth
    args = ["--task", task, "--case", f"eval_{task}_stub", "--model", "org/model",
            "--repeat-dir", str(tmp_path / "rep"), "--raw-dir", str(tmp_path / "raw")]  # fmt: skip
    assert rec.main(args, generate=gen) == 0
    return json.loads((tmp_path / "rep/metrics.json").read_text()), (
        json.loads((tmp_path / "raw/eval_slot-12_stub.json").read_text())
        if task == "slot-12"
        else None
    )


def test_numeric_score_and_file_layout(tmp_path):
    mod, samples, _ = rec.load_task("slot-12")
    truth = [mod.extract_ground_truth(s["answer"]) for s in samples]
    metrics, bench = run(tmp_path, "slot-12", truth, 2)
    assert metrics["samples"]["llm.eval.exact_match"] == [0.5]
    row = bench["benchmarks"][0]
    assert (
        row["iterations"] == 50
        and row["props"]["task"] == "slot-12"
        and row["props"]["num_fewshot"] == "2"
    )
    assert row["run_props"]["run.quality.dataset_digest"].startswith("sha256:")
    assert len(json.loads((tmp_path / "rep/answers.json").read_text())) == 50


def test_choice_extracts_a_letter(tmp_path):
    mod, samples, _ = rec.load_task("slot-14")
    answers = iter(s["answer"] for s in samples)
    args = [
        "--task",
        "slot-14",
        "--case",
        "c",
        "--model",
        "m",
        "--repeat-dir",
        str(tmp_path / "r"),
        "--raw-dir",
        str(tmp_path / "w"),
    ]
    assert rec.main(args, generate=lambda prompt: f" {next(answers)}.") == 0
    assert json.loads((tmp_path / "r/metrics.json").read_text())["samples"][
        "llm.eval.exact_match"
    ] == [1.0]


def test_the_recorded_row_derives_the_engine_id_of_its_case():
    results, BenchmarkId = chk.load_ingest()
    rows, bad = chk.table(results, BenchmarkId, chk.declared(), chk.bare_models())
    assert not bad
    assert {r[1] for r in rows if r[2] == "quality"} == {
        "eval_slot-12_qwen3-0.6b_l50",
        "eval_slot-14_qwen3-0.6b_l50",
    }


def test_vendored_question_files_match_the_asset_digests():
    import yaml

    assets = {}
    for doc in yaml.safe_load_all(
        (rec.ROOT / "benchmarks/assets-eval.yaml").read_text()
    ):
        if doc["spec"].get("integrity", {}).get("files"):
            assets[doc["metadata"]["name"]] = doc["spec"]
    for task, asset in (("slot-12", "slot-12-files"), ("slot-14", "slot-14-files")):
        _, _, digest = rec.load_task(task)
        assert "sha256:" + assets[asset]["integrity"]["files"][0]["sha256"] == digest
