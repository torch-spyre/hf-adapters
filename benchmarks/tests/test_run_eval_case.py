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

"""run_eval_case.py translates the eval script's own result; it must not score anything itself."""

import json

import check_identity_ids as chk
import pytest
import run_eval_case as rec

pytest.importorskip("spyre_bench")


def fake_run(correct, total):
    """Stand in for the eval script: write its output file and return its metadata block."""

    def run(task, model, limit, out):
        meta = {
            "backend": "spyre_adapter",
            "model": model,
            "num_samples": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
        }
        results = [{"index": i, "correct": i < correct} for i in range(total)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metadata": meta, "results": results}))
        return meta

    return run


def invoke(tmp_path, task, correct, total, limit=50, case="eval_case_stub"):
    args = ["--task", task, "--case", case, "--model", "org/model", "--limit", str(limit),
            "--repeat-dir", str(tmp_path / "rep"), "--raw-dir", str(tmp_path / "raw")]  # fmt: skip
    return rec.main(args, run=fake_run(correct, total))


def test_the_score_and_the_file_layout(tmp_path):
    assert invoke(tmp_path, "gsm8k", correct=25, total=50) == 0
    metrics = json.loads((tmp_path / "rep/metrics.json").read_text())
    assert metrics["samples"]["llm.eval.exact_match"] == [0.5]
    assert metrics["props"]["run.quality.dataset_digest"].startswith("sha256:")

    row = json.loads((tmp_path / "raw/eval_case_stub.json").read_text())["benchmarks"][
        0
    ]
    assert row["iterations"] == 50
    assert row["props"]["task"] == "gsm8k"
    assert row["props"]["num_fewshot"] == "2"
    assert row["props"]["num_questions"] == "50"

    # The script's own per-answer output is kept, and stays out of the recorded result.
    assert len(json.loads((tmp_path / "rep/answers.json").read_text())["results"]) == 50


def test_mmlu_is_zero_shot_in_the_recorded_identity(tmp_path):
    assert invoke(tmp_path, "mmlu", correct=50, total=50) == 0
    row = json.loads((tmp_path / "raw/eval_case_stub.json").read_text())["benchmarks"][
        0
    ]
    assert row["props"]["num_fewshot"] == "0"
    assert row["measurements"]["llm.eval.exact_match"] == [1.0]


def test_a_question_count_that_misses_the_declared_limit_is_an_error(tmp_path):
    """The identity hashes num_questions, so recording 49 under a benchmark that promises 50
    would silently start a new series. Fail instead."""
    assert invoke(tmp_path, "gsm8k", correct=20, total=49, limit=50) == 3
    assert not (tmp_path / "rep/metrics.json").exists()
    assert not (tmp_path / "raw/eval_case_stub.json").exists()


def test_the_declared_limit_reaches_the_eval_script():
    argv = rec.script_argv("gsm8k", "org/model", 12, "/tmp/out.json")
    assert "--num-samples" in argv and argv[argv.index("--num-samples") + 1] == "12"
    assert argv[1].endswith("evals/gsm8k/eval_gsm8k_spyre.py")

    argv = rec.script_argv("mmlu", "org/model", 7, "/tmp/out.json")
    assert argv[argv.index("--num-samples") + 1] == "7"
    assert "--backend" in argv and argv[argv.index("--backend") + 1] == "spyre"
    assert argv[1].endswith("evals/mmlu/mmlu_eval.py")


def test_it_delegates_to_a_script_that_exists():
    for task in rec.TASKS:
        assert rec.script_argv(task, "m", 1, "/tmp/o.json")[1]
        spec = rec.TASKS[task]
        assert (rec.ROOT / "evals" / spec["dir"] / spec["script"]).is_file()
        assert (rec.ROOT / "evals" / spec["dir"] / spec["samples"]).is_file()


def test_the_recorded_row_derives_the_engine_id_of_its_case():
    results, BenchmarkId = chk.load_ingest()
    rows, bad = chk.table(results, BenchmarkId, chk.declared(), chk.bare_models())
    assert not bad
    assert {r[1] for r in rows if r[2] == "quality"} == {
        "eval_gsm8k_qwen3-0.6b_l50",
        "eval_mmlu_qwen3-0.6b_l50",
    }


def test_vendored_question_files_match_the_asset_digests():
    import yaml

    assets = {}
    for doc in yaml.safe_load_all(
        (rec.ROOT / "benchmarks/assets-eval.yaml").read_text()
    ):
        if doc["spec"].get("integrity", {}).get("files"):
            assets[doc["metadata"]["name"]] = doc["spec"]
    for task, asset in (("gsm8k", "gsm8k-files"), ("mmlu", "mmlu-files")):
        want = assets[asset]["integrity"]["files"][0]["sha256"]
        assert rec.samples_digest(task) == "sha256:" + want


def test_a_failing_eval_script_is_a_message_not_a_traceback(tmp_path):
    def boom(task, model, limit, out):
        raise rec.EvalError("eval_gsm8k_spyre.py exited 1")

    args = ["--task", "gsm8k", "--case", "c", "--model", "m", "--limit", "50",
            "--repeat-dir", str(tmp_path / "r"), "--raw-dir", str(tmp_path / "w")]  # fmt: skip
    assert rec.main(args, run=boom) == 1
    assert not (tmp_path / "r/metrics.json").exists()
