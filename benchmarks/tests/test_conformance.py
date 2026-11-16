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

"""Identity conformance: definitions derive the ids the ingest writes, and a drift is caught."""

import shutil
import subprocess
import sys

import check_identity_ids as chk
import pytest

pytest.importorskip("spyre_bench")
pytest.importorskip("spyre_clickhouse_ingest")
pytest.importorskip("lxml")


def test_the_catalog_matches_the_golden_table():
    code, msgs = chk.check()
    assert code == 0, msgs
    assert "164 cases" in msgs[0]


def test_every_model_case_derives_the_id_of_the_ingest_report_row():
    results, BenchmarkId = chk.load_ingest()
    cases = chk.declared()
    rows, bad = chk.table(results, BenchmarkId, cases, chk.bare_models())
    assert not bad
    assert sum(r[2] == "report" for r in rows) == 162


def edited_copy(tmp_path, old, new):
    cat = tmp_path / "benchmarks"
    shutil.copytree(
        chk.CATALOG, cat, ignore=shutil.ignore_patterns("tests", "tools", "__pycache__")
    )
    for path in cat.rglob("gpt2.yaml"):
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))
    return cat


def test_a_changed_discriminator_fails_the_check(tmp_path):
    cat = edited_copy(tmp_path, "value: model}", "value: op}")
    code, msgs = chk.check(catalog=cat)
    assert code == 1 and any("gpt2" in m for m in msgs)


def test_a_changed_name_fails_the_check(tmp_path):
    cat = edited_copy(tmp_path, "    name: gpt2\n", "    name: gpt-two\n")
    code, msgs = chk.check(catalog=cat)
    assert code == 1


def test_a_schedule_annotation_is_a_lint_error(tmp_path):
    cat = edited_copy(
        tmp_path,
        "metadata: {name: gpt2}",
        "metadata: {name: gpt2, annotations: {schedule.preferred: nightly}}",
    )
    assert any("cadence" in p for p in chk.lint(cat))


def test_the_check_runs_as_a_script():
    out = subprocess.run(
        [sys.executable, str(chk.ROOT / "benchmarks/tools/check_identity_ids.py")],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


def test_a_changed_question_file_is_caught_by_its_digest(tmp_path, monkeypatch):
    sample = tmp_path / "q_samples.json"
    sample.write_text("[]")
    monkeypatch.setattr(chk, "ROOT", tmp_path)
    cat = tmp_path / "benchmarks"
    cat.mkdir()
    (cat / "assets-eval.yaml").write_text(
        "kind: Asset\nmetadata: {name: a}\nspec:\n  source: {path: .}\n  integrity: {files: [{path: q_samples.json, sha256: "
        + "0" * 64
        + "}]}\n"
    )
    assert any("differs" in p for p in chk.asset_problems(cat))
