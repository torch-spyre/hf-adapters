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

"""collect_ingest_xml.py gathers the report XML of the cases that ran ok."""

import json

import collect_ingest_xml as collect


def test_only_ok_cases_are_collected(tmp_path):
    for name in ("a-1", "b-2"):
        raw = tmp_path / "cases" / name / "raw"
        raw.mkdir(parents=True)
        (raw / "report.xml").write_text("<x/>")
        (raw / "notes.txt").write_text("ignored")
    run = {
        "run_id": "r",
        "stored_run_ids": ["s"],
        "cases": [
            {"status": "ok", "dir": "cases/a-1"},
            {"status": "failed", "dir": "cases/b-2"},
        ],
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    assert collect.collect(tmp_path, tmp_path / "out") == ("r", "s", 1)
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["a-1-report.xml"]
