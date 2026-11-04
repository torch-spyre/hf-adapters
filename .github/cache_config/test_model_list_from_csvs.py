# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline tests for building a --model-list-file from catalog CSVs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_list_from_csvs as convert
import warm_top_models_cache as warm

_HEADER: str = "rank,model_id,downloads,parameters,model_type,config_class\n"
_SETTINGS: dict[str, int] = {
    "top_k_generative": 2,
    "top_k_embedding": 5,
    "max_params": 60_000_000_000,
}


class ModelListFromCsvsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dir: Path = Path(temporary.name)
        self.gen: Path = self.dir / "gen.csv"
        self.emb: Path = self.dir / "emb.csv"
        self.gen.write_text(
            _HEADER
            + "1,org/gen-a,500,1000,llama,LlamaConfig\n"
            + "2,org/too-big,400,70000000000,llama,LlamaConfig\n"
            + "3,org/gen-c,300,,llama,LlamaConfig\n",
            encoding="utf-8",
        )
        self.emb.write_text(
            _HEADER
            + "1,org/emb-a,900,10,bert,BertConfig\n"
            + "2,org/curated,100,20,bert,BertConfig\n",
            encoding="utf-8",
        )

    def _build(self, include_curated: bool) -> list[dict[str, object]]:
        with (
            patch.object(convert, "load_curated_generative_models", return_value=[]),
            patch.object(
                convert, "load_curated_embedding_models", return_value=["org/curated"]
            ),
        ):
            return convert.build_from_csvs(
                self.gen, self.emb, 2, 5, 60_000_000_000, include_curated
            )

    def test_top_k_cuts_each_csv_then_merges_by_downloads(self) -> None:
        rows = self._build(include_curated=False)
        # org/gen-c is row 3, past --top-k-generative 2; org/too-big is over 60B.
        self.assertEqual(
            [r["model_id"] for r in rows], ["org/emb-a", "org/gen-a", "org/curated"]
        )
        self.assertEqual(
            [r["mode"] for r in rows], ["embedding", "generative", "embedding"]
        )

    def test_curated_come_first_and_are_not_duplicated(self) -> None:
        rows = self._build(include_curated=True)
        self.assertEqual(rows[0]["model_id"], "org/curated")
        self.assertTrue(rows[0]["curated"])
        self.assertEqual([r["model_id"] for r in rows].count("org/curated"), 1)

    def test_output_loads_with_the_same_flags_only(self) -> None:
        out: Path = self.dir / "list.json"
        warm.save_model_list(out, self._build(include_curated=False), _SETTINGS)
        loaded = warm.load_model_list(out, _SETTINGS)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[1]["parameters"], 1000)
        with self.assertRaises(ValueError):
            warm.load_model_list(out, {**_SETTINGS, "top_k_generative": 3})

    def test_missing_column_is_refused(self) -> None:
        bad: Path = self.dir / "bad.csv"
        bad.write_text("rank,model_id\n1,org/x\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing column"):
            convert.read_catalog_csv(bad, "generative", 10)


if __name__ == "__main__":
    unittest.main()
