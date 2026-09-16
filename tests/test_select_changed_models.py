# Copyright 2025 The Torch-Spyre Authors.
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

"""Unit tests for .github/scripts/select_changed_models.py's ``select()``.

Lives at the tests/ root, like test_adapter_coverage.py, and is run with
``pytest --noconftest`` for the same reason: the root conftest imports/patches
torch-adjacent machinery that this pure, data-only test doesn't need. The
script itself only imports ``tests.model_registry`` (which needs ``pytest``,
nothing torch-related), so this stays runnable on an interpreter with no
Spyre/torch stack.

``select_changed_models.py`` lives under ``.github/scripts/`` -- not an
importable package (the directory name starts with a dot) -- so it's loaded
via ``importlib`` from its file path rather than a normal import.
"""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "scripts"
    / "select_changed_models.py"
)
_spec = importlib.util.spec_from_file_location("select_changed_models", _SCRIPT_PATH)
select_changed_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_changed_models)

select = select_changed_models.select


class TestDocOnly:
    def test_single_doc_file_skips(self) -> None:
        models, skip_tests = select(["README.md"])
        assert models == []
        assert skip_tests is True

    def test_docs_dir_skips(self) -> None:
        models, skip_tests = select(["docs/fms_comparison.md"])
        assert skip_tests is True


class TestWeeklyOnly:
    def test_weekly_generation_file_skips(self) -> None:
        models, skip_tests = select(
            ["tests/spyre/weekly_generation/model_prefilter.py"]
        )
        assert skip_tests is True

    def test_weekly_and_doc_mix_skips(self) -> None:
        models, skip_tests = select(
            ["tests/spyre/weekly_generation/weekly_test.py", "README.md"]
        )
        assert skip_tests is True


class TestSingleAdapter:
    def test_single_adapter_file_restricts_to_its_models(self) -> None:
        models, skip_tests = select(["hf_adapters/hf_qwen3.py"])
        assert skip_tests is False
        assert models
        assert "Qwen/Qwen3-0.6B" in models

    def test_adapter_change_plus_docs_still_restricts(self) -> None:
        with_docs, skip_tests = select(["hf_adapters/hf_qwen3.py", "README.md"])
        without_docs, _ = select(["hf_adapters/hf_qwen3.py"])
        assert with_docs == without_docs
        assert skip_tests is False

    def test_two_unrelated_adapters_union_their_models(self) -> None:
        qwen3_only, _ = select(["hf_adapters/hf_qwen3.py"])
        llama_only, _ = select(["hf_adapters/hf_llama.py"])
        combined, skip_tests = select(
            ["hf_adapters/hf_qwen3.py", "hf_adapters/hf_llama.py"]
        )
        assert skip_tests is False
        assert set(combined) == set(qwen3_only) | set(llama_only)


class TestAdapterDependencyGraph:
    """A changed adapter file must also pull in every adapter that imports
    from it, transitively -- e.g. hf_granite_swa.py does
    ``from hf_adapters.hf_granite import _run_backbone_forward, _run_forward``,
    so a hf_granite.py-only change that skipped hf_granite_swa's models would
    silently under-test a real coupling. See hf_adapters/*.py's own import
    statements for the ground truth this locks in.
    """

    def test_granite_change_pulls_in_its_dependents(self) -> None:
        models, skip_tests = select(["hf_adapters/hf_granite.py"])
        assert skip_tests is False
        # hf_granite_swa.py, hf_granite_vision.py, hf_granitemoehybrid.py all
        # import from hf_granite.py.
        assert "ibm-granite/granite-3.3-8b-instruct" in models  # hf_granite
        assert "ibm-research/granite-4.1-20b" in models  # hf_granite_swa
        assert "ibm-granite/granite-vision-4.1-4b" in models  # hf_granite_vision
        assert "ibm-granite/granite-4.0-1b-base" in models  # hf_granitemoehybrid

    def test_bert_change_pulls_in_xlm_roberta(self) -> None:
        # hf_xlm_roberta.py imports _make_compiled_encoder_block from hf_bert.py.
        models, skip_tests = select(["hf_adapters/hf_bert.py"])
        assert skip_tests is False
        assert "BAAI/bge-base-en-v1.5" in models  # hf_bert
        assert "BAAI/bge-m3" in models  # hf_xlm_roberta

    def test_mistral_change_pulls_in_mistral3(self) -> None:
        # hf_mistral3.py does `from hf_adapters import hf_mistral`.
        models, skip_tests = select(["hf_adapters/hf_mistral.py"])
        assert skip_tests is False
        assert "ministral/Ministral-3B-Instruct" in models  # hf_mistral
        assert "mistralai/Mistral-Small-3.2-24B-Instruct-2506" in models  # hf_mistral3

    def test_gemma4_change_pulls_in_dspark_and_mm_variants(self) -> None:
        # hf_dspark_gemma4.py and hf_gemma4_mm.py both import from hf_gemma4.py.
        models, skip_tests = select(["hf_adapters/hf_gemma4.py"])
        assert skip_tests is False
        assert "google/gemma-4-12B-it" in models  # hf_gemma4 / hf_gemma4_mm
        assert "deepseek-ai/dspark_gemma4_12b_block7" in models  # hf_dspark_gemma4

    def test_pixtral_vision_change_pulls_in_mistral3_vision_mm(self) -> None:
        models, skip_tests = select(["hf_adapters/hf_pixtral_vision.py"])
        assert skip_tests is False
        assert (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503" in models
        )  # hf_mistral3_vision_mm (kind="vlm")

    def test_siglip_vision_change_pulls_in_granite_vision_mm(self) -> None:
        models, skip_tests = select(["hf_adapters/hf_siglip_vision.py"])
        assert skip_tests is False
        assert "ibm-granite/granite-vision-4.1-4b" in models  # hf_granite_vision_mm

    def test_leaf_adapter_change_has_no_extra_dependents(self) -> None:
        # Nothing in hf_adapters/ imports from hf_qwen3.py.
        models, _ = select(["hf_adapters/hf_qwen3.py"])
        assert models == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B"]


class TestFallbackToFullMatrix:
    def test_shared_non_adapter_file_forces_full_matrix(self) -> None:
        # hf_common.py is shared by every adapter -- must not be treated as
        # "just one adapter changed".
        models, skip_tests = select(["hf_adapters/hf_common.py"])
        assert models == []
        assert skip_tests is False

    def test_non_hf_prefixed_adapter_infra_forces_full_matrix(self) -> None:
        # auto_spyre_model.py doesn't match the hf_*.py adapter pattern.
        models, skip_tests = select(["hf_adapters/auto_spyre_model.py"])
        assert models == []
        assert skip_tests is False

    def test_registry_change_forces_full_matrix(self) -> None:
        models, skip_tests = select(["tests/model_registry.py"])
        assert models == []
        assert skip_tests is False

    def test_ci_workflow_change_forces_full_matrix(self) -> None:
        models, skip_tests = select([".github/workflows/_test_matrix.yaml"])
        assert models == []
        assert skip_tests is False

    def test_adapter_plus_shared_file_forces_full_matrix(self) -> None:
        models, skip_tests = select(
            ["hf_adapters/hf_bert.py", "hf_adapters/hf_common.py"]
        )
        assert models == []
        assert skip_tests is False

    def test_empty_diff_forces_full_matrix(self) -> None:
        models, skip_tests = select([])
        assert models == []
        assert skip_tests is False
