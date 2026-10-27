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

"""Shared multimodal accuracy tests; one reference/device load per checkpoint."""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import transformers
from _pytest.mark.structures import ParameterSet

from tests._multimodal_embedding_helpers import (
    TEXT_PROMPTS,
    CosineAccuracyError,
    PreparedEmbeddingModel,
    RetrievalAccuracyError,
    _assert_embeddings,
    _assert_hidden_states,
    _assert_logits,
    _assert_retrieval,
    _cosine_scores,
    bounded_cpu_threads,
    embedding_cases,
    prepare_model_cases,
)
from tests.model_registry import MULTIMODAL_EMBED_PATHS

pytestmark = pytest.mark.model_harness("multimodal_embedding")


class _ModelRunCache:
    """Hold at most one checkpoint, and remember a failed load.

    Tests are grouped by ``model_path`` at collection (see
    ``tests/spyre/conftest.py``), so each checkpoint loads once and is
    released before the next one loads. A failed load is replayed as a
    failure for that checkpoint's remaining tests instead of being retried.
    """

    def __init__(self) -> None:
        self.model_path: str | None = None
        self.run: PreparedEmbeddingModel | None = None
        self.error: str | None = None

    def get(
        self, model_path: str, trust_remote_code: bool | None
    ) -> PreparedEmbeddingModel:
        if model_path != self.model_path:
            self.release()
            self.model_path = model_path
            try:
                self.run = self._load(model_path, trust_remote_code)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise
        if self.error is not None:
            pytest.fail(
                f"Model load failed earlier for {model_path}: {self.error}",
                pytrace=False,
            )
        assert self.run is not None
        return self.run

    def release(self) -> None:
        self.model_path, self.run, self.error = None, None, None
        gc.collect()

    @staticmethod
    def _load(
        model_path: str, trust_remote_code: bool | None
    ) -> PreparedEmbeddingModel:
        print(f"Transformers: {transformers.__version__} ({transformers.__file__})")
        print("Embedding API: Hugging Face dual encoder; CPU reference: float32")
        with bounded_cpu_threads():
            return prepare_model_cases(
                model_path,
                embedding_cases(),
                trust_remote_code=trust_remote_code,
            )


_MODEL_RUNS: _ModelRunCache = _ModelRunCache()


@pytest.fixture(scope="module", autouse=True)
def _release_model_runs() -> Iterator[None]:
    # Free the last CPU reference and Spyre model when the module finishes.
    yield
    _MODEL_RUNS.release()


def _model_run(
    model_path: str, trust_remote_code: bool | None
) -> PreparedEmbeddingModel:
    """Load the CPU reference and Spyre model once per checkpoint; reuse across tests."""
    return _MODEL_RUNS.get(model_path, trust_remote_code)


# Keep temporary accuracy exceptions beside the checks they affect. These
# parameters mark only this checkpoint; new/off-registry models remain blocking.
_SIGLIP_PATH = "google/siglip2-giant-opt-patch16-384"
_VISION_REASON = (
    "SigLIP fp16 vision drift vs fp32 CPU: minimum cat/dog patch cosine "
    "0.104/0.558 after 40 blocks; full-fp32 Spyre compilation is unsupported. "
    "See ARCHITECTURE.md, Multimodal Embedding."
)
_UNIFORM_REASON = (
    "SigLIP uniform-color vision outputs can exceed accuracy tolerances on Spyre"
)


def _accuracy_params(reason: str, raises: type[AssertionError]) -> list[ParameterSet]:
    return [
        pytest.param(
            path,
            marks=(
                pytest.mark.xfail(reason=reason, strict=False, raises=raises)
                if path == _SIGLIP_PATH
                else ()
            ),
        )
        for path in MULTIMODAL_EMBED_PATHS
    ]


@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_text_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    model_run = _model_run(model_path, trust_remote_code)
    actual, reference = model_run.compare("text")
    assert actual.embeddings["text"].ndim == 2
    assert actual.embeddings["text"].shape[0] == len(TEXT_PROMPTS)
    _assert_embeddings(actual, reference)


@pytest.mark.parametrize(
    "case_name", ["scalar_image", "image_list", "cat_dog", "three_captions"]
)
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_embeddings_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    actual, reference = _model_run(model_path, trust_remote_code).compare(case_name)
    _assert_embeddings(actual, reference)


@pytest.mark.parametrize(
    "model_path", _accuracy_params(_UNIFORM_REASON, CosineAccuracyError)
)
def test_multimodal_uniform_embeddings_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    actual, reference = _model_run(model_path, trust_remote_code).compare("uniform")
    _assert_embeddings(actual, reference)


def _compare_retrieval(model_run: PreparedEmbeddingModel, case_name: str) -> None:
    actual, reference = model_run.compare(case_name)
    for prefix in ("", "joint_"):
        text_key, image_key = f"{prefix}text", f"{prefix}image"
        print(f"Retrieval using {prefix or 'standalone_'}features")
        _assert_retrieval(
            _cosine_scores(actual.embeddings[text_key], actual.embeddings[image_key]),
            _cosine_scores(
                reference.embeddings[text_key], reference.embeddings[image_key]
            ),
            image_case=case_name,
        )


@pytest.mark.parametrize("case_name", ["cat_dog", "three_captions"])
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_retrieval_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    _compare_retrieval(_model_run(model_path, trust_remote_code), case_name)


@pytest.mark.parametrize(
    "model_path", _accuracy_params(_UNIFORM_REASON, RetrievalAccuracyError)
)
def test_multimodal_uniform_retrieval_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    _compare_retrieval(_model_run(model_path, trust_remote_code), "uniform")


@pytest.mark.parametrize("case_name", ["cat_dog", "three_captions", "uniform"])
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_logits_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    actual, reference = _model_run(model_path, trust_remote_code).compare(case_name)
    _assert_logits(actual, reference)


@pytest.mark.parametrize("case_name", ["text", "cat_dog", "three_captions", "uniform"])
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_text_hidden_states_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    actual, reference = _model_run(model_path, trust_remote_code).compare(case_name)
    _assert_hidden_states(actual, reference, modality="text")


@pytest.mark.parametrize(
    "case_name", ["scalar_image", "image_list", "cat_dog", "three_captions", "uniform"]
)
@pytest.mark.parametrize(
    "model_path", _accuracy_params(_VISION_REASON, CosineAccuracyError)
)
def test_multimodal_image_hidden_states_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    actual, reference = _model_run(model_path, trust_remote_code).compare(case_name)
    _assert_hidden_states(actual, reference, modality="image")
