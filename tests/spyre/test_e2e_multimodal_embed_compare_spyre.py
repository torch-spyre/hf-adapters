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

from tests._multimodal_embedding_helpers import (
    TEXT_PROMPTS,
    CosineAccuracyError,
    EmbeddingTestSettings,
    PreparedEmbeddingModel,
    _assert_embeddings,
    _assert_hidden_states,
    _assert_retrieval,
    _cosine_scores,
    bounded_cpu_threads,
    embedding_cases,
    prepare_model_cases,
    settings_for_model,
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
        settings: EmbeddingTestSettings = settings_for_model(model_path)
        print(f"Embedding API: {settings.backend}; CPU reference: float32")
        with bounded_cpu_threads():
            return prepare_model_cases(
                model_path,
                settings,
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


@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_text_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    model_run: PreparedEmbeddingModel = _model_run(model_path, trust_remote_code)
    actual, reference = model_run.compare("text")
    assert actual.embeddings["text"].ndim == 2
    assert actual.embeddings["text"].shape[0] == len(TEXT_PROMPTS)
    _assert_embeddings(actual, reference, model_run.settings)


@pytest.mark.parametrize("case_name", ["scalar_image", "image_list"])
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_image_compare_spyre(
    model_path: str, case_name: str, trust_remote_code: bool | None
) -> None:
    model_run: PreparedEmbeddingModel = _model_run(model_path, trust_remote_code)
    actual, reference = model_run.compare(case_name)
    _assert_embeddings(actual, reference, model_run.settings)


@pytest.mark.parametrize("image_case", ["cat_dog", "three_captions", "uniform"])
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_embeddings_compare_spyre(
    model_path: str,
    image_case: str,
    trust_remote_code: bool | None,
    request: pytest.FixtureRequest,
) -> None:
    model_run: PreparedEmbeddingModel = _model_run(model_path, trust_remote_code)
    settings = model_run.settings
    if image_case in settings.xfail_cases:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=settings.xfail_cases[image_case],
                strict=False,
                raises=AssertionError,
            )
        )
    actual, reference = model_run.compare(image_case)
    _assert_embeddings(actual, reference, settings)
    for prefix in ("", "joint_"):
        text_key, image_key = f"{prefix}text", f"{prefix}image"
        if text_key in reference.embeddings and image_key in reference.embeddings:
            print(f"Retrieval using {prefix or 'standalone_'}features")
            _assert_retrieval(
                _cosine_scores(
                    actual.embeddings[text_key], actual.embeddings[image_key]
                ),
                _cosine_scores(
                    reference.embeddings[text_key], reference.embeddings[image_key]
                ),
                image_case=image_case,
                score_atol=settings.score_atol,
                ranking_atol=settings.ranking_atol,
            )


@pytest.mark.parametrize(
    "case_name,modality",
    [
        ("text", "text"),
        ("scalar_image", "image"),
        ("image_list", "image"),
        ("cat_dog", "text"),
        ("cat_dog", "image"),
        ("three_captions", "text"),
        ("three_captions", "image"),
        ("uniform", "text"),
        ("uniform", "image"),
    ],
)
@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_hidden_states_compare_spyre(
    model_path: str,
    case_name: str,
    modality: str,
    trust_remote_code: bool | None,
    request: pytest.FixtureRequest,
) -> None:
    model_run: PreparedEmbeddingModel = _model_run(model_path, trust_remote_code)
    reference = model_run.reference[case_name]
    if modality not in reference.hidden_states:
        pytest.skip(
            "The public embedding API does not expose this tower's hidden states"
        )
    reason = model_run.settings.hidden_state_xfails.get(modality)
    if reason is not None:
        request.node.add_marker(
            pytest.mark.xfail(reason=reason, strict=False, raises=CosineAccuracyError)
        )
    actual, reference = model_run.compare(case_name)
    _assert_hidden_states(actual, reference, model_run.settings, modality=modality)
