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

from collections.abc import Iterator
from typing import cast

import pytest
import transformers

from tests._multimodal_embedding_helpers import (
    TEXT_PROMPTS,
    CosineAccuracyError,
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


@pytest.fixture(scope="module")
def model_path(request: pytest.FixtureRequest) -> str:
    # Indirect parametrization keeps one model index across all scenario axes.
    # Direct parameters in pytest 9 are reindexed per test's Cartesian product.
    return cast(str, request.param)


@pytest.fixture(scope="module")
def model_run(
    model_path: str, request: pytest.FixtureRequest
) -> Iterator[PreparedEmbeddingModel]:
    print(f"Transformers: {transformers.__version__} ({transformers.__file__})")
    settings = settings_for_model(model_path)
    print(f"Embedding API: {settings.backend}; CPU reference: float32")
    trust_remote_code = (
        True if request.config.getoption("--trust-remote-code") else None
    )
    with bounded_cpu_threads():
        yield prepare_model_cases(
            model_path, settings, embedding_cases(), trust_remote_code=trust_remote_code
        )


@pytest.mark.parametrize(
    "model_path",
    MULTIMODAL_EMBED_PATHS,
    ids=MULTIMODAL_EMBED_PATHS,
    scope="module",
    indirect=True,
)
def test_multimodal_text_compare_spyre(model_run: PreparedEmbeddingModel) -> None:
    actual, reference = model_run.compare("text")
    assert actual.embeddings["text"].ndim == 2
    assert actual.embeddings["text"].shape[0] == len(TEXT_PROMPTS)
    _assert_embeddings(actual, reference, model_run.settings)


@pytest.mark.parametrize("case_name", ["scalar_image", "image_list"])
@pytest.mark.parametrize(
    "model_path",
    MULTIMODAL_EMBED_PATHS,
    ids=MULTIMODAL_EMBED_PATHS,
    scope="module",
    indirect=True,
)
def test_multimodal_image_compare_spyre(
    model_run: PreparedEmbeddingModel, case_name: str
) -> None:
    actual, reference = model_run.compare(case_name)
    _assert_embeddings(actual, reference, model_run.settings)


@pytest.mark.parametrize("image_case", ["cat_dog", "three_captions", "uniform"])
@pytest.mark.parametrize(
    "model_path",
    MULTIMODAL_EMBED_PATHS,
    ids=MULTIMODAL_EMBED_PATHS,
    scope="module",
    indirect=True,
)
def test_multimodal_embeddings_compare_spyre(
    model_run: PreparedEmbeddingModel, image_case: str, request: pytest.FixtureRequest
) -> None:
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
@pytest.mark.parametrize(
    "model_path",
    MULTIMODAL_EMBED_PATHS,
    ids=MULTIMODAL_EMBED_PATHS,
    scope="module",
    indirect=True,
)
def test_multimodal_hidden_states_compare_spyre(
    model_run: PreparedEmbeddingModel,
    case_name: str,
    modality: str,
    request: pytest.FixtureRequest,
) -> None:
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
