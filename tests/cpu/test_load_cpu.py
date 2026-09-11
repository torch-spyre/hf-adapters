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

"""
CPU loading test: every auto-class entry loads cleanly without forward.

Picks one small representative per adapter module from the shared
``CAUSAL_LM_MODELS`` / ``EMBEDDING_MODELS`` registries in ``conftest.py``,
and asserts that ``from_pretrained`` returns a model. Causal-LM entries
also verify that ``AutoSpyreModelForCausalLM`` attached a ``generate``
method.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``.
"""

import gc
import sys
from typing import Any

import pytest

from tests.model_registry import (
    CAUSAL_PATHS,
    EMBED_PATHS,
    MASKED_LM_PATHS,
    QUESTION_ANSWERING_PATHS,
    REMOTE_CODE_PATHS,
    TOKEN_CLASSIFICATION_PATHS,
)


@pytest.mark.model_harness("causal")
@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_load_causal_lm(model_path, trust_remote_code):
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    assert model is not None
    assert callable(
        getattr(model, "generate", None)
    ), "AutoSpyreModelForCausalLM should attach a generate method"
    del model
    gc.collect()


@pytest.mark.model_harness("embedding")
@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_load_embedding(model_path, trust_remote_code):
    model = load_embedding(model_path=model_path, trust_remote_code=trust_remote_code)
    assert model is not None
    del model
    gc.collect()


def load_embedding(model_path: str, trust_remote_code: bool | None = None) -> Any:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    model = auto_spyre_model.AutoSpyreModel.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    return model


@pytest.mark.model_harness("masked_lm")
@pytest.mark.parametrize("model_path", MASKED_LM_PATHS, ids=MASKED_LM_PATHS)
def test_load_masked_lm(model_path, trust_remote_code):
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    model = auto_spyre_model.AutoSpyreModelForMaskedLM.from_pretrained(
        model_path,
        dtype=auto_spyre_model.dtype_for_model_path(
            model_path, target_device="cpu", trust_remote_code=trust_remote_code
        ),
        trust_remote_code=trust_remote_code,
    )
    assert callable(model.forward)
    del model
    gc.collect()


@pytest.mark.model_harness("question_answering")
@pytest.mark.parametrize(
    "model_path", QUESTION_ANSWERING_PATHS, ids=QUESTION_ANSWERING_PATHS
)
def test_load_question_answering(model_path, trust_remote_code):
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    model = auto_spyre_model.AutoSpyreModelForQuestionAnswering.from_pretrained(
        model_path,
        dtype=auto_spyre_model.dtype_for_model_path(
            model_path, target_device="cpu", trust_remote_code=trust_remote_code
        ),
        trust_remote_code=trust_remote_code,
    )
    assert callable(model.forward)
    assert next(model.qa_outputs.parameters()).device.type == "cpu"
    del model
    gc.collect()


@pytest.mark.model_harness("token_classification")
@pytest.mark.parametrize(
    "model_path", TOKEN_CLASSIFICATION_PATHS, ids=TOKEN_CLASSIFICATION_PATHS
)
def test_load_token_classification(model_path, trust_remote_code):
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    model = auto_spyre_model.AutoSpyreModelForTokenClassification.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    assert callable(model.forward)
    assert next(model.classifier.parameters()).device.type == "cpu"
    del model
    gc.collect()
