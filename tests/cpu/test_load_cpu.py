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

import pytest
from conftest import torch_dtype_for
from model_registry import CAUSAL_LM_MODELS, CAUSAL_PATHS, EMBED_PATHS, EMBEDDING_MODELS


@pytest.mark.parametrize("model_key", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_load_causal_lm(model_key, auto_spyre_model):
    info = CAUSAL_LM_MODELS[model_key]
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        info["path"], dtype=torch_dtype_for(info)
    )
    assert model is not None
    assert callable(
        getattr(model, "generate", None)
    ), "AutoSpyreModelForCausalLM should attach a generate method"
    del model
    gc.collect()


@pytest.mark.parametrize("model_key", EMBED_PATHS, ids=EMBED_PATHS)
def test_load_embedding(model_key, auto_spyre_model):
    info = EMBEDDING_MODELS[model_key]
    model = auto_spyre_model.AutoSpyreModel.from_pretrained(
        info["path"], dtype=torch_dtype_for(info)
    )
    assert model is not None
    del model
    gc.collect()
