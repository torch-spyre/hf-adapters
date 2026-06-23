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
Spyre loading test: every auto-class entry loads cleanly onto Spyre.

No forward pass — just verifies that ``AutoSpyreModelForCausalLM`` and
``AutoSpyreModel`` resolve, prepare, and move the model onto Spyre without
error. Causal-LM entries also check that a ``generate`` method is attached.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_load_spyre.py
    pytest -s -vvv tests/spyre/test_load_spyre.py -k qwen3
"""

import time

import pytest
import torch
from model_registry import (
    CAUSAL_KEYS,
    CAUSAL_LM_MODELS,
    EMBED_KEYS,
    EMBEDDING_MODELS,
)


@pytest.mark.parametrize("model_key", CAUSAL_KEYS, ids=CAUSAL_KEYS)
def test_load_causal_lm(model_key):
    from hf_adapters import AutoSpyreModelForCausalLM

    info = CAUSAL_LM_MODELS[model_key]
    path = info["path"]
    dtype = torch.float32 if model_key == "granite4" else torch.float16

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(path, dtype=dtype)
    load_s = time.time() - t0

    assert model is not None, f"{model_key}: from_pretrained returned None"
    assert callable(
        getattr(model, "generate", None)
    ), f"{model_key}: AutoSpyreModelForCausalLM did not attach generate()"
    print(f"  [{model_key}] causal-LM load time: {load_s:.1f}s")


@pytest.mark.parametrize("model_key", EMBED_KEYS, ids=EMBED_KEYS)
def test_load_embedding(model_key):
    from hf_adapters import AutoSpyreModel

    info = EMBEDDING_MODELS[model_key]
    path = info["path"]

    t0 = time.time()
    model = AutoSpyreModel.from_pretrained(path, dtype=torch.float16)
    load_s = time.time() - t0

    assert model is not None, f"{model_key}: from_pretrained returned None"
    print(f"  [{model_key}] embedding load time: {load_s:.1f}s")
