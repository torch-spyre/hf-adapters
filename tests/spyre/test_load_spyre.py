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

"""Spyre loading test: every auto-class entry loads cleanly onto Spyre.

No forward pass — just verifies that ``AutoSpyreModelForCausalLM`` and
``AutoSpyreModel`` resolve, prepare, and move the model onto Spyre without
error. Causal-LM entries also check that a ``generate`` method is attached.

Parametrized over the full ``CAUSAL_LM_MODELS`` / ``EMBEDDING_MODELS`` registries
so the CI matrix's ``-k <model_key>`` filter always matches a parametrize id;
the per-adapter ``CAUSAL_KEYS`` / ``EMBED_KEYS`` selection used by the CPU side
would silently deselect models like ``tiny_llama`` whose adapter family has
another representative.

Usage (on Spyre pod):
    pytest -s -vvv tests/spyre/test_load_spyre.py
    pytest -s -vvv tests/spyre/test_load_spyre.py -k qwen3
"""

import time

import pytest
import torch
from model_registry import CAUSAL_LM_MODELS, EMBEDDING_MODELS


def _causal_dtype(key):
    return torch.float32 if key == "granite4" else torch.float16


@pytest.mark.parametrize("key", list(CAUSAL_LM_MODELS.keys()))
def test_load_causal_lm(key):
    from hf_adapters import AutoSpyreModelForCausalLM

    path = CAUSAL_LM_MODELS[key]["path"]
    dtype = _causal_dtype(key)

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(path, dtype=dtype)
    load_s = time.time() - t0
    print(f"  [{key}] load time: {load_s:.1f}s ({path})")

    assert model is not None, f"{key}: from_pretrained returned None"
    assert callable(
        getattr(model, "generate", None)
    ), f"{key}: AutoSpyreModelForCausalLM did not attach generate()"


@pytest.mark.parametrize("key", list(EMBEDDING_MODELS.keys()))
def test_load_embedding(key):
    from hf_adapters import AutoSpyreModel

    path = EMBEDDING_MODELS[key]["path"]

    t0 = time.time()
    model = AutoSpyreModel.from_pretrained(path, dtype=torch.float16)
    load_s = time.time() - t0
    print(f"  [{key}] load time: {load_s:.1f}s ({path})")

    assert model is not None, f"{key}: from_pretrained returned None"
