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
    pytest -s -vvv "tests/spyre/test_load_spyre.py::test_load_causal_lm[Qwen/Qwen3-0.6B]"
"""

import time

import pytest
from model_registry import CAUSAL_PATHS, EMBED_PATHS

from tests.conftest import torch_dtype_for_model_path


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_load_causal_lm(model_path: str) -> None:
    from hf_adapters import AutoSpyreModelForCausalLM

    dtype = torch_dtype_for_model_path(model_path)

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    load_s = time.time() - t0

    assert model is not None, f"{model_path}: from_pretrained returned None"
    assert callable(
        getattr(model, "generate", None)
    ), f"{model_path}: AutoSpyreModelForCausalLM did not attach generate()"
    print(f"  [{model_path}] causal-LM load time: {load_s:.1f}s")
    print("\n## Spyre Load Test Results\n")
    print("| Path | Kind | Status | Load (s) |")
    print("|------|------|--------|----------|")
    print(f"| {model_path} | causal-LM | PASS | {load_s:.1f} |")


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_load_embedding(model_path: str) -> None:
    from hf_adapters import AutoSpyreModel

    dtype = torch_dtype_for_model_path(model_path)
    t0 = time.time()
    model = AutoSpyreModel.from_pretrained(model_path, dtype=dtype)
    load_s = time.time() - t0

    assert model is not None, f"{model_path}: from_pretrained returned None"
    print(f"  [{model_path}] embedding load time: {load_s:.1f}s")
    print("\n## Spyre Load Test Results\n")
    print("| Path | Kind | Status | Load (s) |")
    print("|------|------|--------|----------|")
    print(f"| {model_path} | embedding | PASS | {load_s:.1f} |")
