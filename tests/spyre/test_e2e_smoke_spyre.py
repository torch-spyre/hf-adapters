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

"""E2E smoke test: load HF model on Spyre, generate tokens, verify non-trivial.

For each registered causal-LM, loads the model, generates 5 tokens for the
prompt "The capital of France is", and asserts the output is non-empty,
contains tokens, and is not all-zero / all-same.

Usage (on Spyre pod):
    pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k qwen3
"""

import time

import pytest
from _helpers import torch_dtype_for
from model_registry import CAUSAL_LM_MODELS


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_smoke_generate(model_key):
    """Load model, generate 5 tokens, assert each diversity check individually."""
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM

    info = CAUSAL_LM_MODELS[model_key]
    print(f"\n  {info['name']}: loading from {info['path']}")

    dtype = torch_dtype_for(info)
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(info["path"], dtype=dtype)
    load_s = time.time() - t0
    print(f"  Load time: {load_s:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompt = "The capital of France is"

    t0 = time.time()
    outputs = model.generate(
        tokenizer,
        [prompt],
        max_new_tokens=5,
        do_sample=False,
        timing=True,
    )
    gen_s = time.time() - t0
    print(f"  Generate time: {gen_s:.1f}s")

    assert outputs, f"{model_key}: generate() returned empty list"
    output_text = outputs[0]
    print(f"  Output: {output_text!r}")

    assert len(output_text.strip()) > 0, f"{model_key}: output was empty / all-spaces"

    gen_ids = tokenizer.encode(output_text, add_special_tokens=False)
    assert len(gen_ids) > 0, f"{model_key}: tokenizer produced no ids from output"
    assert not all(
        t == 0 for t in gen_ids
    ), f"{model_key}: all generated token ids are 0 (token_ids={gen_ids})"
    assert (
        len(set(gen_ids)) > 1 or len(gen_ids) <= 1
    ), f"{model_key}: all generated token ids are identical (token_ids={gen_ids})"
