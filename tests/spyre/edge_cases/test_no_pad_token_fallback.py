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

"""``pad_token`` is None on the tokenizer: triggers ``pad_token = eos_token`` fallback."""

import pytest
from _common import free, load_hf_reference, load_spyre_model, model_info
from _generate_edge_case_helpers import (
    NoPadTokenizer,
    hf_reference_outputs,
    make_prompts,
)
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoTokenizer


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_no_pad_token_fallback(model_key):
    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])  # mixed lengths -> left-padding
    max_new_tokens = 16

    ref = load_hf_reference(info)
    try:
        hf_outputs = hf_reference_outputs(ref, tokenizer, prompts, max_new_tokens)
    finally:
        free(ref)

    wrapped = NoPadTokenizer(tokenizer)

    model = load_spyre_model(info)
    try:
        spyre_outputs = model.generate(
            wrapped, prompts, max_new_tokens=max_new_tokens, do_sample=False
        )
    finally:
        free(model)

    assert len(spyre_outputs) == len(prompts)
    for i, (hf_out, sp_out) in enumerate(zip(hf_outputs, spyre_outputs)):
        assert hf_out.strip() == sp_out.strip(), (
            f"{model_key} no-pad-token prompt[{i}]:\n"
            f"  HF:    {hf_out!r}\n  Spyre: {sp_out!r}"
        )
