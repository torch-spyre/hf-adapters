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

"""EOS id appearing inside the prompt: must NOT trip the ``finished`` mask.

``generate()`` updates ``finished`` only on emitted next_tokens, not on prompt
content. A prompt containing the model's eos_token_id should still produce a
normal greedy continuation matching HF.
"""

import pytest
from _common import free, load_hf_reference, load_spyre_model, model_info
from _generate_edge_case_helpers import (
    BLOCK_SIZE,
    hf_reference_outputs,
    make_prompt_with_eos_inside,
)
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoTokenizer


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_eos_inside_prompt(model_key):
    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    if tokenizer.eos_token_id is None:
        pytest.skip(f"{model_key}: tokenizer has no eos_token_id")

    prompts = [
        make_prompt_with_eos_inside(tokenizer, tokenizer.eos_token_id, target_tokens=12)
    ]
    max_new_tokens = BLOCK_SIZE + 8  # cross a block boundary

    ref = load_hf_reference(info)
    try:
        hf_outputs = hf_reference_outputs(ref, tokenizer, prompts, max_new_tokens)
    finally:
        free(ref)

    model = load_spyre_model(info)
    try:
        spyre_outputs = model.generate(
            tokenizer, prompts, max_new_tokens=max_new_tokens, do_sample=False
        )
    finally:
        free(model)

    assert hf_outputs[0].strip() == spyre_outputs[0].strip(), (
        f"{model_key} eos-in-prompt:\n"
        f"  HF:    {hf_outputs[0]!r}\n  Spyre: {spyre_outputs[0]!r}"
    )
