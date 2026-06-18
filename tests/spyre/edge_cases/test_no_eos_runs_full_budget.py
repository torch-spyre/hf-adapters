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

"""``eos_token_id=None``: must run full max_new_tokens with no early stop.

Crosses a block boundary (BLOCK_SIZE + 7) so the no-EOS path is exercised
across an expansion step.
"""

import pytest
import torch
from _common import free, load_hf_reference, load_spyre_model, model_info
from _generate_edge_case_helpers import BLOCK_SIZE, make_prompts
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoTokenizer


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_no_eos_runs_full_budget(model_key):
    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])
    max_new_tokens = BLOCK_SIZE + 7

    # HF reference with eos_token_id=None so HF also runs the full budget.
    ref = load_hf_reference(info)
    try:
        hf_outputs = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = ref.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                )
            new_ids = out[0][encoded["input_ids"].shape[1] :]
            assert (
                len(new_ids) == max_new_tokens
            ), f"HF reference truncated early for prompt {prompt!r}"
            hf_outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    finally:
        free(ref)

    model = load_spyre_model(info)
    try:
        spyre_outputs = model.generate(
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=None,
        )
    finally:
        free(model)

    assert len(spyre_outputs) == len(prompts)
    for i, (hf_out, sp_out) in enumerate(zip(hf_outputs, spyre_outputs)):
        assert hf_out.strip() == sp_out.strip(), (
            f"{model_key} no-eos prompt[{i}]:\n"
            f"  HF:    {hf_out!r}\n  Spyre: {sp_out!r}"
        )
