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

"""``max_new_tokens=0``: the loop must short-circuit and return empty strings."""

import pytest
from _common import free, load_spyre_model, model_info
from _generate_edge_case_helpers import make_prompts
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoTokenizer


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_zero_new_tokens(model_key):
    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])

    # No HF reference needed: the expected output is constant (empty strings).
    model = load_spyre_model(info)
    try:
        outputs = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
    finally:
        free(model)

    assert len(outputs) == len(
        prompts
    ), f"{model_key}: got {len(outputs)} outputs, expected {len(prompts)}"
    for i, out in enumerate(outputs):
        assert out == "", f"{model_key} prompt[{i}]: expected empty output, got {out!r}"
