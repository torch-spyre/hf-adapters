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

"""Sampling determinism: same seed → identical outputs; different seed → different.

Adapter-vs-adapter comparison; no HF reference needed.
"""

import pytest
import torch
from _common import free, load_spyre_model, model_info
from _generate_edge_case_helpers import (
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
    make_prompts,
)
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoTokenizer


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_sampling_determinism(model_key):
    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, SAMPLING_TARGETS)

    model = load_spyre_model(info)
    try:
        torch.manual_seed(1234)
        out_a1 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
        torch.manual_seed(1234)
        out_a2 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
        torch.manual_seed(9999)
        out_b = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
    finally:
        free(model)

    assert out_a1 == out_a2, (
        f"{model_key}: sampling not reproducible with the same seed:\n"
        f"  run1: {out_a1}\n  run2: {out_a2}"
    )
    assert out_a1 != out_b, (
        f"{model_key}: different seeds produced identical outputs (top_k=20 "
        f"should diverge):\n  seed=1234: {out_a1}\n  seed=9999: {out_b}"
    )
