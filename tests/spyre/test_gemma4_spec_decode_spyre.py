# Copyright 2026 The Torch-Spyre Authors.
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

"""Gemma 4 26B-A4B tree speculative decoding with XPress on Spyre (real checkpoints).

With the lossy knobs unset, the emitted tokens must agree with plain greedy
``generate`` on the same card until the first near-tie, so the test asserts
an agreeing prefix and reports where the two first differ.

Usage::

    pytest -s -vvv tests/spyre/test_gemma4_spec_decode_spyre.py
"""

import os

import pytest
import torch
import torch_spyre  # noqa: F401
from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3
from hf_adapters.gemma4_spec_decode import TreeSpeculativeDecoder
from hf_adapters.hf_common import encode_prompts
from tests.model_registry import CAUSAL_LM_MODELS, XPRESS_PATHS

pytestmark = pytest.mark.requires_spyre

PROMPT = "Explain, step by step, how a sliding-window KV cache works in a decoder."


@pytest.mark.skipif(
    os.environ.get("SPYRE_MOE_UNION_CAP") or os.environ.get("SPYRE_GEMMA4_FP8"),
    reason="compares against plain greedy: run with the lossy knobs unset",
)
@pytest.mark.parametrize("drafter_path", XPRESS_PATHS)
def test_tree_speculative_decoding_agrees_with_greedy(drafter_path):
    target_path = next(
        i["target"] for i in CAUSAL_LM_MODELS.values() if i["path"] == drafter_path
    )
    tok = AutoTokenizer.from_pretrained(target_path)
    ids = encode_prompts(tok, [PROMPT])["input_ids"]
    target = AutoSpyreModelForCausalLM.from_pretrained(target_path, dtype=torch.float16)
    greedy = target.generate(ids, max_new_tokens=48, do_sample=False)
    greedy = greedy[0, ids.shape[1] :].tolist()
    drafter = hf_xpress_qwen3.load_model(
        drafter_path, target_path=target_path, dtype=torch.float32
    )
    decoder = TreeSpeculativeDecoder(target, drafter, width=16)
    out = decoder.generate(ids, max_new_tokens=48)[0].tolist()
    first = next((i for i, (a, b) in enumerate(zip(out, greedy)) if a != b), None)
    print(
        f"first difference vs greedy: {first}; tokens/step {decoder.stats.mean_accepted:.2f}"
    )
    assert first is None or first >= 8, (tok.decode(out[:16]), tok.decode(greedy[:16]))
    assert decoder.stats.mean_accepted > 1.5
