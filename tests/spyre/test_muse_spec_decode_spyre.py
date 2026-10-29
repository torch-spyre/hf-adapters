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

"""Muse Glimmer 30B tree speculative decoding with its DFlash assistant on Spyre.

With ``SPYRE_MUSE_FP8`` unset, the emitted tokens must agree with plain greedy
``generate`` on the same card until the first near-tie, so the test asserts an
agreeing prefix and reports where the two first differ.

Usage::

    pytest -s -vvv tests/spyre/test_muse_spec_decode_spyre.py
    # TP 2 (target and assistant sharded over two cards):
    SPYRE_DEVICES=2,3 python -m torch.distributed.run --standalone \\
        --nproc-per-node=2 -m pytest -s tests/spyre/test_muse_spec_decode_spyre.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401
from transformers import AutoTokenizer, MuseGlimmerAssistantModel

from hf_adapters import AutoSpyreModelForCausalLM
from hf_adapters.hf_common import encode_prompts
from hf_adapters.muse_spec_decode import TreeSpeculativeDecoder
from tests.model_registry import MUSE_SPEC_PAIRS

pytestmark = pytest.mark.requires_spyre

PROMPT = "Explain, step by step, how a sliding-window KV cache works in a decoder."


@pytest.mark.skipif(
    bool(os.environ.get("SPYRE_MUSE_FP8")),
    reason="compares against plain greedy: run with SPYRE_MUSE_FP8 unset",
)
@pytest.mark.parametrize("target_path, assistant_path", MUSE_SPEC_PAIRS)
def test_tree_speculative_decoding_agrees_with_greedy(target_path, assistant_path):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("cpu:gloo,spyre:spyreccl")
    tok = AutoTokenizer.from_pretrained(target_path)
    ids = encode_prompts(tok, [PROMPT])["input_ids"]
    target = AutoSpyreModelForCausalLM.from_pretrained(target_path, dtype=torch.float16)
    tp = getattr(target, "_spyre_tp", None)
    assert (tp.world if tp is not None else 1) == world
    greedy = target.generate(ids, max_new_tokens=48, do_sample=False)
    greedy = greedy[0, ids.shape[1] :].tolist()
    assistant = MuseGlimmerAssistantModel.from_pretrained(
        assistant_path, dtype=torch.float32
    ).eval()
    decoder = TreeSpeculativeDecoder(target, assistant, width=16)
    out = decoder.generate(ids, max_new_tokens=48)[0].tolist()
    first = next((i for i, (a, b) in enumerate(zip(out, greedy)) if a != b), None)
    print(
        f"first difference vs greedy: {first}; tokens/step {decoder.stats.mean_accepted:.2f}"
    )
    assert first is None or first >= 8, (tok.decode(out[:16]), tok.decode(greedy[:16]))
    assert decoder.stats.mean_accepted > 1.5
