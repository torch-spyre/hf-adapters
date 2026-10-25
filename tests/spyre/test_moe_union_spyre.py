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

"""Compile the union-capped multi-token MoE region and check its Spyre numerics.

Usage::

    pytest -s -vvv tests/spyre/test_moe_union_spyre.py
"""

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch_spyre  # noqa: F401
from torch_spyre.model_utils import dma_moe_per_expert_scale_to_spyre

from hf_adapters import hf_common
from hf_adapters.hf_common import (
    DEVICE,
    moe_union_experts,
    moe_union_pack,
    moe_union_selected_experts,
    prepare_moe_expert_weights,
)

pytestmark = pytest.mark.requires_spyre

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, STICK, CHUNKS = 32, 256, 128, 4, 64, 4


def _cpu_union(x, probs, weights, cap, scale):
    expert_ids = torch.arange(EXPERTS, dtype=torch.float32)[:, None].contiguous()
    return moe_union_selected_experts(
        x.float(),
        probs.float(),
        weights.decode_gate_proj.float(),
        weights.decode_up_proj.float(),
        weights.decode_down_proj.float(),
        TOP_K,
        cap,
        32,
        STICK,
        "gelu_tanh",
        expert_ids=expert_ids,
        slot_identity=None,
        per_expert_scale_stick=scale[:, None].expand(-1, STICK).float(),
        decode_chunks=CHUNKS,
    )


@pytest.mark.parametrize(
    "tokens,cap", [(2, 8), (4, 16), (16, 16)], ids=["exact8", "exact16", "capped16"]
)
def test_union_spyre(tokens, cap):
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(tokens)
    gate_up = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
    down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
    gate_up = (gate_up / HIDDEN**0.5).half()
    down = (down / INTERMEDIATE**0.5).half()
    x = (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).half()
    logits = torch.randn(tokens, EXPERTS, generator=generator) * 2
    probs = logits.softmax(-1).half()
    scale = torch.linspace(0.5, 1.5, EXPERTS).half()

    # The reference keeps its decode views on the host.
    host = SimpleNamespace(gate_up_proj=gate_up.clone(), down_proj=down.clone())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hf_common, "DEVICE", "cpu")
        prepare_moe_expert_weights(host, pad_to_multiple=STICK, decode_chunks=CHUNKS)
    expected = _cpu_union(x, probs, host, cap, scale)

    dev = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    prepare_moe_expert_weights(dev, pad_to_multiple=STICK, decode_chunks=CHUNKS)
    expert_ids = (
        torch.arange(EXPERTS, dtype=torch.float16)[:, None].contiguous().to(DEVICE)
    )
    scale_stick = dma_moe_per_expert_scale_to_spyre(scale, device=DEVICE)

    slot_identity = torch.eye(cap, dtype=torch.float16).to(DEVICE)

    # Two compiled regions, as the model's tree layer runs them.
    def pack(probs, expert_ids):
        return moe_union_pack(probs, TOP_K, cap, expert_ids, STICK)

    def experts(x, slot_sticks, routes, gate, up, down, scale_stick, eye, cs, co):
        return moe_union_experts(
            x,
            slot_sticks,
            routes,
            gate,
            up,
            down,
            32,
            STICK,
            "gelu_tanh",
            slot_identity=eye,
            per_expert_scale_stick=scale_stick,
            decode_chunks=CHUNKS,
            chunk_scale=cs,
            chunk_offsets=co,
        )

    pack_c = torch.compile(pack, fullgraph=True, dynamic=False)
    experts_c = torch.compile(experts, fullgraph=True, dynamic=False)
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        slot_sticks, routes = pack_c(probs.to(DEVICE), expert_ids)
        actual = experts_c(
            x.to(DEVICE),
            slot_sticks,
            routes,
            dev.decode_gate_proj,
            dev.decode_up_proj,
            dev.decode_down_proj,
            scale_stick,
            slot_identity,
            dev.decode_chunk_scale,
            dev.decode_chunk_offsets,
        ).to("cpu")
    fallbacks = [w for w in caught if "Fallback" in type(w.message).__name__]
    assert not fallbacks, [str(w.message) for w in fallbacks]
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
