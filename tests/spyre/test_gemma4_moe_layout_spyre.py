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

"""Gemma 4 expert pools must serve prefill and chunked decode from one allocation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_spyre._C import get_spyre_tensor_layout

from hf_adapters.hf_common import (
    moe_decode_selected_experts,
    moe_topk,
    prepare_moe_expert_weights,
)
from hf_adapters.hf_gemma4_moe import _compiled_moe_loop_region

pytestmark = pytest.mark.requires_spyre


@pytest.mark.parametrize("intermediate", [704, 352, 176], ids=["tp1", "tp2", "tp4"])
def test_expert_decode_views_share_weights_and_preserve_chunks(intermediate):
    experts, hidden, chunks, stick = 16, 256, 4, 64
    torch.manual_seed(intermediate)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.float16)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.float16)
    weights = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)

    prepare_moe_expert_weights(weights, pad_to_multiple=stick, decode_chunks=chunks)
    width = intermediate + (-intermediate) % stick
    for role in ("gate", "up", "down"):
        pool = getattr(weights, f"{role}_proj")
        alias = getattr(weights, f"decode_{role}_proj")
        assert alias.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        host = pool.cpu()
        if role == "down":
            expected = (
                host.reshape(experts, width, chunks, hidden // chunks)
                .permute(0, 2, 1, 3)
                .reshape(experts * chunks, width, hidden // chunks)
            )
        else:
            expected = host.reshape(experts * chunks, hidden // chunks, width)
        assert torch.equal(alias.cpu(), expected)
        if role == "down":
            assert not host[:, intermediate:].count_nonzero()
        else:
            assert not host[..., intermediate:].count_nonzero()

    down_layout = get_spyre_tensor_layout(weights.down_proj)
    assert down_layout.device_size == [experts, hidden // stick, width, stick]
    assert get_spyre_tensor_layout(weights.decode_down_proj).device_size == [
        experts * chunks,
        hidden // chunks // stick,
        width,
        stick,
    ]


@pytest.mark.parametrize("decode_chunks", [4, 1], ids=["chunked", "one_chunk"])
@pytest.mark.parametrize("intermediate", [704, 352, 176], ids=["tp1", "tp2", "tp4"])
def test_compiled_chunked_decode_matches_independent_expert_reference(
    intermediate, decode_chunks
):
    experts, hidden, tokens, top_k, stick = 128, 256, 1, 8, 64
    torch.manual_seed(intermediate)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.float16) * 0.05
    down = torch.randn(experts, hidden, intermediate, dtype=torch.float16) * 0.05
    weights = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    prepare_moe_expert_weights(
        weights, pad_to_multiple=stick, decode_chunks=decode_chunks
    )

    x = torch.randn(tokens, hidden, dtype=torch.float16) * 0.5
    scores = (
        torch.arange(1, experts + 1, dtype=torch.float16)
        .expand(tokens, -1)
        .contiguous()
    )
    values, indices = torch.topk(scores, top_k, dim=-1)
    routing = values / values.sum(-1, keepdim=True)
    scale = torch.rand(experts, dtype=torch.float16) + 0.5
    scale_stick = scale[:, None].expand(-1, stick).contiguous().to("spyre")
    expected = torch.zeros(tokens, hidden, dtype=torch.float32)
    for row in range(tokens):
        for slot in range(top_k):
            expert = int(indices[row, slot])
            gate = x[row].float() @ gate_up[expert, :intermediate].float().T
            up = x[row].float() @ gate_up[expert, intermediate:].float().T
            activated = F.gelu(gate, approximate="tanh") * up
            output = activated @ down[expert].float().T
            expected[row] += output * float(routing[row, slot]) * float(scale[expert])

    def decode(x_device, scores_device):
        values, indices = moe_topk(scores_device, top_k)
        return moe_decode_selected_experts(
            x_device,
            values / values.sum(-1, keepdim=True),
            indices,
            weights.decode_gate_proj,
            weights.decode_up_proj,
            weights.decode_down_proj,
            top_k,
            32,
            stick,
            "gelu_tanh",
            per_expert_scale_stick=scale_stick,
            decode_chunks=decode_chunks,
            chunk_scale=weights.decode_chunk_scale,
            chunk_offsets=weights.decode_chunk_offsets,
        )

    compiled = torch.compile(decode, dynamic=False, fullgraph=True)
    inputs = (x.to("spyre"), scores.to("spyre"))
    with torch.no_grad():
        actual = compiled(*inputs).cpu().float()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        repeated = compiled(*inputs).cpu().float()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("tokens", "decode_chunks"),
    [(2, 4), (4, 4), (2, 1)],
    ids=["batch2", "batch4", "one_chunk"],
)
def test_compiled_routed_decode_for_multiple_tokens(tokens, decode_chunks):
    experts, hidden, intermediate, top_k, stick = 128, 256, 192, 8, 64
    generator = torch.Generator().manual_seed(341)
    gate_up = (
        torch.randn(experts, 2 * intermediate, hidden, generator=generator).half()
        * 0.05
    )
    down = torch.randn(experts, hidden, intermediate, generator=generator).half() * 0.05
    pools = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    prepare_moe_expert_weights(
        pools, pad_to_multiple=stick, decode_chunks=decode_chunks
    )
    router = torch.randn(experts, hidden, generator=generator).half() * 0.05
    x_router = torch.randn(tokens, hidden, generator=generator).half() * 0.5
    x_expert = torch.randn(tokens, hidden, generator=generator).half() * 0.5
    scale = torch.rand(experts, generator=generator).half() + 0.5
    scale_stick = scale[:, None].expand(-1, stick).contiguous().to("spyre")

    normalized = x_router.float() * torch.rsqrt(
        x_router.float().square().mean(-1, keepdim=True) + 1e-6
    )
    probabilities = torch.softmax(F.linear(normalized.half(), router), dim=-1)
    values, indices = torch.topk(probabilities, top_k, dim=-1)
    routing = values / values.sum(-1, keepdim=True)
    expected = torch.zeros(tokens, hidden, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(indices[token, slot])
            gate = x_expert[token].float() @ gate_up[expert, :intermediate].float().T
            up = x_expert[token].float() @ gate_up[expert, intermediate:].float().T
            activated = F.gelu(gate, approximate="tanh") * up
            output = activated @ down[expert].float().T
            expected[token] += (
                output * float(routing[token, slot]) * float(scale[expert])
            )

    compiled = torch.compile(_compiled_moe_loop_region, dynamic=False, fullgraph=True)
    with torch.no_grad():
        actual = (
            compiled(
                x_router.to("spyre"),
                x_expert.to("spyre"),
                router.to("spyre"),
                1.0,
                1.0,
                scale_stick,
                pools.decode_gate_proj,
                pools.decode_up_proj,
                pools.decode_down_proj,
                top_k,
                32,
                stick,
                1e-6,
                decode_chunks,
                pools.decode_chunk_scale,
                pools.decode_chunk_offsets,
            )
            .cpu()
            .float()
        )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
