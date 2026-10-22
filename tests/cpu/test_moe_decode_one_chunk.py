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

"""The one-chunk decode layout must bind the same attribute contract as chunked."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    moe_decode_selected_experts,
    moe_topk,
    prepare_moe_expert_weights,
)


def _prepared(experts, hidden, intermediate, top_k, stick, decode_chunks):
    gate_up = torch.randn(experts, 2 * intermediate, hidden) * 0.05
    down = torch.randn(experts, hidden, intermediate) * 0.05
    weights = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    prepare_moe_expert_weights(
        weights, pad_to_multiple=stick, decode_chunks=decode_chunks
    )
    return weights, gate_up, down


@pytest.mark.parametrize("decode_chunks", [1, 4])
def test_prepare_binds_the_decode_attribute_contract(decode_chunks):
    # SPYRE_MOE_CHUNKS=1 (or a derived count of 1) must leave every attribute
    # the decode path reads; the pools are their own one-chunk views.
    experts, hidden, intermediate, top_k, stick = 16, 256, 192, 8, 64
    weights, _, _ = _prepared(
        experts, hidden, intermediate, top_k, stick, decode_chunks
    )
    for role in ("gate", "up", "down"):
        pool = getattr(weights, f"{role}_proj")
        view = getattr(weights, f"decode_{role}_proj")
        assert view is not None
        assert view.shape[0] == experts * decode_chunks
        if decode_chunks == 1:
            # The pools are their own one-chunk views.
            assert view is pool
        # Chunked CPU views are reshape copies; the shared-storage invariant
        # is asserted on Spyre by test_gemma4_moe_layout_spyre.py.
    scale, offsets = weights.decode_chunk_scale, weights.decode_chunk_offsets
    if decode_chunks == 1:
        # The trivial entry map e*1+0: scale one, offset zero.
        assert scale.shape == (1, 1, 1) and float(scale) == 1.0
        assert offsets.shape == (1, 1, 1) and float(offsets) == 0.0
    else:
        assert float(scale) == float(decode_chunks)
        assert offsets.flatten().tolist() == list(range(decode_chunks))


def test_one_chunk_decode_through_bound_attributes():
    experts, hidden, tokens, top_k, stick = 16, 256, 1, 8, 64
    intermediate = 192
    weights, gate_up, down = _prepared(experts, hidden, intermediate, top_k, stick, 1)
    x = torch.randn(tokens, hidden) * 0.5
    scores = torch.arange(1, experts + 1).expand(tokens, -1).contiguous()
    values, indices = moe_topk(scores, top_k)
    routing = values / values.sum(-1, keepdim=True)
    expected = torch.zeros(tokens, hidden)
    for slot in range(top_k):
        expert = int(indices[0, slot])
        gate = x[0] @ gate_up[expert, :intermediate].T
        up = x[0] @ gate_up[expert, intermediate:].T
        active = F.gelu(gate, approximate="tanh") * up
        expected[0] += (active @ down[expert].T) * float(routing[0, slot])
    actual = moe_decode_selected_experts(
        x,
        routing,
        indices,
        weights.decode_gate_proj,
        weights.decode_up_proj,
        weights.decode_down_proj,
        top_k,
        32,
        stick,
        "gelu_tanh",
        decode_chunks=1,
        chunk_scale=weights.decode_chunk_scale,
        chunk_offsets=weights.decode_chunk_offsets,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_chunked_decode_matches_per_expert_reference():
    # The chunked kernel's entry map (e*chunks+c), pre-activation chunk sums,
    # and in-place down reshape must reproduce the dense per-expert math.
    experts, hidden, tokens, top_k, stick, chunks = 16, 256, 3, 8, 64, 4
    intermediate = 192
    weights, gate_up, down = _prepared(
        experts, hidden, intermediate, top_k, stick, chunks
    )
    x = torch.randn(tokens, hidden) * 0.5
    scores = torch.arange(1, experts + 1).expand(tokens, -1).contiguous()
    values, indices = moe_topk(scores, top_k)
    routing = values / values.sum(-1, keepdim=True)
    expected = torch.zeros(tokens, hidden)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(indices[token, slot])
            gate = x[token] @ gate_up[expert, :intermediate].T
            up = x[token] @ gate_up[expert, intermediate:].T
            active = F.gelu(gate, approximate="tanh") * up
            expected[token] += (active @ down[expert].T) * float(routing[token, slot])
    actual = moe_decode_selected_experts(
        x,
        routing,
        indices,
        weights.decode_gate_proj,
        weights.decode_up_proj,
        weights.decode_down_proj,
        top_k,
        32,
        stick,
        "gelu_tanh",
        decode_chunks=chunks,
        chunk_scale=weights.decode_chunk_scale,
        chunk_offsets=weights.decode_chunk_offsets,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    # No host constants: the region must also build them itself.
    actual_selfbuilt = moe_decode_selected_experts(
        x,
        routing,
        indices,
        weights.decode_gate_proj,
        weights.decode_up_proj,
        weights.decode_down_proj,
        top_k,
        32,
        stick,
        "gelu_tanh",
        decode_chunks=chunks,
    )
    torch.testing.assert_close(actual_selfbuilt, expected, atol=2e-2, rtol=2e-2)


def test_chunk_view_rejects_indivisible_dims():
    # A pool whose chunked dim does not divide must fail loudly here, not
    # reshape to a misshapen view. prepare_moe_expert_weights validates the
    # hidden side upstream, so this guards direct view construction.
    import torch

    from hf_adapters.hf_common import _moe_decode_weight_view

    pool = torch.randn(16, 190, 128)
    with pytest.raises(ValueError, match="contract dim 190 must divide into 3"):
        _moe_decode_weight_view(pool, 3)
    down_pool = torch.randn(16, 128, 190)
    with pytest.raises(ValueError, match="free dim 190 must divide into 3"):
        _moe_decode_weight_view(down_pool, 3, down=True)
