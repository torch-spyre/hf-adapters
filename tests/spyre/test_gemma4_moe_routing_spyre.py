# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compiled Gemma routing must retain HF's near-tied expert selection."""

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from hf_adapters.hf_common import prepare_moe_expert_weights
from hf_adapters.hf_gemma4_moe import (
    _compiled_moe_loop_region,
    _moe_route_persistent_packed,
)
from tests._gemma4_moe_routing_helpers import reference_routing, routing_inputs

pytestmark = pytest.mark.requires_spyre


@pytest.mark.parametrize("tokens", [1, 2])
def test_compiled_prefill_routing_matches_hf(tokens: int) -> None:
    router, x = routing_inputs(tokens, torch.float16)
    _, _, expected = reference_routing(router, x)
    compiled = torch.compile(
        _moe_route_persistent_packed, dynamic=False, fullgraph=True
    )
    with torch.no_grad():
        actual = compiled(
            x.to("spyre"),
            router.proj.weight.to("spyre"),
            router.scale.to("spyre"),
            router.scalar_root_size,
            router.per_expert_scale.to("spyre"),
            router.config.top_k_experts,
            64,
            router.eps,
            torch.eye(64, dtype=x.dtype).to("spyre"),
        ).cpu()
    torch.testing.assert_close(actual[..., 0] != 0, expected != 0)
    torch.testing.assert_close(
        actual.float(), expected[..., None].expand_as(actual), atol=0.002, rtol=0.002
    )


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("chunks", [1, 4])
def test_compiled_decode_routing_matches_hf(tokens: int, chunks: int) -> None:
    router, x = routing_inputs(tokens, torch.float16)
    weights, indices, _ = reference_routing(router, x)
    experts = Gemma4TextExperts(router.config).to(x.dtype).eval()
    generator = torch.Generator().manual_seed(773)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.normal_(std=0.05, generator=generator)
        expected = experts(x, indices, weights)
    prepare_moe_expert_weights(experts, pad_to_multiple=64, decode_chunks=chunks)
    scale = router.per_expert_scale[:, None].expand(-1, 64).contiguous().to("spyre")
    compiled = torch.compile(_compiled_moe_loop_region, dynamic=False, fullgraph=True)
    with torch.no_grad():
        actual = compiled(
            x.to("spyre"),
            x.to("spyre"),
            router.proj.weight.to("spyre"),
            router.scale.to("spyre"),
            router.scalar_root_size,
            scale,
            experts.decode_gate_proj,
            experts.decode_up_proj,
            experts.decode_down_proj,
            router.config.top_k_experts,
            32,
            64,
            router.eps,
            chunks,
            experts.decode_chunk_scale,
            experts.decode_chunk_offsets,
        ).cpu()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
