# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gemma MoE must select experts before low-precision probability rounding."""

import pytest
import torch

from hf_adapters import hf_gemma4_moe
from tests._gemma4_moe_routing_helpers import reference_routing, routing_inputs


def _keep_by_index(
    values: torch.Tensor, indices: torch.Tensor, dim: int, fill: float
) -> torch.Tensor:
    indices = indices.long()
    return torch.full_like(values, fill).scatter(
        dim, indices, values.gather(dim, indices)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 3])
def test_prefill_routing_matches_hf(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tokens: int
) -> None:
    router, x = routing_inputs(tokens, dtype)
    _, _, expected = reference_routing(router, x)
    monkeypatch.setattr(torch.ops.spyre, "keep_by_index", _keep_by_index, raising=False)
    with torch.no_grad():
        actual = hf_gemma4_moe._moe_route_persistent_packed(
            x,
            router.proj.weight,
            router.scale,
            router.scalar_root_size,
            router.per_expert_scale,
            router.config.top_k_experts,
            64,
            router.eps,
            torch.eye(64, dtype=dtype),
        )
    torch.testing.assert_close(actual[..., 0] != 0, expected != 0)
    torch.testing.assert_close(
        actual.float(), expected[..., None].expand_as(actual), atol=0.005, rtol=0.005
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 3])
def test_decode_routing_matches_hf(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tokens: int
) -> None:
    router, x = routing_inputs(tokens, dtype)
    expected_weights, expected_indices, _ = reference_routing(router, x)
    selected_indices: list[torch.Tensor] = []
    selected_weights: list[torch.Tensor] = []

    def capture_experts(
        expert_input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        selected_indices.append(indices)
        selected_weights.append(weights * router.per_expert_scale[indices])
        return torch.zeros_like(expert_input)

    monkeypatch.setattr(hf_gemma4_moe, "moe_decode_selected_experts", capture_experts)
    unused = torch.empty(0, dtype=dtype)
    with torch.no_grad():
        hf_gemma4_moe._compiled_moe_loop_region(
            x,
            x,
            router.proj.weight,
            router.scale,
            router.scalar_root_size,
            router.per_expert_scale[:, None].expand(-1, 64),
            unused,
            unused,
            unused,
            router.config.top_k_experts,
            32,
            64,
            router.eps,
        )
    torch.testing.assert_close(torch.cat(selected_indices), expected_indices)
    torch.testing.assert_close(
        torch.cat(selected_weights).float(), expected_weights, atol=0.005, rtol=0.005
    )
