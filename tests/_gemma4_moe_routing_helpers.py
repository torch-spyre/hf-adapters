# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Near-tied Gemma router scores with a distinct HF top-k boundary."""

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter


def routing_inputs(
    tokens: int, dtype: torch.dtype
) -> tuple[Gemma4TextRouter, torch.Tensor]:
    config = Gemma4TextConfig(
        hidden_size=256, num_experts=4, top_k_experts=2, moe_intermediate_size=128
    )
    router = Gemma4TextRouter(config).to(dtype).eval()
    with torch.no_grad():
        router.proj.weight.zero_()
        # Experts 1 and 2 have distinct scores but identical fp16/bf16
        # probabilities. HF's fp32 probabilities retain their ordering.
        router.proj.weight[:, 0] = (
            torch.tensor([0.0, 0.01, 0.0101, 0.02], dtype=dtype)
            / router.scalar_root_size
        )
        router.per_expert_scale.copy_(torch.tensor([0.5, 1.5, 2.0, 0.75]))
    return router, torch.ones(tokens, config.hidden_size, dtype=dtype)


def reference_routing(
    router: Gemma4TextRouter, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the stock HF router, including its non-unit per-expert scaling."""
    with torch.no_grad():
        _, weights, indices = router(x)
    dense = torch.zeros(x.shape[0], router.config.num_experts, dtype=weights.dtype)
    dense.scatter_(-1, indices, weights)
    return weights, indices, dense
