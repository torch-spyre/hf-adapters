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

"""Compile the selected-expert decode loop and check its Spyre numerics."""

import pytest
import torch
import torch_spyre  # noqa: F401
from torch_spyre.model_utils import (
    dma_moe_expert_weight_to_spyre,
    dma_moe_per_expert_scale_to_spyre,
)

from hf_adapters.hf_common import DEVICE, moe_decode_selected_experts, moe_topk
from tests._moe_decode_helpers import decode_inputs, decode_reference


@pytest.mark.parametrize("tile", [32, 2], ids=["small", "multitile"])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("scaled", [False, True])
def test_moe_decode_spyre(tile, activation, scaled):
    torch._dynamo.reset()
    tokens = 1
    x, weights, indices, gate, up, down, scale = decode_inputs(
        tokens, scaled, dtype=torch.float16
    )
    probabilities = torch.zeros(tokens, gate.shape[0], dtype=x.dtype)
    probabilities.scatter_(1, indices, weights)
    x, probabilities = x.to(DEVICE), probabilities.to(DEVICE)
    gate, up, down = (
        dma_moe_expert_weight_to_spyre(weight, device=DEVICE)
        for weight in (gate, up, down)
    )
    if scale is not None:
        scale = dma_moe_per_expert_scale_to_spyre(scale[:, 0], device=DEVICE)
    weights, indices = torch.topk(probabilities.cpu(), 8, dim=-1)
    args = (x, weights, indices, gate, up, down, scale)
    # Account for dl16 rounding at the device transfer boundary.
    reference_args = tuple(arg.cpu() if arg is not None else None for arg in args)
    expected = decode_reference(*reference_args, activation)

    def decode(x, probabilities, gate, up, down, scale):
        # Keep routing in the compiled region, as in the model adapters. Spyre
        # topk returns floating-point indices that need stick widening.
        weights, indices = moe_topk(probabilities, 8)
        return moe_decode_selected_experts(
            x,
            weights,
            indices,
            gate,
            up,
            down,
            top_k=8,
            tile=tile,
            stick_size=64,
            activation=activation,
            per_expert_scale_stick=scale,
        )

    compiled = torch.compile(decode, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = compiled(x, probabilities, gate, up, down, scale).cpu()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
