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

"""CPU numerics for the decode expert combine.

Spyre cannot view ``[T, K]`` as ``[T*K, 1, 1]`` or broadcast a one-lane
slice of a stick. The combine repeats each scale across a full stick and
multiplies stick-by-stick. These checks lock that this matches the
per-expert scalar multiply at Gemma-4's batch=4, K=8 shape.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import moe_decode_selected_experts


def _reference(x, weights, expert_indices, gate, up, down, activation, scale_stick):
    outputs = []
    for token in range(x.shape[0]):
        acc = torch.zeros(x.shape[-1], dtype=x.dtype)
        for slot in range(weights.shape[1]):
            expert = int(expert_indices[token, slot])
            gated = x[token] @ gate[expert]
            projected = x[token] @ up[expert]
            if activation == "silu":
                hidden = F.silu(gated) * projected
            else:
                hidden = F.gelu(gated, approximate="tanh") * projected
            mixed = hidden @ down[expert]
            mixed = mixed * weights[token, slot]
            if scale_stick is not None:
                mixed = mixed * scale_stick[expert, 0]
            acc = acc + mixed
        outputs.append(acc)
    return torch.stack(outputs)


def test_decode_combine_matches_per_expert_multiply():
    torch.manual_seed(0)
    experts, tokens, top_k = 16, 4, 8
    # Hidden spans two sticks so the scale is broadcast across sticks, not
    # only the single-stick case.
    hidden, intermediate, stick = 128, 64, 64
    x = torch.randn(tokens, hidden)
    gate = torch.randn(experts, hidden, intermediate)
    up = torch.randn(experts, hidden, intermediate)
    down = torch.randn(experts, intermediate, hidden)
    weights = torch.rand(tokens, top_k)
    weights = weights / weights.sum(-1, keepdim=True)
    expert_indices = torch.randint(0, experts, (tokens, top_k))
    scale = torch.randn(experts)
    # Device DMA stores the same scale on every lane of the expert stick.
    scale_stick = scale[:, None].expand(-1, stick).contiguous()

    for activation, scale_arg in (
        ("gelu_tanh", scale_stick),
        ("silu", None),
    ):
        got = moe_decode_selected_experts(
            x,
            weights,
            expert_indices,
            gate,
            up,
            down,
            top_k,
            32,
            stick,
            activation,
            per_expert_scale_stick=scale_arg,
        )
        expected = _reference(
            x,
            weights,
            expert_indices,
            gate,
            up,
            down,
            activation,
            scale_arg,
        )
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    # Qwen decode already carries the routing weight on every stick lane.
    widened = weights.unsqueeze(-1).expand(-1, -1, stick).contiguous()
    got = moe_decode_selected_experts(
        x,
        widened,
        expert_indices,
        gate,
        up,
        down,
        top_k,
        32,
        stick,
        "silu",
    )
    expected = _reference(x, weights, expert_indices, gate, up, down, "silu", None)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
