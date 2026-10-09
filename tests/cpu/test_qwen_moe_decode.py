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

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hf_adapters.hf_qwen3_5_moe import _decode_routed_experts


def test_qwen_moe_decode_matches_weighted_selected_experts():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, generator=generator)
    experts = SimpleNamespace(
        gate_proj=torch.randn(3, 4, 8, generator=generator),
        up_proj=torch.randn(3, 4, 8, generator=generator),
        down_proj=torch.randn(3, 8, 4, generator=generator),
    )
    mlp = SimpleNamespace(experts=experts)
    weights = torch.tensor([[0.25, 0.75]])
    selected = torch.tensor([[2, 0]])
    stick_size = 64

    actual = _decode_routed_experts(
        x=x,
        weights=weights[..., None].expand(-1, -1, stick_size).contiguous(),
        expert_indices=selected[..., None].expand(-1, -1, stick_size).contiguous(),
        mlp=mlp,
        top_k=2,
        stick_size=stick_size,
    )

    expected = torch.zeros_like(x)
    for weight, expert in zip(weights[0], selected[0]):
        gate = F.linear(x, experts.gate_proj[expert].T)
        up = F.linear(x, experts.up_proj[expert].T)
        output = F.linear(F.silu(gate) * up, experts.down_proj[expert].T)
        expected += weight * output
    torch.testing.assert_close(actual, expected)
