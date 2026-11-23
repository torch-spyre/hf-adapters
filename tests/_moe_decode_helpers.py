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

"""Small, checkpoint-free inputs and a dense reference for MoE decode tests."""

import torch
import torch.nn.functional as F


def decode_inputs(tokens, scaled, dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    experts, hidden, intermediate, top_k, stick = 16, 128, 192, 8, 64

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    x = randn(tokens, hidden) * 0.5
    weights = torch.softmax(randn(tokens, top_k), dim=-1)
    indices = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)]
    )
    gate = randn(experts, hidden, intermediate) / hidden**0.5
    up = randn(experts, hidden, intermediate) / hidden**0.5
    down = randn(experts, intermediate, hidden) / intermediate**0.5
    scale = None
    if scaled:
        scale = torch.linspace(0.5, 1.5, experts, dtype=dtype)
        scale = scale[:, None].expand(-1, stick).contiguous()
    return x, weights, indices, gate, up, down, scale


def decode_reference(x, weights, indices, gate, up, down, scale, activation):
    # Evaluate all experts with regular matmuls, then route their outputs.
    # This does not use the selected-weight gathers/BMMs of the adapter.
    gate_out = x.float() @ gate.float()
    up_out = x.float() @ up.float()
    if activation == "silu":
        activated = F.silu(gate_out) * up_out
    else:
        activated = F.gelu(gate_out, approximate="tanh") * up_out
    outputs = activated @ down.float()
    if scale is not None:
        outputs = outputs * scale[:, :1, None].float()
    selected = outputs[indices.long(), torch.arange(x.shape[0])[:, None]]
    return (selected * weights[..., None].float()).sum(dim=1)
