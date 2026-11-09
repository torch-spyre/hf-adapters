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

"""CPU coverage for the shared selected-expert decode helper."""

import pytest
import torch

from hf_adapters.hf_common import moe_decode_selected_experts
from tests._moe_decode_helpers import decode_inputs, decode_reference


@pytest.mark.parametrize("tokens", [1, 5, 8])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("scaled", [False, True])
def test_moe_decode(tokens, activation, scaled):
    args = decode_inputs(tokens, scaled)
    x, weights, indices, gate, up, down, scale = args
    expected = decode_reference(*args, activation)
    actual = moe_decode_selected_experts(
        x,
        weights,
        indices,
        gate,
        up,
        down,
        top_k=8,
        tile=32,
        stick_size=64,
        activation=activation,
        per_expert_scale_stick=scale,
    )
    torch.testing.assert_close(actual, expected)
