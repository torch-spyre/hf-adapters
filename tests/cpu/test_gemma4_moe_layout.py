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

import sys
import types
from types import SimpleNamespace

import torch

from hf_adapters import hf_gemma4_moe


def test_prepare_experts_tiles_only_down_projection(monkeypatch):
    calls = []

    def record_move(weight, *, output_stick_tile=None):
        calls.append((tuple(weight.shape), output_stick_tile))
        return weight

    monkeypatch.setattr(hf_gemma4_moe, "_move_expert_weight", record_move)
    experts = SimpleNamespace(
        gate_up_proj=torch.randn(2, 8, 3),
        down_proj=torch.randn(2, 3, 4),
    )

    hf_gemma4_moe._prepare_experts(experts)

    assert calls == [
        ((2, 3, 4), None),
        ((2, 3, 4), None),
        ((2, 4, 3), hf_gemma4_moe._MOE_DOWN_OUTPUT_STICK_TILE),
    ]


def test_move_expert_weight_retries_default_layout(monkeypatch):
    calls = []
    fallback = object()

    def fake_dma(weight, *, output_stick_tile=None):
        calls.append(output_stick_tile)
        return None if output_stick_tile is not None else fallback

    torch_spyre = types.ModuleType("torch_spyre")
    torch_spyre.__path__ = []
    model_utils = types.ModuleType("torch_spyre.model_utils")
    model_utils.dma_moe_expert_weight_to_spyre = fake_dma
    monkeypatch.setitem(sys.modules, "torch_spyre", torch_spyre)
    monkeypatch.setitem(sys.modules, "torch_spyre.model_utils", model_utils)

    moved = hf_gemma4_moe._move_expert_weight(
        torch.randn(2, 4, 3), output_stick_tile=22
    )

    assert moved is fallback
    assert calls == [22, None]


def test_move_expert_weight_supports_older_torch_spyre(monkeypatch):
    calls = []
    fallback = object()

    def fake_dma(weight):
        calls.append(weight)
        return fallback

    torch_spyre = types.ModuleType("torch_spyre")
    torch_spyre.__path__ = []
    model_utils = types.ModuleType("torch_spyre.model_utils")
    model_utils.dma_moe_expert_weight_to_spyre = fake_dma
    monkeypatch.setitem(sys.modules, "torch_spyre", torch_spyre)
    monkeypatch.setitem(sys.modules, "torch_spyre.model_utils", model_utils)
    weight = torch.randn(2, 4, 3)

    moved = hf_gemma4_moe._move_expert_weight(weight, output_stick_tile=22)

    assert moved is fallback
    assert calls == [weight]
