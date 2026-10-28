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

import torch
from torch import nn

from hf_adapters.hf_common import (
    StandardGQABlock,
    make_standard_gqa_block,
    prepare_standard_gqa_blocks,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.o_proj = nn.Identity()
        self.head_dim = 1
        self.scaling = 1.0


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.Identity()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()


def test_standard_gqa_factories_compile_each_complete_block_once(monkeypatch):
    compile_calls = []

    def fake_compile(module, **kwargs):
        result = object()
        compile_calls.append((module, kwargs, result))
        return result

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled = make_standard_gqa_block(_Layer())

    assert len(compile_calls) == 1
    module, kwargs, result = compile_calls[0]
    assert isinstance(module, StandardGQABlock)
    assert kwargs == {"dynamic": False}
    assert compiled is result

    compile_calls.clear()
    layers = nn.ModuleList([_Layer(), _Layer()])
    compiled_blocks = prepare_standard_gqa_blocks(layers)

    assert len(compile_calls) == len(layers) == 2
    for layer, (module, kwargs, result), compiled_block in zip(
        layers, compile_calls, compiled_blocks, strict=True
    ):
        assert layer is module
        assert isinstance(module, StandardGQABlock)
        assert kwargs == {"dynamic": False}
        assert compiled_block is result
