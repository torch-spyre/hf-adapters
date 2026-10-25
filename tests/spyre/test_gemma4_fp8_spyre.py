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

"""Compile the opt-in Gemma 4 fp8 linears and check them against CPU emulation.

Usage::

    pytest -s -vvv tests/spyre/test_gemma4_fp8_spyre.py
"""

import pytest
import torch
import torch.nn as nn
import torch_spyre  # noqa: F401

from hf_adapters import _gemma4_fp8 as fp8
from hf_adapters.hf_common import DEVICE
from hf_adapters.hf_gemma4 import _gemma4_rms_norm

pytestmark = pytest.mark.requires_spyre

# Gemma 4 26B-A4B widths: hidden 2816, sliding q|k|v 8192 + 2 * 2048,
# dense gate|up 2 * 2176 (2112 padded), down 2176 -> 2816.
SHAPES = {
    "qkv": ([("q", 4096), ("k", 2048), ("v", 2048)], 2816),
    "gate_up": ([("gate", 2112), ("up", 2112)], 2816),
    "down": ([("down", 2816)], 2176),
}


def _linears(members, k, seed):
    generator = torch.Generator().manual_seed(seed)
    out = []
    for name, n in members:
        linear = nn.Linear(k, n, bias=False)
        w = torch.randn(n, k, generator=generator) / k**0.5
        linear.weight = nn.Parameter(w.half(), requires_grad=False)
        out.append((name, linear))
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("group", list(SHAPES))
def test_fp8_group_spyre(group, tokens):
    torch._dynamo.reset()
    members, k = SHAPES[group]
    pad = fp8._dense_pad(2112) if group == "gate_up" else None
    host = fp8._fuse(
        _linears(members, k, 1), "cpu", per_row=False, fold=True, pad_rows=pad
    )
    dev = fp8._fuse(
        _linears(members, k, 1), DEVICE, per_row=False, fold=True, pad_rows=pad
    )
    names = [name for name, _ in members]
    x = (torch.randn(1, tokens, k, generator=torch.Generator().manual_seed(2))).half()

    def run(mods, x):
        return tuple(mods[name](x) for name in names)

    expected = run(host, x)
    with torch.no_grad():
        actual = torch.compile(lambda x: run(dev, x), fullgraph=True, dynamic=False)(
            x.to(DEVICE)
        )
    for name, a, e in zip(names, actual, expected):
        a = a.cpu()
        assert a.shape == e.shape, name
        assert torch.isfinite(a.float()).all(), name
        if name in fp8._FOLDED:
            a = _gemma4_rms_norm(a.float(), None, 1e-6)
            e = _gemma4_rms_norm(e.float(), None, 1e-6)
        # The device and the emulation share codes; they differ in fp16
        # accumulation and in the device's activation rounding.
        assert _rel(a, e) < 0.03, (name, _rel(a, e))
