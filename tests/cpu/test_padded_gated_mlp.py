# Copyright 2025 The Torch-Spyre Authors.
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

"""Explicit width preserves local channels, hooks and the logical reduction."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.granite.modeling_granite import GraniteMLP

from hf_adapters.hf_common import PaddedGatedMLP
from hf_adapters.hf_granite import _can_pad_mlp
from hf_adapters.spyre_tensor_parallel import (
    SpyreColwiseGatherOutputParallel,
    SpyreColwiseParallel,
    SpyreRowwiseParallel,
    SpyreRowwiseSplitInputParallel,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padded_gated_mlp_preserves_prefix_and_excludes_nan_tail(dtype):
    torch.random.default_generator.manual_seed(17)
    config = SimpleNamespace(
        hidden_size=16, intermediate_size=20, hidden_act="silu", mlp_bias=True
    )
    mlp = GraniteMLP(config).to(dtype=dtype).eval()
    assert _can_pad_mlp(mlp)
    # Install the real styles without a process group; eligibility never runs
    # their communication hooks. Both unsupported variants must keep the MLP.
    for gate, down_style, supported in (
        (SpyreColwiseParallel(), SpyreRowwiseParallel(), True),
        (SpyreColwiseGatherOutputParallel(), SpyreRowwiseParallel(), False),
        (SpyreColwiseParallel(), SpyreRowwiseSplitInputParallel(), False),
    ):
        hooked = copy.deepcopy(mlp)
        for proj, style in (
            (hooked.gate_proj, gate),
            (hooked.up_proj, SpyreColwiseParallel()),
            (hooked.down_proj, down_style),
        ):
            style.prepare_module_tp(proj, None)
        assert _can_pad_mlp(hooked) is supported
        if supported:
            # Replacing one hook preserves its count and role but invalidates
            # the exact registrations certified by TP preparation.
            key = next(iter(hooked.gate_proj._forward_hooks))
            hooked.gate_proj._forward_hooks[key] = lambda mod, args, output: output
            assert not _can_pad_mlp(hooked)
    hook = mlp.register_forward_hook(lambda mod, args, output: output)
    assert not _can_pad_mlp(mlp)
    hook.remove()
    hook = mlp.gate_proj.register_forward_hook(lambda mod, args, output: output)
    assert not _can_pad_mlp(mlp)
    hook.remove()
    reference = copy.deepcopy(mlp)
    seen = []
    down = mlp.down_proj
    down.register_forward_pre_hook(lambda mod, args: seen.append(args[0]))
    padded = PaddedGatedMLP(mlp, 8)
    assert padded.down_proj is down and down.weight.shape == (16, 20)
    x = torch.randn(2, 3, 16, dtype=dtype)
    expected = padded(x)
    # Check the original formula in fp32: fp16/bf16 GEMM rounding can change
    # with the output width. Tail isolation below remains bitwise in each dtype.
    torch.testing.assert_close(
        copy.deepcopy(padded).float()(x.float()),
        copy.deepcopy(reference).float()(x.float()),
    )
    for name in ("gate_proj", "up_proj"):
        original, widened = getattr(reference, name), getattr(padded, name)
        assert torch.equal(widened.weight[:20], original.weight)
        assert torch.equal(widened.bias[:20], original.bias)
        assert widened.weight.dtype == widened.bias.dtype == dtype
        # A discarded NaN is stronger than assuming zero-valued padding.
        with torch.no_grad():
            widened.weight[20:].fill_(float("nan"))
    actual = padded(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 3, 16)
    assert seen[0].shape[-1] == 20 and seen[0].stride()[-2:] == (24, 1)
