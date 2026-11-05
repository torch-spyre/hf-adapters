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
from transformers import AutoModelForCausalLM, GraniteConfig
from transformers.models.granite.modeling_granite import GraniteMLP

from hf_adapters import hf_common, hf_granite
from hf_adapters.hf_common import PaddedGatedMLP
from hf_adapters.hf_granite import _can_pad_mlp
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_CPU_COLWISE,
    SPYRE_ROWWISE,
    SpyreColwiseGatherOutputParallel,
    SpyreColwiseParallel,
    SpyreCpuColwiseParallel,
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
    # The width helper remains general; automatic adaptation is TP4-only.
    assert not _can_pad_mlp(mlp)
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
        assert padded.state_dict()[f"{name}.weight"].shape == (24, 16)
        # A discarded NaN is stronger than assuming zero-valued padding.
        with torch.no_grad():
            widened.weight[20:].fill_(float("nan"))
    actual = padded(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 3, 16)
    assert seen[0].shape[-1] == 20 and seen[0].stride()[-2:] == (24, 1)


@pytest.mark.parametrize(
    "opt_in,full_width,local_width,supported",
    [
        (None, 12800, 3200, False),
        ("1", 12800, 3200, True),
        ("1", 12800, 1600, False),
        ("1", 3200, 3200, False),
    ],
)
def test_automatic_padding_is_limited_to_granite_tp4(
    monkeypatch, opt_in, full_width, local_width, supported
):
    monkeypatch.delenv("SPYRE_GRANITE_TP4_MLP_PADDING", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("SPYRE_GRANITE_TP4_MLP_PADDING", opt_in)
    config = SimpleNamespace(
        hidden_size=16, intermediate_size=full_width, hidden_act="silu", mlp_bias=False
    )
    mlp = GraniteMLP(config).half().eval()
    for name in ("gate_proj", "up_proj"):
        proj = getattr(mlp, name)
        proj.weight = torch.nn.Parameter(proj.weight[:local_width].detach().clone())
        proj.out_features = local_width
    mlp.down_proj.weight = torch.nn.Parameter(
        mlp.down_proj.weight[:, :local_width].detach().clone()
    )
    mlp.down_proj.in_features = local_width
    assert _can_pad_mlp(mlp) is supported
    if not supported:
        return
    # Install the real styles without a process group; eligibility never runs
    # their communication hooks. Both unsupported variants must keep the MLP.
    for gate, down_style, style_supported in (
        (SpyreColwiseParallel(), SpyreRowwiseParallel(), True),
        (SpyreCpuColwiseParallel(), SpyreRowwiseParallel(), True),
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
            assert proj._spyre_tp_state[0] is style
            assert not hasattr(proj, "_spyre_mlp_width_role")
        assert _can_pad_mlp(hooked) is style_supported
        if style_supported:
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


@pytest.mark.parametrize(
    "opt_in,full_width,tp_size,staged",
    [
        (None, 12800, 4, False),
        ("1", 12800, 4, True),
        ("1", 12800, 8, False),
        ("1", 6400, 4, False),
    ],
)
def test_granite_tp_load_stages_only_qualified_gate_up(
    tmp_path, monkeypatch, opt_in, full_width, tp_size, staged
):
    monkeypatch.delenv("SPYRE_GRANITE_TP4_MLP_PADDING", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("SPYRE_GRANITE_TP4_MLP_PADDING", opt_in)
    config = GraniteConfig(
        hidden_size=16,
        intermediate_size=full_width,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(hf_common, "_resolve_tp_size", lambda: tp_size)
    plan, _ = hf_common._resolve_tp_plan(
        tmp_path, AutoModelForCausalLM, "auto", hf_granite
    )
    for projection in ("gate_proj", "up_proj"):
        assert plan[f"model.layers.*.mlp.{projection}"] == (
            SPYRE_CPU_COLWISE if staged else SPYRE_COLWISE
        )
    assert plan["model.layers.*.self_attn.q_proj"] == SPYRE_COLWISE
    if not staged:
        return

    explicit, _ = hf_common._resolve_tp_plan(
        tmp_path,
        AutoModelForCausalLM,
        {
            "model.layers.*.mlp.gate_proj": "colwise",
            "model.layers.*.mlp.up_proj": "rowwise",
        },
        hf_granite,
    )
    assert explicit["model.layers.*.mlp.gate_proj"] == SPYRE_CPU_COLWISE
    assert explicit["model.layers.*.mlp.up_proj"] == SPYRE_ROWWISE

    # Use the real HF sharder. Staging and padding must not call a device DMA.
    def unexpected_dma(*args, **kwargs):
        pytest.fail("CPU staging copied a parameter to Spyre before padding")

    monkeypatch.setattr(
        "hf_adapters.spyre_tensor_parallel._copy_linear", unexpected_dma
    )
    mesh = SimpleNamespace(shape=(4,), size=lambda: 4)
    mlp = GraniteMLP(config).half().eval()
    for projection in ("gate_proj", "up_proj"):
        proj = getattr(mlp, projection)
        style = SpyreCpuColwiseParallel(
            device_mesh=mesh, rank=1, empty_param=torch.empty(3200, 16)
        )
        original = proj.weight.detach().clone()
        shard = style.shard_tensor(original, device="spyre", dtype=torch.float16)
        torch.testing.assert_close(shard, original[3200:6400], rtol=0, atol=0)
        assert shard.device.type == "cpu"
        proj.weight = torch.nn.Parameter(shard)
        proj.out_features = 3200
        style.prepare_module_tp(proj, None)
    mlp.down_proj.weight = torch.nn.Parameter(mlp.down_proj.weight[:, 3200:6400])
    mlp.down_proj.in_features = 3200
    assert _can_pad_mlp(mlp)
    padded = PaddedGatedMLP(mlp, 256)
    for projection in ("gate_proj", "up_proj"):
        proj = getattr(padded, projection)
        assert proj.weight.device.type == "cpu" and proj.weight.shape == (3328, 16)
        assert torch.count_nonzero(proj.weight[3200:]) == 0
        assert proj._spyre_tp_state[1:] == (
            tuple(proj._forward_pre_hooks.items()),
            tuple(proj._forward_hooks.items()),
        )
