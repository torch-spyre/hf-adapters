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

"""Opt-in hints retain the model, TP hook, mesh and prefill guards."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers.models.granite.modeling_granite import GraniteMLP

from hf_adapters.hf_common import PaddedGatedMLP
from hf_adapters.hf_granite import (
    _can_force_mlp,
    _GraniteTP4PrefillMLP,
    _prepare_mlp,
)
from hf_adapters.spyre_tensor_parallel import (
    SpyreColwiseParallel,
    SpyreCpuColwiseParallel,
    SpyreRowwiseParallel,
)


@pytest.mark.parametrize(
    "colwise_style", [SpyreColwiseParallel, SpyreCpuColwiseParallel]
)
def test_granite_forced_mlp_default_and_guards(monkeypatch, colwise_style):
    config = SimpleNamespace(
        model_type="granite",
        hidden_size=4096,
        intermediate_size=12800,
        num_hidden_layers=40,
        hidden_act="silu",
        mlp_bias=False,
    )
    local = SimpleNamespace(**vars(config))
    with torch.device("meta"):
        mlp = GraniteMLP(local).to(dtype=torch.float16).eval()
        for name in ("gate_proj", "up_proj"):
            getattr(mlp, name).weight = torch.nn.Parameter(
                torch.empty(3200, 4096, dtype=torch.float16)
            )
        mlp.down_proj.weight = torch.nn.Parameter(
            torch.empty(4096, 3200, dtype=torch.float16)
        )
    mesh = SimpleNamespace(ndim=1, size=lambda: 4)
    for name, style in (
        ("gate_proj", colwise_style()),
        ("up_proj", colwise_style()),
        ("down_proj", SpyreRowwiseParallel()),
    ):
        # Match the Transformers loader, which binds this before hook setup.
        style.device_mesh = mesh
        style.prepare_module_tp(getattr(mlp, name), mesh)
        assert getattr(mlp, name)._spyre_tp_mesh is mesh
    monkeypatch.delenv("SPYRE_GRANITE_TP4_MLP_PADDING", raising=False)
    assert not _can_force_mlp(mlp, config)
    assert _prepare_mlp(mlp, config) is mlp
    monkeypatch.setenv("SPYRE_GRANITE_TP4_MLP_PADDING", "0")
    assert not _can_force_mlp(mlp, config)
    monkeypatch.setenv("SPYRE_GRANITE_TP4_MLP_PADDING", "1")
    monkeypatch.delenv("SPYRE_GRANITE_TP4_PREFILL_MLP", raising=False)
    assert _can_force_mlp(mlp, config)
    # The old hint option does not override the required padding opt-in.
    for value in ("0", "1", "invalid"):
        monkeypatch.setenv("SPYRE_GRANITE_TP4_PREFILL_MLP", value)
        assert _can_force_mlp(mlp, config)
    monkeypatch.delenv("SPYRE_GRANITE_TP4_PREFILL_MLP")
    for field, value in (
        ("model_type", "llama"),
        ("hidden_size", 2048),
        ("intermediate_size", 14336),
        ("num_hidden_layers", 32),
    ):
        other = SimpleNamespace(**vars(config))
        setattr(other, field, value)
        assert not _can_force_mlp(mlp, other)
    mesh.size = lambda: 1
    assert not _can_force_mlp(mlp, config)
    mesh.size = lambda: 4
    mesh.ndim = 2
    assert not _can_force_mlp(mlp, config)
    mesh.ndim = 1
    mlp.up_proj._spyre_tp_mesh = None
    assert not _can_force_mlp(mlp, config)
    mlp.up_proj._spyre_tp_mesh = SimpleNamespace(ndim=1, size=lambda: 4)
    assert not _can_force_mlp(mlp, config)
    mlp.up_proj._spyre_tp_mesh = mesh
    for dtype in (torch.bfloat16, torch.float32):
        mlp.to(dtype=dtype)
        assert not _can_force_mlp(mlp, config)
    mlp.half()
    mlp.up_proj.bias = torch.nn.Parameter(torch.empty(3200, device="meta"))
    assert not _can_force_mlp(mlp, config)
    mlp.up_proj.bias = None
    weight = mlp.up_proj.weight
    mlp.up_proj.weight = torch.nn.Parameter(
        torch.empty(3199, 4096, device="meta", dtype=torch.float16)
    )
    assert not _can_force_mlp(mlp, config)
    mlp.up_proj.weight = weight
    mlp.train()
    assert _prepare_mlp(mlp, config) is mlp
    mlp.eval()

    # Allocate only for the actual preparation; padding must retain TP hooks.
    mlp.to_empty(device="cpu")
    down = mlp.down_proj
    prepared = _prepare_mlp(mlp, config)
    assert type(prepared) is _GraniteTP4PrefillMLP
    assert prepared.down_proj is down
    assert prepared.gate_proj.weight.shape == (3328, 4096)
    assert prepared.down_proj.weight.shape == (4096, 3200)
    assert prepared.gate_proj._forward_hooks == mlp.gate_proj._forward_hooks
    x = SimpleNamespace(
        device=SimpleNamespace(type="spyre"), dtype=torch.float16, shape=(1, 512, 4096)
    )
    assert prepared._use_forced_division(x)
    for shape in (
        (1, 1, 4096),
        (2, 512, 4096),
        (1, 256, 4096),
        (1, 512, 2048),
        (512, 4096),
    ):
        x.shape = shape
        assert not prepared._use_forced_division(x)
    x.shape = (1, 512, 4096)
    x.device.type = "cpu"
    assert not prepared._use_forced_division(x)
    x.device.type = "spyre"
    x.dtype = torch.bfloat16
    assert not prepared._use_forced_division(x)
    x.dtype = torch.float16
    prepared.train()
    assert not prepared._use_forced_division(x)

    # Unsupported weight shapes retain their original module.
    tiny = SimpleNamespace(
        hidden_size=16, intermediate_size=20, hidden_act="silu", mlp_bias=False
    )
    tiny_mlp = GraniteMLP(tiny).half().eval()
    assert _prepare_mlp(tiny_mlp, config) is tiny_mlp


def test_granite_forced_mlp_forward_scopes_and_fallback(monkeypatch):
    # Small CPU weights let us check the exact forward calls without a device.
    config = SimpleNamespace(
        hidden_size=16, intermediate_size=20, hidden_act="silu", mlp_bias=False
    )
    mlp = _GraniteTP4PrefillMLP(GraniteMLP(config).half().eval(), 8)
    x = torch.randn(1, 3, 16, dtype=torch.float16)
    with torch.no_grad():
        expected = PaddedGatedMLP.forward(mlp, x)
    events = []

    @contextmanager
    def record_hint(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append(("exit",))

    hints = ModuleType("torch_spyre._inductor.propagate_hints")
    hints.spyre_hint = record_hint
    monkeypatch.setitem(sys.modules, hints.__name__, hints)
    # The real CPU guard must fall back and must not enter either hint scope.
    with torch.no_grad():
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)
    assert events == []

    # Eligibility is checked above; bypass only that guard for this CPU replay.
    monkeypatch.setattr(mlp, "_use_forced_division", lambda value: True)
    for name in ("gate_proj", "act_fn", "up_proj", "down_proj"):
        getattr(mlp, name).register_forward_pre_hook(
            lambda mod, args, name=name: events.append((name, tuple(args[0].shape)))
        )
    with torch.no_grad():
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)
    assert events == [
        (
            "enter",
            {
                "named_dims": [
                    "granite_mlp_batch",
                    "granite_mlp_rows",
                    "granite_mlp_channels",
                ],
                "work_div": {"granite_mlp_rows": 8, "granite_mlp_channels": 4},
            },
        ),
        ("gate_proj", (1, 3, 16)),
        ("act_fn", (1, 3, 24)),
        ("up_proj", (1, 3, 16)),
        ("exit",),
        (
            "enter",
            {
                "named_dims": [
                    "granite_mlp_batch",
                    "granite_mlp_rows",
                    "granite_mlp_hidden",
                ],
                "work_div": {"granite_mlp_rows": 8, "granite_mlp_hidden": 4},
            },
        ),
        ("down_proj", (1, 3, 20)),
        ("exit",),
    ]
