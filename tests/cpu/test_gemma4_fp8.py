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

"""CPU coverage for the opt-in Gemma 4 MoE W8A8 fp8 linears."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters import _gemma4_fp8 as fp8
from hf_adapters.hf_gemma4 import _gemma4_rms_norm

H, D, HEADS, KV_HEADS, INTER = 256, 128, 2, 1, 192  # INTER pads to 256


def _linear(n, k, seed):
    linear = nn.Linear(k, n, bias=False)
    generator = torch.Generator().manual_seed(seed)
    linear.weight = nn.Parameter(
        torch.randn(n, k, generator=generator) / k**0.5, requires_grad=False
    )
    return linear


def _block(with_v=True):
    attn = SimpleNamespace(
        q_proj=_linear(HEADS * D, H, 1),
        k_proj=_linear(KV_HEADS * D, H, 2),
        v_proj=_linear(KV_HEADS * D, H, 3) if with_v else None,
        o_proj=_linear(H, HEADS * D, 4),
    )
    mlp = SimpleNamespace(
        gate_proj=_linear(INTER, H, 5),
        up_proj=_linear(INTER, H, 6),
        down_proj=_linear(H, INTER, 7),
    )
    return SimpleNamespace(self_attn=attn, mlp=mlp)


def _x(tokens, width=H, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, tokens, width, generator=generator)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_parts_are_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv("SPYRE_GEMMA4_FP8", raising=False)
    assert fp8.fp8_parts() == ()
    assert fp8.fp8_parts("attn, lm_head") == ("attn", "lm_head")
    with pytest.raises(ValueError, match="unknown"):
        fp8.fp8_parts("attn,experts")


def test_widths_that_are_not_whole_fp8_sticks_are_refused():
    with pytest.raises(ValueError, match="multiples of 128"):
        fp8._fuse([("o", _linear(H, 192, 0))], "cpu", per_row=False, fold=True)


def test_weight_codes_round_trip_within_e4m3_precision():
    w = _linear(256, 512, 0).weight
    codes, scale = fp8.quantize_weight(w)
    assert codes.dtype is torch.float8_e4m3fn
    assert _rel(codes.float() * scale.float()[:, None], w) < 0.04
    codes, scale = fp8.quantize_weight(w, per_row=True)
    assert scale.shape == (256,)
    assert _rel(codes.float() * scale.float()[:, None], w) < 0.04


@pytest.mark.parametrize("tokens", [1, 2, 16])
def test_rescaled_output_matches_the_fp32_linear(tokens):
    linear = _linear(256, H, 0)
    x = _x(tokens)
    out = fp8._fuse([("gate", linear)], "cpu", per_row=False, fold=True)["gate"](x)
    assert out.shape == (1, tokens, 256)
    assert _rel(out, F.linear(x, linear.weight)) < 0.06


@pytest.mark.parametrize("tokens", [1, 16])
def test_folded_outputs_match_after_their_rmsnorm(tokens):
    # q / k / v / o / down return raw codes products: a per-row factor that
    # the consumer RMSNorm removes.
    linear = _linear(256, H, 0)
    x = _x(tokens)
    raw = fp8._fuse([("o", linear)], "cpu", per_row=False, fold=True)["o"](x)
    ref = F.linear(x, linear.weight)
    assert _rel(raw, ref) > 0.5  # really not rescaled
    normed = _gemma4_rms_norm(raw, None, 1e-6)
    assert _rel(normed, _gemma4_rms_norm(ref, None, 1e-6)) < 0.06


@pytest.mark.parametrize("with_v", [True, False])
def test_converted_attention_and_mlp_match_fp16_up_to_rounding(with_v):
    block = _block(with_v)
    ref = {
        name: getattr(block.self_attn, name).weight.clone()
        for name in ("q_proj", "k_proj", "o_proj") + (("v_proj",) if with_v else ())
    }
    mlp_ref = {
        n: getattr(block.mlp, n).weight.clone()
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    fp8.convert_block(block, "cpu", ("attn", "dense"))
    attn = block.self_attn
    assert isinstance(attn.q_proj, fp8.Fp8Linear)
    assert attn.k_proj.weight is None  # followers hold no weight
    x = _x(4)
    q = attn.q_proj(x)
    k = attn.k_proj(x)
    for name, got in (("q_proj", q), ("k_proj", k)):
        want = F.linear(x, ref[name])
        assert (
            _rel(_gemma4_rms_norm(got, None, 1e-6), _gemma4_rms_norm(want, None, 1e-6))
            < 0.06
        )
    if with_v:
        v = attn.v_proj(x)
        assert (
            _rel(
                _gemma4_rms_norm(v, None, 1e-6),
                _gemma4_rms_norm(F.linear(x, ref["v_proj"]), None, 1e-6),
            )
            < 0.06
        )
    assert getattr(x, "_fp8_out", None) is None  # the last follower cleared it

    mlp = block.mlp
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    assert gate.shape[-1] == 256  # 192 padded to two fp8 sticks
    torch.testing.assert_close(gate[..., INTER:], torch.zeros_like(gate[..., INTER:]))
    act = F.gelu(gate, approximate="tanh") * up
    down = mlp.down_proj(act)
    want = F.linear(
        F.gelu(F.linear(x, mlp_ref["gate_proj"]), approximate="tanh")
        * F.linear(x, mlp_ref["up_proj"]),
        mlp_ref["down_proj"],
    )
    assert (
        _rel(_gemma4_rms_norm(down, None, 1e-6), _gemma4_rms_norm(want, None, 1e-6))
        < 0.08
    )


def test_a_follower_before_its_leader_is_an_error():
    block = _block()
    fp8.convert_block(block, "cpu", ("attn",))
    with pytest.raises(RuntimeError, match="before its leader"):
        block.self_attn.k_proj(_x(2))


def test_lm_head_is_replaced_not_kept_and_rescaled_before_the_softcap():
    head = _linear(512, H, 9)
    w = head.weight.clone()
    model = nn.Module()
    model.lm_head = head
    model.layer = nn.Linear(H, H, bias=False)
    cap = 30.0

    def softcap(logits):
        return torch.tanh(logits / cap) * cap

    replaced, fp8_bytes = fp8.convert(model, [], ("lm_head",), "cpu", softcap)
    assert replaced == w.numel() * w.element_size()
    assert fp8_bytes == w.numel()  # one byte per code
    assert not any(p is head.weight for p in model.parameters())
    x = _x(3)
    logits = model._spyre_lm_head_forward(x)
    assert _rel(logits, softcap(F.linear(x, w))) < 0.06
    assert logits.argmax(-1).tolist() == softcap(F.linear(x, w)).argmax(-1).tolist()


def test_fp8_weights_survive_the_device_dtype_cast():
    block = _block()
    fp8.convert_block(block, "cpu", ("attn",))
    leader = block.self_attn.q_proj
    leader.to(dtype=torch.float16)
    assert leader.weight.dtype is torch.float8_e4m3fn
