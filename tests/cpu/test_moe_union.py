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

"""CPU coverage for the opt-in union-capped multi-token MoE form."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    moe_decode_selected_experts,
    moe_topk,
    moe_union_cap,
    moe_union_selected_experts,
    prepare_moe_expert_weights,
)

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, STICK = 32, 256, 128, 4, 64


def _experts(chunks, dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    gate_up = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
    down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
    weights = SimpleNamespace(
        gate_up_proj=(gate_up / HIDDEN**0.5).to(dtype),
        down_proj=(down / INTERMEDIATE**0.5).to(dtype),
    )
    raw = (weights.gate_up_proj.clone(), weights.down_proj.clone())
    prepare_moe_expert_weights(weights, pad_to_multiple=STICK, decode_chunks=chunks)
    return weights, raw


def _inputs(tokens, skew, dtype=torch.float64):
    generator = torch.Generator().manual_seed(100 + tokens)
    x = torch.randn(tokens, HIDDEN, generator=generator).to(dtype) * 0.5
    logits = torch.randn(tokens, EXPERTS, generator=generator).to(dtype)
    # A skew concentrates routing on a few experts, so tokens share them.
    logits = logits + skew * torch.linspace(0, 1, EXPERTS, dtype=dtype)
    scale = torch.linspace(0.5, 1.5, EXPERTS, dtype=dtype)
    return x, logits.softmax(-1), scale[:, None].expand(-1, STICK).contiguous()


def _constants(dtype=torch.float64):
    expert_ids = torch.arange(EXPERTS, dtype=dtype)[:, None].contiguous()
    return expert_ids, torch.eye(STICK, dtype=dtype)


def _union(x, probs, weights, cap, chunks, scale, activation="gelu_tanh"):
    expert_ids, identity = _constants(x.dtype)
    return moe_union_selected_experts(
        x,
        probs,
        weights.decode_gate_proj,
        weights.decode_up_proj,
        weights.decode_down_proj,
        TOP_K,
        cap,
        32,
        STICK,
        activation,
        expert_ids=expert_ids,
        slot_identity=None,
        per_expert_scale_stick=scale,
        decode_chunks=chunks,
        chunk_scale=weights.decode_chunk_scale,
        chunk_offsets=weights.decode_chunk_offsets,
    )


def _per_token(x, probs, weights, chunks, scale, activation="gelu_tanh"):
    rows = []
    for t in range(x.shape[0]):
        w, idx = moe_topk(probs[t : t + 1], TOP_K)
        w = w / w.sum(-1, keepdim=True)
        rows.append(
            moe_decode_selected_experts(
                x[t : t + 1],
                w,
                idx,
                weights.decode_gate_proj,
                weights.decode_up_proj,
                weights.decode_down_proj,
                TOP_K,
                32,
                STICK,
                activation,
                per_expert_scale_stick=scale,
                decode_chunks=chunks,
                chunk_scale=weights.decode_chunk_scale,
                chunk_offsets=weights.decode_chunk_offsets,
            )
        )
    return torch.cat(rows)


def _dense_reference(x, route, raw, scale, activation="gelu_tanh"):
    """Every expert with plain matmuls, weighted by a dense ``[T, E]`` route."""
    gate_up, down = raw
    gate, up = gate_up[:, :INTERMEDIATE], gate_up[:, INTERMEDIATE:]
    g = torch.einsum("th,emh->etm", x, gate)
    u = torch.einsum("th,emh->etm", x, up)
    act = F.silu(g) * u if activation == "silu" else F.gelu(g, approximate="tanh") * u
    out = torch.einsum("etm,ehm->eth", act, down) * scale[:, :1, None]
    return torch.einsum("te,eth->th", route, out)


def _topk_route(probs):
    w, idx = probs.topk(TOP_K, -1)
    return torch.zeros_like(probs).scatter(-1, idx, w / w.sum(-1, keepdim=True))


def _capped_route(probs, cap):
    """Independent statement of the slot rule: top-1s, then route mass."""
    route = _topk_route(probs)
    top1 = set(probs.argmax(-1).tolist())
    mass = route.sum(0)
    rest = [e for e in mass.argsort(descending=True).tolist() if e not in top1]
    slots = sorted(top1) + rest[: cap - len(top1)]
    keep = torch.zeros(EXPERTS, dtype=probs.dtype)
    keep[slots] = 1
    kept = route * keep
    return kept / kept.sum(-1, keepdim=True)


@pytest.mark.parametrize("chunks", [1, 4])
@pytest.mark.parametrize("tokens", [2, 4, 8])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_union_matches_the_per_token_form_when_the_cap_covers_the_union(
    chunks, tokens, activation
):
    weights, _ = _experts(chunks)
    x, probs, scale = _inputs(tokens, skew=0.0)
    cap = EXPERTS  # covers any union
    actual = _union(x, probs, weights, cap, chunks, scale, activation)
    expected = _per_token(x, probs, weights, chunks, scale, activation)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("tokens", [2, 3, 4])
def test_union_is_exact_at_a_small_cap_when_tokens_times_top_k_fit(tokens):
    weights, raw = _experts(4)
    x, probs, scale = _inputs(tokens, skew=4.0)
    cap = 16  # tokens * TOP_K <= 16
    actual = _union(x, probs, weights, cap, 4, scale)
    expected = _dense_reference(x, _topk_route(probs), raw, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("cap", [8, 16])
@pytest.mark.parametrize("tokens", [8])
def test_a_cap_below_the_union_keeps_every_top1_expert_and_renormalizes(cap, tokens):
    weights, raw = _experts(4)
    x, probs, scale = _inputs(tokens, skew=0.0)
    w, idx = probs.topk(TOP_K, -1)
    assert torch.unique(idx).numel() > cap  # the cap really drops experts
    actual = _union(x, probs, weights, cap, 4, scale)
    expected = _dense_reference(x, _capped_route(probs, cap), raw, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_more_tokens_than_slots_are_rejected():
    weights, _ = _experts(1)
    x, probs, scale = _inputs(9, skew=0.0)
    with pytest.raises(ValueError, match="tokens"):
        _union(x, probs, weights, 8, 1, scale)


def test_union_cap_is_unset_by_default_and_must_hold_top_k(monkeypatch):
    monkeypatch.delenv("SPYRE_MOE_UNION_CAP", raising=False)
    assert moe_union_cap(8, 128) is None
    monkeypatch.setenv("SPYRE_MOE_UNION_CAP", "7")
    with pytest.raises(ValueError, match="top_k"):
        moe_union_cap(8, 128)
    monkeypatch.setenv("SPYRE_MOE_UNION_CAP", "16")
    assert moe_union_cap(8, 128) == 16
    assert moe_union_cap(8, 12, raw="16") == 12


def test_the_decode_ffn_takes_the_union_form_only_inside_its_token_range(
    monkeypatch,
):
    from hf_adapters import hf_gemma4_moe

    calls = []

    def fake_union(x, *args, **kwargs):
        calls.append(("union", x.shape[0]))
        return torch.zeros_like(x)

    def fake_loop(router_input, expert_input, *args):
        calls.append(("loop", router_input.shape[0]))
        return torch.zeros_like(expert_input)

    monkeypatch.setattr(hf_gemma4_moe, "moe_union_selected_experts", fake_union)
    monkeypatch.setattr(hf_gemma4_moe, "_compiled_moe_loop_region", fake_loop)
    monkeypatch.setattr(
        hf_gemma4_moe, "_router_probs", lambda x, *a: torch.zeros(x.shape[0], 8)
    )
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), eps=1e-6)
    experts = SimpleNamespace(
        decode_gate_proj=None,
        decode_up_proj=None,
        decode_down_proj=None,
        decode_chunk_scale=None,
        decode_chunk_offsets=None,
    )
    router = SimpleNamespace(
        proj=SimpleNamespace(weight=None),
        scale=None,
        scalar_root_size=None,
        per_expert_scale_stick=None,
        expert_ids=None,
        slot_identity=None,
    )
    block = SimpleNamespace(
        pre_feedforward_layernorm_2=norm,
        experts=experts,
        router=router,
        _moe_k=TOP_K,
        _stick_size=STICK,
        _moe_rms_eps=1e-6,
        _decode_chunks=4,
        _dense_forward=lambda residual: residual,
    )
    ffn = hf_gemma4_moe.Gemma4MoEBlock._decode_ffn_local
    for cap, tokens in ((None, 4), (16, 1), (16, 2), (16, 16), (16, 17)):
        block._moe_union_cap = cap
        ffn(block, torch.zeros(1, tokens, HIDDEN))
    assert calls == [
        ("loop", 4),
        ("loop", 1),
        ("union", 2),
        ("union", 16),
        ("loop", 17),
    ]


def test_a_row_left_without_any_slot_expert_gets_zero_not_nan():
    # Nine tokens through 8 slots. Tokens 0-7 route to experts 0-15 (top-1s
    # 0, 2, ..., 14, heavy); token 8 routes only to experts 28-31, lightly.
    # The slots go to the eight heavy top-1s, so token 8 keeps nothing.
    weights, _ = _experts(1)
    x, _, scale = _inputs(9, skew=0.0)
    logits = torch.full((9, EXPERTS), -30.0, dtype=torch.float64)
    for t in range(8):
        logits[t, 2 * t] = 6.0
        logits[t, 2 * t + 1] = 5.0
        logits[t, (2 * t + 2) % 16] = 4.0
        logits[t, (2 * t + 3) % 16] = 3.0
    logits[8, 28:32] = torch.tensor([1.0, 0.9, 0.8, 0.7])
    probs = logits.softmax(-1)
    expert_ids, identity = _constants(x.dtype)
    out = moe_union_selected_experts(
        x,
        probs,
        weights.decode_gate_proj,
        weights.decode_up_proj,
        weights.decode_down_proj,
        TOP_K,
        8,
        32,
        STICK,
        "gelu_tanh",
        expert_ids=expert_ids,
        slot_identity=None,
        per_expert_scale_stick=scale,
        decode_chunks=1,
        chunk_scale=weights.decode_chunk_scale,
        chunk_offsets=weights.decode_chunk_offsets,
        allow_dropped_rows=True,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[8], torch.zeros_like(out[8]))
