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

"""Spyre-resident Gated DeltaNet core: one token per compiled graph call.

Ported from sglang-spyre's ``gdn_device`` (the form it runs on the card for
decode and for tree verify), adapted to the Hugging Face Qwen3.5 projection
layout. The causal convolution, gates, q/k L2 norm, delta rule and gated RMSNorm
of one token run as one compiled Spyre graph, shared by every layer: the
per-layer values are tensor inputs and the head counts trace-time constants.
A prefill chunk is stepped through token by token with the same graph; the
token's projection row is picked in-graph by a one-hot selector and its output
lands in its row of an accumulator, so no eager view of a card tensor enters or
leaves the graph.

Each form below is a device constraint, not a style choice:

* No fp32 matmul. The only matmul is an exact one-hot copy of the gate inputs
  into per-head rows.
* Every tensor keeps whole 64-element sticks on its last dimension. A row
  vector becomes a column with ``(eye * row).sum(-1, keepdim)`` and back with
  ``(col * eye).sum(-2)``, both exact.
* fp16-only graph boundary: an in-graph fp16 -> fp32 cast yields an fp32
  element arrangement that cannot be mixed with an fp32 graph input, so the
  fp32 state and constants cross as fp16 ``(hi, lo)`` pairs
  (``x = hi + lo / 1024``, about 20 significant bits).
* Spyre's fp32 ``exp`` has about 11 bits, so ``exp`` is Newton-refined against
  the fp32-accurate ``log``, and the decay near 1 uses a Taylor ``1 + expm1``
  so long-memory heads do not drift. ``softplus`` is rebuilt from
  ``relu`` / ``abs`` / ``exp`` / ``log``.

State layouts: the convolution state is ``[B, K-1, C]`` (oldest token first,
``C`` on the stick), and the recurrent state ``[2 * B * HV, Dv, Dk]`` fp16 holds
the ``hi`` rows followed by the ``lo`` rows of the fp32 ``[B, HV, Dv, Dk]``
state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

STICK = 64
L2NORM_EPS = 1e-6
HILO_SCALE = 1024.0
# exp(-20) = 2e-9 stands in for anything smaller, so log never sees 0.
EXP_CLAMP = 20.0


def core_dtype(dtype: torch.dtype) -> torch.dtype:
    """The core's activation dtype for a model of ``dtype``.

    16-bit models run the core in fp16, the form whose fp16 ``(hi, lo)`` state
    boundary is proven on the card: values cast in-graph from bf16 and from fp16
    do not share an fp32 element arrangement. fp32 (CPU references) is kept.
    """
    return torch.float32 if dtype == torch.float32 else torch.float16


def ba_layout(num_v_heads: int) -> tuple[int, int]:
    """``(width, a_offset)`` of the stick-padded ``[b | 0 | a | 0]`` projection."""
    half = STICK // 2 * math.ceil(num_v_heads / (STICK // 2))
    return 2 * half, half


def gate_expand_matrix(
    num_v_heads: int, head_k_dim: int, head_v_dim: int
) -> torch.Tensor:
    """One-hot ``[BA, HV*Dk + HV*Dv]`` copying ``a`` to ``Dk`` and ``b`` to ``Dv`` lanes."""
    width, offset = ba_layout(num_v_heads)
    expand = torch.zeros(width, num_v_heads * (head_k_dim + head_v_dim))
    base = num_v_heads * head_k_dim
    for head in range(num_v_heads):
        expand[offset + head, head * head_k_dim : (head + 1) * head_k_dim] = 1
        expand[head, base + head * head_v_dim : base + (head + 1) * head_v_dim] = 1
    return expand.half()


def split_hilo(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Host split of fp32 ``x`` into fp16 ``(hi, lo)`` with ``x ~= hi + lo / 1024``."""
    x = x.float()
    hi = x.half().float()
    lo = ((x - hi) * HILO_SCALE).half()
    return hi.half(), lo


def join_hilo(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    return hi.to(torch.float32) + lo.to(torch.float32) * (1.0 / HILO_SCALE)


def _split_in_graph(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hi = x.to(torch.float16)
    lo = ((x - hi.to(x.dtype)) * HILO_SCALE).to(torch.float16)
    return hi, lo


def _l2norm(x):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + L2NORM_EPS)


def _repeat_heads(x, repeats):
    """``[B, HK, D]`` -> ``[B, HK*repeats, D]``, repeating each head in place."""
    if repeats == 1:
        return x
    batch_size, num_heads, dim = x.shape
    return (
        x.unsqueeze(2)
        .expand(batch_size, num_heads, repeats, dim)
        .reshape(batch_size, num_heads * repeats, dim)
    )


def _exp(x):
    """fp32-accurate ``exp`` from the card's ``exp`` and ``log``."""
    x = torch.clamp(x, -EXP_CLAMP, EXP_CLAMP)
    e0 = torch.exp(x)
    return e0 * (1.0 + (x - torch.log(e0)))


def _exp_nonpos(x):
    """``exp(x)`` for ``x <= 0`` with an error relative to ``1 - exp(x)``."""
    t = 1.0 + x / 9.0
    for n in (8, 7, 6, 5, 4, 3, 2):
        t = 1.0 + (x / float(n)) * t
    near = 1.0 + x * t
    return torch.where(x > -0.5, near, _exp(x))


def _sigmoid(x):
    return 1.0 / (1.0 + _exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _softplus(x):
    return torch.relu(x) + torch.log(1.0 + _exp(-torch.abs(x)))


def gdn_token_step(
    qkvz,
    ba,
    conv_state,
    ssm_state,
    taps,
    gate_expand,
    neg_a,
    dt_bias,
    eye_v,
    norm_weight,
    *,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    norm_eps: float,
):
    """One token of convolution, gates, L2 norm, delta rule and gated norm.

    Args:
        qkvz: ``[B, 2*HK*Dk + 2*HV*Dv]`` projection of the token (``q|k|v|z``).
        ba: ``[B, BA]`` padded ``[b | 0 | a | 0]`` gate projection.
        conv_state: ``[B, K-1, C]``, oldest token first.
        ssm_state: ``[B, HV, Dv, Dk]`` fp32.
        taps: ``[K, C]`` fp32; ``taps[K-1]`` multiplies the current token.
        gate_expand: one-hot ``[BA, HV*Dk + HV*Dv]``.
        neg_a: ``[HV, 1, Dk]`` fp32 ``-exp(A_log)`` per head.
        dt_bias: ``[HV, 1, Dk]`` fp32.
        eye_v: ``[Dv, Dv]`` identity.
        norm_weight: ``[Dv]`` fp32 gated-RMSNorm weight.

    Returns:
        ``(y [B, HV*Dv], new_conv [B, K-1, C], new_ssm [B, HV, Dv, Dk] fp32)``.
    """
    f32 = torch.float32
    act_dtype = qkvz.dtype
    batch_size = qkvz.shape[0]
    key_dim = hk * dk
    conv_dim = 2 * key_dim + hv * dv
    history = conv_state.shape[1]
    x_raw = qkvz[:, :conv_dim]
    z = qkvz[:, conv_dim:].reshape(batch_size, hv, dv)

    # Depthwise causal convolution, current token first.
    state = conv_state.to(f32)
    acc = x_raw.to(f32) * taps[history]
    for lag in range(1, history + 1):
        acc = acc + state[:, history - lag] * taps[history - lag]
    act = _silu(acc)
    new_conv = torch.cat(
        [conv_state[:, 1:], x_raw.unsqueeze(1).to(conv_state.dtype)], dim=1
    )

    q = _l2norm(act[:, :key_dim].reshape(batch_size, hk, dk))
    k = _l2norm(act[:, key_dim : 2 * key_dim].reshape(batch_size, hk, dk))
    v = act[:, 2 * key_dim :].reshape(batch_size, hv, dv)
    q = _repeat_heads(q, hv // hk) * (dk**-0.5)
    k = _repeat_heads(k, hv // hk)

    gates = torch.matmul(ba, gate_expand.to(ba.dtype)).to(f32)  # exact one-hot copy
    a_rep = gates[:, : hv * dk].reshape(-1, hv, 1, dk)
    b_rep = gates[:, hv * dk :].reshape(-1, hv, 1, dv)
    # The gate inputs are replicated across each head's lanes; an exact amax
    # over the lanes yields one [B, HV, 1, 1] scalar per head. It broadcasts at
    # the stick, so it combines with operands of either element arrangement
    # (torch-spyre gives the gate path a standard one and the state, cast from
    # fp16, a staggered one).
    decay = _exp_nonpos(neg_a * _softplus(a_rep + dt_bias)).amax(-1, keepdim=True)
    beta = _sigmoid(b_rep).amax(-1, keepdim=True)
    eye = eye_v.to(f32)

    h = ssm_state * decay  # [B, HV, Dv, Dk] * [B, HV, 1, 1]
    k_row = k.unsqueeze(2)
    kv = (h * k_row).sum(-1, keepdim=True)  # [B, HV, Dv, 1]
    v_col = (eye * v.unsqueeze(2)).sum(-1, keepdim=True)
    h = h + (v_col - kv) * beta * k_row
    core = ((h * q.unsqueeze(2)).sum(-1, keepdim=True) * eye).sum(-2)  # [B, HV, Dv]

    # Qwen3.5 gated RMSNorm: weight * rmsnorm(core), gated by silu(z).
    xf = core.to(act_dtype).to(f32)
    xf = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + norm_eps)
    y = (norm_weight * xf.to(act_dtype).to(f32) * _silu(z.to(f32))).to(act_dtype)
    return y.reshape(batch_size, hv * dv), new_conv, h


def gdn_token_graph(
    qkvz,
    ba,
    selector,
    y_acc,
    conv_state,
    ssm_state,
    taps_hi,
    taps_lo,
    gate_expand,
    neg_a_hi,
    neg_a_lo,
    dt_bias_hi,
    dt_bias_lo,
    eye_v,
    norm_hi,
    norm_lo,
    *,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    norm_eps: float,
):
    """The Spyre graph: one token of a ``[B, T, W]`` chunk, fp16 state boundary.

    ``selector`` is a ``[1, T, 1]`` one-hot of the token's row (an exact masked
    sum picks the row); ``y_acc`` ``[B, T, HV*Dv]`` carries the rows done so far
    and receives this token's output. Returns ``(y_acc, conv_state, ssm_state)``.
    """
    batch_size = qkvz.shape[0]
    x = (qkvz * selector).sum(1)
    b = (ba * selector).sum(1)
    hilo = ssm_state.reshape(2, batch_size, hv, dv, dk)
    y, new_conv, h = gdn_token_step(
        x,
        b,
        conv_state,
        join_hilo(hilo[0], hilo[1]),
        join_hilo(taps_hi, taps_lo),
        gate_expand,
        join_hilo(neg_a_hi, neg_a_lo),
        join_hilo(dt_bias_hi, dt_bias_lo),
        eye_v,
        join_hilo(norm_hi, norm_lo),
        hk=hk,
        hv=hv,
        dk=dk,
        dv=dv,
        norm_eps=norm_eps,
    )
    hi, lo = _split_in_graph(h)
    new_ssm = torch.cat([hi, lo], dim=0).reshape(2 * batch_size * hv, dv, dk)
    return y_acc + selector * y.unsqueeze(1), new_conv, new_ssm


@dataclass
class GDNSpyreConstants:
    """Per-layer device constants and static sizes of :func:`gdn_token_graph`."""

    tensors: tuple[torch.Tensor, ...]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    kernel_size: int
    norm_eps: float

    @property
    def conv_dim(self) -> int:
        return (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    def static(self) -> dict:
        return dict(
            hk=self.num_k_heads,
            hv=self.num_v_heads,
            dk=self.head_k_dim,
            dv=self.head_v_dim,
            norm_eps=self.norm_eps,
        )


def prepare_gdn_constants(linear_attn, device) -> GDNSpyreConstants:
    """Build the fp16 ``(hi, lo)`` constants of one HF Qwen3.5 DeltaNet layer on ``device``."""
    num_k_heads = linear_attn.num_k_heads
    num_v_heads = linear_attn.num_v_heads
    head_k_dim = linear_attn.head_k_dim
    head_v_dim = linear_attn.head_v_dim
    conv_weight = linear_attn.conv1d.weight.detach().to("cpu", torch.float32)
    taps = conv_weight[:, 0, :].t().contiguous()  # [K, C]
    neg_a = -linear_attn.A_log.detach().to("cpu", torch.float32).exp()
    dt_bias = linear_attn.dt_bias.detach().to("cpu", torch.float32)
    per_head = (num_v_heads, 1, head_k_dim)
    tensors = (
        *split_hilo(taps),
        gate_expand_matrix(num_v_heads, head_k_dim, head_v_dim),
        *split_hilo(neg_a.reshape(num_v_heads, 1, 1).expand(per_head).contiguous()),
        *split_hilo(dt_bias.reshape(num_v_heads, 1, 1).expand(per_head).contiguous()),
        torch.eye(head_v_dim, dtype=torch.float16),
        *split_hilo(linear_attn.norm.weight.detach().to("cpu", torch.float32)),
    )
    return GDNSpyreConstants(
        tensors=tuple(t.contiguous().to(device) for t in tensors),
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        kernel_size=linear_attn.conv_kernel_size,
        norm_eps=float(linear_attn.norm.variance_epsilon),
    )


def token_selectors(
    sequence_length: int, dtype: torch.dtype, device
) -> tuple[torch.Tensor, ...]:
    """The ``[1, T, 1]`` one-hot row selectors of a ``T``-token chunk, on ``device``."""
    eye = torch.eye(sequence_length, dtype=dtype)
    return tuple(
        eye[index].reshape(1, sequence_length, 1).contiguous().to(device)
        for index in range(sequence_length)
    )


def pad_ba_weight(b_weight: torch.Tensor, a_weight: torch.Tensor) -> torch.Tensor:
    """``in_proj_b`` / ``in_proj_a`` rows -> one stick-padded ``[b | 0 | a | 0]`` weight."""
    num_v_heads = b_weight.shape[0]
    width, offset = ba_layout(num_v_heads)
    weight = b_weight.new_zeros((width, b_weight.shape[1]))
    weight[:num_v_heads] = b_weight
    weight[offset : offset + num_v_heads] = a_weight
    return weight


__all__ = [
    "GDNSpyreConstants",
    "core_dtype",
    "ba_layout",
    "gdn_token_graph",
    "gdn_token_step",
    "pad_ba_weight",
    "prepare_gdn_constants",
    "token_selectors",
]
