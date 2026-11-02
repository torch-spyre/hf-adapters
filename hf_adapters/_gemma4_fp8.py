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

"""Opt-in W8A8 fp8 for the Gemma 4 MoE adapter's non-expert linears and LM head.

Lossy. Off unless ``SPYRE_GEMMA4_FP8`` names the parts to convert, e.g.
``SPYRE_GEMMA4_FP8=attn,dense,lm_head``.

At TP=1, batch 1, the attention projections, the dense MLP and the 262K-row LM
head stream about 4.8 GB of fp16 weights per forward, which bounds their
kernels by memory bandwidth. Spyre's FMMA multiplies fp8 by fp8 only, so the
native fp8 matmul is W8A8: halve the weight bytes and quantize the activations
per token. The expert pools stay fp16 (an fp8 gathered expert currently
streams slower than fp16).

Per linear, inside the existing compiled regions (no extra graphs)::

    xs  = amax|x| / act_max per token          (clamped to a small floor)
    xq  = spyre::quantize_fp8_with_scale(x, xs)
    raw = spyre::scaled_mm(xq, Wq [K, N])       (fp16 out, fp8 x fp8 products)
    y   = raw * xs * ws

Choices, each forced by a Spyre constraint:

- ``scaled_mm`` is only correct when **N and K are multiples of 128** (one fp8
  stick); every other width is refused on the host. Gemma 4 26B-A4B
  qualifies everywhere except the dense intermediate 2112, which is
  zero-padded to 2176. The pad is exact: gelu(0) * 0 = 0, and the extra down
  input rows are zero.
- Quantization keeps the input's rank (``[B, T, K]``, scale ``[B, T, 1]``)
  and reshapes only the fp8 codes to 2-D, so no fused pointwise op reads a
  higher-rank producer. A single-token input is padded to two rows: with
  every leading dim 1, Inductor otherwise inlines the reshape into the
  quantize chain and the layout pass asserts.
- Weights are rounded to e4m3 on the host and uploaded once with the QFP8WT
  kernel layout ``scaled_mm`` reads. They **replace** the fp16 weights; the
  two copies are never resident together.
- q|k|v (k|v-shared global layers: q|k) and gate|up are fused into one
  weight each, so one activation quantization serves each fused group.
- Folding: q, k, v, o and down each feed an RMSNorm before anything else
  reads them. A per-tensor weight scale times a per-token activation scale
  is a per-row factor that RMSNorm removes up to its epsilon, so those
  outputs skip the rescale. gate (before GELU), up (multiplied by gelu(gate)
  in fp16) and the LM head (before the softcap) are rescaled. The LM head
  uses a per-row scale.

TP > 1 is refused: partial sums from different ranks would carry different
activation scales.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
FP8_STICK = 128
PARTS = ("attn", "dense", "lm_head")
# Target code range. 16 x 64 keeps the raw fp8 x fp8 sums of Gemma 4's
# widest contraction far below the fp16 maximum; e4m3 is floating point, so
# a smaller code range costs no relative precision above the subnormals.
ACT_MAX = 16.0
W_MAX = 64.0
# torch-spyre's own fp8 tests use this per-token scale floor.
SCALE_EPS = 1e-4
_FOLDED = frozenset({"q", "k", "v", "o", "down"})


def fp8_parts(raw=None):
    """Parts named by ``SPYRE_GEMMA4_FP8`` (a comma list from ``PARTS``)."""
    if raw is None:
        raw = os.getenv("SPYRE_GEMMA4_FP8", "")
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(f"SPYRE_GEMMA4_FP8: unknown parts {unknown}; use {PARTS}")
    return parts


def quantize_weight(w_nk, w_max=W_MAX, per_row=False):
    """Round a ``[N, K]`` weight to e4m3. Returns ``(codes [N, K], scale [N] fp16)``.

    The codes are made with the fp16-rounded scale, so dequantizing with the
    same fp16 scale reproduces exactly what was coded.
    """
    w = w_nk.detach().float()
    if per_row:
        scale = (w.abs().amax(dim=1) / w_max).clamp_min(2.0**-14).half()
    else:
        amax = max(float(w.abs().amax()), 1e-12)
        scale = torch.full((w.shape[0],), amax / w_max).half()
    if not bool(torch.isfinite(scale.float()).all()):
        raise ValueError("fp8 weight scale is not finite in fp16")
    codes = (w / scale.float()[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return codes, scale


def _check_aligned(name, n, k):
    if n % FP8_STICK or k % FP8_STICK:
        raise ValueError(
            f"fp8 {name}: N={n}, K={k} must both be multiples of {FP8_STICK}; "
            "spyre::scaled_mm is not correct otherwise"
        )


def _to_device(codes_nk, device):
    """``[N, K]`` codes -> ``[K, N]`` weight on ``device`` (QFP8WT on Spyre)."""
    kn = codes_nk.t().contiguous()
    if str(device).startswith("spyre"):
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        return _dma_to_spyre_fp8_kernel(kn)
    return kn.to(device)


def _scale_to_device(scale, device):
    if str(device).startswith("spyre"):
        from torch_spyre.model_utils import _dma_to_spyre_default

        return _dma_to_spyre_default(scale.contiguous())
    return scale.to(device)


def _quantize_activation(x):
    """Per-token codes and scale at the input's rank."""
    xs = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / ACT_MAX)).clamp(min=SCALE_EPS)
    if x.device.type == "spyre":
        return torch.ops.spyre.quantize_fp8_with_scale(x, xs), xs
    xs = xs.float()
    return (x.float() / xs).clamp(-FP8_MAX, FP8_MAX).to(FP8), xs


def _raw_matmul(xq, wq):
    if xq.device.type == "spyre":
        return torch.ops.spyre.scaled_mm(xq, wq, out_dtype=torch.float16)
    # CPU emulation: exact fp8 products, fp32 accumulation.
    return xq.float() @ wq.float()


def fp8_matmul(x, wq, members):
    """``x [..., K]`` -> ``{name: y [..., n]}`` for one fused fp8 weight.

    ``members`` holds ``(name, start, end, scale)`` slices of the fused
    output; ``scale`` is ``None`` (folded: the raw output goes to an RMSNorm),
    a python float (per-tensor) or an ``[n]`` tensor (per-row).
    """
    k, n = wq.shape
    if x.device.type == "spyre" and x.dtype != torch.float16:
        raise ValueError("fp8 linears on Spyre need an fp16 model (scaled_mm out)")
    if x.shape[-1] != k:
        raise ValueError(f"fp8 linear: input width {x.shape[-1]} != K {k}")
    if x.dim() == 3 and x.numel() == k:
        # One token: run on [x; 0] (see the module docstring) and keep row 0.
        padded = torch.cat([x, torch.zeros_like(x)], dim=1)
        return {name: y[:, :1] for name, y in fp8_matmul(padded, wq, members).items()}
    lead = tuple(x.shape[:-1])
    xq, xs = _quantize_activation(x)
    raw = _raw_matmul(xq.reshape(-1, k), wq).reshape(*lead, n)
    on_spyre = raw.device.type == "spyre"
    out = {}
    for name, start, end, scale in members:
        y = raw if (start, end) == (0, n) else raw[..., start:end]
        if scale is not None:
            y = y * xs
            y = y * (scale if on_spyre or isinstance(scale, float) else scale.float())
        out[name] = y if on_spyre else y.to(x.dtype)
    return out


class Fp8Linear(nn.Module):
    """Leader of a fused fp8 group: computes every member, returns the first.

    The other members' outputs ride on the input tensor (``_fp8_out``) until
    their followers read them. The followers run next, on the same input, in
    every Gemma 4 attention and MLP path, and the last one clears the memo,
    so a compiled graph keeps no extra outputs.
    """

    def __init__(self, weight, members, out_features):
        super().__init__()
        # A plain attribute, not a Parameter: ``move_model_to_spyre`` casts
        # every floating-point Parameter to the model dtype after
        # ``prepare_for_spyre``, which would turn the fp8 codes into fp16.
        # The expert pools stay put the same way.
        self.weight = weight
        self.members = tuple(members)
        self.in_features = int(weight.shape[0])
        self.out_features = int(out_features)

    def forward(self, x):
        if x.device.type == "spyre" and not torch.compiler.is_compiling():
            raise RuntimeError("fp8 linears run inside compiled regions only")
        outs = fp8_matmul(x, self.weight, self.members)
        if len(outs) > 1:
            x._fp8_out = outs
        return outs[self.members[0][0]]


class Fp8Follower(nn.Module):
    """A fused member other than the leader: returns its slice, holds no weight."""

    def __init__(self, name, out_features, last):
        super().__init__()
        self.name = name
        self.out_features = int(out_features)
        self.last = bool(last)
        self.weight = None

    def forward(self, x):
        memo = getattr(x, "_fp8_out", None)
        if memo is None or self.name not in memo:
            raise RuntimeError(
                f"fp8 follower {self.name!r} called before its leader on this input"
            )
        y = memo[self.name]
        if self.last:
            x._fp8_out = None
        return y


def _fuse(named, device, *, per_row, fold, pad_rows=None, pad_cols=None):
    """Quantize ``[(name, Linear)]`` into one fp8 weight. Returns modules by name."""
    parts, members, start = [], [], 0
    for name, linear in named:
        if getattr(linear, "bias", None) is not None:
            raise ValueError(f"fp8 {name}: biased linear")
        if getattr(linear, "_hf_device_mesh", None) is not None:
            raise ValueError("fp8 linears support TP=1 only")
        w = linear.weight.detach().to("cpu")
        if pad_rows and w.shape[0] < pad_rows:
            w = F.pad(w, (0, 0, 0, pad_rows - w.shape[0]))
        if pad_cols and w.shape[1] < pad_cols:
            w = F.pad(w, (0, pad_cols - w.shape[1]))
        parts.append((name, w))
        members.append((name, start, start + w.shape[0]))
        start += w.shape[0]
    n, k = start, parts[0][1].shape[1]
    _check_aligned("+".join(m for m, _ in named), n, k)
    codes, member_scales = [], []
    for name, w in parts:
        c, s = quantize_weight(w, per_row=per_row)
        codes.append(c)
        member_scales.append(s)
    weight = _to_device(torch.cat(codes, 0), device)
    resolved = []
    for (name, a, b), s in zip(members, member_scales):
        if fold and name in _FOLDED:
            scale = None
        elif per_row:
            scale = _scale_to_device(s, device)
        else:
            scale = float(s[0])
        resolved.append((name, a, b, scale))
    leader = Fp8Linear(weight, resolved, resolved[0][2] - resolved[0][1])
    out = {named[0][0]: leader}
    for i, (name, a, b, _) in enumerate(resolved[1:], start=2):
        out[name] = Fp8Follower(name, b - a, last=i == len(resolved))
    return out


def _dense_pad(intermediate):
    return -(-intermediate // FP8_STICK) * FP8_STICK


def convert_block(block, device, parts):
    """Replace one ``Gemma4MoEBlock``'s attention / dense-MLP linears in place."""
    attn, mlp = block.self_attn, block.mlp
    if "attn" in parts:
        qkv = [("q", attn.q_proj), ("k", attn.k_proj)]
        if attn.v_proj is not None:
            qkv.append(("v", attn.v_proj))
        fused = _fuse(qkv, device, per_row=False, fold=True)
        attn.q_proj, attn.k_proj = fused["q"], fused["k"]
        if attn.v_proj is not None:
            attn.v_proj = fused["v"]
        attn.o_proj = _fuse([("o", attn.o_proj)], device, per_row=False, fold=True)["o"]
    if "dense" in parts:
        pad = _dense_pad(mlp.gate_proj.weight.shape[0])
        fused = _fuse(
            [("gate", mlp.gate_proj), ("up", mlp.up_proj)],
            device,
            per_row=False,
            fold=True,
            pad_rows=pad,
        )
        mlp.gate_proj, mlp.up_proj = fused["gate"], fused["up"]
        mlp.down_proj = _fuse(
            [("down", mlp.down_proj)], device, per_row=False, fold=True, pad_cols=pad
        )["down"]


def convert_lm_head(model, device, logits_processor=None):
    """Replace the stick-padded LM head with a per-row-scaled fp8 one.

    ``prepare_lm_head_for_spyre`` compiled a closure over the fp16 head, so
    the compiled head forward is rebuilt over the new module.
    """
    if getattr(model, "_spyre_lm_head_tp_mesh", None) is not None:
        raise ValueError("fp8 LM head supports TP=1 only")
    model.lm_head = _fuse(
        [("lm_head", model.lm_head)], device, per_row=True, fold=False
    )["lm_head"]
    head = model.lm_head

    def lm_head_forward(hidden_states):
        logits = head(hidden_states)
        if logits_processor is not None:
            logits = logits_processor(logits)
        return logits

    model._spyre_lm_head_forward = torch.compile(
        lm_head_forward, dynamic=False, fullgraph=True
    )


def convert(model, blocks, parts, device, logits_processor=None):
    """Convert ``parts`` of a prepared Gemma 4 MoE model in place.

    Returns ``(fp16_bytes_replaced, fp8_bytes)`` of the converted weights.
    """
    if not parts:
        return 0, 0
    before = _linear_bytes(model)
    for block in blocks:
        convert_block(block, device, parts)
    if "lm_head" in parts:
        convert_lm_head(model, device, logits_processor)
    after = _linear_bytes(model)
    return before[0] - after[0], after[1]


def _linear_bytes(model):
    """``(nn.Linear bytes, Fp8Linear bytes)`` over the model's modules."""
    dense = fp8 = 0
    for module in model.modules():
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor):
            continue
        if isinstance(module, nn.Linear):
            dense += w.numel() * w.element_size()
        elif isinstance(module, Fp8Linear):
            fp8 += w.numel() * w.element_size()
    return dense, fp8


def emulated_reference_error(w_nk, x, per_row=False):
    """Relative error of the CPU W8A8 emulation vs fp32 ``x @ w.T`` (diagnostic)."""
    codes, scale = quantize_weight(w_nk, per_row=per_row)
    members = [("y", 0, w_nk.shape[0], scale.float() if per_row else float(scale[0]))]
    y = fp8_matmul(x.float(), codes.t().contiguous(), members)["y"].float()
    ref = x.float() @ w_nk.float().t()
    return float((y - ref).norm() / ref.norm().clamp_min(math.ulp(1.0)))
