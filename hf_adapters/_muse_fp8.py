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

"""Opt-in W8A8 fp8 for the Muse Glimmer text decoder's linears and LM head.

Lossy. Off unless ``SPYRE_MUSE_FP8`` names the parts to convert, e.g.
``SPYRE_MUSE_FP8=attn,mlp,lm_head``.

Muse Glimmer 30B is dense: at batch 1 every forward streams its ~57 GB of
16-bit text weights, which bounds decode by memory bandwidth. Spyre's FMMA
multiplies fp8 by fp8 only, so the native fp8 matmul is W8A8: halve the
weight bytes and quantize the activations per token::

    xs  = amax|x| / ACT_MAX per token          (clamped to a small floor)
    xq  = spyre::quantize_fp8_with_scale(x, xs)
    raw = spyre::scaled_mm(xq, Wq [K, N])       (fp16 out)
    y   = raw * xs * ws[N]                      (per-output-row weight scale)

Choices:

- ``scaled_mm`` is only correct when **N and K are multiples of 128**; every
  other width is refused on the host. Muse qualifies everywhere (6656,
  19968, 4096, 256); the LM head's 202048 rows are zero-padded to a 128
  multiple whose stick count keeps the per-core span under the EAR limit
  (202240: 202112 is 1579 sticks, a prime).
- q|k|v|attention-gate and mlp gate|up each share one fused weight, so one
  activation quantization serves each group.
- Every output is rescaled back to its fp16 magnitude (no folding into the
  next RMSNorm), so the norms see the same values as the 16-bit model.
  Scales are per output row, as tensors.
- ``hf_muse_glimmer`` calls ``maybe_convert`` at the end of preparation.
  Converted layers run in fp8 blocks with the adapter's block signature
  (plain SDPA), so the adapter's driver and the tree verifier run them
  unchanged; the head becomes the adapter's ``_spyre_lm_head_forward``.
- Weights are rounded to e4m3 on the host and uploaded once in the QFP8WT
  kernel layout. They **replace** the 16-bit weights (the ``nn.Linear``
  modules are dropped before the model moves to the device).
- Quantization keeps the input's rank; a single-token input is padded to two
  rows (Inductor otherwise inlines the reshape into the quantize chain).

Under tensor parallelism (``_muse_tp``) every fp8 output is rescaled to fp16
units on its own rank before the o / down all-reduce and the LM-head gather,
so no activation scale has to be shared between ranks.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters import hf_common

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
FP8_STICK = 128
PARTS = ("attn", "mlp", "lm_head")
# Code ranges. scaled_mm returns fp16, so the raw fp8 x fp8 sums must stay
# below 65504. On the real 30B checkpoint the Gemma 4 ranges (16 x 64) peak at
# ~5.3e4 raw (layer 0 q / v, layer 51 up); 8 x 32 quarters that. e4m3 is
# floating point, so a smaller code range costs no relative precision above
# the subnormals. Overridable for audits.
ACT_MAX = float(os.getenv("SPYRE_MUSE_FP8_ACT_MAX", "8"))
W_MAX = float(os.getenv("SPYRE_MUSE_FP8_W_MAX", "32"))
SCALE_EPS = 1e-4


def fp8_parts(raw=None):
    """Parts named by ``SPYRE_MUSE_FP8`` (a comma list from ``PARTS``)."""
    if raw is None:
        raw = os.getenv("SPYRE_MUSE_FP8", "")
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(f"SPYRE_MUSE_FP8: unknown parts {unknown}; use {PARTS}")
    return parts


def fp8_vocab(vocab, hidden):
    """Smallest 128-multiple >= ``vocab`` whose fp8 head fits the EAR limit."""
    max_residual = hf_common._EAR_LIMIT_BYTES // (hidden * FP8_STICK)
    sticks = -(-vocab // FP8_STICK)
    while hf_common._largest_prime_factor(sticks) > max_residual:
        sticks += 1
    return sticks * FP8_STICK


def quantize_weight(w_nk, w_max=W_MAX):
    """``[N, K]`` -> ``(codes [N, K] e4m3, per-row scale [N] fp16)``.

    Codes are made with the fp16-rounded scale, so dequantizing with that
    same scale reproduces exactly what was coded.
    """
    w = w_nk.detach().float()
    scale = (w.abs().amax(dim=1) / w_max).clamp_min(2.0**-14).half()
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


def _scale_to_device(scale, device, dtype):
    if str(device).startswith("spyre"):
        from torch_spyre.model_utils import _dma_to_spyre_default

        return _dma_to_spyre_default(scale.contiguous())
    return scale.to(device, dtype)


def _quantize_activation(x):
    xs = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / ACT_MAX)).clamp(min=SCALE_EPS)
    if x.device.type == "spyre":
        return torch.ops.spyre.quantize_fp8_with_scale(x, xs), xs
    xs = xs.float()
    return (x.float() / xs).clamp(-FP8_MAX, FP8_MAX).to(FP8), xs


def _raw_matmul(xq, wq):
    if xq.device.type == "spyre":
        return torch.ops.spyre.scaled_mm(xq, wq, out_dtype=torch.float16)
    return xq.float() @ wq.float()  # CPU emulation: exact products, fp32 sums


def fp8_matmul(x, wq, scale, splits=None):
    """``x [B, T, K]`` -> ``y [B, T, N]`` (or its ``splits`` column slices)."""
    k, n = wq.shape
    if x.device.type == "spyre" and x.dtype != torch.float16:
        raise ValueError("fp8 linears on Spyre need an fp16 model (scaled_mm out)")
    if x.shape[-1] != k:
        raise ValueError(f"fp8 linear: input width {x.shape[-1]} != K {k}")
    if x.dim() == 3 and x.numel() == k:
        # One token: run on [x; 0] and keep row 0 (see the module docstring).
        padded = torch.cat([x, torch.zeros_like(x)], dim=1)
        out = fp8_matmul(padded, wq, scale, splits)
        return [y[:, :1] for y in out] if splits else out[:, :1]
    xq, xs = _quantize_activation(x)
    raw = _raw_matmul(xq.reshape(-1, k), wq).reshape(*x.shape[:-1], n)
    if raw.device.type == "spyre":
        y = raw * xs * scale
    else:
        y = (raw * xs * scale.float()).to(x.dtype)
    if not splits:
        return y
    out, start = [], 0
    for width in splits:
        out.append(y[..., start : start + width])
        start += width
    return out


def _fuse(linears, device, dtype, pad_rows=None):
    """Quantize ``[nn.Linear]`` into one ``(codes [K, N], scale [N])``."""
    rows = []
    for linear in linears:
        if linear.bias is not None:
            raise ValueError("fp8 linears must be bias-free")
        if getattr(linear, "_hf_device_mesh", None) is not None:
            raise ValueError("fp8 linears need host-sharded TP (_muse_tp)")
        rows.append(linear.weight.detach().to("cpu"))
    w = torch.cat(rows, 0)
    if pad_rows is not None and w.shape[0] < pad_rows:
        w = F.pad(w, (0, 0, 0, pad_rows - w.shape[0]))
    _check_aligned("+".join(type(x).__name__ for x in linears), *w.shape)
    codes, scale = quantize_weight(w)
    return _to_device(codes, device), _scale_to_device(scale, device, dtype)


# -- text blocks -------------------------------------------------------------


def _make_block(layer, parts, num_q_heads, num_kv_heads, head_dim, use_rope, tp):
    """One decoder layer with fp8 linears for ``parts``.

    The plain-SDPA path of ``hf_gemma3._make_compiled_block`` as Muse Glimmer
    configures it (stock Muse norm modules, weightless Q/K norm, query
    multiplier, sigmoid gate in head layout, RoPE or NoPE), with the same
    call signature, so ``hf_muse_glimmer``'s driver and the tree verifier run
    it unchanged. q|k|v|gate and MLP gate|up are one fused fp8 matmul each.
    """
    from hf_adapters.hf_common import apply_rope_matmul, kv_cache_update

    fp8 = layer._spyre_fp8
    attn, mlp = layer.self_attn, layer.mlp
    qk_norm, query_scale = attn.qk_norm, attn.qk_scale_factor
    scaling = attn.scaling
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    nq, nkv = num_q_heads * head_dim, num_kv_heads * head_dim
    attn_fp8, mlp_fp8 = "attn" in parts, "mlp" in parts
    group = None if tp is None else tp.group_name

    def reduce(x):
        # Under TP the fp8 o / down outputs are rescaled to 16-bit units on
        # each rank first, so the partial sums share units. (16-bit o / down
        # are ``_muse_tp.AllReduceLinear`` and reduce themselves.)
        return x if group is None else all_reduce(x, group)

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)
        bsz, seq_len, _ = h.shape
        if attn_fp8:
            q, k, v, g = fp8_matmul(h, *fp8["attn"], (nq, nkv, nkv, nq))
        else:
            q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
            g = attn.gate_proj(h)
        q = q.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q = qk_norm(q) * query_scale
        k = qk_norm(k)
        if use_rope:
            q = apply_rope_matmul(q, selected_freqs)
            k = apply_rope_matmul(k, selected_freqs)
        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache, cache_index
        )
        attn_out = F.scaled_dot_product_attention(
            q.contiguous(),
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
        gate = torch.sigmoid(g).view(bsz, seq_len, num_q_heads, head_dim)
        attn_out = attn_out * gate.transpose(1, 2)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, nq)
        if attn_fp8:
            attn_out = reduce(fp8_matmul(attn_out, *fp8["o"]))
        else:
            attn_out = attn.o_proj(attn_out)
        h = residual + post_attn_ln(attn_out)
        residual = h
        h = pre_ff_ln(h)
        if mlp_fp8:
            inter = fp8["mlp"][0].shape[1] // 2
            gate_out, up_out = fp8_matmul(h, *fp8["mlp"], (inter, inter))
            h = reduce(fp8_matmul(F.silu(gate_out) * up_out, *fp8["down"]))
        else:
            h = mlp(h)
        return residual + post_ff_ln(h), key_cache, value_cache

    return torch.compile(block_forward, dynamic=False, fullgraph=True)


def all_reduce(tensor, group_name):
    from hf_adapters._muse_tp import all_reduce as reduce

    return reduce(tensor, group_name)


class Fp8Linear(nn.Module):
    """An fp8 ``[K, N]`` weight with per-row scales; runs in compiled regions."""

    def __init__(self, weight, scale, out_features):
        super().__init__()
        # Plain attributes, not Parameters: the model move after
        # ``prepare_for_spyre`` casts every floating Parameter to the model
        # dtype, which would turn the fp8 codes into fp16.
        self.weight = weight
        self.scale = scale
        self.in_features = int(weight.shape[0])
        self.out_features = int(out_features)

    def forward(self, x):
        if x.device.type == "spyre" and not torch.compiler.is_compiling():
            raise RuntimeError("fp8 linears run inside compiled regions only")
        return fp8_matmul(x, self.weight, self.scale)


def _convert_lm_head(model, device, dtype):
    cfg = hf_common.text_config(model.config)
    head = model.lm_head
    if getattr(model, "_spyre_tp", None) is not None:
        rows = head.weight.shape[0]  # an equal, 128-aligned vocabulary shard
    else:
        rows = fp8_vocab(head.weight.shape[0], head.weight.shape[1])
    weight, scale = _fuse([head], device, dtype, pad_rows=rows)
    model.lm_head = Fp8Linear(weight, scale, rows)
    fp8_head = model.lm_head
    multiplier, cap = cfg.output_multiplier, cfg.final_logit_softcapping

    tp = getattr(model, "_spyre_tp", None)

    def finish(out):
        out = out * multiplier
        if cap is not None:
            out = torch.tanh(out / cap) * cap
        return out

    if tp is None:

        def logits(h):
            return finish(fp8_head(h))

        model._spyre_lm_head_forward = torch.compile(
            logits, dynamic=False, fullgraph=True
        )
        return

    # Under TP the fp8 head and the vocabulary all-gather must be separate
    # graphs: fused, scaled_mm's output feeds the gather in a layout it reads
    # wrongly (relative error > 1 on two cards, both shards). The shard is
    # copied to the default layout between them.
    from hf_adapters._muse_tp import gather_vocab

    local = torch.compile(lambda h: fp8_head(h), dynamic=False, fullgraph=True)
    gathered = torch.compile(
        lambda out: finish(gather_vocab(model, out)), dynamic=False, fullgraph=True
    )

    def tp_logits(h):
        return gathered(hf_common.to_default_device_layout(local(h)))

    model._spyre_lm_head_forward = tp_logits
    raw = torch.compile(
        lambda out: gather_vocab(model, out), dynamic=False, fullgraph=True
    )
    model._spyre_tp_raw_head = lambda h: raw(
        hf_common.to_default_device_layout(local(h))
    )


def maybe_convert(model):
    """``convert`` for the parts ``SPYRE_MUSE_FP8`` names; a no-op when unset."""
    return convert(model, fp8_parts())


def convert(model, parts, device=None, dtype=None):
    """Convert ``parts`` of a ``hf_muse_glimmer``-prepared model in place.

    The adapter's decoder layers are replaced by fp8 ones (one compiled
    block per layer, plain SDPA on every layer), and plain ``generate`` then
    uses band-masked SDPA for every request: no fp8 form of the anchored
    sliding-window blocks has been validated. Returns ``(16-bit bytes
    replaced, fp8 bytes)``.
    """
    if not parts:
        return 0, 0
    device = device or hf_common.DEVICE
    dtype = dtype or hf_common.get_model_dtype(model)
    backbone = hf_common.get_backbone(model)
    cfg = hf_common.text_config(model.config)
    before = _linear_bytes(model)
    tp = getattr(model, "_spyre_tp", None)
    if "attn" in parts or "mlp" in parts:
        blocks = []
        for i, layer in enumerate(backbone.layers):
            attn, mlp = layer.self_attn, layer.mlp
            head_dim = cfg.head_dim
            num_q = hf_common._projection_head_count(
                attn.q_proj, cfg.num_attention_heads, head_dim
            )
            num_kv = hf_common._projection_head_count(
                attn.k_proj, cfg.num_key_value_heads, head_dim
            )
            fp8 = {}
            if "attn" in parts:
                fp8["attn"] = _fuse(
                    [attn.q_proj, attn.k_proj, attn.v_proj, attn.gate_proj],
                    device,
                    dtype,
                )
                fp8["o"] = _fuse([attn.o_proj], device, dtype)
                for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj"):
                    setattr(attn, name, nn.Identity())
            if "mlp" in parts:
                fp8["mlp"] = _fuse([mlp.gate_proj, mlp.up_proj], device, dtype)
                fp8["down"] = _fuse([mlp.down_proj], device, dtype)
                for name in ("gate_proj", "up_proj", "down_proj"):
                    setattr(mlp, name, nn.Identity())
            layer._spyre_fp8 = fp8
            blocks.append(
                _make_block(
                    layer,
                    parts,
                    num_q,
                    num_kv,
                    head_dim,
                    bool(cfg.layer_rope_theta[i]),
                    tp,
                )
            )
        # Replace every block list, so the 16-bit blocks (and, through their
        # closures, the 16-bit weights) are released.
        model._spyre_muse_full_blocks = blocks
        model._spyre_muse_swa_blocks = blocks
        model._spyre_compiled_blocks = blocks
        model._spyre_swa_mode = None
        if hasattr(model, "_spyre_cache_allocator"):
            del model._spyre_cache_allocator
    if "lm_head" in parts:
        if getattr(model, "_spyre_lm_head_tp_mesh", None) is not None:
            raise ValueError("fp8 LM head needs host-sharded TP (_muse_tp)")
        _convert_lm_head(model, device, dtype)
    after = _linear_bytes(model)
    model._spyre_fp8_parts = tuple(parts)
    return before - after, _fp8_bytes(model)


def _linear_bytes(model):
    text = hf_common.get_backbone(model)
    total = sum(
        m.weight.numel() * m.weight.element_size()
        for m in text.modules()
        if isinstance(m, nn.Linear)
    )
    head = model.lm_head
    if isinstance(head, nn.Linear):
        total += head.weight.numel() * head.weight.element_size()
    return total


def _fp8_bytes(model):
    total = 0
    for layer in hf_common.get_backbone(model).layers:
        for weight, _ in getattr(layer, "_spyre_fp8", {}).values():
            total += weight.numel() * weight.element_size()
    if isinstance(model.lm_head, Fp8Linear):
        total += model.lm_head.weight.numel()
    return total


def emulated_relative_error(w_nk, x):
    """Relative error of the CPU W8A8 emulation vs fp32 ``x @ w.T`` (diagnostic)."""
    codes, scale = quantize_weight(w_nk)
    y = fp8_matmul(x.float(), codes.t().contiguous(), scale.float()).float()
    ref = x.float() @ w_nk.float().t()
    return float((y - ref).norm() / ref.norm().clamp_min(math.ulp(1.0)))


def raw_peak(w_nk, x):
    """Largest ``|raw|`` fp8 x fp8 sum for ``x`` (fp16-range audit, CPU)."""
    codes, _ = quantize_weight(w_nk)
    xq, _ = _quantize_activation(x.float())
    return float((xq.float().reshape(-1, x.shape[-1]) @ codes.float().t()).abs().max())
