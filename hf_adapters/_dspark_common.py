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

"""
Shared Spyre machinery for DSpark speculative-decoding *drafter* models
(``hf_dspark_qwen3`` / ``hf_dspark_gemma4`` / ``hf_dspark_granite``).

A DSpark drafter is an EAGLE-3-style block proposer, NOT a standard causal-LM:

- It consumes the *target* model's intermediate hidden states (concatenated over
  ``target_layer_ids``) as context, projects them with ``fc`` + ``hidden_norm``,
  and in every decoder layer builds ``K``/``V`` by concatenating the projected
  context K/V with the noise-block K/V — a NON-causal attention over
  ``ctx_len + block_size`` positions (there is no per-token KV cache; the whole
  block is proposed in one shot).
- After the 5-layer backbone it runs an ``lm_head`` for base block logits, then a
  small **markov head** (``markov_w1`` Embedding + ``markov_w2`` Linear(rank →
  vocab)) that autoregressively sharpens each block position from the previous
  token, and an optional confidence head.

Because of the block-propose shape, the adapter's public forward is
``_run_draft_block`` (ctx features + noise block → block hidden states), not the
token-by-token ``_run_forward``/``generate`` surface of a target adapter. The
family adapters wire ``prepare_for_spyre`` (below) + a thin ``_run_draft_block``.

Everything Spyre-specific reuses ``hf_common``: ``patch_rmsnorm`` (pow → x*x),
``pad_lm_head`` (stick-padded single-kernel head, for the draft head AND the
markov ``w2``), ``PrecomputedRotaryEmbedding`` + ``apply_rope_matmul`` (RoPE),
and the "compute-on-CPU, move to device" idiom for the embedding lookup and the
markov sample loop. The two things with no ready-made helper — the concat-KV
non-causal block and the markov sample loop — live here.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    DEVICE,
    apply_rope_matmul,
    pad_lm_head,
    patch_rmsnorm,
)


def build_ctx_block_mask(
    ctx_valid_len: int,
    ctx_pad: int,
    block_size: int,
    kv_pad: int,
    q_len: int,
    dtype: torch.dtype = torch.float16,
):
    """Additive ``-inf`` mask for the drafter's concat-KV attention.

    K/V is ``concat(k_ctx[ctx_pad], k_noise[block_size])`` padded to a fixed,
    stick-aligned ``kv_pad`` so the block compiles once. Real keys are the
    ``ctx_valid_len`` leading context rows and the ``block_size`` noise rows;
    everything else (context zero-pad ``[ctx_valid_len:ctx_pad)`` and the kv tail
    ``[ctx_pad+block_size:kv_pad)``) is masked. The attention is non-causal within
    the block — every query attends every real key.

    Built on CPU and moved as one contiguous tensor: an on-device slice-assign at
    a non-stick-aligned offset is rejected by the Spyre backend
    (``copy_from_d2d`` storage_offset not a multiple of 64), so the ``-inf`` fills
    happen host-side. Mirrors ``hf_common.build_prefill_mask``'s zeros + ``-inf``
    idiom.
    """
    m = torch.zeros((1, 1, q_len, kv_pad), dtype=dtype)
    if ctx_valid_len < ctx_pad:
        m[:, :, :, ctx_valid_len:ctx_pad] = float("-inf")
    if ctx_pad + block_size < kv_pad:
        m[:, :, :, ctx_pad + block_size :] = float("-inf")
    return m


def make_dspark_block(
    layer, *, ctx_pad, block_size, kv_pad, use_qk_norm, scaling, res_mult=1.0
):
    """Compiled DSpark drafter decoder block (concat-KV, non-causal, no cache).

    Reproduces the drafter's ``self_attn``: ``K``/``V`` = concat of the projected
    context (``k_proj``/``v_proj`` of ``target_hidden_states``, padded to
    ``ctx_pad``) with the noise-block projections, RoPE applied over the full
    ``ctx_pad + block_size`` window via ``apply_rope_matmul``, then a single
    non-causal ``scaled_dot_product_attention`` with the ``build_ctx_block_mask``
    additive mask (passed in as ``attn_mask``). No ``kv_cache_update`` — the block
    is proposed in one shot.

    ``use_qk_norm`` toggles the Qwen3-style per-head q/k RMSNorm (Granite/Gemma4
    have none). ``scaling`` is the SDPA temperature (head_dim**-0.5, or the family
    attention_multiplier for Granite).
    """
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    head_dim = attn.head_dim
    q_norm = attn.q_norm if use_qk_norm else None
    k_norm = attn.k_norm if use_qk_norm else None

    def block_forward(hidden_states, target_hidden_states, selected_freqs, attn_mask):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, q_len, _ = h.shape
        ctx_len = target_hidden_states.shape[1]

        q = attn.q_proj(h).view(bsz, q_len, -1, head_dim)
        # K/V = concat(context[ctx_pad], noise-block[q_len]) along the sequence
        # axis, then padded on the seq axis to the FIXED stick-aligned kv_pad so
        # the compiled block is shape-static across steps. The kv-tail pad rows
        # are neutralized by attn_mask (build_ctx_block_mask).
        k_ctx = attn.k_proj(target_hidden_states)
        k_noise = attn.k_proj(h)
        v_ctx = attn.v_proj(target_hidden_states)
        v_noise = attn.v_proj(h)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)

        if q_norm is not None:
            q = q_norm(q)
            k = k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE over the full ctx+block window (matmul form — the slice form
        # mis-lowers in fp16 on Spyre). selected_freqs covers ctx_pad+block_size;
        # the block queries take the trailing q_len rows.
        q = apply_rope_matmul(q, selected_freqs[:, :, -q_len:])
        k = apply_rope_matmul(k, selected_freqs)

        # Pad K/V to the fixed kv_pad on the seq axis (mask handles the pad rows).
        pad = kv_pad - k.shape[-2]
        if pad > 0:
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_out = attn.o_proj(attn_out)

        # Granite scales each residual branch by residual_multiplier (1.0 for
        # families without it — a plain residual add).
        h = residual + attn_out * res_mult
        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        return residual + h * res_mult

    return torch.compile(block_forward, dynamic=False)


def prepare_dspark_common(model, rmsnorm_cls, *, ctx_pad, kv_pad, use_qk_norm):
    """Shared ``prepare_for_spyre`` steps for a DSpark drafter.

    - patch the family RMSNorm (pow → x*x) — covers layer norms, ``hidden_norm``,
      and any q/k norm;
    - stick-pad the draft ``lm_head`` and the markov ``markov_w2`` (both are
      full-vocab matmuls that would otherwise overflow the per-core EAR limit);
    - precompute RoPE and build the compiled concat-KV blocks.

    The caller (family adapter) sets ``model._spyre_rope`` first (families differ
    on the rotary source) and passes its RMSNorm class + attention scaling.
    """
    patch_rmsnorm(rmsnorm_cls)
    pad_lm_head(model)
    _pad_markov_w2(model)

    cfg = model.config
    block_size = int(model.block_size)
    # Default SDPA scaling is head_dim**-0.5; a family may stash an override
    # (e.g. Granite's attention_multiplier) on ``model._spyre_attn_scaling``.
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    scaling = getattr(model, "_spyre_attn_scaling", head_dim**-0.5)

    model._spyre_dspark = {
        "ctx_pad": ctx_pad,
        "kv_pad": kv_pad,
        "block_size": block_size,
    }
    model._spyre_compiled_blocks = [
        make_dspark_block(
            layer,
            ctx_pad=ctx_pad,
            block_size=block_size,
            kv_pad=kv_pad,
            use_qk_norm=use_qk_norm,
            scaling=scaling,
            # Granite scales each residual branch; other families use 1.0.
            res_mult=float(getattr(layer, "residual_multiplier", 1.0)),
        )
        for layer in model.layers
    ]


def _pad_markov_w2(model):
    """Stick-pad the markov head's ``markov_w2`` (Linear rank -> vocab).

    Same EAR-limit concern as the LM head: an unpadded full-vocab output matmul
    overflows the per-core span and the block-correction collapses. We pad the
    output dim to a stick multiple in place; the markov forward slices the real
    vocab back off on CPU (see ``markov_bias``).
    """
    mh = getattr(model, "markov_head", None)
    if mh is None or not hasattr(mh, "markov_w2"):
        return
    w = mh.markov_w2.weight  # [vocab, rank]
    vocab, rank = w.shape
    stick = 64
    padded = ((vocab + stick - 1) // stick) * stick
    if padded != vocab:
        mh.markov_w2.weight = nn.Parameter(
            F.pad(w, (0, 0, 0, padded - vocab)), requires_grad=False
        )
    mh._spyre_real_vocab = vocab


def embed_noise_block(model, draft_input_ids):
    """CPU embedding lookup for the noise block, returned on device.

    ``nn.Embedding`` is a gather with no Spyre kernel (CPU fallback), so we look
    up on CPU and move the result to the device — the same idiom the target
    adapters use for embeddings. Applies the family ``embedding_multiplier`` (1.0
    when the family has none) since the drafter's ``forward`` — which the block
    path bypasses — is where that scale normally lives.
    """
    emb = model.embed_tokens(draft_input_ids.to("cpu"))
    mult = float(getattr(model, "embedding_multiplier", 1.0))
    if mult != 1.0:
        emb = emb * mult
    return emb.to(DEVICE, model.embed_tokens.weight.dtype)


def markov_bias(model, prev_token_ids):
    """One markov-head step bias ``markov_w2(markov_w1(prev))`` on device.

    ``markov_w1`` is a CPU gather (like ``embed_noise_block``); ``markov_w2`` is
    the stick-padded on-device matmul whose output is sliced back to the real
    vocab on CPU (where the sample loop runs).
    """
    mh = model.markov_head
    latent = mh.markov_w1(prev_token_ids.to("cpu")).to(
        DEVICE, mh.markov_w2.weight.dtype
    )
    bias = mh.markov_w2(latent).to("cpu")
    real_vocab = getattr(mh, "_spyre_real_vocab", bias.shape[-1])
    return bias[..., :real_vocab]


def run_draft_block(
    model,
    draft_input_ids,
    target_hidden_states,
    selected_freqs,
    ctx_valid_len: int,
):
    """Drafter block-propose forward: ctx features + noise block -> block hidden.

    The adapter's public forward (the DSpark analogue of ``_run_forward``). Runs
    ``fc`` + ``hidden_norm`` on the concatenated target context once, embeds the
    noise block (CPU lookup), builds the fixed-width concat-KV mask, and loops the
    compiled blocks. Returns the post-``norm`` block hidden states ``[1, block, H]``
    that DSpark's ``compute_logits`` + markov head consume.

    Shapes are fixed (``ctx_pad`` / ``kv_pad`` / ``block_size`` from
    ``prepare_dspark_common``) so the block compiles once; ``ctx_valid_len`` (real
    context rows) only drives the additive mask, never a shape.
    """
    spec = model._spyre_dspark
    ctx_pad, kv_pad, block_size = spec["ctx_pad"], spec["kv_pad"], spec["block_size"]

    # Context projection fc + hidden_norm, once, on the [1, ctx_pad, n*hidden]
    # concatenated target features. hidden_norm is the patched family RMSNorm.
    ctx = model.hidden_norm(model.fc(target_hidden_states))

    h = embed_noise_block(model, draft_input_ids)  # [1, block, H] on device
    q_len = h.shape[1]
    mask = build_ctx_block_mask(
        ctx_valid_len,
        ctx_pad,
        block_size,
        kv_pad,
        q_len,
        dtype=model.embed_tokens.weight.dtype,
    ).to(DEVICE)

    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, ctx, selected_freqs, mask)

    return model.norm(h)
