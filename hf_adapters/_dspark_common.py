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
    PrecomputedRotaryEmbedding,
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

        # RoPE (matmul form — the slice form mis-lowers in fp16 on Spyre).
        # selected_freqs is [B, L, 2, 2, D/2] over the ctx+block window; slice on
        # the SEQUENCE axis (1) to match each tensor's length: K spans the whole
        # ctx_len+q_len window, Q the trailing q_len (block) positions.
        kv_len = ctx_len + q_len
        q = apply_rope_matmul(q, selected_freqs[:, ctx_len:kv_len])
        k = apply_rope_matmul(k, selected_freqs[:, :kv_len])

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

    RoPE defaults to ``PrecomputedRotaryEmbedding(model.rotary_emb)`` (the standard
    ``inv_freq`` rotary used by Qwen3/Granite). A family whose rotary source differs
    may pre-set ``model._spyre_rope`` before calling this (e.g. Gemma4 wraps its
    per-attention-type ``full_attention_inv_freq`` via ``InvFreqShim``); an already
    set ``_spyre_rope`` is left untouched. The caller also passes its RMSNorm class
    and may stash an attention-scaling override on ``model._spyre_attn_scaling``.
    """
    patch_rmsnorm(rmsnorm_cls)
    pad_lm_head(model)
    _pad_markov_w2(model)
    snapshot_cpu_embeddings(model)
    snapshot_cpu_fc(model)
    install_spyre_compute_logits(model)
    install_spyre_markov(model)
    # Disable the confidence head: at confidence_threshold=0 the full block is
    # always kept, so it is unused — and its _predict_confidence_logits path runs
    # get_prev_embeddings (the Spyre markov_w1) against CPU ids, a device mix.
    # Re-enable + route through markov_bias if a nonzero threshold is ever needed.
    model.confidence_head = None

    # Standard inv_freq rotary (Qwen3/Granite); a family with a different rotary
    # source pre-sets ``_spyre_rope`` before calling (e.g. Gemma4's InvFreqShim).
    if not hasattr(model, "_spyre_rope"):
        model._spyre_rope = PrecomputedRotaryEmbedding(model.rotary_emb)

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


def snapshot_cpu_embeddings(model):
    """Keep CPU copies of the gather-only embedding weights.

    ``nn.Embedding`` is a gather with no Spyre kernel (CPU fallback), so the
    noise-block and markov ``w1`` lookups must run on CPU. But
    ``_move_to_spyre_with_layout`` (which runs AFTER ``prepare_for_spyre``) moves
    every parameter — including these embeddings — onto the Spyre device, which
    would collide CPU ids with a Spyre weight. Snapshot the weights on CPU here so
    the lookups are device-independent of where the move leaves the module; the
    on-device matmuls (``markov_w2``, ``lm_head``, backbone) use the moved weights.
    """
    model._spyre_embed_w_cpu = model.embed_tokens.weight.detach().to("cpu")
    mh = getattr(model, "markov_head", None)
    if mh is not None and hasattr(mh, "markov_w1"):
        model._spyre_markov_w1_cpu = mh.markov_w1.weight.detach().to("cpu")


def embed_noise_block(model, draft_input_ids):
    """CPU embedding lookup for the noise block, returned on device.

    Gathers against the CPU weight snapshot (``snapshot_cpu_embeddings``) and moves
    the result to the device. Applies the family ``embedding_multiplier`` (1.0 when
    the family has none) since the drafter's ``forward`` — which the block path
    bypasses — is where that scale normally lives.
    """
    emb = F.embedding(draft_input_ids.to("cpu"), model._spyre_embed_w_cpu)
    mult = float(getattr(model, "embedding_multiplier", 1.0))
    if mult != 1.0:
        emb = emb * mult
    return emb.to(DEVICE, model._spyre_embed_w_cpu.dtype)


def markov_bias(model, prev_token_ids):
    """One markov-head step bias ``markov_w2(markov_w1(prev))`` on device.

    ``markov_w1`` is a CPU gather against the weight snapshot; ``markov_w2`` is the
    stick-padded on-device matmul whose output is sliced back to the real vocab on
    CPU (where the sample loop runs).
    """
    mh = model.markov_head
    latent = F.embedding(prev_token_ids.to("cpu"), model._spyre_markov_w1_cpu)
    latent = latent.to(DEVICE, mh.markov_w2.weight.dtype)
    bias = mh.markov_w2(latent).to("cpu")
    real_vocab = getattr(mh, "_spyre_real_vocab", bias.shape[-1])
    return bias[..., :real_vocab]


def snapshot_cpu_fc(model):
    """Snapshot the context-projection ``fc`` + ``hidden_norm`` weights on CPU.

    ``fc`` is ``Linear(num_layers*hidden -> hidden)`` applied once to the
    concatenated target context. The reduction dim (num_layers*hidden, e.g.
    12800/19200/20480) has no viable Spyre matmul layout when the context tensor
    arrives with a non-canonical layout (e.g. a P2P-received buffer) —
    ``aten.bmm: no supported output layout``. It's a one-shot projection on a tiny
    tensor, off the autoregressive loop, so we run it on CPU (fp32, exact) and
    return the result on device — the same host boundary spyre_draft.py used, and
    analogous to the RoPE / embedding CPU fallbacks. Snapshot the weights here so
    the projection is independent of where ``_move_to_spyre_with_layout`` leaves
    ``fc``/``hidden_norm``.
    """
    fc = getattr(model, "fc", None)
    hn = getattr(model, "hidden_norm", None)
    if fc is None or hn is None:
        return
    model._spyre_fc_w_cpu = fc.weight.detach().to("cpu", torch.float32).contiguous()
    model._spyre_fc_b_cpu = (
        None if fc.bias is None else fc.bias.detach().to("cpu", torch.float32)
    )
    cfg = getattr(model, "config", None)
    model._spyre_hn_eps = float(
        getattr(hn, "eps", None)
        or getattr(hn, "variance_epsilon", None)
        or getattr(cfg, "rms_norm_eps", 1e-6)
    )
    model._spyre_hn_w_cpu = (
        hn.weight.detach().to("cpu", torch.float32).contiguous()
        if getattr(hn, "with_scale", True) and hasattr(hn, "weight")
        else None
    )


def project_context(model, target_hidden_states):
    """``hidden_norm(fc(ctx))`` — on CPU when snapshotted, else on device.

    When ``snapshot_cpu_fc`` ran, compute the projection + RMSNorm on host (fp32)
    against the CPU snapshots and return on device (dtype-matched). Otherwise fall
    back to the model's on-device ``fc``/``hidden_norm``.
    """
    if not hasattr(model, "_spyre_fc_w_cpu"):
        return model.hidden_norm(model.fc(target_hidden_states))
    xf = target_hidden_states.to("cpu", torch.float32)
    proj = torch.nn.functional.linear(xf, model._spyre_fc_w_cpu, model._spyre_fc_b_cpu)
    var = (proj * proj).mean(-1, keepdim=True)
    proj = proj * torch.rsqrt(var + model._spyre_hn_eps)
    if model._spyre_hn_w_cpu is not None:
        proj = proj * model._spyre_hn_w_cpu
    dtype = model.embed_tokens.weight.dtype
    return proj.to(DEVICE, dtype)


def install_spyre_compute_logits(model):
    """Override ``compute_logits`` to trim the padded lm-head vocab + family scale.

    ``pad_lm_head`` widens ``lm_head`` to a stick multiple, so the stock
    ``compute_logits = lm_head(h)`` returns padded-vocab logits — which would let
    the proposal index pad tokens and mismatches the real-vocab markov bias. Slice
    back to the true vocab (``config.vocab_size``), apply the family
    ``final_logit_softcapping`` (Gemma4) and ``logits_scaling`` divide (Granite);
    Qwen3 has neither. Runs the head on-device, returns real-vocab logits on CPU
    (where the markov loop + sampling live).
    """
    real_vocab = int(model.config.vocab_size)
    softcap = getattr(model.config, "final_logit_softcapping", None)
    softcap = float(softcap) if softcap is not None else None
    logits_scaling = getattr(model, "logits_scaling", None)
    logits_scaling = float(logits_scaling) if logits_scaling is not None else None
    lm_head = model.lm_head

    def compute_logits(hidden_states):
        logits = lm_head(hidden_states).to("cpu")[..., :real_vocab]
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        if logits_scaling is not None:
            logits = logits / logits_scaling
        return logits

    model.compute_logits = compute_logits


def install_spyre_markov(model):
    """Override the markov head's ``sample_block_tokens`` for Spyre.

    The stock loop calls ``markov_w1(prev)`` directly — but ``markov_w1`` is a
    Spyre-device Embedding after the layout move, colliding with CPU token ids in
    the (CPU-fallback) gather. Replace the loop with one that computes each step
    bias via ``markov_bias`` (CPU-snapshot ``w1`` gather + stick-padded on-device
    ``w2`` matmul, real vocab sliced back on CPU). ``base_logits`` already arrive on
    CPU (from the padded lm-head + downstream slicing), so the whole per-block
    correction runs on host with only the ``w2`` matmul on-device — no device mix.
    No-op when the drafter has no markov head.
    """
    from deepspec.utils.sampling import sample_tokens

    mh = getattr(model, "markov_head", None)
    if mh is None or not hasattr(mh, "markov_w2"):
        return

    def sample_block_tokens(
        base_logits, *, first_prev_token_ids, hidden_states, temperature=0.0
    ):
        base_logits = base_logits.to("cpu")
        bsz, n, vocab = base_logits.shape
        if n == 0:
            return torch.empty(bsz, 0, dtype=torch.long), base_logits
        corrected = torch.zeros(bsz, n, vocab, dtype=base_logits.dtype)
        sampled = torch.zeros(bsz, n, dtype=torch.long)
        prev = first_prev_token_ids.to("cpu").long()
        for k in range(n):
            step_logits = base_logits[:, k, :] + markov_bias(model, prev)
            corrected[:, k, :] = step_logits
            nxt = sample_tokens(
                step_logits.unsqueeze(1), temperature=temperature
            ).squeeze(1)
            sampled[:, k] = nxt
            prev = nxt
        return sampled, corrected

    mh.sample_block_tokens = sample_block_tokens


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

    # Context projection fc + hidden_norm, once (CPU when snapshotted — the wide
    # reduction dim has no Spyre matmul layout for a P2P-received ctx tensor).
    ctx = project_context(model, target_hidden_states)

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
