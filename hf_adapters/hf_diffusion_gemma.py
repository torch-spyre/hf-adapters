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
HuggingFace Transformers adapter for ``google/diffusiongemma-26B-A4B-it`` on Spyre.

DiffusionGemma is a block-diffusion LLM — **not** an autoregressive causal LM.
Key differences from every other adapter in this repo:

Architecture
~~~~~~~~~~~~
- **Encoder-decoder split.** A causal *encoder* processes all previous tokens and
  builds a KV cache (same as a standard AR decoder).  A fully *bidirectional
  decoder* re-runs a fixed-width "canvas" of ``config.canvas_length`` (256) tokens
  at every denoising step, attending to the encoder's frozen KV cache.  The decoder
  has no KV cache of its own; every denoising step is a full ``[B, 256, H]``
  bidirectional forward.
- **MoE at every layer.** Both the encoder and decoder have a dense MLP (SwiGLU,
  ``intermediate_size=2112``) and a sparse MoE (128 experts, top-8,
  ``moe_intermediate_size=704``) whose outputs are summed.  The MoE router and
  experts use ``nonzero()`` + a Python loop over alive experts and are not
  compilable on Spyre — they run on CPU via the stock HF eager dispatch after
  each compiled attention+dense-MLP block, matching the pattern other adapters
  use for non-compilable ops.
- **Alternating sliding / global attention** (identical to Gemma 4):
  ``sliding_attention`` layers use ``head_dim=256``, ``full_attention`` layers use
  ``head_dim=512`` with ``k_eq_v`` (no ``v_proj``), and per-head Q/K/V RMSNorm.
  Unscaled SDPA (``scale=1.0``).
- **Sandwich norms + ``layer_scalar``** (same as Gemma 4 dense).
- **Self-conditioning.** At each denoising step, the previous step's logits
  ``[B, 256, vocab]`` are embedded via a ``self_conditioning_emb`` linear and added
  to the canvas embedding before the decoder layers.

Generation loop
~~~~~~~~~~~~~~~
DiffusionGemma's ``generate`` is a *block-diffusion loop*, not AR token-by-token:

1. Encode all previously generated tokens → KV cache (causal).
2. Randomly initialize a canvas of 256 tokens.
3. For ``max_denoising_steps`` (default 48) denoising steps:
   a. Run the bidirectional decoder over the canvas + encoder KV cache.
   b. Apply temperature schedule + entropy-bound acceptance/renoising.
   c. Early-exit if the stable-and-confident stopping criterion fires.
4. Append the accepted canvas to ``input_ids``; go to step 1.

The Spyre adapter implements a custom generate loop (acceptance / renoising /
stopping on CPU) and routes all **compiled forward passes** (encoder and decoder)
through Spyre blocks instead of stock HF layers.

Spyre adaptations
~~~~~~~~~~~~~~~~~
- ``_patch_diffusion_gemma_rmsnorm``: the Gemma4-style RMSNorm (``self.eps``,
  optionally scale-free) patched to stay fp16 on Spyre (same as ``hf_gemma4``).
- Encoder blocks: compiled ``_DiffGemmaBlock`` (attention + dense MLP + KV cache write).
  Sliding layers: causal + sliding-window attention, separate V.
  Full-attention layers: causal, ``k_eq_v``, ``head_dim=512``.
- Decoder blocks: compiled ``_DiffGemmaBlock`` (attention + dense MLP, bidirectional, no KV write).
- MoE: runs on CPU after each compiled block via the stock HF ``DiffusionGemmaTextExperts``
  eager dispatch (``nonzero()`` + Python loop).  Not compiled — matches the pattern
  other adapters use for non-compilable ops.
- LM head: smooth-padded (``pad_lm_head``), runs on Spyre.
- Self-conditioning: soft embeddings computed via ``embed_tokens`` on Spyre,
  combined via ``dec.self_conditioning`` on Spyre before the decoder layers.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer, AutoProcessor

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "google/diffusiongemma-26B-A4B-it"
    )
    processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")
    chat = [{"role": "user", "content": "Why is the sky blue?"}]
    input_ids = processor.apply_chat_template(
        chat, tokenize=True, return_tensors="pt"
    )
    output = model.generate(input_ids, max_new_tokens=256)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    DEVICE,
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    apply_rope_matmul,
    kv_cache_update,
    pad_lm_head,
    text_config,
)

# ---------------------------------------------------------------------------
# RMSNorm patch (Gemma4-style: ``self.eps``, optionally scale-free)
# ---------------------------------------------------------------------------


def _patch_diffusion_gemma_rmsnorm(rmsnorm_cls):
    """Patch ``DiffusionGemmaRMSNorm.forward`` to avoid the keepdim [B,S,1] intermediate on Spyre.

    The stock ``forward`` upcasts to fp32, calls ``_norm`` (which produces a
    ``[B, S, 1]`` keepdim tensor), then downcasts.  On Spyre the keepdim tensor
    has a mis-matched eager layout that triggers RetileWarning when it crosses
    into a compiled graph.

    Replacement: use ``F.rms_norm`` (torch >= 2.4) which fuses the reduction
    inside a single op and never exposes a ``[B, S, 1]`` intermediate to eager.
    Falls back to the manual keepdim path on CPU (no retile risk there).
    """

    def _forward_patched(self, hidden_states):
        if hidden_states.device.type == "spyre":
            # F.rms_norm fuses mean+rsqrt; no [B,S,1] keepdim tensor escapes eager.
            normed = torch.nn.functional.rms_norm(
                hidden_states,
                (hidden_states.shape[-1],),
                weight=None,
                eps=self.eps,
            )
            if self.with_scale:
                normed = normed * self.weight
            return normed
        # CPU path: match stock HF fp32 upcasting exactly.
        xf = hidden_states.float()
        rms = (xf * xf).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
        xf = xf * rms
        if self.with_scale:
            xf = xf * self.weight.float()
        return xf.type_as(hidden_states)

    rmsnorm_cls.forward = _forward_patched


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------


class _DiffGemmaAttn(nn.Module):
    """Shared attention core used by both encoder and decoder compiled blocks.

    Identical to Gemma4Attention (see ``hf_gemma4``) but:
    - works for both encoder (KV-cache write) and decoder (read-only from
      encoder cache, no write) via the ``encoder_mode`` flag.
    - ``is_kv_eq_v`` mirrors the Gemma 4 global-layer pattern.
    """

    def __init__(
        self, attn, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, encoder_mode: bool
    ):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj  # None when is_kv_eq_v
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.v_norm = attn.v_norm
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_kv_eq_v = is_kv_eq_v
        self.encoder_mode = encoder_mode
        # DiffusionGemma uses scale=1.0 (same as Gemma 4)
        self.scaling = attn.scaling

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_q_heads, self.head_dim
        )
        k_lin = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        if self.is_kv_eq_v:
            # Global layers: V reuses k_proj output (pre-k_norm, pre-RoPE) through v_norm.
            v = self.v_norm(k_lin).transpose(1, 2)
        else:
            v = self.v_proj(hidden_states).view(
                bsz, seq_len, self.num_kv_heads, self.head_dim
            )
            v = self.v_norm(v).transpose(1, 2)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k_lin).transpose(1, 2)
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        if self.encoder_mode:
            # Encoder writes the KV cache (standard AR-style).
            key_cache, value_cache = kv_cache_update(
                k, v, key_cache, value_cache, cache_index
            )
            attn_k, attn_v = key_cache, value_cache
        else:
            # Decoder: write the current canvas K/V into the slots immediately
            # after the encoder prefix so SDPA sees [encoder KV | canvas KV].
            # cache_index points at positions [cache_len, cache_len + canvas_len).
            # kv_cache_update is in-place, so the same cache tensor gets the
            # canvas tokens written fresh at each denoising step before SDPA.
            attn_k, attn_v = kv_cache_update(k, v, key_cache, value_cache, cache_index)

        attn_out = F.scaled_dot_product_attention(
            q,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), key_cache, value_cache


class _DiffGemmaBlock(nn.Module):
    """Compiled block for one DiffusionGemma layer on Spyre: attention + dense MLP only.

    The MoE path (router + experts) uses ``nonzero()`` and a Python loop over
    alive experts, which the Spyre Inductor backend cannot compile.  It runs on
    CPU after this block returns, matching the pattern used by other adapters
    for non-compilable ops.

    Returns ``(dense_1, residual, key_cache, value_cache)``.  The caller
    finishes the layer by running the MoE on CPU and combining.

    ``encoder_mode`` is a compile-time constant — encoder blocks write the KV
    cache, decoder blocks read it without writing.
    """

    def __init__(
        self, layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, encoder_mode: bool
    ):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.self_attn = _DiffGemmaAttn(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
            encoder_mode,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.train(layer.training)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            h,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        # Sandwich: norm the attention output BEFORE adding the residual.
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        # Dense MLP branch (on Spyre).
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        dense_1 = self.post_feedforward_layernorm_1(h)

        # MoE runs on CPU after this block returns.
        return dense_1, residual, key_cache, value_cache


def _compile_block(
    layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, encoder_mode
):
    block = _DiffGemmaBlock(
        layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, encoder_mode
    )
    return block, torch.compile(block, dynamic=False)


# ---------------------------------------------------------------------------
# KV-cache shapes
# ---------------------------------------------------------------------------


def _kv_shapes_for_model(model, cfg):
    """Per-layer ``(num_kv_heads, head_dim, v_head_dim)`` for the encoder/decoder.

    ``head_dim`` and ``num_key_value_heads`` are per-layer attributes on
    ``DiffusionGemmaTextConfig`` — accessing them globally raises
    ``AmbiguousGlobalPerLayerAttributeError``. Always read from
    ``per_layer_config[i]``.

    Under tensor parallelism the ``k_proj`` weight is sharded across ranks, so
    the effective ``num_kv_heads`` per device is ``config_kv_heads / tp_size``.
    We derive it from the actual weight shape rather than the config value so
    both TP and non-TP paths are handled automatically.
    """
    enc_text = model.model.encoder.language_model
    num_layers = cfg.num_hidden_layers
    shapes = []
    for i in range(num_layers):
        layer_cfg = cfg.per_layer_config[i]
        head_dim = layer_cfg.head_dim
        # Read actual kv_heads from the loaded (possibly sharded) k_proj weight.
        k_proj = enc_text.layers[i].self_attn.k_proj
        kv_heads = k_proj.weight.shape[0] // head_dim
        shapes.append((kv_heads, head_dim, head_dim))
    return shapes


# ---------------------------------------------------------------------------
# Forward helpers: build masks and run compiled blocks
# ---------------------------------------------------------------------------


def _build_encoder_masks(model, attn_mask, seq_len, batch_size, block_base, cfg):
    """Per-layer-type causal (+ optional sliding) masks for the encoder."""
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(
        batch_size, seq_len
    )
    sliding_mask = add_causal_sliding_window_band(
        attn_mask, query_coords, cfg.sliding_window
    )
    return {"full_attention": attn_mask, "sliding_attention": sliding_mask}


def _build_decoder_mask(
    seq_len, batch_size, encoder_cache_len, prompt_offsets, dtype, max_cache_len
):
    """Bidirectional decoder mask: canvas attends to all encoder keys + itself.

    Shape: ``[B, 1, canvas_len, max_cache_len]``.
    The mask spans the full KV cache width so it broadcasts against the
    pre-allocated ``key_cache`` tensor (shape ``[B, n_kv, max_cache_len, hd]``).
    Positions beyond ``encoder_cache_len + canvas_len`` are masked out (cache tail).
    The canvas positions are fully bidirectional (no causal restriction).
    Encoder positions masked for left-padding only.
    """
    from hf_adapters.hf_common import _mask_fill_value

    fill = _mask_fill_value(dtype)
    active_kv = encoder_cache_len + seq_len
    mask = torch.full((batch_size, 1, seq_len, max_cache_len), fill, dtype=dtype)
    # Unmask the active region (left-pad of encoder + canvas)
    mask[:, :, :, :active_kv] = 0
    # Re-mask left-pad of the encoder portion
    if isinstance(prompt_offsets, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, : int(prompt_offsets[b].item())] = fill
    elif prompt_offsets > 0:
        mask[:, :, :, :prompt_offsets] = fill
    return mask


def _finish_layer(layer, dense_1, residual, layer_scalar):
    """Combine dense MLP + MoE outputs and apply the layer residual + scalar.

    The MoE weights (router, experts, post-norms) live on CPU
    (``_spyre_cpu_submodules``).  ``dense_1`` and ``residual`` come from the
    Spyre-compiled block.  We move both to CPU, run the full MoE path there,
    then move the combined result back to Spyre.

    Mirrors the tail of ``DiffusionGemmaEncoderTextLayer.forward``:
        flat      = residual.reshape(-1, H)
        expert_in = pre_feedforward_layernorm_2(flat)
        _, w, idx = router(flat)
        moe_out   = experts(expert_in, idx, w).reshape(residual.shape)
        moe_2     = post_feedforward_layernorm_2(moe_out)
        combined  = post_feedforward_layernorm(dense_1 + moe_2)
        h         = (residual + combined) * layer_scalar
    """
    dev = residual.device
    r_cpu = residual.to("cpu")
    d_cpu = dense_1.to("cpu")

    flat = r_cpu.reshape(-1, r_cpu.shape[-1])
    expert_in = layer.pre_feedforward_layernorm_2(flat)
    _, top_k_weights, top_k_index = layer.router(flat)
    moe_out = layer.experts(expert_in, top_k_index, top_k_weights)
    moe_out = moe_out.reshape(r_cpu.shape)
    moe_2 = layer.post_feedforward_layernorm_2(moe_out)

    combined = layer.post_feedforward_layernorm(d_cpu + moe_2)
    h_cpu = (r_cpu + combined) * layer_scalar
    # .contiguous() is required: a plain .to(dev) on a CPU tensor produces
    # row-major layout; the compiled block's next input expects stickified
    # layout — mismatch triggers RetileWarning and corrupts activations.
    return h_cpu.to(dev).contiguous()


def _run_encoder_blocks(
    model,
    h,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Run compiled encoder blocks (attention + dense MLP on Spyre, MoE on CPU)."""
    cfg = text_config(model.config)
    # encoder layers/norm live under model.model.encoder.language_model
    enc_text = model.model.encoder.language_model

    freqs = {
        lt: model._spyre_enc_rope[lt](h, position_ids) for lt in model._spyre_enc_rope
    }

    bsz, seq_len = h.shape[0], h.shape[1]
    block_base = int(cache_index[0])
    masks = _build_encoder_masks(model, attn_mask, seq_len, bsz, block_base, cfg)

    for i, compiled_block in enumerate(model._spyre_enc_compiled_blocks):
        lt = cfg.layer_types[i]
        ls = float(enc_text.layers[i].layer_scalar)
        dense_1, residual, key_caches[i], value_caches[i] = compiled_block(
            h, freqs[lt], masks[lt], key_caches[i], value_caches[i], cache_index, ls
        )
        h = _finish_layer(enc_text.layers[i], dense_1, residual, ls)

    enc_text.norm(h)  # final encoder norm — result discarded, KV caches are the output
    return h


def _make_soft_embeddings(model, logits_cpu, dtype):
    """Convert raw CPU logits to soft embeddings on CPU.

    Mirrors ``DiffusionGemmaDecoder.forward``:
        soft = softmax(logits) @ embed_weight * embed_scale

    Computed on CPU from CPU logits so no Spyre retile occurs when the result
    is moved to device — it enters the compiled graph as a plain [B, 256, H]
    tensor with no prior eager-op layout annotation.
    """
    dec = model.model.decoder
    embed_w = dec.embed_tokens.weight.to("cpu")
    embed_scale = float(dec.embed_tokens.embed_scale)
    soft = (
        torch.matmul(
            logits_cpu.softmax(dim=-1, dtype=torch.float32).to(embed_w.dtype),
            embed_w,
        )
        * embed_scale
    )
    return soft.to(dtype).contiguous()


def _run_decoder_blocks(
    model,
    canvas_embeds,
    position_ids,
    decoder_attn_mask,
    key_caches,
    value_caches,
    decoder_cache_index,  # [canvas_len] pointing at [cache_len, cache_len+canvas_len)
    soft_conditioning,  # [B, 256, H] on device, or None
):
    """Run compiled decoder blocks (attention + dense MLP on Spyre, MoE on CPU)."""
    cfg = text_config(model.config)
    # decoder layers/norm live directly on model.model.decoder
    dec = model.model.decoder

    h = canvas_embeds

    # Fix #2: always run self_conditioning, even on the first denoising step.
    # When there are no previous logits HF uses zeros, not an identity bypass.
    # dec.self_conditioning ends with an RMSNorm that is NOT a no-op on zeros.
    if soft_conditioning is None:
        soft_conditioning = torch.zeros_like(h)
    h = dec.self_conditioning(h, soft_conditioning)

    freqs = {
        lt: model._spyre_dec_rope[lt](h, position_ids) for lt in model._spyre_dec_rope
    }

    for i, compiled_block in enumerate(model._spyre_dec_compiled_blocks):
        lt = cfg.layer_types[i]
        ls = float(dec.layers[i].layer_scalar)
        # Decoder mask per-layer-type (full_attention uses plain bidirectional,
        # sliding_attention intersects with the sliding window).
        if isinstance(decoder_attn_mask, dict):
            mask = decoder_attn_mask[lt]
        else:
            mask = decoder_attn_mask
        dense_1, residual, _, _ = compiled_block(
            h, freqs[lt], mask, key_caches[i], value_caches[i], decoder_cache_index, ls
        )
        h = _finish_layer(dec.layers[i], dense_1, residual, ls)

    return h


# ---------------------------------------------------------------------------
# Encoder forward (called by the diffusion generate loop)
# ---------------------------------------------------------------------------


def _run_encoder_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Encode ``input_ids`` into the KV cache and return hidden states."""
    # embed_tokens lives under encoder.language_model
    h = model.model.encoder.language_model.embed_tokens(input_ids)
    return _run_encoder_blocks(
        model, h, position_ids, attn_mask, key_caches, value_caches, cache_index
    )


# ---------------------------------------------------------------------------
# Decoder forward (called at every denoising step)
# ---------------------------------------------------------------------------


def _run_decoder_forward(
    model,
    canvas_ids,
    position_ids,
    decoder_attn_mask,
    key_caches,
    value_caches,
    decoder_cache_index,  # [canvas_len] pointing at [cache_len, cache_len+canvas_len)
    soft_conditioning,  # [B, 256, H] on device, or None
):
    """Run the bidirectional decoder over a canvas, return logits."""
    # embed_tokens lives directly on model.model.decoder
    canvas_embeds = model.model.decoder.embed_tokens(canvas_ids)

    h = _run_decoder_blocks(
        model,
        canvas_embeds,
        position_ids,
        decoder_attn_mask,
        key_caches,
        value_caches,
        decoder_cache_index,
        soft_conditioning,
    )

    return model._spyre_dec_head(h)


# ---------------------------------------------------------------------------
# generate — Spyre-compatible block-diffusion loop
# ---------------------------------------------------------------------------


def generate(
    model, tokenizer, prompts, max_new_tokens=256, max_denoising_steps=48, **kwargs
):
    """Spyre block-diffusion generate loop for DiffusionGemma.

    Outer loop: AR canvas generation (encode + N denoising steps per canvas).
    Inner loop: ``max_denoising_steps`` denoising passes per canvas.

    Sampling uses the HF ``EntropyBoundSampler`` and
    ``StableAndConfidentStoppingCriteria`` directly — identical behaviour to
    stock HF inference, no hand-rolled approximation.

    Args:
        model: prepared DiffusionGemma model on Spyre.
        tokenizer: HF tokenizer.
        prompts: list of str prompts.
        max_new_tokens: total number of new tokens to generate.
        max_denoising_steps: denoising steps per canvas (default 48).
        entropy_bound: entropy bound for the EB sampler (default 0.1).
        stability_threshold: stability steps for stopping (default 1).
        confidence_threshold: confidence threshold for stopping (default 0.005).

    Returns:
        list[str]: decoded outputs (one per prompt).
    """
    from transformers import StableAndConfidentStoppingCriteria
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )

    from hf_adapters.hf_common import (
        BLOCK_SIZE,
        allocate_kv_caches,
        build_prefill_mask,
        generation_cache_len,
        make_cache_index,
    )

    entropy_bound = kwargs.pop("entropy_bound", 0.1)
    stability_threshold = kwargs.pop("stability_threshold", 1)
    confidence_threshold = kwargs.pop("confidence_threshold", 0.005)
    t_min = kwargs.pop("t_min", 0.4)
    t_max = kwargs.pop("t_max", 0.8)

    canvas_length = model.config.canvas_length  # 256
    vocab_size = text_config(model.config).vocab_size
    dtype = next(model.parameters()).dtype
    batch_size = len(prompts)

    # Apply chat template for instruct models (Gemma chat format).
    if getattr(tokenizer, "chat_template", None) is not None:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    # Tokenize + left-pad to BLOCK_SIZE multiple
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"]
    actual_lengths = enc["attention_mask"].sum(dim=1).long()
    prompt_length = input_ids.shape[1]
    padded_len = math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
    block_pad = padded_len - prompt_length
    if block_pad > 0:
        input_ids = torch.cat(
            [input_ids.new_zeros((batch_size, block_pad)), input_ids], dim=1
        )
    prompt_offsets = padded_len - actual_lengths  # [B] left-pad counts

    position_ids_prompt = torch.zeros((batch_size, padded_len), dtype=torch.long)
    for b in range(batch_size):
        n = int(actual_lengths[b])
        position_ids_prompt[b, int(prompt_offsets[b]) :] = torch.arange(n)

    # KV caches
    max_canvases = math.ceil(max_new_tokens / canvas_length)
    max_cache_len = generation_cache_len(padded_len, max_new_tokens + canvas_length)
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, dtype
    )

    # HF sampler + stopping criteria (CPU, pure Python — identical to stock HF)
    sampler = EntropyBoundSampler(
        config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
        canvas_length=canvas_length,
        vocab_size=vocab_size,
        max_denoising_steps=max_denoising_steps,
    )
    stopping_criteria = StableAndConfidentStoppingCriteria(
        stability_threshold=stability_threshold,
        confidence_threshold=confidence_threshold,
    )

    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    finished = [False] * batch_size
    generated_ids = input_ids.tolist()

    # --- Encode prompt ---
    attn_mask = build_prefill_mask(
        batch_size, padded_len, max_cache_len, prompt_offsets, dtype=dtype
    ).to(DEVICE)
    _run_encoder_forward(
        model,
        input_ids.to(DEVICE),
        position_ids_prompt.to(DEVICE),
        attn_mask,
        key_caches,
        value_caches,
        make_cache_index(0, padded_len, device=DEVICE),
    )
    cache_len = padded_len

    # --- Canvas generation loop ---
    for _canvas_idx in range(max_canvases):
        if all(finished):
            break

        canvas_pos_start = cache_len
        canvas_position_ids = (
            torch.arange(
                canvas_pos_start, canvas_pos_start + canvas_length, dtype=torch.long
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(DEVICE)
        )
        decoder_attn_mask = _build_decoder_mask(
            canvas_length, batch_size, cache_len, prompt_offsets, dtype, max_cache_len
        ).to(DEVICE)

        # Index pointing the decoder canvas writes at [cache_len, cache_len+canvas_len).
        # Fix #1: the decoder uses this so SDPA sees [encoder KV | canvas KV].
        decoder_cache_index = make_cache_index(cache_len, canvas_length, device=DEVICE)

        # Initialize canvas + reset stopping criteria
        current_canvas = (
            sampler.initialize_canvas(batch_size, device="cpu").to(DEVICE).contiguous()
        )
        soft_conditioning = None  # [B, 256, H] CPU→Spyre, built each step
        argmax_canvas = current_canvas.clone()
        stopping_criteria.reset()

        # --- Denoising loop ---
        for step_idx in range(max_denoising_steps, 0, -1):
            logits = _run_decoder_forward(
                model,
                current_canvas,
                canvas_position_ids,
                decoder_attn_mask,
                key_caches,
                value_caches,
                decoder_cache_index,
                soft_conditioning,
            )
            # All sampling runs on CPU with the HF classes.
            # Fix #3: apply the HF temperature schedule before every downstream use
            # (sampling, argmax, acceptance check, self-conditioning).
            logits_cpu = logits.to("cpu")
            temperature = t_min + (t_max - t_min) * (step_idx / max_denoising_steps)
            processed_logits = logits_cpu / temperature

            # Sample denoiser canvas with temperature-scaled logits
            probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
            denoiser_canvas = (
                torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
                .squeeze(-1)
                .view(batch_size, canvas_length)
            )
            new_argmax = torch.argmax(processed_logits, dim=-1)

            # HF acceptance + renoising (uses processed logits for entropy bound)
            accepted = sampler.accept_canvas(
                current_canvas.cpu(), denoiser_canvas, processed_logits, step_idx
            )
            current_canvas = (
                sampler.renoise_canvas(accepted, step_idx).to(DEVICE).contiguous()
            )
            argmax_canvas = new_argmax.to(DEVICE)

            # Self-conditioning: build soft embeddings on CPU from processed logits,
            # then move to Spyre as a plain [B, 256, H] tensor — avoids the
            # RetileWarning that fired when moving vocab-sized logits to Spyre.
            soft_conditioning = (
                _make_soft_embeddings(model, processed_logits, dtype)
                .to(DEVICE)
                .contiguous()
            )

            # HF stopping criteria (uses processed logits)
            if stopping_criteria(new_argmax, processed_logits):
                break

        # Append accepted canvas
        new_tokens = argmax_canvas.cpu()
        for b in range(batch_size):
            if not finished[b]:
                generated_ids[b].extend(new_tokens[b].tolist())
                if eos_ids and any(t in eos_ids for t in new_tokens[b].tolist()):
                    finished[b] = True

        # Encode canvas into KV cache for next AR step
        _run_encoder_forward(
            model,
            new_tokens.to(DEVICE),
            canvas_position_ids,
            build_prefill_mask(
                batch_size, canvas_length, max_cache_len, prompt_offsets, dtype=dtype
            ).to(DEVICE),
            key_caches,
            value_caches,
            make_cache_index(cache_len, canvas_length, device=DEVICE),
        )
        cache_len += canvas_length

    # Decode — strip the padded prompt, return only generated tokens
    return [
        tokenizer.decode(generated_ids[b][padded_len:], skip_special_tokens=True)
        for b in range(batch_size)
    ]


# ---------------------------------------------------------------------------
# Custom loader + prepare_for_spyre
# ---------------------------------------------------------------------------


def _apply_tp_sharding(model):
    """Slice attention/MLP Linear weights in-place for this TP rank.

    ``"colwise"`` → shard output dim 0 of weight [out, in].
    ``"rowwise"`` → shard input dim 1 of weight [out, in].

    MoE experts are NOT sharded — they run on CPU where all ranks have the
    full expert tensors and execute the same router + expert dispatch.
    """
    tp_plan = getattr(model, "_spyre_tp_plan", None)
    if not tp_plan:
        return

    import os

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    tp_size = model._spyre_tp_size

    for module_path, style in tp_plan.items():
        try:
            parent_path, _, attr = module_path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            submod = getattr(parent, attr)
        except (AttributeError, ValueError):
            continue
        if not isinstance(submod, nn.Linear):
            continue

        w = submod.weight.data  # [out, in]
        b = submod.bias.data if submod.bias is not None else None

        if "colwise" in style:
            out_dim = w.shape[0]
            chunk = out_dim // tp_size
            start, end = rank * chunk, (rank + 1) * chunk
            submod.weight = nn.Parameter(w[start:end].clone(), requires_grad=False)
            if b is not None:
                submod.bias = nn.Parameter(b[start:end].clone(), requires_grad=False)
        elif "rowwise" in style:
            in_dim = w.shape[1]
            chunk = in_dim // tp_size
            start, end = rank * chunk, (rank + 1) * chunk
            submod.weight = nn.Parameter(w[:, start:end].clone(), requires_grad=False)


def load_hf_model(model_path, dtype, tp_plan=None):
    """Custom loader: ``DiffusionGemmaForBlockDiffusion`` is not in AutoModelForCausalLM.

    Always loads weights to CPU regardless of ``tp_plan``.  HF's TP path
    (``distributed_config``) targets Spyre directly and runs a
    ``caching_allocator_warmup`` that tries to pre-allocate ~50 GB on device
    before any weights land — OOM on a 103 GB card when the full model is in
    flight.  Instead, we load to CPU, stash the resolved TP plan on the model,
    and let ``prepare_for_spyre`` slice the Linear weights before
    ``move_model_to_spyre`` moves everything to Spyre via ``load_model_to_spyre``.
    Inference runs on Spyre; the CPU phase is weight loading only.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion,
    )

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)

    if tp_plan is not None:
        from transformers import AutoConfig

        from hf_adapters.hf_common import _resolve_tp_size

        if tp_plan == "auto":
            # DiffusionGemmaForBlockDiffusion has no public ``from_config``;
            # use ``_from_config`` on a meta-device probe instead.
            cfg = AutoConfig.from_pretrained(model_path)
            with torch.device("meta"):
                probe = DiffusionGemmaForBlockDiffusion._from_config(cfg)
            resolved_plan = dict(probe.tp_plan or {})
            resolved_plan.pop("lm_head", None)
            resolved_plan.pop("model.embed_tokens", None)
        else:
            resolved_plan = dict(tp_plan)
        # Drop both embed_tokens variants (sharding involves unsupported bool int32 ops)
        resolved_plan.pop("model.encoder.language_model.embed_tokens", None)
        resolved_plan.pop("model.decoder.embed_tokens", None)
        # Stash for prepare_for_spyre, which applies sharding before moving to Spyre.
        model._spyre_tp_plan = resolved_plan
        model._spyre_tp_size = _resolve_tp_size()

    return model


def prepare_for_spyre(model):
    """Apply Spyre adaptations to ``DiffusionGemmaForBlockDiffusion`` in-place.

    Steps:
    1. Patch ``DiffusionGemmaRMSNorm`` for the fp16 Spyre path.
    2. Build per-type ``PrecomputedRotaryEmbedding`` for encoder and decoder.
    3. Record per-layer KV-cache shapes.
    4. Pad the LM head.
    5. Apply TP sharding of attention/MLP linears (if requested).
    6. Compile each layer block (attention + dense MLP) for encoder and decoder.
       MoE runs on CPU via the stock HF eager dispatch after each compiled block.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaRMSNorm,
    )

    cfg = text_config(model.config)

    # 1. Patch RMSNorm for the fp16 Spyre path.
    _patch_diffusion_gemma_rmsnorm(DiffusionGemmaRMSNorm)

    # encoder text layers live under model.model.encoder.language_model
    enc_text = model.model.encoder.language_model
    # decoder layers live directly on model.model.decoder
    dec = model.model.decoder
    layer_types = cfg.layer_types
    # head_dim and num_key_value_heads are per-layer — read from per_layer_config[i].
    num_q_heads = cfg.num_attention_heads

    # 2. Apply TP sharding of attention/MLP Linear weights on CPU so only
    #    ~1/tp_size of the parameters land on each Spyre device.
    _apply_tp_sharding(model)

    # 3. Per-type RoPE (Gemma4/DiffusionGemma store per-type inv_freq on the
    #    rotary_emb module as ``<layer_type>_inv_freq`` + ``<layer_type>_attention_scaling``).
    def _build_rope_dict(rotary_emb):
        rope_dict = {}
        for lt in set(layer_types):
            inv_freq = getattr(rotary_emb, f"{lt}_inv_freq")
            scaling = float(getattr(rotary_emb, f"{lt}_attention_scaling", 1.0))
            rope_dict[lt] = PrecomputedRotaryEmbedding(InvFreqShim(inv_freq, scaling))
        return rope_dict

    model._spyre_enc_rope = _build_rope_dict(enc_text.rotary_emb)
    model._spyre_dec_rope = _build_rope_dict(dec.rotary_emb)

    # Set freq-cache dtype to match model weights and prebuild the caches on CPU
    # now, before move_model_to_spyre runs.  The common set_rope_dtype /
    # prebuild_rope_cache helpers only inspect model._spyre_rope and would miss
    # these two per-encoder/decoder dicts entirely, leaving the caches at the
    # default fp16 dtype (wrong for bf16 models) and unbuilt (so the first
    # on-device forward triggers lazy construction inside the compiled graph,
    # corrupting results — the same bug fixed by prebuild_rope_cache for other
    # adapters).
    _model_dtype = next(model.parameters()).dtype
    _context_len = max(
        getattr(cfg, "max_position_embeddings", 0),
        2048,
    )
    for _rope_dict in (model._spyre_enc_rope, model._spyre_dec_rope):
        for _rope in _rope_dict.values():
            _rope.set_dtype(_model_dtype)
            _rope._extend_cache(_context_len)

    # 3. Per-layer KV-cache shapes (reads actual weight shapes to handle TP sharding)
    kv_shapes = _kv_shapes_for_model(model, cfg)
    model._spyre_kv_shapes = kv_shapes

    # 4. Pad LM head
    pad_lm_head(model)

    # 5. Compile blocks (encoder + decoder): attention + dense MLP only.
    #    MoE (router + experts) stays in the HF layer objects and runs on CPU
    #    via _finish_layer after each compiled block returns.
    is_kv_eq_v_per_layer = [
        (lt == "full_attention") and (enc_text.layers[i].self_attn.v_proj is None)
        for i, lt in enumerate(layer_types)
    ]

    enc_compiled = []
    dec_compiled = []
    for i, (layer_enc, layer_dec) in enumerate(zip(enc_text.layers, dec.layers)):
        layer_cfg = cfg.per_layer_config[i]
        head_dim = layer_cfg.head_dim
        kv_heads = layer_cfg.num_key_value_heads
        is_kv_eq_v = is_kv_eq_v_per_layer[i]

        _, compiled_enc = _compile_block(
            layer_enc, num_q_heads, kv_heads, head_dim, is_kv_eq_v, encoder_mode=True
        )
        _, compiled_dec = _compile_block(
            layer_dec, num_q_heads, kv_heads, head_dim, is_kv_eq_v, encoder_mode=False
        )
        enc_compiled.append(compiled_enc)
        dec_compiled.append(compiled_dec)

    model._spyre_enc_compiled_blocks = enc_compiled
    model._spyre_dec_compiled_blocks = dec_compiled

    # Compile dec_norm + lm_head + softcap together so the norm's [B, S, 1]
    # keepdim intermediate never crosses the eager→compiled boundary.
    # Without this, the norm runs in eager on a Spyre tensor, producing a
    # mis-laid-out result that retiles (and corrupts) when lm_head consumes it.
    dec_norm = dec.norm
    lm_head = model.lm_head
    softcap = getattr(cfg, "final_logit_softcapping", None)

    @torch.compile(dynamic=False)
    def _dec_head(h):
        h = dec_norm(h)
        logits = lm_head(h)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    model._spyre_dec_head = _dec_head

    # Keep MoE submodules on CPU — they run via the stock HF eager dispatch
    # (nonzero + Python loop) which is not compatible with Spyre.
    #
    # Strategy: detach them from the module tree NOW (before move_model_to_spyre
    # calls model.to(DEVICE)) by replacing each with a bare nn.Module()
    # placeholder.  The real objects are saved in a dict keyed by
    # (parent_module, attr_name).  After the device move, _spyre_post_move
    # restores every real module back so _finish_layer can call them.
    #
    # This avoids trying to DMA ~24 GB of expert weights to Spyre only to move
    # them back — which OOMs before _spyre_cpu_submodules would have a chance
    # to run.
    _moe_saved: dict = {}
    _moe_attr_names = [
        "router",
        "experts",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_2",
        "post_feedforward_layernorm",
    ]
    for i, (layer_enc, layer_dec) in enumerate(zip(enc_text.layers, dec.layers)):
        for layer in (layer_enc, layer_dec):
            for attr in _moe_attr_names:
                real = getattr(layer, attr)
                _moe_saved[(id(layer), attr)] = (layer, attr, real)
                setattr(layer, attr, nn.Module())

    def _restore_moe():
        for layer, attr, real in _moe_saved.values():
            setattr(layer, attr, real)

    model._spyre_post_move = _restore_moe
    model._spyre_cpu_submodules = []
