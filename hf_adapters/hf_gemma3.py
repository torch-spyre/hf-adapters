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
HuggingFace Transformers adapter for Gemma 3 (dense) causal-LM models on Spyre.

Targets every dense Gemma 3 size — 1B (``model_type`` ``gemma3_text``) and the
4B/12B/27B text decoders nested in the multimodal ``gemma3`` checkpoints. Gemma 3
is structurally Gemma 4 (``hf_gemma4``) minus its hardest features, so it gets
its own compiled block rather than ``make_standard_gqa_block``:

- **Local / global alternating attention.** ``config.layer_types`` mixes
  ``sliding_attention`` (the majority) with ``full_attention``. Each type
  carries its own RoPE theta (sliding 1e4, global 1e6) but, unlike Gemma 4, a
  *single* ``head_dim`` — so both layer types share one KV-cache shape. See
  ``model._spyre_kv_shapes``.
- **Q / K RMSNorm.** Per-head RMSNorm on Q and K (no V-norm, unlike Gemma 4),
  applied before RoPE. Runs inside the compiled block.
- **Embedding scaling.** ``embed_tokens`` multiplies by ``sqrt(hidden_size)``
  (``Gemma3TextScaledWordEmbedding``); part of the loaded module, runs as-is.
- **"Sandwich" norms.** Four norms per layer: ``input_layernorm`` (pre-attn),
  ``post_attention_layernorm`` (on the *attn output* before the residual add),
  ``pre_feedforward_layernorm`` (pre-MLP), and ``post_feedforward_layernorm``
  (on the *MLP output* before the residual add).
- **Unit-offset RMSNorm.** ``Gemma3RMSNorm`` scales by ``(1.0 + weight)`` (weights
  stored centered at 0) and is *always* scaled (no ``with_scale=False`` V-norm).
  This is the one substantive numeric difference from ``hf_common.patch_rmsnorm``.
- **Scaled attention via ``query_pre_attn_scalar``.** ``scaling ==
  query_pre_attn_scalar ** -0.5``, which is NOT ``head_dim ** -0.5`` in general
  (e.g. 27B: ``head_dim=128`` but ``query_pre_attn_scalar=168``). Captured from
  ``attn.scaling`` so the per-checkpoint value is used.
- **Large vocab.** 262K vocab → chunked LM head (like ``hf_gemma4`` / ``hf_phi3``).
  ``final_logit_softcapping`` is ``None`` on published Gemma 3 (dropped from
  Gemma 2); the cap is applied only if a checkpoint sets it.

``AutoModelForCausalLM`` loads the text-only ``Gemma3ForCausalLM`` (1B) whose
decoder is ``model.model``, or — for 4B/12B/27B — the multimodal
``Gemma3ForConditionalGeneration`` whose text decoder is nested at
``model.model.language_model``. The shared ``get_backbone`` resolves both (it
descends into ``.language_model`` when present); ``lm_head`` stays at the top
level (``model.lm_head``), matching where ``chunk_lm_head`` looks.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    apply_rope_matmul,
    chunk_lm_head,
    get_backbone,
    kv_cache_update,
)


def _text_config(model):
    """Return the Gemma 3 *text* decoder config.

    The multimodal checkpoints carry a composite ``Gemma3Config`` whose decoder
    fields (``head_dim``, ``layer_types``, ``sliding_window``, ...) live on the
    nested ``text_config``. The text-only ``Gemma3TextConfig`` (1B) exposes them
    directly. This returns whichever holds the decoder fields.
    """
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def _patch_gemma3_rmsnorm(rmsnorm_cls):
    """Patch a Gemma3 ``RMSNorm`` class to stay in fp16 on Spyre.

    Mirrors ``hf_common.patch_rmsnorm`` but for Gemma3's RMSNorm, which:
      - uses ``self.eps`` (not ``variance_epsilon``),
      - is **unit-offset**: scales by ``(1.0 + weight)`` rather than ``weight``
        (Gemma stores norm weights centered at 0),
      - is always scaled (no scale-free variant — there is no V-norm).

    On Spyre we stay in fp16; on CPU we upcast to fp32 to match stock HF, whose
    ``Gemma3RMSNorm`` computes the norm and the ``(1.0 + weight)`` multiply in
    fp32 before casting back.
    """

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            normed = hidden_states * torch.rsqrt(variance + self.eps)
            return normed * (1.0 + self.weight)
        # CPU path: fp32 for numerical parity with stock HF.
        xf = hidden_states.float()
        variance = (xf * xf).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        xf = xf * (1.0 + self.weight.float())
        return xf.type_as(hidden_states)

    rmsnorm_cls.forward = _forward_fp16


def _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim):
    """Compile one Gemma 3 dense decoder layer.

    Block signature carries the per-layer mask and RoPE freqs (which differ
    between sliding and global layers), so the caller selects them:

        block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position)
            -> (hidden_states, key_cache, value_cache)

    Gemma applies Q/K RMSNorm before RoPE and uses the four-norm "sandwich"
    structure. Attention is scaled by ``query_pre_attn_scalar ** -0.5`` (captured
    from ``attn.scaling``), which is not ``head_dim ** -0.5`` in general.
    """
    attn = layer.self_attn
    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj
    o_proj = attn.o_proj
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    scaling = attn.scaling  # query_pre_attn_scalar ** -0.5

    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    mlp = layer.mlp

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape

        # Q/K/V projections viewed as [B, L, n_heads, head_dim] then transposed
        # to [B, n_heads, L, head_dim]. Q/K norms run per-head over head_dim
        # (the last dim) — invariant to transpose order, so we norm after the
        # transpose to match stock HF's ordering exactly.
        q = q_proj(h).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q = q_norm(q)
        k = k_norm(k)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)
        # Sandwich: norm the attention output BEFORE adding the residual.
        attn_out = post_attn_ln(attn_out)
        h = residual + attn_out

        residual = h
        h = pre_ff_ln(h)
        h = mlp(h)
        h = post_ff_ln(h)
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Gemma 3 backbone: scaled embedding, per-type RoPE + masks, blocks, norm.

    The base ``attn_mask`` is the causal mask the caller built (column index =
    cache slot). Sliding layers intersect it with a causal sliding-window band
    using each query row's cache coordinate ``block_base + j`` where
    ``block_base = cache_position - token_index`` (see ``hf_gemma4`` /
    ``add_causal_sliding_window_band``). Global layers use the base mask as-is.
    """
    backbone = get_backbone(model)
    cfg = _text_config(model)

    h = backbone.embed_tokens(input_ids)

    # Per-layer-type RoPE freqs (sliding theta vs global theta).
    freqs = {
        layer_type: rope(h, position_ids)
        for layer_type, rope in model._spyre_rope.items()
    }

    # Sliding mask: base causal mask restricted to a backward window. Query row
    # j occupies cache coordinate block_base + j. Built on CPU (int arange +
    # scalar offset); add_causal_sliding_window_band keeps the int/bool work off
    # Spyre and returns a float additive mask on attn_mask's device.
    bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
    block_base = cache_position - token_index
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(bsz, seq_len)
    sliding_mask = add_causal_sliding_window_band(
        attn_mask, query_coords, cfg.sliding_window
    )
    masks = {"full_attention": attn_mask, "sliding_attention": sliding_mask}

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = backbone.norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Gemma 3 causal-LM forward: backbone + chunked LM head + optional softcap."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )

    # Chunked LM head: 262K vocab exceeds Spyre's per-core EAR limit. Split into
    # N chunks, run each, cat on CPU.
    logits_parts = []
    for lm_chunk, real_sz in zip(
        model._spyre_lm_head_chunks, model._spyre_lm_chunk_sizes
    ):
        logits_parts.append(lm_chunk(h).to("cpu")[..., :real_sz])
    logits = torch.cat(logits_parts, dim=-1)

    # final_logit_softcapping is None on published Gemma 3 (dropped from Gemma 2);
    # applied defensively if a checkpoint sets it.
    cap = getattr(_text_config(model), "final_logit_softcapping", None)
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a dense Gemma 3 causal-LM model in-place.

    1. Patch ``Gemma3RMSNorm`` for the fp16 Spyre path.
    2. Build one ``PrecomputedRotaryEmbedding`` per layer type from the model's
       per-type ``inv_freq`` buffers (no head padding — D/2 >= 64 already).
    3. Record per-layer KV-cache shapes (single head_dim for all layers).
    4. Chunk the LM head for the large vocab.
    5. Compile each decoder layer's block.
    """
    backbone = get_backbone(model)
    cfg = _text_config(model)

    # Patch whichever concrete RMSNorm class this model uses. The norm module
    # closest to a decoder layer's input_layernorm is representative.
    rmsnorm_cls = type(backbone.layers[0].input_layernorm)
    _patch_gemma3_rmsnorm(rmsnorm_cls)

    head_dim = cfg.head_dim
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads

    assert head_dim % 2 == 0 and head_dim // 2 >= 64, (
        f"Gemma 3 head_dim={head_dim}: head_dim/2 must be >= 64 (one Spyre "
        "stick). A padded variant is not implemented for this adapter."
    )

    # One PrecomputedRotaryEmbedding per layer type, reading the model's per-type
    # inv_freq + attention_scaling buffers via a shim.
    rope = backbone.rotary_emb
    model._spyre_rope = {}
    for layer_type in set(cfg.layer_types):
        inv_freq = getattr(rope, f"{layer_type}_inv_freq")
        scaling = getattr(rope, f"{layer_type}_attention_scaling")
        model._spyre_rope[layer_type] = PrecomputedRotaryEmbedding(
            InvFreqShim(inv_freq, scaling)
        )

    # Per-layer KV-cache shapes. Unlike Gemma 4, all layers share one head_dim
    # and num_key_value_heads, so every entry is identical — but we keep the
    # per-layer list so the shared allocator / KV machinery sees the same shape.
    model._spyre_kv_shapes = [
        (num_kv_heads, head_dim, head_dim) for _ in cfg.layer_types
    ]

    chunk_lm_head(model, num_chunks=8)

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim)
        for layer in backbone.layers
    ]
