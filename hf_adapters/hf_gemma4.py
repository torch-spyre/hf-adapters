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
HuggingFace Transformers adapter for Gemma 4 (dense) causal-LM models on Spyre.

Targets the **12B dense** variant (``model_type`` ``gemma4_text`` /
``gemma4``). Gemma 4 departs from the standard GQA decoder (``hf_qwen2`` etc.)
in several ways, so it gets a custom compiled block rather than reusing
``make_standard_gqa_block``:

- **Local / global alternating attention.** ``config.layer_types`` mixes
  ``sliding_attention`` (4 of every 5 layers) with ``full_attention``. Each
  type carries its own RoPE *and its own head_dim* (sliding ``head_dim``,
  global ``global_head_dim``), so the two layer types use different KV-cache
  shapes — see ``model._spyre_kv_shapes`` and ``hf_common.allocate_kv_caches``.
- **Partial rotary on global layers.** The global RoPE is "proportional" with
  ``partial_rotary_factor=0.25`` (theta 1e6); HF builds an ``inv_freq`` of
  length ``global_head_dim/2`` whose tail is zeros, so those dims rotate by
  angle 0 (identity). The existing ``PrecomputedRotaryEmbedding`` /
  ``apply_rope_matmul`` handle that unchanged — no special casing needed.
  Sliding layers use full rotary (theta 1e4) over ``head_dim``.
- **Q / K / V RMSNorm.** Per-head RMSNorm on Q and K (scaled) and on V
  (``with_scale=False``), applied before RoPE. The norms run in the compiled
  block.
- **K == V on global layers** (12B ``attention_k_eq_v=true``). Global layers
  have no ``v_proj``; V is the *raw* ``k_proj`` output (pre-k_norm, pre-RoPE)
  reshaped to the KV-head layout. Sliding layers keep a separate V.
- **Embedding scaling.** ``embed_tokens`` multiplies by ``sqrt(hidden_size)``
  (``Gemma4TextScaledWordEmbedding``); this is part of the loaded module and
  runs as-is.
- **"Sandwich" norms.** Four norms per layer: ``input_layernorm`` (pre-attn),
  ``post_attention_layernorm`` (applied to the *attn output* before the
  residual add), ``pre_feedforward_layernorm`` (pre-MLP), and
  ``post_feedforward_layernorm`` (applied to the *MLP output* before the
  residual add). Unlike the 2-norm pre-norm of standard GQA.
- **Per-layer scalar.** Each decoder layer multiplies its output by a learned
  ``layer_scalar`` buffer (init 1.0).
- **Unscaled attention.** ``Gemma4TextAttention.scaling == 1.0`` — Q·Kᵀ is NOT
  divided by ``sqrt(head_dim)``. SDPA is called with ``scale=1.0``.
- **Large vocab + logit softcap.** 262K vocab → chunked LM head (like
  ``hf_phi3``); ``final_logit_softcapping`` (30.0) applies a
  ``cap * tanh(logits / cap)`` after the head.

Out of scope (E2B / 26B-A4B features): per-layer embeddings (PLE), KV-sharing
across layers, and MoE blocks. ``prepare_for_spyre`` asserts these are absent
so an unsupported checkpoint fails loudly instead of running incorrectly.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("google/gemma-4-12B-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
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


def _gemma4_backbone(model):
    """Return the Gemma 4 text decoder backbone.

    ``AutoModelForCausalLM`` loads the *as-published* multimodal model
    (``Gemma4ForConditionalGeneration`` / ``Gemma4UnifiedForConditionalGeneration``)
    whose text decoder is nested at ``model.model.language_model`` (a
    ``Gemma4TextModel`` / ``Gemma4UnifiedTextModel`` with ``layers``,
    ``embed_tokens``, ``norm``, ``rotary_emb``). The shared ``get_backbone``
    descends into ``.language_model`` for exactly this case; this wrapper names
    the intent. The ``lm_head`` stays at the top level (``model.lm_head``),
    matching where ``chunk_lm_head`` looks.
    """
    return get_backbone(model)


def _text_config(model):
    """Return the Gemma 4 *text* decoder config.

    The as-published multimodal model carries a composite config
    (``Gemma4Config`` / ``Gemma4UnifiedConfig``) whose decoder fields
    (``head_dim``, ``layer_types``, ``sliding_window``, ...) live on the nested
    ``text_config``. A text-only causal-LM config exposes them directly. This
    returns whichever holds the decoder fields.
    """
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def _patch_gemma4_rmsnorm(rmsnorm_cls):
    """Patch a Gemma4 ``RMSNorm`` class to stay in fp16 on Spyre.

    Mirrors ``hf_common.patch_rmsnorm`` but for Gemma4's RMSNorm, which:
      - uses ``self.eps`` (not ``variance_epsilon``),
      - is optionally scale-free (``with_scale=False`` for V-norm and a couple
        of MoE/router norms — those carry no ``weight``),
      - computes ``x * pow(meansq + eps, -0.5)`` (equivalent to
        ``rsqrt(meansq + eps)``).

    On Spyre we stay in fp16; on CPU we upcast to fp32 to match stock HF.
    ``rmsnorm_cls`` is the concrete class the loaded model uses
    (``Gemma4RMSNorm`` or ``Gemma4UnifiedRMSNorm``) so the patch lands on the
    type the instances actually dispatch through.
    """

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            normed = hidden_states * torch.rsqrt(variance + self.eps)
            if self.with_scale:
                normed = normed * self.weight
            return normed
        # CPU path: fp32 for numerical parity with stock HF.
        xf = hidden_states.float()
        variance = (xf * xf).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            xf = xf * self.weight.float()
        return xf.type_as(hidden_states)

    rmsnorm_cls.forward = _forward_fp16


def _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
    """Compile one Gemma 4 dense decoder layer.

    Block signature carries the per-layer mask and RoPE freqs (which differ
    between sliding and global layers), so the caller selects them:

        block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position)
            -> (hidden_states, key_cache, value_cache)

    Gemma applies Q/K/V RMSNorm before RoPE, uses the four-norm "sandwich"
    structure, an unscaled (scale=1.0) attention, and a final per-layer scalar.
    On global ``attention_k_eq_v`` layers (``is_kv_eq_v=True``) there is no
    ``v_proj``: V is the raw ``k_proj`` output (before k_norm and RoPE).
    """
    attn = layer.self_attn
    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj  # None when is_kv_eq_v
    o_proj = attn.o_proj
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    v_norm = attn.v_norm
    scaling = attn.scaling  # 1.0 for Gemma 4

    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    mlp = layer.mlp
    # Capture the per-layer scalar as a Python float, not the buffer tensor:
    # the tensor is captured here (pre-Spyre-move), so a captured tensor would
    # stay the old CPU buffer while the move rebinds layer.layer_scalar to a new
    # Spyre tensor — mixing devices in ``h * layer_scalar``. A float folds into
    # the graph as a constant, like Granite's ``residual_multiplier``.
    layer_scalar = float(layer.layer_scalar)

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

        # Q/K/V projections viewed as [B, L, n_heads, head_dim]; norms are
        # applied per-head (last dim = head_dim) before the transpose.
        q = q_proj(h).view(bsz, seq_len, num_q_heads, head_dim)
        k_lin = k_proj(h).view(bsz, seq_len, num_kv_heads, head_dim)

        if is_kv_eq_v:
            # V reuses the raw k_proj output (pre-norm, pre-RoPE).
            v = k_lin.transpose(1, 2)
        else:
            v = v_proj(h).view(bsz, seq_len, num_kv_heads, head_dim)
            v = v_norm(v).transpose(1, 2)

        q = q_norm(q).transpose(1, 2)
        k = k_norm(k_lin).transpose(1, 2)

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

        h = h * layer_scalar
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
    """Gemma 4 backbone: scaled embedding, per-type RoPE + masks, blocks, norm.

    The base ``attn_mask`` is the causal mask the caller built (column index =
    cache slot). Sliding layers intersect it with a causal sliding-window band
    using each query row's cache coordinate ``block_base + j`` where
    ``block_base = cache_position - token_index`` (see the module docstring /
    ``add_causal_sliding_window_band``). Global layers use the base mask as-is.
    """
    backbone = _gemma4_backbone(model)
    cfg = _text_config(model)

    h = backbone.embed_tokens(input_ids)

    # Per-layer-type RoPE freqs (sliding theta vs global proportional theta).
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
    """Gemma 4 causal-LM forward: backbone + chunked LM head + logit softcap."""
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

    # Chunked LM head: 262K vocab exceeds Spyre's per-core EAR limit. Split
    # into N chunks, run each, cat on CPU.
    logits_parts = []
    for lm_chunk, real_sz in zip(
        model._spyre_lm_head_chunks, model._spyre_lm_chunk_sizes
    ):
        logits_parts.append(lm_chunk(h).to("cpu")[..., :real_sz])
    logits = torch.cat(logits_parts, dim=-1)

    cap = _text_config(model).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a dense Gemma 4 causal-LM model in-place.

    1. Assert the unsupported (E2B / MoE) features are absent.
    2. Patch ``Gemma4RMSNorm`` for the fp16 Spyre path.
    3. Build one ``PrecomputedRotaryEmbedding`` per layer type from the model's
       per-type ``inv_freq`` buffers (no head padding — D/2 >= 64 already).
    4. Record per-layer KV-cache shapes (sliding vs global differ).
    5. Chunk the LM head for the large vocab.
    6. Compile each decoder layer's block.
    """
    backbone = _gemma4_backbone(model)
    cfg = _text_config(model)

    assert not getattr(cfg, "hidden_size_per_layer_input", 0), (
        "Gemma 4 adapter does not support per-layer embeddings (PLE); "
        f"hidden_size_per_layer_input={cfg.hidden_size_per_layer_input}. "
        "This adapter targets the dense 12B/31B variants, not E2B/E4B."
    )
    assert not getattr(cfg, "num_kv_shared_layers", 0), (
        "Gemma 4 adapter does not support KV-sharing across layers; "
        f"num_kv_shared_layers={cfg.num_kv_shared_layers}."
    )
    assert not getattr(
        cfg, "enable_moe_block", False
    ), "Gemma 4 adapter does not support MoE blocks (enable_moe_block=True)."

    # Patch whichever concrete RMSNorm class this model uses. The norm module
    # closest to a decoder layer's input_layernorm is representative.
    rmsnorm_cls = type(backbone.layers[0].input_layernorm)
    _patch_gemma4_rmsnorm(rmsnorm_cls)

    head_dim = cfg.head_dim
    global_head_dim = getattr(cfg, "global_head_dim", None) or head_dim
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    num_global_kv_heads = (
        getattr(cfg, "num_global_key_value_heads", None) or num_kv_heads
    )
    attention_k_eq_v = getattr(cfg, "attention_k_eq_v", False)

    # Both head_dims must be stick-aligned for the RoPE [2, D/2] reshape.
    for hd, name in ((head_dim, "head_dim"), (global_head_dim, "global_head_dim")):
        assert hd % 2 == 0 and hd // 2 >= 64, (
            f"Gemma 4 {name}={hd}: head_dim/2 must be >= 64 (one Spyre stick). "
            "A padded variant is not implemented for this adapter."
        )

    # One PrecomputedRotaryEmbedding per layer type, reading the model's
    # per-type inv_freq + attention_scaling buffers via a shim. No padding:
    # the global proportional RoPE already encodes its NoPE tail as zero freqs.
    rope = backbone.rotary_emb
    model._spyre_rope = {}
    for layer_type in set(cfg.layer_types):
        inv_freq = getattr(rope, f"{layer_type}_inv_freq")
        scaling = getattr(rope, f"{layer_type}_attention_scaling")
        model._spyre_rope[layer_type] = PrecomputedRotaryEmbedding(
            InvFreqShim(inv_freq, scaling)
        )

    # Per-layer KV-cache shapes. Global (full_attention) layers use
    # global_head_dim and, when attention_k_eq_v, num_global_key_value_heads;
    # sliding layers use head_dim and num_key_value_heads.
    kv_shapes = []
    is_kv_eq_v_per_layer = []
    for lt in cfg.layer_types:
        is_global = lt == "full_attention"
        use_kv_eq_v = attention_k_eq_v and is_global
        is_kv_eq_v_per_layer.append(use_kv_eq_v)
        if is_global:
            n_kv = num_global_kv_heads if use_kv_eq_v else num_kv_heads
            hd = global_head_dim
        else:
            n_kv = num_kv_heads
            hd = head_dim
        kv_shapes.append((n_kv, hd, hd))
    model._spyre_kv_shapes = kv_shapes

    chunk_lm_head(model, num_chunks=8)

    model._spyre_compiled_blocks = [
        _make_compiled_block(
            layer,
            num_q_heads,
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
        )
        for i, layer in enumerate(backbone.layers)
    ]
