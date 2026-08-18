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
  reshaped to the KV-head layout, then passed through ``v_norm`` (matching
  stock HF, which applies ``v_norm`` to the aliased value tensor). Sliding
  layers keep a separate V.
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
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    text_config,
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
    matching where ``pad_lm_head`` looks.
    """
    return get_backbone(model)


def _patch_gemma4_rmsnorm(rmsnorm_cls):
    """Patch a Gemma4 ``RMSNorm`` class to stay in fp16 on Spyre.

    Mirrors ``hf_common.patch_rmsnorm`` but for Gemma4's RMSNorm, which:
      - uses ``self.eps`` (not ``variance_epsilon``),
      - is optionally scale-free (``with_scale=False`` for V-norm and a couple
        of MoE/router norms — those carry no ``weight``),
      - computes ``x * pow(meansq + eps, -0.5)`` (equivalent to
        ``rsqrt(meansq + eps)``).

    On Spyre we keep the reduction at input dtype; on CPU we upcast to fp32 to
    match stock HF. ``rmsnorm_cls`` is the concrete class the loaded model uses
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


class Gemma4Attention(nn.Module):
    """Attention executed by the dense Gemma 4 Spyre adapter path.

    On global ``attention_k_eq_v`` layers (``is_kv_eq_v=True``) there is no
    ``v_proj``: V is the raw ``k_proj`` output (before k_norm and RoPE) put
    through ``v_norm``, mirroring stock HF.
    """

    def __init__(self, attn, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
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
        self.scaling = attn.scaling  # 1.0 for Gemma 4

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
        # Q/K/V projections viewed as [B, L, n_heads, head_dim]; norms are
        # applied per-head (last dim = head_dim) before the transpose.
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_q_heads, self.head_dim
        )
        k_lin = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        if self.is_kv_eq_v:
            # V reuses the raw k_proj output (pre-k_norm, pre-RoPE) but still
            # passes through v_norm: stock HF aliases value_states = key_states
            # *before* k_norm/RoPE, then applies self.v_norm(value_states)
            # unconditionally (modeling_gemma4 Gemma4TextAttention.forward). The
            # norm exists on these layers even though v_proj is None.
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

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )
        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), key_cache, value_cache


class Gemma4Block(nn.Module):
    """Registered dense Gemma 4 decoder block used by the Spyre adapter."""

    def __init__(self, layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
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
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        return h * layer_scalar, key_cache, value_cache


def prepare_gemma4_blocks(
    layers, num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer
):
    """Replace Gemma 4 decoder layers with registered blocks and compile them."""
    blocks = []
    for i, layer in enumerate(list(layers)):
        block = Gemma4Block(
            layer,
            num_q_heads_per_layer[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
        )
        layers[i] = block
        blocks.append(torch.compile(block, dynamic=False))
    return blocks


def _build_layer_masks(
    model,
    attn_mask,
    seq_len,
    batch_size,
    block_base,
):
    """Build the text-only per-layer-type mask dict {full_attention, sliding_attention}.

    ``attn_mask`` is the base causal mask the caller built (column index = cache
    slot). Global ("full_attention") layers use it as-is (plain causal). Sliding
    ("sliding_attention") layers intersect it with a causal sliding-window band
    using each query row's cache coordinate ``block_base + j`` (see
    ``add_causal_sliding_window_band``).

    ``block_base`` is the cache column the first query row occupies — the first
    entry of the block's ``cache_index`` (``int(cache_index[0])``).

    This is the text-decoder mask policy. The unified VLM adapter
    (``hf_gemma4_mm``) needs a bidirectional vision overlay OR-ed into both mask
    types, so it builds its own mask dict and passes it to
    ``_run_blocks_over_embeds(..., masks=...)`` rather than calling this.
    """
    cfg = text_config(model.config)
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(
        batch_size, seq_len
    )
    sliding_mask = add_causal_sliding_window_band(
        attn_mask, query_coords, cfg.sliding_window
    )
    return {"full_attention": attn_mask, "sliding_attention": sliding_mask}


def _run_blocks_over_embeds(
    model,
    h,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    masks=None,
):
    """Run the compiled Gemma 4 decoder blocks over precomputed embeddings.

    Shared by the text-only causal LM (``_run_backbone_forward``) and the VLM
    adapter (``hf_gemma4_mm``, which drives the decoder from image-scattered
    ``inputs_embeds``). Builds per-type RoPE freqs, then runs the blocks under a
    per-layer-type mask dict and applies the final norm.

    ``masks`` (optional ``{layer_type: mask}``) lets a caller supply its own
    per-type masks — the VLM passes masks with the bidirectional vision overlay
    OR-ed in. When ``None``, the text-only causal + sliding masks are built from
    ``attn_mask`` via ``_build_layer_masks`` (``attn_mask`` is ignored when
    ``masks`` is given).
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    # Per-layer-type RoPE freqs (sliding theta vs global proportional theta).
    freqs = {
        layer_type: rope(h, position_ids)
        for layer_type, rope in model._spyre_rope.items()
    }

    if masks is None:
        bsz, seq_len = h.shape[0], h.shape[1]
        # The sliding-window band needs each query row's cache coordinate: row j
        # sits at block_base + j, where block_base is the first cache slot this
        # block writes — the first entry of cache_index.
        #
        # The scalar read syncs from the device; deliberately not optimized. This
        # runs once per step (not per layer — the mask dict is reused across all
        # layers) in eager code outside the compiled block, and
        # add_causal_sliding_window_band already round-trips the whole mask
        # through CPU by necessity. See the same note in hf_gemma3.
        block_base = int(cache_index[0])
        masks = _build_layer_masks(model, attn_mask, seq_len, bsz, block_base)

    backbone_layers = backbone.layers
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        # Pass the per-layer scalar as a tensor read fresh from the registered,
        # device-moved block — NOT as a Python float — so Dynamo guards on tensor
        # metadata instead of recompiling for each distinct learned value.
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            cache_index,
            backbone_layers[i].layer_scalar,
        )

    h = backbone.norm(h)
    return h


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Gemma 4 backbone: scaled embedding, per-type RoPE + masks, blocks, norm.

    Text-only path: embed the ids (scaled word embedding) then delegate to
    ``_run_blocks_over_embeds`` (no blockwise vision band).
    """
    backbone = _gemma4_backbone(model)
    h = backbone.embed_tokens(input_ids)
    return _run_blocks_over_embeds(
        model,
        h,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Gemma 4 causal-LM forward: backbone + LM head + logit softcap."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )

    logits = model.lm_head(h)

    cap = text_config(model.config).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


def prepare_text_decoder_for_spyre(model):
    """Prepare ONLY the Gemma 4 text decoder for Spyre (in-place).

    1. Assert the unsupported (E2B / MoE) features are absent.
    2. Patch ``Gemma4RMSNorm`` for the fp16 Spyre path.
    3. Build one ``PrecomputedRotaryEmbedding`` per layer type from the model's
       per-type ``inv_freq`` buffers (no head padding — D/2 >= 64 already).
    4. Record per-layer KV-cache shapes (sliding vs global differ).
    5. Chunk the LM head for the large vocab.
    6. Compile each decoder layer's block.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

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

    attention_k_eq_v = getattr(cfg, "attention_k_eq_v", False)
    layer_configs = cfg.per_layer_config
    num_q_heads_per_layer = []
    kv_shapes = []
    is_kv_eq_v_per_layer = []
    for i, (layer_type, layer_cfg) in enumerate(zip(cfg.layer_types, layer_configs)):
        num_q_heads_per_layer.append(layer_cfg.num_attention_heads)
        head_dim = layer_cfg.head_dim
        assert head_dim % 2 == 0 and head_dim // 2 >= 64, (
            f"Gemma 4 layer {i} head_dim={head_dim}: head_dim/2 must be >= 64 "
            "(one Spyre stick). A padded variant is not implemented for this adapter."
        )
        num_kv_heads = layer_cfg.num_key_value_heads
        kv_shapes.append((num_kv_heads, head_dim, head_dim))
        is_kv_eq_v_per_layer.append(attention_k_eq_v and layer_type == "full_attention")
    model._spyre_kv_shapes = kv_shapes

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

    # LM head: smooth-padded to a stick-aligned vocab whose per-core span fits
    # the 256 MB EAR limit (see hf_common.pad_lm_head).
    pad_lm_head(model)

    model._spyre_compiled_blocks = prepare_gemma4_blocks(
        backbone.layers,
        num_q_heads_per_layer,
        kv_shapes,
        is_kv_eq_v_per_layer,
    )


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a dense Gemma 4 causal-LM model in-place."""
    prepare_text_decoder_for_spyre(model)
