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
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
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


def _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
    """Compile one Gemma 4 dense decoder layer.

    Block signature carries the per-layer mask and RoPE freqs (which differ
    between sliding and global layers), so the caller selects them:

        block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position,
                      layer_scalar)
            -> (hidden_states, key_cache, value_cache)

    Gemma applies Q/K/V RMSNorm before RoPE, uses the four-norm "sandwich"
    structure, an unscaled (scale=1.0) attention, and a final per-layer scalar.
    The per-layer scalar is a *tensor argument* (not a captured constant) so all
    layers share one compiled binary — see the note in the body.
    On global ``attention_k_eq_v`` layers (``is_kv_eq_v=True``) there is no
    ``v_proj``: V is the raw ``k_proj`` output (before k_norm and RoPE) put
    through ``v_norm``, mirroring stock HF.
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
    # NOTE: the per-layer ``layer_scalar`` is NOT captured here. It is passed as
    # a tensor *argument* to ``block_forward`` instead (see below). All 48 layers
    # share one ``block_forward`` ``__code__`` (loop-created closures). If the
    # scalar were folded in as a Python float constant, dynamo would guard each
    # compiled entry on its exact value (``layer_scalar == 0.296875``) — and
    # because the 48 layers carry ~43 *distinct* learned scalars, layer N's call
    # would miss every prior layer's guard and recompile. That banks ~N_layers
    # cache entries per forward on the shared frame, crossing dynamo's
    # ``accumulated_recompile_limit`` (256) after only a handful of decode steps
    # and dropping the tail of generation onto the slow/inaccurate eager path.
    # Passing the scalar as a *tensor* makes dynamo guard on tensor metadata
    # (shape/dtype — identical across layers), so all layers reuse one binary per
    # offset (~1 entry/step, like Granite/Qwen3). Granite's ``residual_multiplier``
    # can be a captured float because it is a single config value shared by every
    # layer; Gemma 4's is per-layer, so it must be a tensor arg.
    #
    # The scalar tensor is read fresh from ``layer.layer_scalar`` at call time in
    # ``_run_backbone_forward`` (NOT captured), so it is always the post-Spyre-move
    # buffer on the right device — avoiding the device-mismatch the old float
    # capture sidestepped.

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
        layer_scalar,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape

        # Q/K/V projections viewed as [B, L, n_heads, head_dim]; norms are
        # applied per-head (last dim = head_dim) before the transpose.
        q = q_proj(h).view(bsz, seq_len, num_q_heads, head_dim)
        k_lin = k_proj(h).view(bsz, seq_len, num_kv_heads, head_dim)

        if is_kv_eq_v:
            # V reuses the raw k_proj output (pre-k_norm, pre-RoPE) but still
            # passes through v_norm: stock HF aliases value_states = key_states
            # *before* k_norm/RoPE, then applies self.v_norm(value_states)
            # unconditionally (modeling_gemma4 Gemma4TextAttention.forward). The
            # norm exists on these layers even though v_proj is None.
            v = v_norm(k_lin).transpose(1, 2)
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


def _build_layer_masks(
    model,
    attn_mask,
    seq_len,
    batch_size,
    token_index,
    cache_position,
):
    """Build the text-only per-layer-type mask dict {full_attention, sliding_attention}.

    ``attn_mask`` is the base causal mask the caller built (column index = cache
    slot). Global ("full_attention") layers use it as-is (plain causal). Sliding
    ("sliding_attention") layers intersect it with a causal sliding-window band
    using each query row's cache coordinate ``block_base + j`` where
    ``block_base = cache_position - token_index`` (see
    ``add_causal_sliding_window_band``).

    This is the text-decoder mask policy. The unified VLM adapter
    (``hf_gemma4_mm``) needs a bidirectional vision overlay OR-ed into both mask
    types, so it builds its own mask dict and passes it to
    ``_run_blocks_over_embeds(..., masks=...)`` rather than calling this.
    """
    cfg = text_config(model.config)
    block_base = cache_position - token_index
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
    is_filling,
    token_index,
    cache_position,
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
        masks = _build_layer_masks(
            model, attn_mask, seq_len, bsz, token_index, cache_position
        )

    backbone_layers = backbone.layers
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        # Pass the per-layer scalar as a tensor read fresh from the (already
        # device-moved) buffer — NOT a captured float — so the 48 layers share
        # one compiled binary instead of recompiling per distinct scalar value.
        # See the note in _make_compiled_block.
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
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
    is_filling,
    token_index,
    cache_position,
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
        is_filling,
        token_index,
        cache_position,
    )


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
    """Gemma 4 causal-LM forward: backbone + LM head + logit softcap."""
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

    # LM head: smooth-padded to a stick-aligned vocab whose per-core span fits
    # the 256 MB EAR limit (see hf_common.pad_lm_head).
    pad_lm_head(model)

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


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a dense Gemma 4 causal-LM model in-place."""
    prepare_text_decoder_for_spyre(model)
