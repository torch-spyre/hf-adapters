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
Unified (encoder-free) HuggingFace adapter for Gemma 4 12B on Spyre — image→text.

Where ``hf_gemma4`` runs only the text decoder (``AutoSpyreModelForCausalLM``),
this module loads the full unified multimodal model
(``Gemma4UnifiedForConditionalGeneration``, ``model_type=gemma4_unified``) via
``AutoModelForImageTextToText`` and runs the image→text pipeline. It is the
adapter behind ``AutoSpyreModelForImageTextToText``.

Gemma 4 is **encoder-free**: there is no vision tower. Vision is a pure
projection of raw (processor-merged) pixel patches into the LM embedding space,
scattered into the ``<image>`` token slots of the text embeddings:

    pixel_values [B, P, 48²·3]                    (processor already merged
      │   image_position_ids [B, P, 2]             pooling_kernel_size² raw
      ▼                                            16×16 patches per token)
    Gemma4UnifiedVisionEmbedder                   (LN → Dense → LN → +posemb
      │   LN/Dense/RMSNorm/Linear — no attention   → LN → RMSNorm → Linear)
      ▼
    image_features [valid_patches, 3840]          (padding patches stripped)
      │
    input_ids ──► scaled word embeddings ──► masked_scatter into <image> slots
      │
      ▼
    Gemma 4 text decoder   (hf_gemma4 compiled blocks, Spyre)
      │   + bidirectional vision mask on sliding layers at prefill
      ▼
    logits  ──► final_logit_softcapping

**Bidirectional vision attention.** ``text_config.use_bidirectional_attention ==
"vision"``: within one image, the soft-tokens attend to each other
bidirectionally. Stock builds every layer-type mask via
``create_causal_mask(block_sequence_ids=...)``, which OR-s a "blockwise" overlay
(same image group ⇒ allowed, from ``mm_token_type_ids``) into the causal mask for
**both** full and sliding layers — NOT sliding-only. So at prefill we OR the
blockwise band into the base causal mask for both types:

  - full_attention  = OR(causal, blockwise)
  - sliding_attention = AND(sliding_window, OR(causal, blockwise))

Decode steps are pure text (one new causal token), so no blockwise band is
needed after prefill. (Verified against stock ``create_causal_mask``: the
``create_masks_for_vision_model`` docstring claiming globals stay causal is not
the path the forward takes — traced directly through
``create_masks_for_generate`` → ``create_causal_mask``.)

**Vision embedder on Spyre.** The compilable core (LN₁→Dense→LN₂→+posemb→
pos_norm→RMSNorm→Linear) is ``torch.compile``d and runs on Spyre. The
position-embedding gather (integer XY ``image_position_ids`` with ``-1``
padding, validity masking) and the final padding-patch strip are computed on
**CPU** — those integer-gather / boolean-index ops don't lower on the Spyre
backend (same doctrine as the SigLIP CPU patch-embed in ``hf_siglip_vision``).
The CPU-built per-patch positional-embedding tensor is passed into the compiled
core as a device argument.

The text decoder reuses ``hf_gemma4`` unchanged. Both towers live under the one loaded VLM, so a
single ``prepare_for_spyre`` covers them. Exposes ``prefill_logits``
(first-token forward) and ``generate`` (full autoregressive decode — image
features scattered at prefill; decode steps are pure text).

Scope: **text + image**. Audio and video are asserted out loudly
(``prepare_for_spyre`` / forward raise if audio/video inputs are present).
"""

import math

import torch

from hf_adapters import hf_gemma4
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _resolve_generation_params,
    allocate_kv_caches,
    build_expansion_mask,
    build_prefill_mask,
    decode_block_walk,
    get_backbone,
    get_model_dtype,
    pad_and_position,
    patch_layernorm,
    select_next_token,
    text_config,
)


def _vision_embedder(model):
    """The Gemma4UnifiedVisionEmbedder (``model.model.embed_vision``)."""
    return model.model.embed_vision


def _make_compiled_vision_core(embedder):
    """Compile the attention-free vision projection core (Spyre).

    Signature::

        core(pixel_values, pos_embs) -> features [B, P, mm_embed_dim]

    where ``pixel_values`` is ``[B, P, 48²·3]`` (processor-merged raw patches)
    and ``pos_embs`` is ``[B, P, mm_embed_dim]`` — the factorized positional
    embedding for each patch, prebuilt on CPU (the integer XY gather + ``-1``
    padding validity masking don't lower on Spyre) and moved to the device.

    Reproduces ``Gemma4UnifiedVisionEmbedder.forward`` minus the pos-emb gather:
    ``LN₁ → Dense → LN₂ → (+pos_embs) → pos_norm → multimodal_embedder``
    (the multimodal embedder is ``RMSNorm(with_scale=False) → Linear``). Padding
    patches are stripped by the caller *after* this core (boolean index on CPU).
    """
    patch_ln1 = embedder.patch_ln1
    patch_dense = embedder.patch_dense
    patch_ln2 = embedder.patch_ln2
    pos_norm = embedder.pos_norm
    mm_embedder = embedder.multimodal_embedder

    def core(pixel_values, pos_embs):
        h = patch_ln1(pixel_values)
        h = patch_dense(h)
        h = patch_ln2(h)
        h = h + pos_embs
        h = pos_norm(h)
        h = mm_embedder(h)
        return h

    return torch.compile(core, dynamic=False)


def prepare_for_spyre(model):
    """Prepare a loaded Gemma 4 unified VLM for Spyre in-place (text + image).

    Text decoder → ``hf_gemma4.prepare_text_decoder_for_spyre`` (RMSNorm patch,
    per-type RoPE, per-layer KV shapes, padded LM head, compiled blocks). Vision
    → compile the attention-free vision projection core. Asserts the model is a
    vision-capable unified checkpoint and that audio is out of scope.
    """
    cfg = text_config(model.config)
    assert getattr(cfg, "use_bidirectional_attention", None) == "vision", (
        "hf_gemma4_mm expects a unified Gemma 4 with "
        "use_bidirectional_attention='vision'; got "
        f"{getattr(cfg, 'use_bidirectional_attention', None)!r}."
    )
    assert getattr(model.model, "embed_vision", None) is not None, (
        "hf_gemma4_mm requires a vision embedder (model.model.embed_vision); "
        "this checkpoint has no vision_config."
    )

    # Shared text decoder (mirrors hf_gemma4.prepare_for_spyre).
    hf_gemma4.prepare_text_decoder_for_spyre(model)

    # Vision projection core, compiled for Spyre. The three vision LayerNorms
    # (patch_ln1/patch_ln2/pos_norm) must be patched to the un-fused
    # decomposition BEFORE compiling: the fused F.layer_norm lowering NaNs on
    # near-constant patch rows (see patch_layernorm / the doc). Patch first, then
    # compile so the core captures the patched forward.
    embedder = _vision_embedder(model)
    patch_layernorm(embedder.patch_ln1, embedder.patch_ln2, embedder.pos_norm)
    model._spyre_vision_core = _make_compiled_vision_core(embedder)


def _build_pos_embs(embedder, image_position_ids):
    """Factorized 2D positional embeddings per patch, on CPU.

    Mirrors the pos-emb block of ``Gemma4UnifiedVisionEmbedder.forward``:
    ``pos_embedding[clamped_xy, axes] * valid`` summed over the 2 axes. Kept on
    CPU because the integer gather + ``-1`` validity masking do not lower on the
    Spyre backend. Returns ``[B, P, mm_embed_dim]`` in the pos_embedding's dtype.
    """
    pos_embedding = embedder.pos_embedding.detach().cpu()  # [posemb_size, 2, D]
    ipos = image_position_ids.to("cpu")
    clamped = ipos.clamp(min=0).long()
    valid = (ipos != -1).to(pos_embedding.dtype).unsqueeze(-1)
    axes = torch.arange(2)
    pos_embs = (pos_embedding[clamped, axes] * valid).sum(-2)  # [B, P, D]
    return pos_embs


def _image_features(model, pixel_values, image_position_ids):
    """Run the Spyre vision core and return stripped features [valid_patches, H].

    CPU: build the positional embeddings and the padding mask. Spyre: the
    LN/Dense/RMSNorm projection core. CPU: strip padding patches
    (``image_position_ids == -1`` on both axes), matching stock
    ``get_image_features``.
    """
    embedder = _vision_embedder(model)
    dtype = get_model_dtype(model)

    # anyres / multi-image: [B, T, P, ...] -> [B*T, P, ...] (stock flattens too)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.flatten(0, 1)
        image_position_ids = image_position_ids.flatten(0, 1)

    pos_embs = _build_pos_embs(embedder, image_position_ids).to(dtype)
    features = model._spyre_vision_core(
        pixel_values.to(dtype).to(DEVICE), pos_embs.to(DEVICE)
    )
    features = features.to("cpu")

    padding_mask = (image_position_ids.to("cpu") == -1).all(dim=-1)  # [B, P]
    features = features[~padding_mask]  # [valid_patches, H]
    return features


def _embed_and_scatter(model, input_ids, image_features):
    """Scaled word embeddings with image features scattered into <image> slots.

    ``embed_tokens`` is ``Gemma4UnifiedTextScaledWordEmbedding`` (×√hidden runs
    as-is). Stock does ``inputs_embeds.masked_scatter(image_mask, features)``;
    Spyre can't ``masked_scatter``, so we zero the image-token slots (elementwise
    mul by a CPU-built keep factor) and add a CPU-built additive tensor holding
    the features at the image positions — bit-identical given the zeroed slots
    (same doctrine as hf_granite_vision_mm._inject_deepstack). Asserts the
    token/feature counts match (mirrors stock's shape check).
    """
    backbone = get_backbone(model)
    image_token_id = model.config.image_token_id
    dtype = get_model_dtype(model)

    ids = input_ids.to(backbone.embed_tokens.weight.device)
    h = backbone.embed_tokens(ids)  # scaled word embeddings, on embed device

    image_mask = input_ids == image_token_id  # [B, L] bool, CPU
    n_image_tokens = int(image_mask.sum())
    hidden = h.shape[-1]
    feats = image_features.to("cpu", dtype)
    if n_image_tokens * hidden != feats.numel():
        raise ValueError(
            "image tokens and features do not match: tokens "
            f"{n_image_tokens}, features {tuple(feats.shape)}"
        )

    keep = (~image_mask).to(dtype).unsqueeze(-1).to(h.device)
    h = h * keep

    additive = torch.zeros(h.shape[0], h.shape[1], hidden, dtype=dtype)
    additive[image_mask] = feats.view(n_image_tokens, hidden)
    return h + additive.to(h.device)


def _blockwise_band(mm_token_type_ids, padded_len, max_cache_len, dtype):
    """Additive bidirectional blockwise band ``[B, 1, padded_len, max_cache_len]``.

    Reproduces stock ``blockwise_overlay(get_block_sequence_ids_for_mask(...))``:
    two tokens attend to each other iff they share the same image group id
    (>= 0). Built and kept on CPU (int/bool ops don't lower on Spyre; also avoids
    the bf16 ``-inf + -inf`` NaN hazard when OR-combined). Only used at prefill.

    ``mm_token_type_ids`` is the left-padded ``[B, padded_len]`` type map
    (0=text, 1=image). Cache column ``c`` at prefill holds the token at padded
    position ``c`` (cache filled in order), so the query row and key column index
    the same group vector; columns beyond ``padded_len`` (unused cache) stay
    masked (band = -inf there, harmless since the base causal mask masks them
    too).

    Returns 0 where the blockwise overlay *allows* a pair, -inf elsewhere — an
    additive mask suitable for an elementwise-max OR against the causal mask.
    """
    tt = mm_token_type_ids.to("cpu")
    # Group ids: contiguous runs of vision tokens get an incrementing id; text = -1.
    is_vision = tt >= 1
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_starts = is_vision & ~is_prev_vision
    group_ids = torch.cumsum(new_starts.int(), dim=1) - 1  # [B, padded_len]
    group_ids = torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))

    bsz = tt.shape[0]
    # Key columns span the full cache; pad group ids for unused cache slots with -1.
    if max_cache_len > padded_len:
        pad = torch.full((bsz, max_cache_len - padded_len), -1, dtype=group_ids.dtype)
        key_groups = torch.cat([group_ids, pad], dim=1)  # [B, max_cache_len]
    else:
        key_groups = group_ids[:, :max_cache_len]

    q = group_ids[:, :, None]  # [B, padded_len, 1]
    k = key_groups[:, None, :]  # [B, 1, max_cache_len]
    allowed = (q == k) & (q >= 0)  # [B, padded_len, max_cache_len] bool
    band = torch.zeros(allowed.shape, dtype=dtype)
    band = band.masked_fill(~allowed, -torch.inf)
    return band[:, None, :, :]  # [B, 1, padded_len, max_cache_len]


def _sliding_window_lower_band(mask, sliding_window):
    """Restrict an additive prefill mask to the sliding-window *lower bound* only.

    Masks keys further back than ``sliding_window`` (``q - k >= window``) but —
    unlike ``hf_common.add_causal_sliding_window_band`` — does NOT mask future
    keys (``q - k < 0``). This is stock's ``sliding_window_overlay``
    (``kv_idx > q_idx - window``), an ``and_mask`` applied *after* the base
    ``OR(causal, blockwise)`` composition: the causal upper bound already lives
    in that base, so the window only supplies the backward cutoff. Applying the
    causal window band instead would re-mask the forward-attending bidirectional
    image pairs.

    Prefill only (``cache_position == token_index == 0``), so a query row's cache
    coordinate is its row index ``q`` and the key column is the cache slot ``k``.
    ``mask`` is ``[B, 1, Lq, Lk]`` where ``Lk`` (the cache length) may exceed
    ``Lq`` (unused decode slots), so the band is the rectangular ``q - k`` over
    ``[Lq, Lk]``. Built and combined on CPU (int/bool off Spyre; add off-device
    to dodge the bf16 ``-inf + -inf`` NaN).
    """
    lq, lk = mask.shape[-2], mask.shape[-1]
    q = torch.arange(lq)[:, None]  # [Lq, 1]
    k = torch.arange(lk)[None, :]  # [1, Lk]
    out_of_band = (q - k) >= sliding_window  # [Lq, Lk]
    band = torch.zeros((lq, lk), dtype=mask.dtype)
    band = band.masked_fill(out_of_band, -torch.inf)
    orig_device = mask.device
    return (mask.to("cpu") + band[None, None, :, :]).to(orig_device)


def _build_mm_masks(prefill_mask, blockwise_band, sliding_window):
    """Per-layer-type masks for a multimodal prefill: {full, sliding}.

    Stock builds every mask type via ``create_causal_mask(block_sequence_ids)``,
    which OR-s the blockwise vision overlay into the causal mask for **both** full
    and sliding layers (traced through ``create_masks_for_generate`` →
    ``create_causal_mask``):

      - full_attention  = OR(causal, blockwise)
      - sliding_attention = AND(sliding_window_lowerbound, OR(causal, blockwise))

    The OR is an elementwise ``max`` of the two additive (0 / -inf) masks, done on
    CPU to avoid the bf16 ``-inf + -inf`` NaN hazard. Prefill only.
    ``prefill_mask``/``blockwise_band`` are ``[B, 1, padded_len, max_cache_len]``.
    """
    orig_device = prefill_mask.device
    full_mask = torch.maximum(prefill_mask.to("cpu"), blockwise_band.to("cpu")).to(
        orig_device
    )
    sliding_mask = _sliding_window_lower_band(full_mask, sliding_window)
    return {"full_attention": full_mask, "sliding_attention": sliding_mask}


def _logits_from_embeds(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
    masks=None,
):
    """Text decoder over image-scattered embeds → logits (+ softcap).

    Delegates the block walk to ``hf_gemma4._run_blocks_over_embeds``
    (shared with the text-only adapter), then applies the LM head and Gemma 4's
    ``final_logit_softcapping``. ``masks``
    (``{layer_type: mask}``) carries the bidirectional vision overlay at prefill;
    decode steps pass ``None`` and let
    the shared walk build the plain text-only causal + sliding masks from
    ``attn_mask``.
    """
    h = hf_gemma4._run_blocks_over_embeds(
        model,
        inputs_embeds,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
        masks=masks,
    )
    logits = model.lm_head(h)
    cap = text_config(model.config).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


def _prefill_forward(
    model,
    padded_ids,
    padded_len,
    prompt_offsets,
    position_ids,
    pixel_values,
    image_position_ids,
    mm_token_type_ids,
    key_caches,
    value_caches,
    max_cache_len,
):
    """Shared multimodal prefill: padded ids + image → full-sequence logits.

    Builds scaled text embeddings with the image features scattered into the
    ``<image>`` slots, then the per-layer-type masks with the bidirectional
    vision overlay OR-ed into both full and sliding layers, and runs the decoder
    once (writing the KV caches). ``mm_token_type_ids`` is the *unpadded* batch
    tensor; it is left-padded here to match ``padded_ids``.
    """
    dtype = get_model_dtype(model)
    cfg = text_config(model.config)
    image_features = _image_features(model, pixel_values, image_position_ids)
    inputs_embeds = _embed_and_scatter(model, padded_ids, image_features)

    mm_padded = _pad_mm_token_type_ids(mm_token_type_ids, padded_len)
    prefill_mask = build_prefill_mask(
        padded_ids.shape[0], padded_len, max_cache_len, prompt_offsets, dtype=dtype
    )
    blockwise = _blockwise_band(mm_padded, padded_len, max_cache_len, dtype)
    masks = _build_mm_masks(prefill_mask, blockwise, cfg.sliding_window)
    masks = {lt: m.to(DEVICE) for lt, m in masks.items()}
    return _logits_from_embeds(
        model,
        inputs_embeds.to(DEVICE),
        position_ids.to(DEVICE),
        prefill_mask.to(DEVICE),
        key_caches,
        value_caches,
        is_filling=False,
        token_index=0,
        cache_position=0,
        masks=masks,
    )


def _pad_mm_token_type_ids(mm_token_type_ids, padded_len):
    """Left-pad ``mm_token_type_ids`` to ``padded_len`` with 0 (text), matching
    ``pad_and_position``'s left block-pad of ``input_ids``."""
    bsz, seq = mm_token_type_ids.shape
    if padded_len == seq:
        return mm_token_type_ids
    pad = mm_token_type_ids.new_zeros((bsz, padded_len - seq))
    return torch.cat([pad, mm_token_type_ids], dim=1)


def prefill_logits(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    image_position_ids,
    mm_token_type_ids,
):
    """One-shot prefill of the (text + scattered image) sequence → logits.

    Left-pads to a BLOCK_SIZE multiple (same convention as ``generate``),
    scatters image features into ``<image>`` slots, runs the decoder once with
    the bidirectional vision band on sliding layers, and returns full-sequence
    logits ``[B, L, padded_vocab]`` (callers take ``[:, -1, :true_vocab]``).
    """
    actual_lengths = attention_mask.sum(dim=1)
    padded_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
        input_ids, actual_lengths
    )
    key_caches, value_caches = allocate_kv_caches(
        model, padded_ids.shape[0], padded_len, get_model_dtype(model)
    )
    logits = _prefill_forward(
        model,
        padded_ids,
        padded_len,
        prompt_offsets,
        position_ids,
        pixel_values,
        image_position_ids,
        mm_token_type_ids,
        key_caches,
        value_caches,
        max_cache_len=padded_len,
    )
    return logits, padded_len, input_ids.shape[1]


def generate(
    model,
    processor,
    input_ids,
    attention_mask,
    pixel_values,
    image_position_ids,
    mm_token_type_ids,
    max_new_tokens,
    do_sample=None,
    temperature=None,
    top_k=None,
    top_p=None,
):
    """Autoregressive image→text generation on Spyre (greedy / top-k/p sampling).

    Mirrors ``hf_granite_vision_mm.generate``'s 64-block padded decode, driven by
    embeddings so the prefill step carries the image scatter + bidirectional
    vision mask:

    - **Prefill** (step 0): scaled text embeds with ``<image>`` slots filled by
      the vision features; decoder runs once with the blockwise band on sliding
      layers.
    - **Decode** (steps ≥1): each new token id is embedded (scaled) and fed back
      — pure text, causal, no image band.

    Inputs come pre-tokenized from the checkpoint's ``AutoProcessor`` (chat
    template + image-token expansion). Assumes **left-padded** input
    (``processor.tokenizer.padding_side='left'``). Returns EOS-trimmed strings.
    """
    tokenizer = processor.tokenizer
    params = _resolve_generation_params(
        model,
        tokenizer,
        {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
    )
    do_sample = params["do_sample"]
    temperature = params["temperature"]
    top_k = params["top_k"]
    top_p = params["top_p"]
    eos_ids = params["eos_ids"]

    backbone = get_backbone(model)
    model_d_type = get_model_dtype(model)

    batch_size, prompt_length = input_ids.shape
    actual_prompt_lengths = attention_mask.sum(dim=1)  # [B]

    max_cache_len = (
        math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
        + math.ceil(max_new_tokens / BLOCK_SIZE) * BLOCK_SIZE
    )
    input_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
        input_ids, actual_prompt_lengths
    )

    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, model_d_type
    )

    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None
    finished = torch.zeros(batch_size, dtype=torch.bool)
    num_generated = torch.zeros(batch_size, dtype=torch.long)

    def embed_ids(ids):
        """Token ids -> scaled embeddings (decode steps; pure text)."""
        return backbone.embed_tokens(ids)

    for i in range(max_new_tokens):
        if i == 0:
            # --- PREFILL: text embeds with image scatter + blockwise vision band ---
            logits = _prefill_forward(
                model,
                input_ids,
                padded_len,
                prompt_offsets,
                position_ids,
                pixel_values,
                image_position_ids,
                mm_token_type_ids,
                key_caches,
                value_caches,
                max_cache_len,
            )
            next_logits = logits.to("cpu")[:, -1, :]
            current_cache_len = padded_len
            decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
            for b in range(batch_size):
                actual_len = actual_prompt_lengths[b].item()
                for j in range(BLOCK_SIZE):
                    decode_pos[b, j] = actual_len + j - BLOCK_SIZE
        else:
            is_filling = tokens_in_block > 0
            next_input = result[:, -BLOCK_SIZE:].to(DEVICE)
            next_embeds = embed_ids(next_input)
            if is_filling:
                fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
                logits = _logits_from_embeds(
                    model,
                    next_embeds,
                    decode_pos.to(DEVICE),
                    fill_mask_device,
                    key_caches,
                    value_caches,
                    is_filling=True,
                    token_index=tokens_in_block,
                    cache_position=fill_pos,
                )
                grab_idx = BLOCK_SIZE - tokens_in_block
                next_logits = logits.to("cpu")[:, -grab_idx, :]
            else:
                current_cache_len += BLOCK_SIZE
                decode_pos = decode_pos + BLOCK_SIZE
                exp_mask = build_expansion_mask(
                    batch_size,
                    BLOCK_SIZE,
                    max_cache_len,
                    current_cache_len,
                    prompt_offsets,
                    dtype=model_d_type,
                )
                logits = _logits_from_embeds(
                    model,
                    next_embeds,
                    decode_pos.to(DEVICE),
                    exp_mask.to(DEVICE),
                    key_caches,
                    value_caches,
                    is_filling=False,
                    token_index=0,
                    cache_position=current_cache_len - BLOCK_SIZE,
                )
                next_logits = logits.to("cpu")[:, -BLOCK_SIZE, :]
                fill_mask_device = exp_mask.to(DEVICE)

        next_tokens = select_next_token(
            next_logits, do_sample, temperature, top_k, top_p
        )

        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        if tokens_in_block == 0:
            result = torch.nn.functional.pad(result, (0, BLOCK_SIZE))
        grab_idx = (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        result[:, -grab_idx] = next_tokens
        if eos_ids is not None:
            finished |= torch.isin(next_tokens, eos_ids)
        num_generated += (~finished).long()
        if finished.all():
            break

    return decode_block_walk(result, num_generated, padded_len, eos_ids, tokenizer)
