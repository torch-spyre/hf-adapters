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
Combined (two-tower) HuggingFace adapter for Mistral3 Vision models on Spyre.

Where ``hf_mistral3`` extracts only the text backbone (text-only causal LM,
used by ``AutoSpyreModelForCausalLM``), this module loads BOTH towers from the
multimodal checkpoint via ``AutoModelForImageTextToText`` and runs the full
image→text pipeline. It is the adapter behind
``AutoSpyreModelForImageTextToText``.

    pixel_values ──► Pixtral vision tower         (hf_pixtral_vision, Spyre)
                       │  last_hidden_state [B, P, hidden]  (selected layer)
                       ▼
                     RMSNorm + PatchMerger + linear projector  (stock, CPU)
                       │  image_features [num_img_tokens, text_hidden]
                       ▼
    input_ids ──► text embeddings ──► scatter into <image> slots
                       │  inputs_embeds [B, L, text_hidden]
                       ▼
                     Mistral text decoder           (hf_mistral3 blocks, Spyre)
                       ▼
                     logits

Mistral3 uses a **flat single-injection** pattern (contrast with Granite
Vision's deepstack multi-layer injection):

1. Run the Pixtral vision tower once with ``output_hidden_states=True``.
2. Select the configured ``vision_feature_layer`` (or last layer).
3. Apply ``multi_modal_projector`` (RMSNorm → PatchMerger → linear × 2)
   on CPU to produce ``image_features``.
4. Zero the ``<image>`` token slots in the text embeddings, then scatter
   ``image_features`` into those slots via a CPU-built additive tensor
   (same technique as ``hf_granite_vision_mm._inject_deepstack`` — avoids
   Spyre-incompatible boolean indexing).
5. Run the full text decoder with the resulting ``inputs_embeds``; no
   further per-layer injection.

Decode steps are pure text (no image re-encoding).

Verified on CPU to match stock ``Mistral3ForConditionalGeneration.generate``
(token-exact, greedy) and ``forward`` (first-token logits cosine ≥ 0.999).
"""

import math

import torch

from hf_adapters import hf_pixtral_vision
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
    make_standard_gqa_block,
    pad_and_position,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    select_next_token,
)

# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------


def prepare_for_spyre(model):
    """Prepare BOTH towers of a loaded Mistral3 VLM in-place.

    Vision tower → ``hf_pixtral_vision.prepare_for_spyre`` (compiled pre-LN
    blocks, head padding, CPU patch-embed, 2D RoPE matrices).
    Text decoder → standard-GQA RoPE/head prep + compiled Mistral blocks +
    padded LM head, mirroring ``hf_mistral3.prepare_for_spyre`` but applied
    against the VLM's nested text backbone.
    """
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    # --- Vision tower ---
    hf_pixtral_vision.prepare_for_spyre(model)

    # --- Text decoder ---
    # Re-pin the multi_modal_projector to CPU: _move_to_spyre_with_layout
    # will blanket-move every param; the projector must run on CPU because
    # it processes CPU vision features (same pattern as granite_vision_mm's
    # layerwise_projectors pin).
    if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        model.model.multi_modal_projector.to("cpu")

    # Do NOT call prepare_standard_gqa: it writes model._spyre_compiled_blocks,
    # which would collide with the vision tower's compiled blocks stored there
    # by hf_pixtral_vision.prepare_for_spyre.  Mirror what hf_granite_vision_mm
    # does: call the constituent parts individually and store text blocks in
    # model._spyre_text_blocks.
    prepare_rope_and_heads(model)
    patch_rmsnorm(MistralRMSNorm)
    pad_lm_head(model)

    backbone = get_backbone(model)
    model._spyre_text_blocks = [
        make_standard_gqa_block(layer) for layer in backbone.layers
    ]


# ---------------------------------------------------------------------------
# Vision feature extraction
# ---------------------------------------------------------------------------


def _image_features(model, pixel_values, image_sizes):
    """Run the Pixtral tower + projector → image features for scatter injection.

    Reproduces ``Mistral3Model.get_image_features``:
    1. Run the prepared Pixtral tower with ``output_hidden_states=True``.
    2. Select ``vision_feature_layer`` hidden state(s).
    3. Apply the checkpoint's own ``multi_modal_projector`` on CPU
       (RMSNorm + PatchMerger + two linear layers).

    Returns ``image_features`` ``[total_image_tokens, text_hidden]`` on CPU.
    """
    cfg = model.config
    vision_feature_layer = cfg.vision_feature_layer

    _last_hidden, all_hidden_states = hf_pixtral_vision.prefill_vision_tower(
        model, pixel_values, image_sizes, output_hidden_states=True
    )

    # Select feature layer(s) — all_hidden_states is a tuple indexed from 0
    # (input embeddings after ln_pre) to N (last encoder layer).
    if isinstance(vision_feature_layer, int):
        selected = all_hidden_states[vision_feature_layer]  # [1, P, hidden]
        selected = selected.squeeze(0)  # [P, hidden]
    else:
        parts = [all_hidden_states[i].squeeze(0) for i in vision_feature_layer]
        selected = torch.cat(parts, dim=-1)  # [P, hidden * N]

    # Move to CPU for the projector (CPU module; see prepare_for_spyre note).
    projector = model.model.multi_modal_projector.to("cpu")
    dtype = get_model_dtype(model)
    image_features = projector(selected.to("cpu").to(dtype), image_sizes)
    return image_features  # [total_image_tokens, text_hidden] on CPU


# ---------------------------------------------------------------------------
# Image-token injection
# ---------------------------------------------------------------------------


def _vision_mask(model, input_ids):
    """``[B, L, 1]`` bool mask, True at ``image_token_index`` positions."""
    return (input_ids == model.config.image_token_index).unsqueeze(-1)


def _inject_image_features(hidden_states, features, vision_mask_cpu):
    """Scatter ``features`` into the image-token slots of ``hidden_states``.

    Mirrors ``Mistral3Model.forward``'s ``inputs_embeds.masked_scatter``.
    On Spyre the on-device boolean reduction and ``masked_scatter`` don't
    lower; we build an additive tensor on **CPU** and move it to the device
    for a plain elementwise add — same technique as
    ``hf_granite_vision_mm._inject_deepstack``.

    The image-token slots are zeroed at embed time, so the injection reduces
    to ``h + additive`` where ``additive`` is ``features`` scattered at image
    positions and zero elsewhere.
    """
    flat_mask = vision_mask_cpu.squeeze(-1)  # [B, L] bool on CPU
    hidden = hidden_states.shape[-1]
    features = features.to("cpu", hidden_states.dtype)
    n_image_tokens = int(flat_mask.sum())
    if n_image_tokens * hidden != features.numel():
        raise ValueError(
            f"image tokens and features do not match: "
            f"tokens {n_image_tokens}, features {tuple(features.shape)}"
        )
    additive = torch.zeros(
        flat_mask.shape[0], flat_mask.shape[1], hidden, dtype=hidden_states.dtype
    )
    additive[flat_mask] = features.view(n_image_tokens, hidden)
    return hidden_states + additive.to(hidden_states.device)


# ---------------------------------------------------------------------------
# Text backbone forward (embeds-in)
# ---------------------------------------------------------------------------


def _embed_text(model, input_ids):
    """Token embeddings (Mistral has no embedding multiplier)."""
    backbone = get_backbone(model)
    ids = input_ids.to(backbone.embed_tokens.weight.device)
    return backbone.embed_tokens(ids)


def _run_text_backbone(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
    image_features=None,
    vision_mask=None,
):
    """Mistral text backbone over pre-computed ``inputs_embeds``.

    At prefill ``image_features`` + ``vision_mask`` carry the image injection:
    image-token slots (zeroed at embed time) receive the projected features via
    a CPU additive scatter before the first decoder layer. Decode steps pass
    ``image_features=None`` (pure text).
    """
    backbone = get_backbone(model)
    h = inputs_embeds

    # Single flat injection before layer 0 (unlike Granite's per-layer deepstack)
    if image_features is not None and vision_mask is not None:
        h = _inject_image_features(h, image_features, vision_mask)

    selected_freqs = model._spyre_rope(h, position_ids)
    for i, compiled_block in enumerate(model._spyre_text_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )
    return backbone.norm(h)


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
    image_features=None,
    vision_mask=None,
):
    """Run text backbone over embeds + LM head → logits."""
    h = _run_text_backbone(
        model,
        inputs_embeds,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
        image_features=image_features,
        vision_mask=vision_mask,
    )
    return model.lm_head(h)


# ---------------------------------------------------------------------------
# Prefill (shared between prefill_logits and generate)
# ---------------------------------------------------------------------------


def _prefill_forward(
    model,
    padded_ids,
    padded_len,
    prompt_offsets,
    position_ids,
    pixel_values,
    image_sizes,
    key_caches,
    value_caches,
    max_cache_len,
):
    """Shared multimodal prefill: padded ids + image → first-step logits.

    Builds scaled text embeddings, zeroes the ``<image>`` slots, runs the
    Pixtral tower + projector for image features, then runs the Mistral
    decoder once with the features injected before layer 0.  Returns
    full-sequence logits ``[B, padded_len, padded_vocab]``.
    """
    model_dtype = get_model_dtype(model)

    inputs_embeds = _embed_text(model, padded_ids)
    vision_mask = _vision_mask(model, padded_ids)
    # Zero the <image> slots: multiply by a (0/1) keep factor built on CPU
    # (masked_fill_ and boolean ops don't lower on Spyre).
    keep = (~vision_mask).to(model_dtype).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds * keep

    image_feats = _image_features(model, pixel_values, image_sizes)

    prefill_mask = build_prefill_mask(
        padded_ids.shape[0],
        padded_len,
        max_cache_len,
        prompt_offsets,
        dtype=model_dtype,
    )
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
        image_features=image_feats,  # on CPU; _inject_image_features moves it
        vision_mask=vision_mask,  # on CPU
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prefill_logits(model, input_ids, attention_mask, pixel_values, image_sizes):
    """One-shot prefill of the (text + image-injected) sequence → logits.

    Left-pads to a BLOCK_SIZE multiple, zeroes image-token embedding slots,
    runs the Pixtral tower + projector, injects image features before layer 0,
    and returns full-sequence logits ``[B, L, padded_vocab]``.

    Callers take ``[:, -1, :true_vocab]`` for the first generated token.
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
        image_sizes,
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
    image_sizes,
    max_new_tokens,
    do_sample=None,
    temperature=None,
    top_k=None,
    top_p=None,
):
    """Autoregressive image→text generation on Spyre (greedy / top-k/p sampling).

    Mirrors ``hf_common.generate``'s 64-block padded decode, but driven by
    **embeddings** so the prefill step can carry the image injection:

    - **Prefill** (step 0): build text embeddings, zero the ``<image>`` slots,
      run the Pixtral tower + projector, inject image features before decoder
      layer 0, run the Mistral decoder once.
    - **Decode** (steps ≥1): each newly generated token id is embedded with
      the text embedding table and fed back — pure text, no image re-encoding.

    Inputs come pre-tokenized from ``AutoProcessor`` (handles chat template +
    ``[IMG]`` token expansion). Assumes **left-padded** input (set
    ``processor.tokenizer.padding_side = 'left'``).

    Returns a list of decoded strings (one per batch row), EOS-trimmed.
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
    model_dtype = get_model_dtype(model)

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
        model, batch_size, max_cache_len, model_dtype
    )

    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None
    finished = torch.zeros(batch_size, dtype=torch.bool)
    num_generated = torch.zeros(batch_size, dtype=torch.long)

    def embed_ids(ids):
        """Token ids → embeddings (decode steps; pure text, no multiplier)."""
        return backbone.embed_tokens(ids)

    for i in range(max_new_tokens):
        if i == 0:
            # --- PREFILL: text embeds with image slots zeroed, flat injection ---
            logits = _prefill_forward(
                model,
                input_ids,
                padded_len,
                prompt_offsets,
                position_ids,
                pixel_values,
                image_sizes,
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
                    dtype=model_dtype,
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

        # Token selection (CPU) — mirrors hf_common.generate.
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
