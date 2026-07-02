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
Combined (two-tower) HuggingFace adapter for Granite Vision 4.1 on Spyre.

Where ``hf_granite_vision`` extracts only the text backbone (text-only causal
LM, used by ``AutoSpyreModelForCausalLM``), this module loads BOTH towers from
the one multimodal checkpoint via ``AutoModelForImageTextToText`` and runs the
full image→text pipeline. It is the adapter behind
``AutoSpyreModelForImageTextToText``.

    pixel_values ──► SigLIP vision tower            (hf_siglip_vision, Spyre)
                       │  output_hidden_states
                       ▼
                     layerwise + spatial projectors  (stock modules, CPU)
                       │  pack_image_features
                       ▼
                     deepstack_features dict {text_layer: features}
                       │
    input_ids ──► text embeddings ──► zero <image> slots
                       │
                       ▼
                     Granite text decoder            (hf_granite blocks, Spyre)
                       │  + deepstack injection at mapped layers
                       ▼
                     logits

This reproduces granite-vision-4.1's native **deepstack + spatial** injection:
``get_image_features`` projects several vision layers, and each projected set is
*summed into* the image-token positions before a specific decoder layer
(``deepstack_layer_map`` + ``spatial_target_layers``; image-token embedding
slots are zeroed first). The decoder layer forward itself is unchanged Granite,
so the injection is a per-layer ``masked_scatter`` between compiled blocks.

The vision tower is prepared by ``hf_siglip_vision``; the text decoder reuses
``hf_granite``'s compiled block. Both live under the one loaded VLM, so a single
``load_model`` / ``prepare_for_spyre`` covers them. Exposes ``prefill_logits``
(first-token forward) and ``generate`` (full autoregressive decode — image
features injected at prefill; decode steps are pure text).

Verified on CPU to match stock full-deepstack ``model.forward`` (first-token
logits cosine ≥ 0.999, argmax match) and stock ``model.generate`` (token-exact).
"""

import math

import torch

from hf_adapters import hf_siglip_vision
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
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    select_next_token,
)
from hf_adapters.hf_granite import _make_compiled_block


def prepare_for_spyre(model):
    """Prepare BOTH towers of a loaded Granite Vision VLM in-place.

    Vision tower → ``hf_siglip_vision.prepare_for_spyre`` (compiled pre-LN
    blocks, head padding, CPU patch-embed). Text decoder → Granite RoPE/head
    prep + compiled Granite blocks + padded LM head, mirroring
    ``hf_granite.prepare_for_spyre`` but against the VLM's nested text backbone.
    """
    from transformers.models.granite4_vision.modeling_granite4_vision import (
        Granite4VisionTextRMSNorm,
    )

    # --- Vision tower (resolves model.model.vision_tower) ---
    hf_siglip_vision.prepare_for_spyre(model)

    # --- Text decoder (model.model.language_model via get_backbone) ---
    prepare_rope_and_heads(model)
    patch_rmsnorm(Granite4VisionTextRMSNorm)
    pad_lm_head(model)
    backbone = get_backbone(model)
    model._spyre_text_blocks = [
        _make_compiled_block(layer) for layer in backbone.layers
    ]


def _embed_text(model, input_ids):
    """Token embeddings * embedding_multiplier (Granite scales its embeddings).

    The gather runs on ``embed_tokens``' device — after ``_move_to_spyre_with_layout``
    the table lives on Spyre, so ``input_ids`` is moved to match (mirrors the
    decode-step ``embed_ids``). Returns embeddings on the embedding's device.
    """
    backbone = get_backbone(model)
    ids = input_ids.to(backbone.embed_tokens.weight.device)
    h = backbone.embed_tokens(ids)
    return h * backbone.embedding_multiplier


def _deepstack_features(model, pixel_values, image_sizes):
    """Run the Spyre vision tower once, build ALL deepstack + spatial features.

    Reproduces stock ``Granite4VisionModel.get_image_features``, but the vision
    tower is the Spyre-prepared SigLIP adapter (so we can't call stock
    ``get_image_features``, whose attention path assumes unpadded heads). We run
    the tower once with ``output_hidden_states=True`` and then apply the
    checkpoint's own projectors + ``pack_image_features`` on CPU per the
    ``deepstack_layer_map`` (layerwise projectors) and ``spatial_target_layers``
    (spatial projectors from ``spatial_vision_layer``).

    Returns ``{text_layer_idx: features[num_image_tokens, hidden]}`` — one entry
    per injection point (8 for granite-vision-4.1: 4 deepstack + 4 spatial).
    """
    cfg = model.config
    inner = model.model  # Granite4VisionModel
    from transformers.models.granite4_vision.modeling_granite4_vision import (
        image_size_to_num_patches,
    )

    # anyrES: pixel_values [B, T, C, H, W] -> [B*T, C, H, W]
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.flatten(0, 1)

    _, hidden_states = hf_siglip_vision.prefill_vision_tower(
        model, pixel_values, output_hidden_states=True
    )
    dtype = get_model_dtype(model)

    # The deepstack/spatial projectors (Blip2 QFormers) and the image_newline
    # parameter are stock CPU modules: _project_and_pack / pack_image_features run
    # them on CPU (vision features are moved to CPU first). _move_to_spyre_with_layout
    # blanket-moves every param to Spyre, so re-pin these to CPU before use — the
    # same CPU-fallback contract as the patch-embed conv (idempotent; .to(cpu) on an
    # already-CPU module is a no-op).
    inner.layerwise_projectors.to("cpu")
    inner.spatial_projectors.to("cpu")
    if getattr(inner, "image_newline", None) is not None:
        # image_newline is a bare nn.Parameter; a Spyre tensor can't be re-homed
        # via .data set_data (incompatible tensor type), so replace the Parameter.
        inner.image_newline = torch.nn.Parameter(
            inner.image_newline.detach().to("cpu"), requires_grad=False
        )
    image_num_patches = [
        image_size_to_num_patches(
            image_size=imsize,
            grid_pinpoints=cfg.image_grid_pinpoints,
            patch_size=cfg.vision_config.image_size,
        )
        for imsize in image_sizes
    ]
    select_default = cfg.vision_feature_select_strategy == "default"

    def _project_and_pack(selected_layer_feature, projector):
        feat = selected_layer_feature.to("cpu")
        if select_default:
            feat = feat[:, 1:]
        projected = projector(feat.to(dtype))
        projected = torch.split(projected, image_num_patches, dim=0)
        packed, _ = inner.pack_image_features(
            projected,
            image_sizes,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            image_newline=inner.image_newline,
        )
        if isinstance(packed, (list, tuple)):
            packed = torch.cat(list(packed), dim=0)
        return packed

    deepstack = {}
    # Deepstack: each vision layer -> its own projector -> a distinct text layer.
    for proj_idx, (vision_layer, llm_layer) in enumerate(cfg.deepstack_layer_map):
        deepstack[llm_layer] = _project_and_pack(
            hidden_states[vision_layer], inner.layerwise_projectors[proj_idx]
        )
    # Spatial: 4 offset groups from a single vision layer -> 4 text layers.
    spatial_feature = hidden_states[cfg.spatial_vision_layer]
    for group_idx, llm_layer in enumerate(cfg.spatial_target_layers):
        deepstack[llm_layer] = _project_and_pack(
            spatial_feature, inner.spatial_projectors[group_idx]
        )
    return deepstack


def _vision_mask(model, input_ids):
    """``[B, L, 1]`` bool mask, True at ``image_token_id`` positions."""
    return (input_ids == model.config.image_token_id).unsqueeze(-1)


def _inject_deepstack(hidden_states, features, vision_mask_cpu):
    """Add ``features`` into image-token positions (stock deepstack injection).

    Stock does ``h.masked_scatter(mask, h[mask] + features)``. On Spyre the
    image-token slots are zeroed at embed time, so the injection is just
    ``h + additive`` where ``additive`` is ``features`` scattered into the
    image-token positions and zero elsewhere. We build that additive tensor on
    **CPU** (the mask is a fixed, statically-known CPU bool tensor) and move it to
    the device for a plain elementwise add — the on-device boolean reduction
    (``mask.sum()``), boolean indexing, and ``masked_scatter`` all fail to lower
    on Spyre. Bit-identical to the stock masked_scatter given the zeroed slots.

    Asserts the image-token count matches the feature count first (mirrors stock
    ``get_placeholder_mask``'s ``torch_compilable_check``): a token/feature
    mismatch — e.g. image-token expansion misaligned with the tiling — would
    otherwise corrupt the scatter silently.
    """
    flat_mask = vision_mask_cpu.squeeze(-1)  # [B, L] bool, on CPU
    hidden = hidden_states.shape[-1]
    features = features.to("cpu", hidden_states.dtype)
    n_image_tokens = int(flat_mask.sum())
    if n_image_tokens * hidden != features.numel():
        raise ValueError(
            f"image tokens and features do not match: tokens {n_image_tokens}, "
            f"features {tuple(features.shape)}"
        )
    additive = torch.zeros(
        flat_mask.shape[0], flat_mask.shape[1], hidden, dtype=hidden_states.dtype
    )
    additive[flat_mask] = features.view(n_image_tokens, hidden)
    return hidden_states + additive.to(hidden_states.device)


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
    deepstack=None,
    vision_mask=None,
):
    """Granite text backbone over precomputed ``inputs_embeds`` (already scaled).

    ``deepstack`` (``{layer_idx: features}``) + ``vision_mask`` (``[B, L, 1]``
    bool, kept on **CPU**) are the multimodal injections: before each mapped
    layer, the projected vision features are summed into the image-token
    positions (which were zeroed at embed time). The injection's scatter runs on
    CPU (see ``_inject_deepstack``). Used at prefill only — decode steps pass
    ``deepstack=None``.
    """
    backbone = get_backbone(model)
    h = inputs_embeds
    selected_freqs = model._spyre_rope(h, position_ids)
    for i, compiled_block in enumerate(model._spyre_text_blocks):
        if deepstack is not None and i in deepstack:
            h = _inject_deepstack(h, deepstack[i], vision_mask)
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
    """The shared multimodal prefill: padded ids + image → first-step logits.

    Builds scaled text embeddings, zeroes the ``<image>`` slots, builds the
    deepstack/spatial features, and runs the Granite decoder once with injection
    at the mapped layers (writing into the supplied KV caches). Returns
    full-sequence logits ``[B, padded_len, padded_vocab]``. The vision mask is
    built on the *padded* ids so it aligns with the embeddings.

    KV caches are passed in (not allocated here) so ``generate`` can size them
    for the whole decode while ``prefill_logits`` sizes them for one forward.
    """
    model_d_type = get_model_dtype(model)
    # _embed_text returns embeds on the embedding table's device (Spyre after
    # the layout move). Zero the <image> slots by multiplying with a keep factor
    # (0 at image positions, 1 elsewhere): masked_fill_ does not lower on the
    # Spyre eager backend, but elementwise mul does. The keep factor is built on
    # CPU (the bool/not op also doesn't lower) then moved to the embeds' device.
    inputs_embeds = _embed_text(model, padded_ids)
    vision_mask = _vision_mask(model, padded_ids)
    keep = (~vision_mask).to(model_d_type).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds * keep
    deepstack = _deepstack_features(model, pixel_values, image_sizes)
    prefill_mask = build_prefill_mask(
        padded_ids.shape[0],
        padded_len,
        max_cache_len,
        prompt_offsets,
        dtype=model_d_type,
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
        deepstack=deepstack,
        vision_mask=vision_mask,  # kept on CPU: _inject_deepstack scatters on CPU
    )


def prefill_logits(model, input_ids, attention_mask, pixel_values, image_sizes):
    """One-shot prefill of the (text + deepstack-injected image) sequence → logits.

    Left-pads to a BLOCK_SIZE multiple (same convention as ``generate``'s
    prefill arm), zeroes image-token embedding slots, runs the Granite decoder
    once with deepstack/spatial injection at the mapped layers, and returns
    full-sequence logits ``[B, L, padded_vocab]`` (callers take
    ``[:, -1, :true_vocab]`` for the first token).
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
    deepstack=None,
    vision_mask=None,
):
    """Run text backbone over embeds + LM head / logits scaling -> logits."""
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
        deepstack=deepstack,
        vision_mask=vision_mask,
    )
    return model.lm_head(h) / model.config.text_config.logits_scaling


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

    Mirrors ``hf_common.generate``'s 64-block padded decode, but the model is
    driven by **embeddings** rather than token ids so the prefill step can carry
    the deepstack image injection:

    - **Prefill** (step 0): build text embeddings, zero the ``<image>`` slots,
      build the deepstack/spatial features, and run the Granite decoder once with
      injection at the mapped layers.
    - **Decode** (steps ≥1): each newly generated token id is embedded with the
      same ``embedding_multiplier`` and fed back — pure text, no image.

    Inputs come pre-tokenized from the checkpoint's ``AutoProcessor`` (which
    handles the chat template + image-token expansion), so this takes
    ``input_ids``/``pixel_values`` directly instead of raw prompt strings.
    Assumes **left-padded** input (set ``processor.tokenizer.padding_side =
    'left'``), matching the decode loop's right-aligned prompt convention.

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
    emb_mult = backbone.embedding_multiplier
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
        return backbone.embed_tokens(ids) * emb_mult

    for i in range(max_new_tokens):
        if i == 0:
            # --- PREFILL: text embeds with image slots zeroed, deepstack
            # injection at the mapped decoder layers (shared with prefill_logits) ---
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

    # Decode generated tokens per sequence (same block-walk as hf_common.generate).
    return decode_block_walk(result, num_generated, padded_len, eos_ids, tokenizer)
