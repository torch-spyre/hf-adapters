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
``load_model`` / ``prepare_for_spyre`` covers them. Exposes the private
prefill/decode hooks used by the auto model's generation loop.

Verified on CPU to match stock full-deepstack ``model.forward`` (first-token
logits cosine ≥ 0.999, argmax match) and stock ``model.generate`` (token-exact).
"""

import torch

from hf_adapters import hf_siglip_vision
from hf_adapters.hf_common import (
    DEVICE,
    get_backbone,
    get_model_dtype,
    pad_lm_head,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
)

_GENERATION_INPUT_NAMES: tuple = ("pixel_values", "image_sizes")
_GENERATION_TOKEN_ALIGNED_INPUTS: dict = {}


def prepare_for_spyre(model):
    """Prepare BOTH towers of a loaded Granite Vision VLM in-place.

    Vision tower → ``hf_siglip_vision.prepare_for_spyre`` (compiled pre-LN
    blocks, head padding, CPU patch-embed). Text decoder → Granite RoPE/head
    prep + compiled Granite blocks + padded LM head, mirroring
    ``hf_granite.prepare_for_spyre`` but against the VLM's nested text backbone.
    """
    # --- Vision tower (resolves model.model.vision_tower) ---
    hf_siglip_vision.prepare_for_spyre(model)

    # --- Text decoder (model.model.language_model via get_backbone) ---
    prepare_rope_and_heads(model)
    pad_lm_head(model)
    backbone = get_backbone(model)
    model._spyre_text_blocks = prepare_standard_gqa_blocks(backbone.layers, True)
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)


def _embed_text(model, input_ids):
    """Token embeddings * embedding_multiplier (Granite scales its embeddings).

    The gather runs on ``embed_tokens``' device — after the Spyre device move
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
    # them on CPU (vision features are moved to CPU first). load_model_to_spyre
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
    cache_index,
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
            cache_index,
        )
    return model._spyre_compiled_norm(h)


def _prefill_forward(
    *,
    model,
    input_ids,
    position_ids,
    attention_mask,
    key_caches,
    value_caches,
    cache_index,
    pixel_values,
    image_sizes,
):
    """The shared multimodal prefill: padded ids + image → first-step logits.

    Builds scaled text embeddings, zeroes the ``<image>`` slots, builds the
    deepstack/spatial features, and runs the Granite decoder once with injection
    at the mapped layers (writing into the supplied KV caches). Returns
    full-sequence logits ``[B, padded_len, padded_vocab]``. The vision mask is
    built on the *padded* ids so it aligns with the embeddings.

    KV caches are passed in so the generation loop can size them for the full
    prefill and decode sequence.
    """
    model_d_type = get_model_dtype(model)
    # _embed_text returns embeds on the embedding table's device (Spyre after
    # the layout move). Zero the <image> slots by multiplying with a keep factor
    # (0 at image positions, 1 elsewhere): masked_fill_ does not lower on the
    # Spyre eager backend, but elementwise mul does. The keep factor is built on
    # CPU (the bool/not op also doesn't lower) then moved to the embeds' device.
    inputs_embeds = _embed_text(model, input_ids)
    vision_mask = _vision_mask(model, input_ids)
    keep = (~vision_mask).to(model_d_type).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds * keep
    deepstack = _deepstack_features(model, pixel_values, image_sizes)
    return _logits_from_embeds(
        model,
        inputs_embeds.to(DEVICE),
        position_ids.to(DEVICE),
        attention_mask.to(DEVICE),
        key_caches,
        value_caches,
        cache_index=cache_index,
        deepstack=deepstack,
        vision_mask=vision_mask,  # kept on CPU: _inject_deepstack scatters on CPU
    )


def _logits_from_embeds(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    deepstack=None,
    vision_mask=None,
):
    """Run text backbone over embeds + LM head / logits scaling -> logits.

    ``cache_index`` is the KV-write coordinate forwarded verbatim to the
    backbone: the int64 destination positions along the cache's sequence dim
    (``0..padded_len`` at prefill, a single slot per decode step).
    """
    h = _run_text_backbone(
        model,
        inputs_embeds,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
        deepstack=deepstack,
        vision_mask=vision_mask,
    )
    return model.lm_head(h) / model.config.text_config.logits_scaling
