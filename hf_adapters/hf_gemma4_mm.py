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
HuggingFace adapter for Gemma 4 multimodal models on Spyre — image→text.

Supports both encoder-free ``Gemma4UnifiedConfig`` checkpoints and full-vision
``Gemma4Config`` checkpoints. Full checkpoints run their transformer encoder
blocks on Spyre while keeping position lookup and spatial pooling on CPU. The
shared multimodal frontend scatters projected image features into ``<image>``
token slots, then delegates to the existing Gemma 4 dense/PLE or MoE text
implementation selected by the nested text config.

For encoder-free checkpoints, vision is a pure projection of raw
(processor-merged) pixel patches into the LM embedding space:

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
"vision"``: within one image, soft tokens attend bidirectionally in sliding
layers. Stock keeps full-attention layers causal and builds sliding masks as
``AND(sliding_window, OR(causal, blockwise))``, where the blockwise overlay allows
positions belonging to the same image group. Decode steps are pure text, so no
blockwise band is needed after prefill.

**Vision embedder on Spyre.** The compilable core (LN₁→Dense→LN₂→+posemb→
pos_norm→RMSNorm→Linear) is ``torch.compile``d and runs on Spyre. The
position-embedding gather (integer XY ``image_position_ids`` with ``-1``
padding, validity masking) and the final padding-patch strip are computed on
**CPU** — those integer-gather / boolean-index ops don't lower on the Spyre
backend (same doctrine as the SigLIP CPU patch-embed in ``hf_siglip_vision``).
The CPU-built per-patch positional-embedding tensor is passed into the compiled
core as a device argument.

The text decoder reuses ``hf_gemma4`` unchanged. Both towers live under the one loaded VLM, so a
single ``prepare_for_spyre`` covers them. Exposes the private prefill/decode
hooks used by the auto model's generation loop.

Scope: **text + image**. Audio and video are asserted out loudly
(``prepare_for_spyre`` / forward raise if audio/video inputs are present).
"""

import torch

from hf_adapters import hf_gemma4, hf_gemma4_moe, hf_gemma4_vision
from hf_adapters.hf_common import (
    DEVICE,
    get_backbone,
    get_model_dtype,
    patch_layernorm,
    text_config,
)

_GENERATION_INPUT_NAMES: tuple = (
    "pixel_values",
    "image_position_ids",
    "mm_token_type_ids",
)
_GENERATION_TOKEN_ALIGNED_INPUTS: dict = {"mm_token_type_ids": 0}


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
    assert getattr(cfg, "use_bidirectional_attention", None) in (None, "vision"), (
        "hf_gemma4_mm supports causal or vision-bidirectional Gemma 4 text; got "
        f"use_bidirectional_attention={getattr(cfg, 'use_bidirectional_attention', None)!r}."
    )
    assert getattr(model.model, "embed_vision", None) is not None, (
        "hf_gemma4_mm requires a vision embedder (model.model.embed_vision); "
        "this checkpoint has no vision_config."
    )

    # Reuse the text adapter selected by the nested decoder configuration.
    if getattr(cfg, "enable_moe_block", False):
        hf_gemma4_moe.prepare_text_decoder_for_spyre(model)
    else:
        hf_gemma4.prepare_text_decoder_for_spyre(model)

    if getattr(model.model, "vision_tower", None) is not None:
        # Full checkpoints run the dense transformer body on Spyre. The position
        # lookup, spatial pooler, and text-space projection remain on CPU.
        hf_gemma4_vision.prepare_for_spyre(model)
        cpu_submodules = list(getattr(model, "_spyre_cpu_submodules", []))
        cpu_submodules.extend(
            [
                "model.vision_tower.patch_embedder",
                "model.vision_tower.pooler",
                "model.embed_vision",
            ]
        )
        if getattr(model.model, "audio_tower", None) is not None:
            cpu_submodules.extend(["model.audio_tower", "model.embed_audio"])
        model._spyre_cpu_submodules = cpu_submodules
        model._spyre_vision_core = None
    else:
        # Unified checkpoints use an attention-free projection core on Spyre.
        # Patch the three LayerNorms before compiling so the graph captures the
        # unfused fp32-reduction implementation.
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
    """Run the checkpoint's vision path and return flattened text-space features."""
    # anyres / multi-image: [B, T, P, ...] -> [B*T, P, ...]
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.flatten(0, 1)
        image_position_ids = image_position_ids.flatten(0, 1)

    if getattr(model.model, "vision_tower", None) is not None:
        features = hf_gemma4_vision.prefill_vision_tower(
            model, pixel_values, image_position_ids
        )
        return model.model.embed_vision(features.to("cpu"))

    embedder = _vision_embedder(model)
    dtype = get_model_dtype(model)

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
    """Apply stock's sliding-window lower bound to an additive prefill mask.

    Masks keys further back than ``sliding_window`` (``q - k >= window``) but —
    unlike ``hf_common.add_causal_sliding_window_band`` — does not independently
    mask future keys. For multimodal prefill, call this after OR-ing the causal and
    blockwise masks so the window gates bidirectional image edges too.

    Prefill only (``cache_index`` starts at cache slot 0), so a query row's cache
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
    """Build Gemma 4's causal full mask and windowed blockwise sliding mask."""
    orig_device = prefill_mask.device
    prefill_cpu = prefill_mask.to("cpu")
    blockwise_cpu = blockwise_band.to("cpu")

    full_mask = prefill_cpu.to(orig_device)
    blockwise_causal = torch.maximum(prefill_cpu, blockwise_cpu)
    sliding_mask = _sliding_window_lower_band(blockwise_causal, sliding_window).to(
        orig_device
    )
    return {"full_attention": full_mask, "sliding_attention": sliding_mask}


def _logits_from_embeds(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    masks=None,
    input_ids=None,
):
    """Text decoder over image-scattered embeds → logits (+ softcap).

    Delegates the block walk to ``hf_gemma4._run_blocks_over_embeds``
    (shared with the text-only adapter), then applies the LM head and Gemma 4's
    ``final_logit_softcapping``. ``masks``
    (``{layer_type: mask}``) carries the bidirectional vision overlay at prefill;
    decode steps pass ``None`` and let
    the shared walk build the plain text-only causal + sliding masks from
    ``attn_mask``. ``cache_index`` is the KV-write coordinate forwarded verbatim
    to the shared walk — the destination cache slots for the computed positions:
    ``[0, padded_len)`` at prefill, a single slot per decode step. Every computed
    position is written, so the shared walk's sliding-window ``block_base`` is
    simply ``cache_index[0]``: the cache column the (only) query row occupies.
    """
    cfg = text_config(model.config)
    query_row_mask = None
    if inputs_embeds.shape[1] > 1 and not getattr(cfg, "enable_moe_block", False):
        query_row_mask = hf_gemma4._query_row_mask(inputs_embeds, attn_mask)
        inputs_embeds = inputs_embeds * query_row_mask

    if model._spyre_has_ple:
        if input_ids is None:
            raise ValueError(
                "Gemma 4 PLE decoding requires input_ids alongside inputs_embeds."
            )

        ple_input_ids = input_ids
        ple_inputs_embeds = inputs_embeds
        if inputs_embeds.shape[1] > 1:
            image_mask = input_ids.to("cpu") == model.config.image_token_id
            if image_mask.any():
                ple_input_ids = input_ids.to("cpu").clone()
                ple_input_ids[image_mask] = cfg.pad_token_id
                ple_input_ids = ple_input_ids.to(inputs_embeds.device)

                backbone = get_backbone(model)
                pad_ids = torch.full_like(ple_input_ids, cfg.pad_token_id)
                pad_embeds = backbone.embed_tokens(pad_ids)
                keep = (~image_mask).to(inputs_embeds.dtype).unsqueeze(-1)
                keep = keep.to(inputs_embeds.device)
                ple_inputs_embeds = inputs_embeds * keep + pad_embeds * (1 - keep)

        per_layer_inputs = hf_gemma4._compute_per_layer_inputs(
            model, ple_inputs_embeds, ple_input_ids
        )
    else:
        per_layer_inputs = None
    h = hf_gemma4._run_blocks_over_embeds(
        model,
        inputs_embeds,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
        masks=masks,
        per_layer_inputs=per_layer_inputs,
        query_row_mask=query_row_mask,
    )
    logits = model.lm_head(h)
    cap = text_config(model.config).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


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
    image_position_ids,
    mm_token_type_ids,
):
    """Shared multimodal prefill: padded ids + image → full-sequence logits.

    Builds scaled text embeddings with the image features scattered into the
    ``<image>`` slots, then the per-layer-type masks with the bidirectional
    vision overlay OR-ed into both full and sliding layers, and runs the decoder
    once (writing the KV caches). ``mm_token_type_ids`` has already undergone
    the same prompt compaction and block padding as ``input_ids``.
    """
    dtype = get_model_dtype(model)
    cfg = text_config(model.config)
    image_features = _image_features(model, pixel_values, image_position_ids)
    inputs_embeds = _embed_and_scatter(model, input_ids, image_features)

    masks = None
    if getattr(cfg, "use_bidirectional_attention", None) == "vision":
        padded_len = input_ids.shape[1]
        max_cache_len = attention_mask.shape[-1]
        blockwise = _blockwise_band(mm_token_type_ids, padded_len, max_cache_len, dtype)
        masks = _build_mm_masks(attention_mask, blockwise, cfg.sliding_window)
        masks = {lt: m.to(DEVICE) for lt, m in masks.items()}
    return _logits_from_embeds(
        model,
        inputs_embeds.to(DEVICE),
        position_ids.to(DEVICE),
        attention_mask.to(DEVICE),
        key_caches,
        value_caches,
        cache_index=cache_index,
        masks=masks,
        input_ids=input_ids.to(DEVICE),
    )
