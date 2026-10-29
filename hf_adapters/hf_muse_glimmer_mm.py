# Copyright 2026 The Torch-Spyre Authors.
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

"""Muse Glimmer image-to-text using its vision tower and shared text decoder.

The vision execution follows the Muse implementation in PR #493: dynamic
patch metadata and pixel shuffle run on CPU, while encoder blocks and the
perception projector compile on Spyre. Both generation APIs share the optimized
decoder, normalization, attention gates, KV caches and LM head in
``hf_muse_glimmer`` and ``hf_gemma3``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    get_vision_bilinear_indices_and_weights,
)
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)

from hf_adapters import hf_gemma3, hf_muse_glimmer
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _mask_fill_value,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    embed_text_tokens,
    get_model_dtype,
    pad_qk_proj_for_rope,
    patch_layernorm,
    run_lm_head,
)

if TYPE_CHECKING:
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerVisionEncoderLayer,
        MuseGlimmerVisionModel,
    )

_GENERATION_INPUT_NAMES: tuple[str, ...] = ("pixel_values", "image_grid_thw")
_GENERATION_REQUIRED_INPUT_NAMES: tuple[str, ...] = ()
_GENERATION_TOKEN_ALIGNED_INPUTS: dict[str, int] = {}

_VisionState = tuple[torch.Tensor, ...]
_VisionBlock = Callable[
    [_VisionState, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _segment_mask(
    cu_seqlens: torch.Tensor, total: int, padded: int, dtype: torch.dtype
) -> torch.Tensor:
    """Build a finite block-diagonal mask for independent image segments."""
    mask = torch.full((padded, padded), _mask_fill_value(dtype), dtype=dtype)
    bounds: list[int] = cu_seqlens.cpu().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask[start:end, start:end] = 0
    if padded > total:
        indices = torch.arange(total, padded)
        mask[indices, indices] = 0
    return mask[None, None]


def _vision_rope_matrices(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    padded_head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build Muse's [width, height, width, height] rotations with its +1 offset."""
    positions = position_ids.cpu().flip(-1) + 1
    width, height = positions[:, 0].float(), positions[:, 1].float()
    inv = inv_freq.cpu().float()
    half_freq = torch.cat((torch.outer(width, inv), torch.outer(height, inv)), dim=-1)
    cos, sin = half_freq.cos(), half_freq.sin()
    rotations = torch.stack((cos, -sin, sin, cos), dim=1).view(
        positions.shape[0], 2, 2, half_freq.shape[-1]
    )
    padding = padded_head_dim // 2 - half_freq.shape[-1]
    if padding > 0:
        identity = torch.zeros(positions.shape[0], 2, 2, padding)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        rotations = torch.cat((rotations, identity), dim=-1)
    return rotations.contiguous().to(dtype)


@dataclass(frozen=True)
class _VisionBlockSpec:
    num_heads: int
    head_dim: int
    scale: float
    norm1_eps: float
    norm2_eps: float


def _vision_block_spec(
    layer: MuseGlimmerVisionEncoderLayer,
    num_heads: int,
    head_dim: int,
    scale: float,
    hidden_act: str,
) -> _VisionBlockSpec:
    if hidden_act != "gelu":
        raise ValueError(f"Muse vision activation {hidden_act!r} is not supported")
    modules = (
        layer.norm1,
        layer.norm2,
        layer.attn.q_proj,
        layer.attn.k_proj,
        layer.attn.v_proj,
        layer.attn.proj,
        layer.mlp.fc1,
        layer.mlp.fc2,
    )
    if any(module.weight is None or module.bias is None for module in modules):
        raise ValueError("Muse vision blocks require affine norms and biased linears")
    return _VisionBlockSpec(
        num_heads, head_dim, scale, layer.norm1.eps, layer.norm2.eps
    )


def _vision_block_state(layer: MuseGlimmerVisionEncoderLayer) -> _VisionState:
    return (
        layer.norm1.weight,
        layer.norm1.bias,
        layer.attn.q_proj.weight,
        layer.attn.q_proj.bias,
        layer.attn.k_proj.weight,
        layer.attn.k_proj.bias,
        layer.attn.v_proj.weight,
        layer.attn.v_proj.bias,
        layer.attn.proj.weight,
        layer.attn.proj.bias,
        layer.norm2.weight,
        layer.norm2.bias,
        layer.mlp.fc1.weight,
        layer.mlp.fc1.bias,
        layer.mlp.fc2.weight,
        layer.mlp.fc2.bias,
    )


def _layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    if x.device.type != "spyre":
        return F.layer_norm(x, weight.shape, weight, bias, eps)
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    centered = xf - mean
    inverse = torch.rsqrt((centered * centered).mean(-1, keepdim=True) + eps)
    h = (x - mean.to(x.dtype)) * inverse.to(x.dtype)
    return h * weight + bias


def _apply_vision_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Match stock Muse's fp32 rotary products and sum before the output cast."""
    return cast(torch.Tensor, apply_rope_matmul(x.float(), rope.float()).to(x.dtype))


def _make_vision_forward(spec: _VisionBlockSpec) -> _VisionBlock:
    """Share one compiled executor across vision layers with matching shapes."""

    def block(
        state: _VisionState,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            norm1_weight,
            norm1_bias,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            norm2_weight,
            norm2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ) = state
        h = _layernorm(hidden_states, norm1_weight, norm1_bias, spec.norm1_eps)
        seq = h.shape[0]
        q = (
            F.linear(h, q_weight, q_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        k = (
            F.linear(h, k_weight, k_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        v = (
            F.linear(h, v_weight, v_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        q = _apply_vision_rope(q, rope).contiguous()
        k = _apply_vision_rope(k, rope).contiguous()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=spec.scale
        )
        out = out.transpose(1, 2).reshape(seq, spec.num_heads * spec.head_dim)
        h = hidden_states + F.linear(out, out_weight, out_bias)
        mlp_input = _layernorm(h, norm2_weight, norm2_bias, spec.norm2_eps)
        return h + F.linear(
            F.gelu(F.linear(mlp_input, fc1_weight, fc1_bias)), fc2_weight, fc2_bias
        )

    return torch.compile(block, dynamic=False, fullgraph=True)


def _make_vision_projector(
    model: PreTrainedModel,
) -> Callable[[torch.Tensor], torch.Tensor]:
    adapter: torch.nn.Module = model.model.vision_adapter
    projection: torch.nn.Module = model.model.vision_projection
    norm: torch.nn.Module = model.model.perception_emb_norm

    def project(hidden_states: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, norm(projection(adapter(hidden_states))))

    return torch.compile(project, dynamic=False)


def _pixel_shuffle_cpu(
    hidden_states: torch.Tensor, grid_thw: torch.Tensor, merge_size: int
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    offset = 0
    dim = hidden_states.shape[-1]
    for frames, height, width in grid_thw.cpu().tolist():
        count: int = frames * height * width
        permutation = torch.arange(height * width).view(
            height // merge_size, merge_size, width // merge_size, merge_size
        )
        permutation = permutation.permute(0, 2, 1, 3).reshape(-1)
        if frames > 1:
            permutation = (
                permutation[None] + torch.arange(frames)[:, None] * height * width
            ).reshape(-1)
        chunk = hidden_states[offset : offset + count][permutation]
        chunk = chunk.view(-1, merge_size * merge_size, dim)
        output.append(chunk.permute(0, 2, 1).contiguous().view(-1, dim * merge_size**2))
        offset += count
    return torch.cat(output)


def _vision_features(
    model: PreTrainedModel,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    tower: MuseGlimmerVisionModel = model.model.vision_tower
    cfg = tower.config
    dtype: torch.dtype = get_model_dtype(model)
    grid = image_grid_thw.cpu()
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.numel() == 0 or (grid <= 0).any():
        raise ValueError(
            "image_grid_thw must contain positive [frames, height, width] rows"
        )
    total = int(grid.prod(-1).sum())
    if pixel_values.shape[0] != total:
        raise ValueError(
            "pixel_values and image_grid_thw describe different patch counts"
        )
    padded = math.ceil(total / BLOCK_SIZE) * BLOCK_SIZE
    indices, weights = get_vision_bilinear_indices_and_weights(
        grid, cfg.pos_emb_height, spatial_merge_size=1
    )
    pos_table = tower.patch_embedder.position_embedding_table.weight.detach().cpu()
    positions = (pos_table[indices] * weights[:, :, None]).sum(0).to(dtype)
    window_index, window_cu = get_vision_window_index(
        grid,
        spatial_merge_size=1,
        window_size=cfg.pos_emb_height * cfg.patch_size,
        patch_size=cfg.patch_size,
    )
    full_cu = get_vision_cu_seqlens(grid)
    position_ids = get_vision_position_ids(grid, spatial_merge_size=1)[window_index]
    # The checkpoint's 1176-wide patch projection is not stick-aligned.
    # Keep this small frontend on CPU; the 1536-wide encoder runs on Spyre.
    pixels = pixel_values.to("cpu", dtype)[window_index]
    h: torch.Tensor = tower.patch_embedder.patch_embedding(pixels).reshape(total, -1)
    h = tower.ln_pre(h + positions[window_index]).to(DEVICE)
    if padded > total:
        h = F.pad(h, (0, 0, 0, padded - total))
    head_dim = cast(int, model._spyre_vision_head_dim)
    rope = _vision_rope_matrices(
        cast(torch.Tensor, model._spyre_vision_inv_freq), position_ids, head_dim, dtype
    )
    if padded > total:
        identity = torch.zeros(padded - total, 2, 2, head_dim // 2, dtype=dtype)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        rope = torch.cat((rope, identity))
    rope = rope[None].to(DEVICE)
    masks: dict[str, torch.Tensor] = {
        "full_attention": _segment_mask(full_cu, total, padded, dtype).to(DEVICE),
        "window_attention": _segment_mask(window_cu, total, padded, dtype).to(DEVICE),
    }
    blocks = cast(list[_VisionBlock], model._spyre_vision_blocks)
    assert cfg.layer_types is not None
    for i, block in enumerate(blocks):
        layer = cast("MuseGlimmerVisionEncoderLayer", tower.layers[i])
        h = block(_vision_block_state(layer), h, rope, masks[cfg.layer_types[i]])
    norm = cast(
        Callable[[torch.Tensor], torch.Tensor], model._spyre_compiled_vision_norm
    )
    h = norm(h[:total]).cpu()
    h = h[torch.argsort(window_index)]
    h = _pixel_shuffle_cpu(h, grid, cfg.merge_size).to(dtype).to(DEVICE)
    projector = cast(
        Callable[[torch.Tensor], torch.Tensor], model._spyre_compiled_vision_projector
    )
    return projector(h)


def _embed_and_scatter(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    image_features: torch.Tensor | None = None,
) -> torch.Tensor:
    image_mask = input_ids.cpu() == model.config.image_token_id
    if bool((input_ids == model.config.video_token_id).any()):
        raise NotImplementedError("Muse Glimmer video inputs are not supported")
    h: torch.Tensor = embed_text_tokens(model, input_ids)
    if image_features is None:
        return h
    image_tokens = int(image_mask.sum())
    hidden = h.shape[-1]
    if image_features.numel() != image_tokens * hidden:
        raise ValueError(
            "Image features and image tokens do not match: "
            f"tokens {image_tokens}, features {tuple(image_features.shape)}"
        )
    keep = (~image_mask).unsqueeze(-1).to(h.dtype).to(h.device)
    additive = torch.zeros(h.shape, dtype=h.dtype)
    additive[image_mask] = image_features.to("cpu", h.dtype).reshape(
        image_tokens, hidden
    )
    return cast(torch.Tensor, h * keep + additive.to(h.device))


def prepare_for_spyre(model: PreTrainedModel) -> None:
    """Prepare both towers, sharing the causal adapter's text execution."""
    if getattr(model, "_spyre_muse_vision_prepared", False):
        return
    hf_muse_glimmer.prepare_for_spyre(model)
    tower: MuseGlimmerVisionModel = model.model.vision_tower
    cfg = tower.config
    original_head_dim: int = cfg.hidden_size // cfg.num_attention_heads
    head_dim = math.ceil(original_head_dim / (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    scale = original_head_dim**-0.5
    compiled: dict[_VisionBlockSpec, _VisionBlock] = {}
    signatures: dict[
        _VisionBlockSpec, tuple[tuple[tuple[int, ...], torch.dtype], ...]
    ] = {}
    blocks: list[_VisionBlock] = []
    for module in tower.layers:
        layer = cast("MuseGlimmerVisionEncoderLayer", module)
        layer.attn.q_proj = pad_qk_proj_for_rope(
            layer.attn.q_proj, cfg.num_attention_heads, original_head_dim, head_dim
        )
        layer.attn.k_proj = pad_qk_proj_for_rope(
            layer.attn.k_proj, cfg.num_attention_heads, original_head_dim, head_dim
        )
        layer.attn.v_proj = _pad_proj_output_simple(
            layer.attn.v_proj, cfg.num_attention_heads, original_head_dim, head_dim
        )
        layer.attn.proj = _pad_proj_input_simple(
            layer.attn.proj, cfg.num_attention_heads, original_head_dim, head_dim
        )
        spec = _vision_block_spec(
            layer, cfg.num_attention_heads, head_dim, scale, cfg.hidden_act
        )
        signature = tuple((tuple(t.shape), t.dtype) for t in _vision_block_state(layer))
        if spec in signatures and signatures[spec] != signature:
            raise ValueError("Muse vision layers have incompatible state signatures")
        signatures[spec] = signature
        if spec not in compiled:
            compiled[spec] = _make_vision_forward(spec)
        blocks.append(compiled[spec])
    patch_layernorm(tower.ln_pre, tower.ln_post)
    setattr(model, "_spyre_vision_inv_freq", tower.rotary_emb.inv_freq.detach().cpu())
    setattr(model, "_spyre_vision_head_dim", head_dim)
    setattr(model, "_spyre_vision_blocks", blocks)
    setattr(
        model,
        "_spyre_compiled_vision_norm",
        torch.compile(tower.ln_post, dynamic=False),
    )
    setattr(model, "_spyre_compiled_vision_projector", _make_vision_projector(model))
    setattr(
        model,
        "_spyre_cpu_submodules",
        [
            "model.vision_tower.patch_embedder.patch_embedding",
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.vision_tower.ln_pre",
        ],
    )
    setattr(model, "_spyre_muse_vision_prepared", True)


def _logits_from_embeds(
    model: PreTrainedModel,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    cache_index: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    *,
    logits_to_keep: int = 0,
) -> torch.Tensor:
    h = hf_gemma3._run_backbone_from_embeds(
        model,
        hf_muse_glimmer._to_decoder_layout(inputs_embeds),
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    logits: torch.Tensor = run_lm_head(model, h, logits_to_keep=logits_to_keep)
    return logits[..., : model.config.text_config.vocab_size]


def _prefill_forward(
    *,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    cache_index: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    logits_to_keep: int = 0,
) -> torch.Tensor:
    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError("pixel_values and image_grid_thw must be supplied together")
    features: torch.Tensor | None = None
    if pixel_values is not None and image_grid_thw is not None:
        features = _vision_features(model, pixel_values, image_grid_thw)
    return _logits_from_embeds(
        model,
        _embed_and_scatter(model, input_ids, features).to(DEVICE),
        position_ids.to(DEVICE),
        attention_mask.to(DEVICE),
        key_caches,
        value_caches,
        cache_index,
        logits_to_keep=logits_to_keep,
    )
