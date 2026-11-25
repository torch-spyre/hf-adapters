# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL embeddings using stock HF media handling and ST last-token pooling.

Text reuses Qwen3's compiled blocks. Vision blocks run on Spyre with padded
axial RoPE heads. Patch embedding, position interpolation, patch mergers,
media scattering, and SentenceTransformers pooling/normalization stay on CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLModel, Qwen3VLTextModel
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLTextDecoderLayer,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
)
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    pad_qk_proj_for_rope,
    patch_cpu_inputs,
    reject_dynamic_rope,
)
from hf_adapters.hf_qwen3 import _make_compiled_block

_is_encoder_only = True
_preserve_native_forward = True

TextBlock = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
VisionBlock = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _rotation_matrices(
    cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Convert HF's recomposed spatial frequencies to shared matrix RoPE.

    Construct on CPU to preserve HF's axis selection and dtype rounding. Q/K
    padding places each native half at the start of the corresponding padded
    half; unused channels receive identity rotations.
    """
    half = cos.shape[-1] // 2
    rotation = torch.stack(
        (cos[..., :half], -sin[..., :half], sin[..., :half], cos[..., :half]),
        dim=-2,
    ).reshape(*cos.shape[:-1], 2, 2, half)
    pad = head_dim // 2 - half
    if pad:
        identity = torch.zeros(*cos.shape[:-1], 2, 2, pad, dtype=cos.dtype)
        identity[..., 0, 0, :] = 1
        identity[..., 1, 1, :] = 1
        rotation = torch.cat((rotation, identity), dim=-1)
    return rotation


def _make_text_block(layer: Qwen3VLTextDecoderLayer) -> TextBlock:
    qwen_block = _make_compiled_block(layer)

    def forward(
        hidden: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
        visual_addition: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _, _ = qwen_block(hidden, rope, mask, None, None, None)
        return hidden + visual_addition

    return torch.compile(forward, dynamic=False, fullgraph=True)


def _text_forward(
    self: Qwen3VLTextModel,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    use_cache: bool | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs: Any,
) -> BaseModelOutputWithPast | tuple[torch.Tensor]:
    if past_key_values is not None or use_cache:
        raise SpyreUnsupportedFeatureError(
            "Qwen3-VL embedding supports cache-free prefill only"
        )
    if kwargs.get("output_attentions", self.config.output_attentions) or kwargs.get(
        "output_hidden_states", self.config.output_hidden_states
    ):
        raise SpyreUnsupportedFeatureError(
            "Qwen3-VL embedding does not return intermediate states on Spyre"
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    hidden: torch.Tensor
    if inputs_embeds is None:
        assert input_ids is not None
        hidden = self.embed_tokens(input_ids.cpu())
    else:
        hidden = inputs_embeds.cpu()
    batch, length, _ = hidden.shape
    if length == 0:
        raise ValueError("Qwen3-VL embedding requires a nonempty sequence")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise SpyreUnsupportedFeatureError(
            "Qwen3-VL embedding requires a 2D padding mask"
        )
    pad = (-length) % BLOCK_SIZE
    valid = (
        torch.ones((batch, length), dtype=torch.bool)
        if attention_mask is None
        else attention_mask.cpu().bool()
    )
    valid = F.pad(valid, (0, pad), value=False)
    positions = (
        torch.arange(length)[None].expand(batch, -1)
        if position_ids is None
        else position_ids.cpu()
    )
    text_positions: torch.Tensor | None = None
    if positions.ndim == 2:
        text_positions = F.pad(positions, (0, pad))
        positions = positions[None].expand(3, -1, -1)
    elif positions.ndim == 3 and positions.shape[0] == 4:
        text_positions = F.pad(positions[0], (0, pad))
        positions = positions[1:]
    elif positions.ndim != 3 or positions.shape[0] != 3:
        raise ValueError("position_ids must have shape [B,L], [3,B,L], or [4,B,L]")
    positions = F.pad(positions, (0, pad))
    hidden = F.pad(hidden, (0, 0, 0, pad))
    hf_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=hidden,
        attention_mask=valid,
        past_key_values=None,
        position_ids=text_positions,
    )
    mask: torch.Tensor
    if hf_mask is None:
        mask = torch.ones(
            (batch, 1, length + pad, length + pad), dtype=torch.bool
        ).tril()
    elif isinstance(hf_mask, torch.Tensor):
        mask = hf_mask
    else:
        raise SpyreUnsupportedFeatureError(
            "Qwen3-VL embedding does not support flex attention masks"
        )
    if mask.dtype == torch.bool:
        mask = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill_(
            ~mask, -torch.inf
        )
    # Left-padding queries can be fully masked. A private diagonal keeps every
    # layer finite with ordinary softmax, without exposing padding to real queries.
    mask.diagonal(dim1=-2, dim2=-1).masked_fill_(~valid[:, None, :], 0)
    cos, sin = self.rotary_emb(hidden, positions)
    rope = _rotation_matrices(cos, sin, self.config.head_dim).to(DEVICE)
    mask = mask.to(DEVICE)
    zero_addition = torch.zeros_like(hidden).to(DEVICE)
    additions: list[torch.Tensor] = []
    if deepstack_visual_embeds is not None:
        if visual_pos_masks is None:
            raise ValueError("DeepStack features require visual_pos_masks")
        visual_mask = F.pad(visual_pos_masks.cpu().bool(), (0, pad), value=False)
        for features in deepstack_visual_embeds:
            addition = torch.zeros_like(hidden)
            addition[visual_mask] = features.to(device="cpu", dtype=hidden.dtype)
            additions.append(addition.to(DEVICE))
    hidden = hidden.to(DEVICE)
    blocks: list[TextBlock] = getattr(self, "_spyre_compiled_blocks")
    for index, block in enumerate(blocks):
        addition = additions[index] if index < len(additions) else zero_addition
        hidden = block(hidden, rope, mask, addition).clone()
    norm: Callable[[torch.Tensor], torch.Tensor] = getattr(self, "_spyre_compiled_norm")
    hidden = norm(hidden).cpu()[:, :length]
    return_dict: bool | None = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = self.config.return_dict
    return (
        BaseModelOutputWithPast(last_hidden_state=hidden) if return_dict else (hidden,)
    )


def _prepare_vision_attention(attn: Qwen3VLVisionAttention) -> None:
    original_dim: int = attn.head_dim
    padded_dim = ((original_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (
        2 * BLOCK_SIZE
    )
    projections: list[nn.Linear] = []
    for index in range(3):
        projection = nn.Linear(
            attn.dim,
            attn.dim,
            bias=attn.qkv.bias is not None,
            dtype=attn.qkv.weight.dtype,
        )
        projection.weight = nn.Parameter(
            attn.qkv.weight.detach()[index * attn.dim : (index + 1) * attn.dim],
            requires_grad=False,
        )
        if attn.qkv.bias is not None:
            projection.bias = nn.Parameter(
                attn.qkv.bias.detach()[index * attn.dim : (index + 1) * attn.dim],
                requires_grad=False,
            )
        pad_projection = pad_qk_proj_for_rope if index < 2 else _pad_proj_output_simple
        projections.append(
            pad_projection(projection, attn.num_heads, original_dim, padded_dim)
        )
    for name, projection in zip(
        ("_spyre_q_proj", "_spyre_k_proj", "_spyre_v_proj"), projections
    ):
        setattr(attn, name, projection)
    delattr(attn, "qkv")
    attn.proj = _pad_proj_input_simple(
        attn.proj, attn.num_heads, original_dim, padded_dim
    )
    attn.head_dim = padded_dim
    # Keep attn.scaling at the original head dimension.


def _make_vision_block(layer: Qwen3VLVisionBlock) -> VisionBlock:
    attn: Qwen3VLVisionAttention = layer.attn

    q_proj: nn.Linear = getattr(attn, "_spyre_q_proj")
    k_proj: nn.Linear = getattr(attn, "_spyre_k_proj")
    v_proj: nn.Linear = getattr(attn, "_spyre_v_proj")

    def forward(
        hidden: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        normalized = layer.norm1(hidden)
        q = (
            q_proj(normalized)
            .view(batch, length, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        k = (
            k_proj(normalized)
            .view(batch, length, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            v_proj(normalized)
            .view(batch, length, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        # HF vision RoPE upcasts Q/K before rotation, then restores their dtype.
        # Match Gemma 4 vision's explicit layout before MHA: matrix RoPE
        # returns a transposed view, while Spyre SDPA's output follows Q's layout.
        q = apply_rope_matmul(q.float(), rope.float()).to(hidden.dtype).contiguous()
        k = apply_rope_matmul(k.float(), rope.float()).to(hidden.dtype).contiguous()
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=attn.scaling
        )
        hidden = hidden + attn.proj(attended.transpose(1, 2).reshape(batch, length, -1))
        output: torch.Tensor = hidden + layer.mlp(layer.norm2(hidden))
        return output

    return torch.compile(forward, dynamic=False, fullgraph=True)


def _vision_forward(
    self: Qwen3VLVisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Any,
) -> (
    BaseModelOutputWithDeepstackFeatures
    | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
):
    if kwargs.get("output_attentions", self.config.output_attentions) or kwargs.get(
        "output_hidden_states", self.config.output_hidden_states
    ):
        raise SpyreUnsupportedFeatureError(
            "Qwen3-VL vision does not return intermediate states on Spyre"
        )
    grid_thw = grid_thw.cpu()
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=self.num_grid_per_side,
        mode=self.interpolation_mode,
        align_corners=self.interpolation_align_corners,
        spatial_merge_size=self.spatial_merge_size,
        kwargs=kwargs,
    )
    positions = get_vision_position_ids(
        grid_thw, self.spatial_merge_size, kwargs=kwargs
    )
    cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
    hidden = self.patch_embed(hidden_states.cpu())
    hidden = hidden + (self.pos_embed(indices) * weights[:, :, None]).sum(1).to(
        hidden.dtype
    )
    length = hidden.shape[0]
    pad = (-length) % BLOCK_SIZE
    cos, sin = self.rotary_pos_emb(hidden, positions)
    first_block = self.blocks[0]
    assert isinstance(first_block, Qwen3VLVisionBlock)
    first_attention: Qwen3VLVisionAttention = first_block.attn
    head_dim: int = first_attention.head_dim
    rope = _rotation_matrices(cos, sin, head_dim)
    rope = F.pad(rope, (0, 0, 0, 0, 0, 0, 0, pad))[None].to(DEVICE)
    mask = torch.full(
        (1, 1, length + pad, length + pad), -torch.inf, dtype=hidden.dtype
    )
    boundaries: list[int] = cu_seqlens.cpu().tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask[..., start:end, start:end] = 0
    mask.diagonal(dim1=-2, dim2=-1)[..., length:] = 0
    mask = mask.to(DEVICE)
    hidden = F.pad(hidden, (0, 0, 0, pad))[None].to(DEVICE)
    deepstack: list[torch.Tensor] = []
    blocks: list[VisionBlock] = getattr(self, "_spyre_compiled_blocks")
    for index, block in enumerate(blocks):
        hidden = block(hidden, rope, mask).clone()
        if index in self.deepstack_visual_indexes:
            merger_index = self.deepstack_visual_indexes.index(index)
            deepstack.append(
                self.deepstack_merger_list[merger_index](hidden.cpu()[0, :length])
            )
    hidden = hidden.cpu()[0, :length]
    merged = self.merger(hidden)
    return_dict: bool | None = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = self.config.return_dict
    if not return_dict:
        return hidden, merged, deepstack
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden, pooler_output=merged, deepstack_features=deepstack
    )


def prepare_for_spyre(model: Qwen3VLModel | Qwen3VLTextModel) -> None:
    """Retain HF's multimodal frontend and opt into PR #802's mixed placement."""
    if not isinstance(model, (Qwen3VLModel, Qwen3VLTextModel)):
        raise TypeError(
            f"Expected Qwen3VLModel or Qwen3VLTextModel, got {type(model).__name__}"
        )
    if getattr(model, "_spyre_qwen3_vl_embedding_prepared", False):
        return
    text: Qwen3VLTextModel = (
        model.language_model if isinstance(model, Qwen3VLModel) else model
    )
    reject_dynamic_rope(text.rotary_emb, "Qwen3-VL embedding")
    if text.config.head_dim % (2 * BLOCK_SIZE):
        raise SpyreUnsupportedModelError(
            "Qwen3-VL embedding text head_dim must be a multiple of 128"
        )
    blocks: list[TextBlock] = []
    for layer in text.layers:
        if not isinstance(layer, Qwen3VLTextDecoderLayer):
            raise TypeError("Expected Qwen3VLTextDecoderLayer")
        blocks.append(_make_text_block(layer))
    setattr(text, "_spyre_compiled_blocks", blocks)
    setattr(
        text,
        "_spyre_compiled_norm",
        torch.compile(text.norm, dynamic=False, fullgraph=True),
    )
    setattr(text, "forward", MethodType(_text_forward, text))
    prefix = "language_model." if isinstance(model, Qwen3VLModel) else ""
    cpu_modules: list[str] = [prefix + "embed_tokens", prefix + "rotary_emb"]
    if isinstance(model, Qwen3VLModel):
        vision: Qwen3VLVisionModel = model.visual
        vision_blocks: list[VisionBlock] = []
        for layer in vision.blocks:
            if not isinstance(layer, Qwen3VLVisionBlock):
                raise TypeError("Expected Qwen3VLVisionBlock")
            _prepare_vision_attention(layer.attn)
            vision_blocks.append(_make_vision_block(layer))
        setattr(vision, "_spyre_compiled_blocks", vision_blocks)
        setattr(vision, "forward", MethodType(_vision_forward, vision))
        cpu_modules.extend(
            [
                "visual.patch_embed",
                "visual.pos_embed",
                "visual.rotary_pos_emb",
                "visual.merger",
                "visual.deepstack_merger_list",
            ]
        )
        patch_cpu_inputs(model)
        for method in ("get_image_features", "get_video_features"):
            patch_cpu_inputs(model, method_name=method)
    setattr(model, "_spyre_cpu_submodules", cpu_modules)
    setattr(
        model,
        "_spyre_device_parameter_name",
        prefix + "layers.0.self_attn.q_proj.weight",
    )
    setattr(model, "_spyre_qwen3_vl_embedding_prepared", True)
