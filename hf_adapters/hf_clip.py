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

"""
HuggingFace Transformers adapter for CLIP text and vision models on Spyre.

Supports:
- ``CLIPModel`` (multimodal text + vision)
- ``CLIPTextModel`` / ``CLIPTextModelWithProjection``
- ``CLIPVisionModel`` / ``CLIPVisionModelWithProjection``

CLIP Transformer encoders are **pre-LN, bidirectional, no-RoPE, no-KV-cache**
transformer encoders.
- Vision tower: Conv2d patch embedding + class token + learned position embeddings on CPU.
- Text tower: Token embedding + learned position embeddings on CPU.
- Encoder blocks: pre-LN SDPA + MLP compiled on Spyre.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    encoder_backbone_forward,
    make_vision_encoder_block,
)


def _pad_clip_heads(layers, num_heads, orig_head_dim, padded_head_dim):
    """Zero-pad CLIP per-layer Q/K/V/O projections to a stick boundary."""
    for layer in layers:
        attn = layer.self_attn
        attn.q_proj = _pad_proj_output_simple(
            attn.q_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.k_proj = _pad_proj_output_simple(
            attn.k_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.v_proj = _pad_proj_output_simple(
            attn.v_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.out_proj = _pad_proj_input_simple(
            attn.out_proj, num_heads, orig_head_dim, padded_head_dim
        )


def _pad_clip_mlp(layers, orig_inter, padded_inter):
    """Zero-pad each CLIP MLP intermediate dim to a stick boundary if needed."""
    for layer in layers:
        mlp = layer.mlp
        mlp.fc1 = _pad_proj_output_simple(mlp.fc1, 1, orig_inter, padded_inter)
        mlp.fc2 = _pad_proj_input_simple(mlp.fc2, 1, orig_inter, padded_inter)


def _make_compiled_clip_encoder_block(layer, orig_head_dim, padded_head_dim, num_heads):
    """Build a compiled pre-LN block for CLIP transformer layer."""
    attn = layer.self_attn
    scale = 1.0 / math.sqrt(orig_head_dim)
    act_fn = getattr(layer.mlp, "activation_fn", getattr(layer.mlp, "act_fn", F.gelu))

    return make_vision_encoder_block(
        q_proj=attn.q_proj,
        k_proj=attn.k_proj,
        v_proj=attn.v_proj,
        o_proj=attn.out_proj,
        layer_norm1=layer.layer_norm1,
        layer_norm2=layer.layer_norm2,
        ffn_in=layer.mlp.fc1,
        act=act_fn,
        ffn_out=layer.mlp.fc2,
        num_heads=num_heads,
        head_dim=padded_head_dim,
        scale=scale,
    )


def _prepare_clip_encoder(encoder_module, config):
    """Prepare a CLIPEncoder sub-module (text or vision) for Spyre execution."""
    orig_head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    stick_aligned_head_dim = (
        (orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    ) * BLOCK_SIZE

    layers = list(encoder_module.layers)

    if stick_aligned_head_dim > orig_head_dim:
        _pad_clip_heads(
            layers,
            config.num_attention_heads,
            orig_head_dim,
            stick_aligned_head_dim,
        )

    orig_inter = config.intermediate_size
    stick_aligned_inter = ((orig_inter + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if stick_aligned_inter > orig_inter:
        _pad_clip_mlp(layers, orig_inter, stick_aligned_inter)

    compiled_blocks = [
        _make_compiled_clip_encoder_block(
            layer,
            orig_head_dim,
            stick_aligned_head_dim,
            config.num_attention_heads,
        )
        for layer in layers
    ]
    encoder_module._spyre_compiled_blocks = compiled_blocks

    # Replace CLIPEncoder.forward (instance-level) with a compiled-block loop.
    # HF's default CLIPEncoder.forward iterates self.layers (original, uncompiled).
    # We bypass it entirely and run our compiled blocks directly.
    _compiled_blocks = compiled_blocks

    def _spyre_encoder_forward(self, inputs_embeds, **kwargs):
        h = inputs_embeds.to(DEVICE)
        h = h.clone()  # canonical layout before first block
        for block in _compiled_blocks:
            h = block(h)
            h = h.clone()  # canonical layout between blocks
        # Move back to CPU before returning. The downstream ops in
        # CLIPVisionTransformer (class-token slice, post_layernorm) and
        # CLIPTextTransformer (final_layer_norm, eos-token slice) are 2D
        # pointwise ops that the Spyre DDL cannot lower (requires ≥3 dims).
        h = h.to("cpu")
        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=h)

    encoder_module.forward = types.MethodType(_spyre_encoder_forward, encoder_module)


def load_hf_model(model_path, dtype=torch.float16):
    """Load CLIPModel from model_path or its subfolder (e.g. 0_CLIPModel)."""
    from transformers import CLIPModel

    try:
        model = CLIPModel.from_pretrained(model_path, dtype=dtype, device_map="cpu")
    except Exception:
        model = CLIPModel.from_pretrained(
            model_path, subfolder="0_CLIPModel", dtype=dtype, device_map="cpu"
        )
    return model


_run_backbone_forward = encoder_backbone_forward
_is_encoder_only = True


def _patch_vision_transformer_forward(vision_model):
    """Patch CLIPVisionTransformer.forward (instance) to force pixel_values to CPU.

    ST places all features on model.device (Spyre) before calling forward, but
    the vision embeddings (Conv2d + position embed) are pinned to CPU. Force
    pixel_values to CPU here so the whole vision tower runs on CPU up to the
    encoder, which moves inputs to Spyre internally.
    """
    _orig_forward = vision_model.__class__.forward

    def _spyre_vision_forward(self, pixel_values=None, **kwargs):
        if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to("cpu")
        return _orig_forward(self, pixel_values=pixel_values, **kwargs)

    vision_model.forward = types.MethodType(_spyre_vision_forward, vision_model)


def _patch_text_transformer_forward(text_model):
    """Patch CLIPTextTransformer.forward (instance) to force input_ids to CPU.

    ST places all features on model.device (Spyre) before calling forward, but
    the text embeddings (token_embedding + position_embedding) are pinned to CPU.
    Force input_ids and position_ids to CPU here so the whole text tower runs on
    CPU up to the encoder, which moves inputs to Spyre internally.
    """
    _orig_forward = text_model.__class__.forward

    def _spyre_text_forward(
        self, input_ids=None, position_ids=None, attention_mask=None, **kwargs
    ):
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to("cpu")
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            position_ids = position_ids.to("cpu")
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to("cpu")
        return _orig_forward(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    text_model.forward = types.MethodType(_spyre_text_forward, text_model)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a CLIP model / text model / vision model in-place."""
    cpu_submods = []

    # 1. Check if model has text_model / vision_model (Full CLIPModel)
    if hasattr(model, "text_model") or hasattr(model, "vision_model"):
        if hasattr(model, "text_model"):
            text_cfg = getattr(model.config, "text_config", model.config)
            _prepare_clip_encoder(model.text_model.encoder, text_cfg)
            model.text_model.encoder._spyre_final_layer_norm = getattr(
                model.text_model, "final_layer_norm", None
            )
            cpu_submods.append("text_model.embeddings")
            if hasattr(model.text_model, "final_layer_norm"):
                cpu_submods.append("text_model.final_layer_norm")
            _patch_text_transformer_forward(model.text_model)
        if hasattr(model, "vision_model"):
            vision_cfg = getattr(model.config, "vision_config", model.config)
            _prepare_clip_encoder(model.vision_model.encoder, vision_cfg)
            model.vision_model.encoder._spyre_final_layer_norm = getattr(
                model.vision_model, "post_layernorm", None
            )
            cpu_submods.append("vision_model.embeddings")
            cpu_submods.append("vision_model.pre_layrnorm")
            cpu_submods.append("vision_model.post_layernorm")
            _patch_vision_transformer_forward(model.vision_model)
        if hasattr(model, "visual_projection"):
            cpu_submods.append("visual_projection")
        if hasattr(model, "text_projection"):
            cpu_submods.append("text_projection")

        model._spyre_cpu_submodules = cpu_submods
        return

    # 2. Bare CLIPTextModel or CLIPVisionModel
    if hasattr(model, "encoder"):
        cfg = model.config
        _prepare_clip_encoder(model.encoder, cfg)
        final_norm = getattr(
            model, "final_layer_norm", getattr(model, "post_layernorm", None)
        )
        model.encoder._spyre_final_layer_norm = final_norm
        model._spyre_compiled_blocks = model.encoder._spyre_compiled_blocks
        if hasattr(model, "embeddings"):
            cpu_submods_bare = ["embeddings"]
            if hasattr(model, "pre_layrnorm"):
                cpu_submods_bare.append("pre_layrnorm")
                _patch_vision_transformer_forward(model)
            if hasattr(model, "post_layernorm"):
                cpu_submods_bare.append("post_layernorm")
            model._spyre_cpu_submodules = cpu_submods_bare
        return

    raise ValueError(f"Unsupported CLIP model architecture: {type(model)}")
