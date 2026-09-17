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
    encoder_backbone_forward,
    pad_attention_heads_linear,
    pad_encoder_mlp,
)


def _make_compiled_clip_encoder_block(layer, orig_head_dim, padded_head_dim, num_heads):
    """Build a compiled pre-LN block for CLIP transformer layer.

    Compiled with ``dynamic=False``. Callers must ensure the sequence dimension
    is already padded to a BLOCK_SIZE multiple before calling the block, so the
    compiler always sees a fixed shape and Spyre's symbolic-shape issues are avoided.

    Accepts an optional additive ``attn_mask`` argument (shape ``[B,1,S,S]``) so
    the text tower's causal mask is honoured. The vision tower passes ``None``.
    """
    attn = layer.self_attn
    scale = 1.0 / math.sqrt(orig_head_dim)
    act_fn = getattr(layer.mlp, "activation_fn", getattr(layer.mlp, "act_fn", F.gelu))
    num_h = num_heads
    hd = padded_head_dim

    def block_forward(hidden_states, attn_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        residual = hidden_states
        h = layer.layer_norm1(hidden_states)
        # permute(0,2,1,3) + contiguous() produces a row-major [B,heads,seq,hd]
        # tensor. This is required on Spyre: a non-contiguous view after transpose
        # causes ``lower_pad_sequence`` to see multiple non-BLOCK_SIZE-aligned dims
        # simultaneously and reject the pad.
        q = (
            attn.q_proj(h)
            .reshape(bsz, seq_len, num_h, hd)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            attn.k_proj(h)
            .reshape(bsz, seq_len, num_h, hd)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            attn.v_proj(h)
            .reshape(bsz, seq_len, num_h, hd)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=scale
        )
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(bsz, seq_len, -1)
        attn_out = attn.out_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        h = layer.layer_norm2(hidden_states)
        h = layer.mlp.fc2(act_fn(layer.mlp.fc1(h)))
        hidden_states = residual + h
        return hidden_states

    return torch.compile(block_forward, dynamic=False)


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
        pad_attention_heads_linear(
            encoder_module,
            [layer.self_attn for layer in layers],
            orig_head_dim,
            stick_aligned_head_dim,
            config.num_attention_heads,
        )

    orig_inter = config.intermediate_size
    stick_aligned_inter = ((orig_inter + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if stick_aligned_inter > orig_inter:
        pad_encoder_mlp(layers, orig_inter, stick_aligned_inter)

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
    #
    # Sequence padding: compiled blocks use dynamic=False, so they must always
    # see a BLOCK_SIZE-aligned seq_len. The vision tower's patch count (50) is
    # already not a multiple of 64, so we pad on entry and crop on exit for both
    # towers. This mirrors what prefill_encoder does for BERT-style models.

    def _spyre_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
        # In the new transformers API (v4.52+), CLIPTextModel.forward calls
        # create_causal_mask() and passes the result as `attention_mask`.
        # When _attn_implementation == "sdpa", create_causal_mask returns None
        # (relying on is_causal=True in SDPA).  Spyre cannot use is_causal=True,
        # so we materialise the causal mask ourselves when is_causal is set and
        # no explicit mask was provided.  The vision tower never sets is_causal.
        is_causal = kwargs.get("is_causal", False)
        combined_mask = attention_mask
        if combined_mask is None and is_causal:
            bsz, seq_len, _ = inputs_embeds.shape
            # Upper-triangular mask: positions j > i get -inf so the EOS token
            # attends causally to all preceding tokens (standard CLIP text mask).
            causal = torch.full(
                (seq_len, seq_len), float("-inf"), dtype=inputs_embeds.dtype
            )
            causal = torch.triu(causal, diagonal=1)
            combined_mask = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)

        h = inputs_embeds.to(DEVICE)
        orig_seq = h.shape[1]
        pad_len = (math.ceil(orig_seq / BLOCK_SIZE) * BLOCK_SIZE) - orig_seq
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))  # pad seq dim on the right
            if combined_mask is None:
                combined_mask = torch.ones((1, 1, orig_seq, orig_seq), dtype=torch.bool)
            # Pad mask from [B,1,S,S] to [B,1,S_pad,S_pad], disallowing
            # compiler-added positions in both the query and key dimensions.
            mask_value = False if combined_mask.dtype == torch.bool else float("-inf")
            combined_mask = F.pad(
                combined_mask, (0, pad_len, 0, pad_len), value=mask_value
            )
        if combined_mask is not None:
            combined_mask = combined_mask.to(DEVICE)
        h = h.clone()  # canonical layout before first block
        for block in self._spyre_compiled_blocks:
            h = block(h, combined_mask)
            h = h.clone()  # canonical layout between blocks
        if pad_len > 0:
            h = h[:, :orig_seq, :]
        # Move back to CPU before returning. The downstream ops in
        # CLIPVisionModel (class-token slice, post_layernorm) and
        # CLIPTextModel (final_layer_norm, eos-token slice) are 2D
        # pointwise ops that the Spyre DDL cannot lower (requires ≥3 dims).
        h = h.to("cpu")
        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=h)

    encoder_module.forward = types.MethodType(_spyre_encoder_forward, encoder_module)


def load_hf_model(model_path, dtype=torch.float16, trust_remote_code=None):
    """Load CLIPModel from model_path or its subfolder (e.g. 0_CLIPModel)."""
    from transformers import CLIPModel

    try:
        model = CLIPModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        model = CLIPModel.from_pretrained(
            model_path,
            subfolder="0_CLIPModel",
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
    return model


_run_backbone_forward = encoder_backbone_forward
_is_encoder_only = True


def _patch_vision_model_forward(vision_model):
    """Patch CLIPVisionModel.forward (instance) to force pixel_values to CPU.

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


def _patch_text_model_forward(text_model):
    """Patch CLIPTextModel.forward (instance) to force input_ids/attention_mask to CPU.

    ST places all features on model.device (Spyre) before calling forward, but
    the text embeddings (token_embedding + position_embedding) are pinned to CPU.
    Force input_ids, position_ids and the 2-D padding attention_mask to CPU here
    so the whole text tower runs on CPU up to the encoder, which moves inputs to
    Spyre internally.
    """
    _orig_forward = text_model.__class__.forward

    def _spyre_text_forward(
        self, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to("cpu")
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to("cpu")
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            position_ids = position_ids.to("cpu")
        return _orig_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            cpu_submods.append("text_model.embeddings")
            if hasattr(model.text_model, "final_layer_norm"):
                cpu_submods.append("text_model.final_layer_norm")
            _patch_text_model_forward(model.text_model)
        if hasattr(model, "vision_model"):
            vision_cfg = getattr(model.config, "vision_config", model.config)
            _prepare_clip_encoder(model.vision_model.encoder, vision_cfg)
            cpu_submods.append("vision_model.embeddings")
            cpu_submods.append("vision_model.pre_layrnorm")
            cpu_submods.append("vision_model.post_layernorm")
            _patch_vision_model_forward(model.vision_model)
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
        model._spyre_compiled_blocks = model.encoder._spyre_compiled_blocks
        if hasattr(model, "embeddings"):
            cpu_submods_bare = ["embeddings"]
            if hasattr(model, "pre_layrnorm"):
                cpu_submods_bare.append("pre_layrnorm")
                _patch_vision_model_forward(model)
            if hasattr(model, "post_layernorm"):
                cpu_submods_bare.append("post_layernorm")
            model._spyre_cpu_submodules = cpu_submods_bare
        return

    raise ValueError(f"Unsupported CLIP model architecture: {type(model)}")
