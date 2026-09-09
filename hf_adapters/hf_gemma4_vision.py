# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre execution for the transformer body of a full Gemma 4 vision tower."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    SpyreUnsupportedFeatureError,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
)
from hf_adapters.hf_gemma4 import _gemma4_rms_norm


def _pad_qk_linear(proj, num_heads, orig_head_dim, padded_head_dim):
    """Pad and reorder two-axis RoPE channels into one matrix-RoPE layout."""
    linear = proj.linear
    weight = linear.weight.detach().view(num_heads, orig_head_dim, -1)
    new_weight = torch.zeros(
        num_heads, padded_head_dim, weight.shape[-1], dtype=weight.dtype
    )
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    new_weight[:, :quarter] = weight[:, :quarter]
    new_weight[:, quarter : 2 * quarter] = weight[:, 2 * quarter : 3 * quarter]
    new_weight[:, padded_half : padded_half + quarter] = weight[
        :, quarter : 2 * quarter
    ]
    new_weight[:, padded_half + quarter : padded_half + 2 * quarter] = weight[
        :, 3 * quarter :
    ]
    padded = nn.Linear(linear.in_features, num_heads * padded_head_dim, bias=False)
    padded.weight = nn.Parameter(
        new_weight.reshape(num_heads * padded_head_dim, -1), requires_grad=False
    )
    return padded


def _pad_norm_weight(norm, orig_head_dim, padded_head_dim):
    weight = norm.weight.detach()
    padded = torch.ones(padded_head_dim, dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    padded[:quarter] = weight[:quarter]
    padded[quarter : 2 * quarter] = weight[2 * quarter : 3 * quarter]
    padded[padded_half : padded_half + quarter] = weight[quarter : 2 * quarter]
    padded[padded_half + quarter : padded_half + 2 * quarter] = weight[3 * quarter :]
    return nn.Parameter(padded, requires_grad=False)


def _padded_rms_norm(hidden_states, weight, eps, orig_head_dim):
    dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    variance = variance * (hidden_states.shape[-1] / orig_head_dim)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        hidden_states = hidden_states * weight.float()
    return hidden_states.to(dtype)


def _clip_bounds(module, which):
    if not module.use_clipped_linears:
        return None
    return (
        float(getattr(module, f"{which}_min").item()),
        float(getattr(module, f"{which}_max").item()),
    )


def _clamp(hidden_states, bounds):
    if bounds is None:
        return hidden_states
    return torch.clamp(hidden_states, min=bounds[0], max=bounds[1])


def _pad_mlp(layers, orig_intermediate, padded_intermediate):
    for layer in layers:
        mlp = layer.mlp
        mlp.gate_proj.linear = _pad_proj_output_simple(
            mlp.gate_proj.linear, 1, orig_intermediate, padded_intermediate
        )
        mlp.up_proj.linear = _pad_proj_output_simple(
            mlp.up_proj.linear, 1, orig_intermediate, padded_intermediate
        )
        mlp.down_proj.linear = _pad_proj_input_simple(
            mlp.down_proj.linear, 1, orig_intermediate, padded_intermediate
        )


def _make_compiled_block(layer, num_heads, orig_head_dim, padded_head_dim):
    attn = layer.self_attn
    q_input_bounds = _clip_bounds(attn.q_proj, "input")
    q_output_bounds = _clip_bounds(attn.q_proj, "output")
    k_input_bounds = _clip_bounds(attn.k_proj, "input")
    k_output_bounds = _clip_bounds(attn.k_proj, "output")
    v_input_bounds = _clip_bounds(attn.v_proj, "input")
    v_output_bounds = _clip_bounds(attn.v_proj, "output")
    o_input_bounds = _clip_bounds(attn.o_proj, "input")
    o_output_bounds = _clip_bounds(attn.o_proj, "output")

    attn.q_proj.linear = _pad_qk_linear(
        attn.q_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.k_proj.linear = _pad_qk_linear(
        attn.k_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.v_proj.linear = _pad_proj_output_simple(
        attn.v_proj.linear, num_heads, orig_head_dim, padded_head_dim
    )
    attn.o_proj.linear = _pad_proj_input_simple(
        attn.o_proj.linear, num_heads, orig_head_dim, padded_head_dim
    )
    attn.q_norm.weight = _pad_norm_weight(attn.q_norm, orig_head_dim, padded_head_dim)
    attn.k_norm.weight = _pad_norm_weight(attn.k_norm, orig_head_dim, padded_head_dim)

    def block_forward(hidden_states, rope_matrices, attn_mask):
        bsz, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = _gemma4_rms_norm(
            hidden_states, layer.input_layernorm.weight, layer.input_layernorm.eps
        )

        query = attn.q_proj.linear(_clamp(hidden_states, q_input_bounds))
        query = _clamp(query, q_output_bounds).view(
            bsz, seq_len, num_heads, padded_head_dim
        )
        query = _padded_rms_norm(
            query, attn.q_norm.weight, attn.q_norm.eps, orig_head_dim
        ).transpose(1, 2)

        key = attn.k_proj.linear(_clamp(hidden_states, k_input_bounds))
        key = _clamp(key, k_output_bounds).view(
            bsz, seq_len, num_heads, padded_head_dim
        )
        key = _padded_rms_norm(
            key, attn.k_norm.weight, attn.k_norm.eps, orig_head_dim
        ).transpose(1, 2)

        value = attn.v_proj.linear(_clamp(hidden_states, v_input_bounds))
        value = _clamp(value, v_output_bounds).view(
            bsz, seq_len, num_heads, padded_head_dim
        )
        value = _padded_rms_norm(value, None, attn.v_norm.eps, orig_head_dim).transpose(
            1, 2
        )

        query = apply_rope_matmul(query, rope_matrices).contiguous()
        key = apply_rope_matmul(key, rope_matrices).contiguous()
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = attn.o_proj.linear(_clamp(attn_output, o_input_bounds))
        attn_output = _clamp(attn_output, o_output_bounds)
        hidden_states = residual + _gemma4_rms_norm(
            attn_output,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
        )

        residual = hidden_states
        hidden_states = _gemma4_rms_norm(
            hidden_states,
            layer.pre_feedforward_layernorm.weight,
            layer.pre_feedforward_layernorm.eps,
        )
        hidden_states = layer.mlp(hidden_states)
        hidden_states = _gemma4_rms_norm(
            hidden_states,
            layer.post_feedforward_layernorm.weight,
            layer.post_feedforward_layernorm.eps,
        )
        return residual + hidden_states

    return torch.compile(block_forward, dynamic=False)


def _build_rope_matrices(inv_freq, position_ids, padded_head_dim, dtype):
    positions = position_ids.to("cpu").clamp(min=0).float()
    angles = positions[..., None] * inv_freq.float()
    cos = angles.cos()
    sin = angles.sin()
    bsz, seq_len, _, quarter = cos.shape
    padded_half = padded_head_dim // 2
    matrices = torch.zeros(bsz, seq_len, 2, 2, padded_half)
    matrices[:, :, 0, 0, :] = 1.0
    matrices[:, :, 1, 1, :] = 1.0
    for axis in range(2):
        start = axis * quarter
        end = start + quarter
        matrices[:, :, 0, 0, start:end] = cos[:, :, axis]
        matrices[:, :, 0, 1, start:end] = -sin[:, :, axis]
        matrices[:, :, 1, 0, start:end] = sin[:, :, axis]
        matrices[:, :, 1, 1, start:end] = cos[:, :, axis]
    return matrices.to(dtype)


def _build_attention_mask(valid, padded_len, dtype):
    bsz, seq_len = valid.shape
    key_mask = F.pad(valid, (0, padded_len - seq_len), value=False)
    mask = torch.zeros((bsz, 1, padded_len, padded_len), dtype=dtype)
    return mask.masked_fill(~key_mask[:, None, None, :], -torch.inf)


def prepare_for_spyre(model):
    tower = model.model.vision_tower
    config = tower.config
    layers = tower.encoder.layers
    if config.num_key_value_heads != config.num_attention_heads:
        raise SpyreUnsupportedFeatureError(
            "Gemma 4 vision GQA is not supported on Spyre; num_key_value_heads "
            "must equal num_attention_heads."
        )
    if config.rope_parameters.get("rope_type", "default") != "default":
        raise SpyreUnsupportedFeatureError(
            "Gemma 4 vision supports only default, unscaled RoPE on Spyre."
        )
    orig_head_dim = config.head_dim
    padded_head_dim = math.ceil(orig_head_dim / (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)

    orig_intermediate = config.intermediate_size
    padded_intermediate = math.ceil(orig_intermediate / BLOCK_SIZE) * BLOCK_SIZE
    if padded_intermediate > orig_intermediate:
        _pad_mlp(layers, orig_intermediate, padded_intermediate)

    model._spyre_gemma4_vision_inv_freq = (
        tower.encoder.rotary_emb.inv_freq.detach().cpu()
    )
    if tower.config.standardize:
        model._spyre_gemma4_vision_std_bias = tower.std_bias.detach().cpu()
        model._spyre_gemma4_vision_std_scale = tower.std_scale.detach().cpu()
    model._spyre_gemma4_vision_head_dim = padded_head_dim
    model._spyre_gemma4_vision_blocks = [
        _make_compiled_block(
            layer, config.num_attention_heads, orig_head_dim, padded_head_dim
        )
        for layer in layers
    ]


def prefill_vision_tower(model, pixel_values, position_ids):
    tower = model.model.vision_tower
    pixel_values = pixel_values.to("cpu")
    position_ids = position_ids.to("cpu")
    padding_positions = (position_ids == -1).all(dim=-1)
    hidden_states = tower.patch_embedder(pixel_values, position_ids, padding_positions)
    output_dtype = hidden_states.dtype

    seq_len = hidden_states.shape[1]
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    rope_matrices = _build_rope_matrices(
        model._spyre_gemma4_vision_inv_freq,
        position_ids,
        model._spyre_gemma4_vision_head_dim,
        hidden_states.dtype,
    )
    if padded_len > seq_len:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_len - seq_len))
        identity = torch.zeros(
            hidden_states.shape[0],
            padded_len - seq_len,
            2,
            2,
            model._spyre_gemma4_vision_head_dim // 2,
            dtype=hidden_states.dtype,
        )
        identity[:, :, 0, 0, :] = 1.0
        identity[:, :, 1, 1, :] = 1.0
        rope_matrices = torch.cat([rope_matrices, identity], dim=1)

    attn_mask = _build_attention_mask(
        ~padding_positions, padded_len, hidden_states.dtype
    )
    hidden_states = hidden_states.to(DEVICE)
    rope_matrices = rope_matrices.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)
    for block in model._spyre_gemma4_vision_blocks:
        hidden_states = block(hidden_states, rope_matrices, attn_mask).clone()

    hidden_states = hidden_states[:, :seq_len].to("cpu")
    output_length = pixel_values.shape[-2] // (tower.config.pooling_kernel_size**2)
    hidden_states, pooler_mask = tower.pooler(
        hidden_states,
        position_ids,
        padding_positions,
        output_length,
    )
    hidden_states = hidden_states[pooler_mask]
    if tower.config.standardize:
        hidden_states = (
            hidden_states - model._spyre_gemma4_vision_std_bias.float()
        ) * model._spyre_gemma4_vision_std_scale.float()
    return hidden_states.to(output_dtype)
