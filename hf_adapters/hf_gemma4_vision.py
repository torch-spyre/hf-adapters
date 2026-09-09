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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
)
from hf_adapters.hf_gemma4 import _gemma4_mlp_activation, _gemma4_rms_norm


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
    padded = nn.Linear(
        linear.in_features,
        num_heads * padded_head_dim,
        bias=linear.bias is not None,
    )
    padded.weight = nn.Parameter(
        new_weight.reshape(num_heads * padded_head_dim, -1), requires_grad=False
    )
    if linear.bias is not None:
        bias = linear.bias.detach().view(num_heads, orig_head_dim)
        new_bias = torch.zeros(num_heads, padded_head_dim, dtype=bias.dtype)
        new_bias[:, :quarter] = bias[:, :quarter]
        new_bias[:, quarter : 2 * quarter] = bias[:, 2 * quarter : 3 * quarter]
        new_bias[:, padded_half : padded_half + quarter] = bias[
            :, quarter : 2 * quarter
        ]
        new_bias[:, padded_half + quarter : padded_half + 2 * quarter] = bias[
            :, 3 * quarter :
        ]
        padded.bias = nn.Parameter(new_bias.reshape(-1), requires_grad=False)
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


def _clamp(hidden_states, bounds):
    if bounds is None:
        return hidden_states
    return torch.maximum(torch.minimum(hidden_states, bounds[1]), bounds[0])


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


@dataclass(frozen=True)
class _Gemma4VisionBlockSpec:
    hidden_size: int
    num_heads: int
    orig_head_dim: int
    padded_head_dim: int
    intermediate_size: int
    activation: str
    scaling: float
    input_norm_eps: float
    q_norm_eps: float
    k_norm_eps: float
    v_norm_eps: float
    post_attention_norm_eps: float
    pre_feedforward_norm_eps: float
    post_feedforward_norm_eps: float
    projection_biases: tuple[bool, ...]
    clipped_projections: tuple[bool, ...]


def _prepare_clip_bounds(module):
    if not module.use_clipped_linears:
        return
    for which in ("input", "output"):
        for limit in ("min", "max"):
            value = getattr(module, f"{which}_{limit}")
            module.register_buffer(
                f"_spyre_{which}_{limit}",
                value.detach().clone(),
                persistent=False,
            )


def _projection_state(module):
    linear = module.linear
    state = [linear.weight]
    if linear.bias is not None:
        state.append(linear.bias)
    if module.use_clipped_linears:
        state.extend(
            [
                module._spyre_input_min,
                module._spyre_input_max,
                module._spyre_output_min,
                module._spyre_output_max,
            ]
        )
    return tuple(state)


def _vision_block_state(layer):
    attn = layer.self_attn
    state = []
    for module in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        state.extend(_projection_state(module))
    state.extend([attn.q_norm.weight, attn.k_norm.weight])
    for module in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
        state.extend(_projection_state(module))
    state.extend(
        [
            layer.input_layernorm.weight,
            layer.post_attention_layernorm.weight,
            layer.pre_feedforward_layernorm.weight,
            layer.post_feedforward_layernorm.weight,
        ]
    )
    return tuple(state)


def _vision_block_spec(layer, num_heads, orig_head_dim, padded_head_dim):
    attn = layer.self_attn
    projections = (
        attn.q_proj,
        attn.k_proj,
        attn.v_proj,
        attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    )
    activation = layer.mlp.config.hidden_activation
    if activation != "gelu_pytorch_tanh":
        raise SpyreUnsupportedModelError(
            "Gemma 4 vision checkpoints must use "
            "hidden_activation='gelu_pytorch_tanh'; "
            f"got {activation!r}"
        )
    if attn.v_norm.with_scale:
        raise SpyreUnsupportedFeatureError(
            "Scaled Gemma 4 vision V normalization is not supported on Spyre."
        )
    return _Gemma4VisionBlockSpec(
        hidden_size=layer.input_layernorm.weight.numel(),
        num_heads=num_heads,
        orig_head_dim=orig_head_dim,
        padded_head_dim=padded_head_dim,
        intermediate_size=layer.mlp.gate_proj.linear.out_features,
        activation=activation,
        scaling=float(attn.scaling),
        input_norm_eps=layer.input_layernorm.eps,
        q_norm_eps=attn.q_norm.eps,
        k_norm_eps=attn.k_norm.eps,
        v_norm_eps=attn.v_norm.eps,
        post_attention_norm_eps=layer.post_attention_layernorm.eps,
        pre_feedforward_norm_eps=layer.pre_feedforward_layernorm.eps,
        post_feedforward_norm_eps=layer.post_feedforward_layernorm.eps,
        projection_biases=tuple(
            module.linear.bias is not None for module in projections
        ),
        clipped_projections=tuple(module.use_clipped_linears for module in projections),
    )


def _take_projection(state, pos, has_bias, is_clipped):
    weight = state[pos]
    pos += 1
    if has_bias:
        bias = state[pos]
        pos += 1
    else:
        bias = None
    if is_clipped:
        input_bounds = (state[pos], state[pos + 1])
        output_bounds = (state[pos + 2], state[pos + 3])
        pos += 4
    else:
        input_bounds = output_bounds = None
    return weight, bias, input_bounds, output_bounds, pos


def _make_vision_forward(spec):
    def block_forward(state, hidden_states, rope_matrices, attn_mask):
        pos = 0
        (
            q_weight,
            q_bias,
            q_input_bounds,
            q_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[0], spec.clipped_projections[0]
        )
        (
            k_weight,
            k_bias,
            k_input_bounds,
            k_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[1], spec.clipped_projections[1]
        )
        (
            v_weight,
            v_bias,
            v_input_bounds,
            v_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[2], spec.clipped_projections[2]
        )
        (
            o_weight,
            o_bias,
            o_input_bounds,
            o_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[3], spec.clipped_projections[3]
        )
        q_norm_weight, k_norm_weight = state[pos : pos + 2]
        pos += 2
        (
            gate_weight,
            gate_bias,
            gate_input_bounds,
            gate_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[4], spec.clipped_projections[4]
        )
        (
            up_weight,
            up_bias,
            up_input_bounds,
            up_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[5], spec.clipped_projections[5]
        )
        (
            down_weight,
            down_bias,
            down_input_bounds,
            down_output_bounds,
            pos,
        ) = _take_projection(
            state, pos, spec.projection_biases[6], spec.clipped_projections[6]
        )
        input_norm_weight, post_attn_norm_weight = state[pos : pos + 2]
        pos += 2
        pre_ffn_norm_weight, post_ffn_norm_weight = state[pos : pos + 2]

        bsz, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = _gemma4_rms_norm(
            hidden_states, input_norm_weight, spec.input_norm_eps
        )

        query = F.linear(_clamp(hidden_states, q_input_bounds), q_weight, q_bias)
        query = _clamp(query, q_output_bounds).view(
            bsz, seq_len, spec.num_heads, spec.padded_head_dim
        )
        query = _padded_rms_norm(
            query, q_norm_weight, spec.q_norm_eps, spec.orig_head_dim
        ).transpose(1, 2)

        key = F.linear(_clamp(hidden_states, k_input_bounds), k_weight, k_bias)
        key = _clamp(key, k_output_bounds).view(
            bsz, seq_len, spec.num_heads, spec.padded_head_dim
        )
        key = _padded_rms_norm(
            key, k_norm_weight, spec.k_norm_eps, spec.orig_head_dim
        ).transpose(1, 2)

        value = F.linear(_clamp(hidden_states, v_input_bounds), v_weight, v_bias)
        value = _clamp(value, v_output_bounds).view(
            bsz, seq_len, spec.num_heads, spec.padded_head_dim
        )
        value = _padded_rms_norm(
            value, None, spec.v_norm_eps, spec.orig_head_dim
        ).transpose(1, 2)

        query = apply_rope_matmul(query, rope_matrices).contiguous()
        key = apply_rope_matmul(key, rope_matrices).contiguous()
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=spec.scaling,
        )
        attn_output = _clamp(attn_output, o_input_bounds)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = F.linear(attn_output, o_weight, o_bias)
        attn_output = _clamp(attn_output, o_output_bounds)
        hidden_states = residual + _gemma4_rms_norm(
            attn_output, post_attn_norm_weight, spec.post_attention_norm_eps
        )

        residual = hidden_states
        hidden_states = _gemma4_rms_norm(
            hidden_states, pre_ffn_norm_weight, spec.pre_feedforward_norm_eps
        )
        gate = F.linear(
            _clamp(hidden_states, gate_input_bounds), gate_weight, gate_bias
        )
        gate = _clamp(gate, gate_output_bounds)
        up = F.linear(_clamp(hidden_states, up_input_bounds), up_weight, up_bias)
        up = _clamp(up, up_output_bounds)
        hidden_states = _gemma4_mlp_activation(gate) * up
        hidden_states = F.linear(
            _clamp(hidden_states, down_input_bounds), down_weight, down_bias
        )
        hidden_states = _clamp(hidden_states, down_output_bounds)
        hidden_states = _gemma4_rms_norm(
            hidden_states, post_ffn_norm_weight, spec.post_feedforward_norm_eps
        )
        return residual + hidden_states

    return block_forward


def _prepare_vision_blocks(layers, num_heads, orig_head_dim, padded_head_dim):
    compiled_by_spec = {}
    state_signature_by_spec = {}
    compiled_blocks = []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
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
        attn.q_norm.weight = _pad_norm_weight(
            attn.q_norm, orig_head_dim, padded_head_dim
        )
        attn.k_norm.weight = _pad_norm_weight(
            attn.k_norm, orig_head_dim, padded_head_dim
        )
        for module in (
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            _prepare_clip_bounds(module)

        spec = _vision_block_spec(layer, num_heads, orig_head_dim, padded_head_dim)
        state = _vision_block_state(layer)
        signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in state)
        if (
            spec in state_signature_by_spec
            and state_signature_by_spec[spec] != signature
        ):
            raise ValueError(
                f"Gemma 4 vision layer {i} state does not match its compile group"
            )
        state_signature_by_spec.setdefault(spec, signature)
        if spec not in compiled_by_spec:
            compiled_by_spec[spec] = torch.compile(
                _make_vision_forward(spec), dynamic=False, fullgraph=True
            )
        compiled_blocks.append(compiled_by_spec[spec])
    return compiled_blocks


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
    mask = torch.zeros((bsz, 1, 1, padded_len), dtype=dtype)
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
    model._spyre_gemma4_vision_blocks = _prepare_vision_blocks(
        layers, config.num_attention_heads, orig_head_dim, padded_head_dim
    )


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
    for layer, block in zip(tower.encoder.layers, model._spyre_gemma4_vision_blocks):
        hidden_states = block(
            _vision_block_state(layer), hidden_states, rope_matrices, attn_mask
        ).clone()

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
