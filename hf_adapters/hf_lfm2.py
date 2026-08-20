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

"""HuggingFace Transformers adapter for dense LFM2 models on Spyre.

LFM2 is dense (non-MoE), but hybrid: decoder layers alternate between full
GQA and gated short depthwise convolutions. The two layer types use different
cache state, so this adapter supplies a heterogeneous cache allocator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    PrecomputedRotaryEmbedding,
    allocate_kv_cache_tensor,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_attention_heads,
    pad_lm_head,
    patch_rmsnorm,
)


class _PaddedHeadRMSNorm(nn.Module):
    """RMSNorm a zero-padded head using the original active dimension."""

    def __init__(self, norm, original_dim, padded_dim):
        super().__init__()
        half = original_dim // 2
        padded_half = padded_dim // 2
        weight = torch.zeros(padded_dim, dtype=norm.weight.dtype)
        weight[:half] = norm.weight[:half]
        weight[padded_half : padded_half + half] = norm.weight[half:]
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.original_dim = original_dim
        self.variance_epsilon = norm.variance_epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        if hidden_states.device.type == "spyre":
            variance = (hidden_states * hidden_states).sum(-1, keepdim=True)
            variance = variance / self.original_dim
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
        else:
            h = hidden_states.float()
            variance = (h * h).sum(-1, keepdim=True) / self.original_dim
            hidden_states = h * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states


def _causal_depthwise_conv(
    hidden_states,
    state,
    weights,
    shift_matrices,
    decode_matrices,
    bias=None,
    decode=None,
):
    """Apply LFM2's three-tap causal convolution using stick-aligned state."""
    seq_len = hidden_states.shape[-1]
    input_seq_len = seq_len
    state_len = state.shape[-1]
    if decode is None:
        decode = seq_len == 1

    if seq_len > state_len:
        assert seq_len % state_len == 0
        outputs = []
        new_state = state
        for start in range(0, seq_len, state_len):
            out, new_state = _causal_depthwise_conv(
                hidden_states[..., start : start + state_len],
                new_state,
                weights,
                shift_matrices,
                decode_matrices,
                bias,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=-1), new_state

    if decode:
        previous_2 = state @ decode_matrices[0]
        previous_1 = state @ decode_matrices[1]
        new_state = state @ decode_matrices[2] + hidden_states @ decode_matrices[3]
    else:
        if seq_len < state_len:
            hidden_states = F.pad(hidden_states, (state_len - seq_len, 0))
        positions = torch.arange(state_len)
        from_state_1 = (positions == 0)[None, None, :].to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        from_state_2 = (positions < 2)[None, None, :].to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        previous_1 = from_state_1 * (state @ shift_matrices[1]) + (1 - from_state_1) * (
            hidden_states @ shift_matrices[1]
        )
        previous_2 = from_state_2 * (state @ shift_matrices[2]) + (1 - from_state_2) * (
            hidden_states @ shift_matrices[2]
        )
        new_state = hidden_states

    out = weights[0] * previous_2 + weights[1] * previous_1 + weights[2] * hidden_states
    if bias is not None:
        out = out + bias[None, :, None]
    if not decode and input_seq_len < state_len:
        out = out[..., -input_seq_len:]
    return out.to(hidden_states.dtype), new_state


def _make_attention_block(layer):
    attn = layer.self_attn
    operator_norm = layer.operator_norm
    ffn_norm = layer.ffn_norm
    feed_forward = layer.feed_forward

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        h = operator_norm(hidden_states)
        bsz, seq_len, _ = h.shape

        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim)
        q = attn.q_layernorm(q).transpose(1, 2)
        k = attn.k_layernorm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)
        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )

        h = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        h = h.transpose(1, 2).reshape(bsz, seq_len, -1)
        h = residual + attn.out_proj(h)

        residual = h
        h = ffn_norm(h)
        h = feed_forward.w2(F.silu(feed_forward.w1(h)) * feed_forward.w3(h))
        return residual + h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _make_conv_block(layer):
    conv = layer.conv
    operator_norm = layer.operator_norm
    conv._spyre_weights = nn.ParameterList(
        [
            nn.Parameter(
                conv.conv.weight[:, 0, i][None, :, None].expand(1, -1, BLOCK_SIZE),
                requires_grad=False,
            )
            for i in range(3)
        ]
    )
    identity = torch.eye(BLOCK_SIZE, dtype=conv.conv.weight.dtype)
    conv._spyre_shift_matrices = nn.ParameterList(
        [
            nn.Parameter(torch.roll(identity, shift, dims=1), requires_grad=False)
            for shift in range(3)
        ]
    )
    select_previous_2 = torch.zeros_like(identity)
    select_previous_2[-2, 0] = 1
    select_previous_1 = torch.zeros_like(identity)
    select_previous_1[-1, 0] = 1
    shift_state = torch.roll(identity, -1, dims=1)
    shift_state[0, -1] = 0
    append_token = torch.zeros_like(identity)
    append_token[0, -1] = 1
    conv._spyre_decode_matrices = nn.ParameterList(
        [
            nn.Parameter(select_previous_2, requires_grad=False),
            nn.Parameter(select_previous_1, requires_grad=False),
            nn.Parameter(shift_state, requires_grad=False),
            nn.Parameter(append_token, requires_grad=False),
        ]
    )
    ffn_norm = layer.ffn_norm
    feed_forward = layer.feed_forward

    def pre_conv_forward(hidden_states, padding_mask):
        h = operator_norm(hidden_states)
        h = h * padding_mask[:, :, None]
        B, C, x = conv.in_proj(h).transpose(1, 2).chunk(3, dim=1)
        return (B * x).contiguous(), C.contiguous()

    def conv_forward(conv_input, conv_state, decode):
        conv_out, new_state = _causal_depthwise_conv(
            conv_input,
            conv_state,
            conv._spyre_weights,
            conv._spyre_shift_matrices,
            conv._spyre_decode_matrices,
            conv.conv.bias,
            decode,
        )
        return conv_out, new_state

    def post_conv_forward(hidden_states, c, conv_out):
        h = conv.out_proj((c * conv_out).transpose(1, 2).contiguous())
        h = hidden_states + h
        residual = h
        h = ffn_norm(h)
        h = feed_forward.w2(F.silu(feed_forward.w1(h)) * feed_forward.w3(h))
        return residual + h

    return (
        torch.compile(pre_conv_forward, dynamic=False),
        torch.compile(conv_forward, dynamic=False),
        torch.compile(post_conv_forward, dynamic=False),
    )


def _padding_mask(attn_mask, seq_len, cache_index):
    """Recover per-query input validity from the additive causal mask."""
    block_start = int(cache_index[0])
    rows = torch.arange(seq_len)
    diagonal = attn_mask.to("cpu")[:, 0, rows, block_start + rows]
    padding_mask = (diagonal == 0).to(dtype=attn_mask.dtype)
    if seq_len == 1:
        padding_mask = F.pad(padding_mask, (0, BLOCK_SIZE - 1))
    return padding_mask.to(device=DEVICE)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    selected_freqs = model._spyre_rope(h, position_ids)
    padding_mask = _padding_mask(attn_mask, h.shape[1], cache_index)

    for i, (layer_type, compiled_block) in enumerate(
        zip(model.config.layer_types, model._spyre_compiled_blocks)
    ):
        if layer_type == "full_attention":
            h, key_caches[i], value_caches[i] = compiled_block(
                h,
                selected_freqs,
                attn_mask,
                key_caches[i],
                value_caches[i],
                cache_index,
            )
        else:
            pre_conv, conv_forward, post_conv = compiled_block
            decode = h.shape[1] == 1
            if decode:
                h_for_conv = F.pad(h, (0, 0, 0, BLOCK_SIZE - 1))
                padding_mask_for_conv = padding_mask
            else:
                h_for_conv = h
                padding_mask_for_conv = padding_mask
            conv_input, C = pre_conv(h_for_conv, padding_mask_for_conv)
            conv_out, key_caches[i] = conv_forward(conv_input, key_caches[i], decode)
            h = post_conv(h_for_conv, C, conv_out)
            if decode:
                h = h[:, :1]

    return backbone.embedding_norm(h)


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    return model.lm_head(h)


def _allocate_caches(model, batch_size, max_cache_len, dtype, device):
    cfg = model.config
    head_dim = model._spyre_head_dim
    key_caches = []
    value_caches = []
    for layer_type in cfg.layer_types:
        if layer_type == "full_attention":
            key_caches.append(
                allocate_kv_cache_tensor(
                    batch_size,
                    cfg.num_key_value_heads,
                    max_cache_len,
                    head_dim,
                    dtype,
                    device,
                )
            )
            value_caches.append(
                allocate_kv_cache_tensor(
                    batch_size,
                    cfg.num_key_value_heads,
                    max_cache_len,
                    head_dim,
                    dtype,
                    device,
                )
            )
        else:
            key_caches.append(
                torch.zeros(
                    batch_size,
                    cfg.hidden_size,
                    BLOCK_SIZE,
                    dtype=dtype,
                    device=device,
                )
            )
            value_caches.append(torch.zeros(1, dtype=dtype, device=device))
    return key_caches, value_caches


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a dense LFM2 causal LM in-place."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm

    cfg = model.config
    unsupported = set(cfg.layer_types) - {"full_attention", "conv"}
    assert not unsupported, f"Unsupported LFM2 layer types: {sorted(unsupported)}"

    backbone = get_backbone(model)
    original_head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    attention_layers = [layer for layer in backbone.layers if layer.is_attention_layer]
    padded_head_dim = ((original_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (
        2 * BLOCK_SIZE
    )
    if padded_head_dim != original_head_dim:
        for layer in attention_layers:
            layer.self_attn.o_proj = layer.self_attn.out_proj
        pad_attention_heads(
            model,
            attention_layers,
            original_head_dim,
            padded_head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )
        for layer in attention_layers:
            layer.self_attn.out_proj = layer.self_attn.o_proj
            del layer.self_attn.o_proj
    model._spyre_rope = PrecomputedRotaryEmbedding(
        backbone.rotary_emb,
        padded_head_dim=(
            padded_head_dim if padded_head_dim != original_head_dim else None
        ),
    )
    if padded_head_dim != original_head_dim:
        for layer in backbone.layers:
            if not layer.is_attention_layer:
                continue
            attn = layer.self_attn
            attn.q_layernorm = _PaddedHeadRMSNorm(
                attn.q_layernorm, original_head_dim, padded_head_dim
            )
            attn.k_layernorm = _PaddedHeadRMSNorm(
                attn.k_layernorm, original_head_dim, padded_head_dim
            )

    patch_rmsnorm(Lfm2RMSNorm)
    pad_lm_head(model)
    model._spyre_cache_allocator = _allocate_caches
    model._spyre_compiled_blocks = [
        (
            _make_attention_block(layer)
            if layer.is_attention_layer
            else _make_conv_block(layer)
        )
        for layer in backbone.layers
    ]
