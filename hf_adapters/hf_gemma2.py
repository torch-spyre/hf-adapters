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

"""HuggingFace Transformers adapter for Gemma 2 causal-LM models on Spyre.

Gemma 2 alternates sliding-window and full-attention layers and uses four
"sandwich" RMSNorms per layer. Unlike Gemma 3, it softcaps both attention
logits and final logits. The attention softcap requires an explicit attention
implementation rather than ``scaled_dot_product_attention``.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
)
from hf_adapters.hf_gemma3 import _patch_gemma_rmsnorm


def _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim):
    attn = layer.self_attn
    num_key_value_groups = num_q_heads // num_kv_heads
    scaling = attn.scaling
    softcap = attn.attn_logit_softcapping

    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    mlp = layer.mlp

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)
        batch_size, seq_len, _ = h.shape

        q = attn.q_proj(h).view(batch_size, seq_len, num_q_heads, head_dim)
        k = attn.k_proj(h).view(batch_size, seq_len, num_kv_heads, head_dim)
        v = attn.v_proj(h).view(batch_size, seq_len, num_kv_heads, head_dim)
        q = apply_rope_matmul(q.transpose(1, 2), selected_freqs)
        k = apply_rope_matmul(k.transpose(1, 2), selected_freqs)
        v = v.transpose(1, 2)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )

        # Keep each KV head's query group in a separate 4-D BMM. Materializing
        # repeat_kv produces an unsupported Spyre layout, while a single grouped
        # 5-D BMM exceeds the backend's rank limit. Concatenating these groups
        # preserves Transformers' repeat_kv head ordering.
        attn_groups = []
        for kv_idx in range(num_kv_heads):
            q_start = kv_idx * num_key_value_groups
            q_end = q_start + num_key_value_groups
            q_group = q[:, q_start:q_end].contiguous()
            k_group = key_cache[:, kv_idx : kv_idx + 1].contiguous()
            v_group = value_cache[:, kv_idx : kv_idx + 1].contiguous()
            attn_weights = torch.matmul(q_group, k_group.transpose(2, 3)) * scaling
            if softcap is not None:
                attn_weights = torch.tanh(attn_weights / softcap) * softcap
            attn_weights = attn_weights + attn_mask
            if attn_weights.device.type == "spyre":
                attn_weights = F.softmax(attn_weights, dim=-1)
            else:
                # Match stock Gemma 2's fp32 attention softmax on CPU.
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    q.dtype
                )
            attn_groups.append(torch.matmul(attn_weights, v_group))
        attn_out = torch.cat(attn_groups, dim=1)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + post_attn_ln(attn_out)
        residual = h
        h = pre_ff_ln(h)
        h = mlp(h)
        h = residual + post_ff_ln(h)
        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


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
    cfg = model.config
    h = backbone.embed_tokens(input_ids)
    selected_freqs = model._spyre_rope(h, position_ids)

    batch_size, seq_len = input_ids.shape
    block_base = int(cache_index[0])
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(
        batch_size, seq_len
    )
    sliding_mask = add_causal_sliding_window_band(
        attn_mask, query_coords, cfg.sliding_window
    )

    # Explicit softmax cannot consume an all--inf row. Block-padded prefill has
    # such rows for left-padding queries, so make each one attend only to its own
    # cache slot. Real queries still mask every padding key, and padding outputs
    # are never read; this only keeps their intermediate K/V values finite.
    def sanitize_fully_masked_rows(mask):
        mask_cpu = mask.to("cpu")
        fully_masked = torch.isneginf(mask_cpu).all(dim=-1).squeeze(1)
        if fully_masked.any():
            mask_cpu = mask_cpu.clone()
            for batch_idx, query_idx in fully_masked.nonzero().tolist():
                cache_idx = int(query_coords[batch_idx, query_idx])
                mask_cpu[batch_idx, 0, query_idx, cache_idx] = 0
        return mask_cpu.to(mask.device)

    masks = {
        "full_attention": sanitize_fully_masked_rows(attn_mask),
        "sliding_attention": sanitize_fully_masked_rows(sliding_mask),
    }

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            masks[cfg.layer_types[i]],
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    return backbone.norm(h)


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
    logits = model.lm_head(h)[..., : model._spyre_original_vocab_size]
    cap = model.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def prepare_for_spyre(model):
    """Apply Gemma 2 adaptations to ``model`` in-place."""
    backbone = get_backbone(model)
    cfg = model.config
    assert (
        getattr(cfg, "is_causal", True) is True
    ), "Gemma 2 adapter only supports causal attention"

    head_dim = cfg.head_dim
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads

    assert head_dim % 2 == 0 and head_dim // 2 >= 64, (
        f"Gemma 2 head_dim={head_dim}: head_dim/2 must be >= 64 (one Spyre "
        "stick). A padded variant is not implemented for this adapter."
    )

    _patch_gemma_rmsnorm(type(backbone.layers[0].input_layernorm))
    model._spyre_rope = PrecomputedRotaryEmbedding(backbone.rotary_emb)
    model._spyre_kv_shapes = [
        (num_kv_heads, head_dim, head_dim) for _ in backbone.layers
    ]
    model._spyre_original_vocab_size = cfg.vocab_size
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim)
        for layer in backbone.layers
    ]
