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
HuggingFace Transformers adapter for Granite 4.1 SWA models on Spyre.

GraniteSWAForCausalLM is identical to Granite 3.x except that layers alternate
between full attention and sliding-window attention (``layer_type`` attribute on
each ``GraniteSWADecoderLayer``). Sliding-window layers receive a local attention
mask that restricts each token to attend only within the ``sliding_window`` window;
full-attention layers receive the normal causal mask.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/tmp/models/granite-4.1-20b")
    tokenizer = AutoTokenizer.from_pretrained("/tmp/models/granite-4.1-20b")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
)


def _make_compiled_block(layer, sliding_window: int):
    """Compiled block for GraniteSWA.

    Full-attention layers receive the normal causal ``attn_mask``.
    Sliding-window layers receive a local mask built from the same buffer but
    restricted to a ``sliding_window``-wide band; positions outside the window
    are masked to ``-inf``.
    """
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    res_mult = layer.residual_multiplier
    v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)
    is_sliding = getattr(layer, "layer_type", "full_attention") == "sliding_attention"

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        # For sliding-window layers, apply a local band mask on top of the
        # causal mask: positions further than sliding_window steps back are
        # set to -inf so the model never attends outside the window.
        if is_sliding:
            cache_len = key_cache.shape[2]
            q_len = q.shape[2]
            # Build position indices for queries and keys
            q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)  # [q, 1]
            k_pos = torch.arange(cache_len, device=q.device).unsqueeze(0)  # [1, k]
            window_mask = (q_pos - k_pos) >= sliding_window  # [q, k]
            swa_mask = attn_mask.clone()
            swa_mask = swa_mask.masked_fill(
                window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            effective_mask = swa_mask
        else:
            effective_mask = attn_mask

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=effective_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out * res_mult

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h * res_mult

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """GraniteSWA causal-LM forward: embedding * multiplier, blocks, norm, head."""
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = backbone.norm(h)
    logits = model.lm_head(h)
    logits = logits / model.config.logits_scaling
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a GraniteSWA model in-place."""
    from transformers.models.granite_swa.modeling_granite_swa import GraniteSWARMSNorm

    sliding_window = model.config.sliding_window
    prepare_rope_and_heads(model)
    patch_rmsnorm(GraniteSWARMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, sliding_window)
        for layer in get_backbone(model).layers
    ]
