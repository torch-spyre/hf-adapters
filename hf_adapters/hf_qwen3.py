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
HuggingFace Transformers adapter for Qwen3 models on Spyre.

Qwen3 differences from Granite:
- Q/K RMSNorm applied before RoPE
- No embedding/residual/logits multipliers
- Standard head_dim**-0.5 scaling
- head_dim < 128 (e.g. the many sub-0.5B Qwen3 fine-tunes with head_dim=64) is
  zero-padded to a stick-aligned size; the per-head Q/K norms are swapped for
  ``PaddedHeadRMSNorm`` so they still normalise over the original head_dim

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from hf_adapters.hf_common import (
    PaddedHeadRMSNorm,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_attention_heads,
    prepare_lm_head_for_spyre,
    rope_padded_head_dim,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)


def _make_compiled_block(layer):
    """Compiled block for Qwen3: Q/K norm before RoPE, no multipliers."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm

    # Qwen3 has per-head RMSNorm on Q and K
    q_norm = attn.q_norm
    k_norm = attn.k_norm

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

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        # Qwen3-specific: RMSNorm on Q and K before RoPE
        q = q_norm(q)
        k = k_norm(k)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def prepare_for_spyre(model: PreTrainedModel) -> None:
    """Apply Spyre adaptations to Qwen3 model in-place."""
    cfg = model.config
    backbone = get_backbone(model)
    original_head_dim: int = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    padded_head_dim: int = rope_padded_head_dim(original_head_dim)
    needs_padding: bool = padded_head_dim != original_head_dim
    if needs_padding:
        # head_dim / 2 must fill a whole stick for the RoPE matmul, and the
        # per-head Q/K RMSNorm over a 64-wide dim fails to lower otherwise.
        pad_attention_heads(
            model,
            backbone.layers,
            original_head_dim,
            padded_head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )
        for layer in backbone.layers:
            attn = layer.self_attn
            attn.q_norm = PaddedHeadRMSNorm(
                attn.q_norm, original_head_dim, padded_head_dim
            )
            attn.k_norm = PaddedHeadRMSNorm(
                attn.k_norm, original_head_dim, padded_head_dim
            )
    setattr(
        model,
        "_spyre_rope",
        PrecomputedRotaryEmbedding(
            backbone.rotary_emb,
            padded_head_dim=padded_head_dim if needs_padding else None,
        ),
    )
    prepare_lm_head_for_spyre(model)
    setattr(
        model,
        "_spyre_compiled_blocks",
        [_make_compiled_block(layer) for layer in backbone.layers],
    )
    setattr(
        model,
        "_spyre_compiled_norm",
        torch.compile(backbone.norm, dynamic=False),
    )
