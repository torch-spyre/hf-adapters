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
HuggingFace Transformers adapter for OLMo2 models on Spyre.

Covers model_type ``olmo2``: OLMo 2 7B.
Post-norm architecture (norm after attention/MLP, before residual add)
with Q/K RMSNorm applied to flattened projections before reshape.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    prepare_rope_and_heads,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)


def _make_compiled_block(layer):
    """Compiled block for OLMo2: post-norm + Q/K RMSNorm."""
    attn = layer.self_attn
    mlp = layer.mlp
    # OLMo2 is post-norm: norms applied after attention/MLP, before residual add
    post_attn_ln = layer.post_attention_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)

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
        bsz, seq_len, _ = hidden_states.shape

        # Q/K norm on flattened projections, then reshape
        q = q_norm(attn.q_proj(hidden_states))
        k = k_norm(attn.k_proj(hidden_states))
        v = attn.v_proj(hidden_states)

        q = q.view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

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

        # Post-norm: norm after attention, before residual add
        h = residual + post_attn_ln(attn_out)

        residual = h
        h = residual + post_ff_ln(mlp(h))

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def prepare_for_spyre(model):
    """Apply Spyre adaptations to OLMo2 model in-place."""
    prepare_rope_and_heads(model)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in get_backbone(model).layers
    ]
    model._spyre_compiled_norm = torch.compile(get_backbone(model).norm, dynamic=False)
