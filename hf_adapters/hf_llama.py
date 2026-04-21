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
HuggingFace Transformers adapter for Llama models on Spyre.

Covers model_type ``llama``: Llama 1/2/3, Code Llama, Yi, and other
models that register as ``llama`` in HF Transformers.

Usage::

    from hf_adapters.hf_llama import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    generate as _generate,
    kv_cache_update,
    load_model_common,
    pad_attention_heads,
    pad_lm_head,
    patch_rmsnorm,
)


def _make_compiled_block(layer):
    """Compiled block for Llama: separate QKV, standard residual add."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm

    def block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache,
            is_filling, token_index, cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q, key_cache, value_cache,
            attn_mask=attn_mask, dropout_p=0.0, scale=attn.scaling, enable_gqa=True,
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


def _run_forward(model, input_ids, position_ids, attn_mask,
                 key_caches, value_caches,
                 is_filling, token_index, cache_position):
    """Llama forward: embedding, blocks, norm, head."""
    h = model.model.embed_tokens(input_ids)

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h, selected_freqs, attn_mask,
            key_caches[i], value_caches[i],
            is_filling, token_index, cache_position,
        )

    h = model.model.norm(h)
    logits = model.lm_head(h)
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Llama model in-place."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    cfg = model.config
    orig_head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads,
    )

    padded_head_dim = None
    stick_aligned_head_dim = (
        ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    )
    if stick_aligned_head_dim > orig_head_dim:
        padded_head_dim = stick_aligned_head_dim
        pad_attention_heads(
            model, model.model.layers, orig_head_dim, padded_head_dim,
            cfg.num_attention_heads, cfg.num_key_value_heads,
        )

    model._spyre_rope = PrecomputedRotaryEmbedding(
        model.model.rotary_emb, padded_head_dim=padded_head_dim,
    )
    patch_rmsnorm(LlamaRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in model.model.layers
    ]


def load_model(model_path, dtype=torch.float16):
    """Load Llama model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Llama on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
