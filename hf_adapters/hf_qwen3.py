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

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    DenseAttention,
    PrecomputedRotaryEmbedding,
    compile_block,
    get_backbone,
    prepare_lm_head_for_spyre,
    project_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)


def _make_compiled_block(layer, runtime=None):
    """Compiled block for Qwen3: Q/K norm before RoPE, no multipliers."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    attention = (
        DenseAttention(attn.scaling) if runtime is None else runtime.attention(attn)
    )

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
        q, k, v = project_standard_gqa(attn, h, selected_freqs)

        attn_out, key_cache, value_cache = attention(
            q, k, v, attn_mask, key_cache, value_cache, cache_index
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h

        return h, key_cache, value_cache

    return compile_block(block_forward, runtime)


_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def prepare_for_spyre(model, *, runtime=None):
    """Apply Spyre adaptations to Qwen3 model in-place."""
    model._spyre_rope = PrecomputedRotaryEmbedding(get_backbone(model).rotary_emb)
    if runtime is None:
        prepare_lm_head_for_spyre(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, runtime) for layer in get_backbone(model).layers
    ]
    model._spyre_compiled_norm = compile_block(get_backbone(model).norm, runtime)
