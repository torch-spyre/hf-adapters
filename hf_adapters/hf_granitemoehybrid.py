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
HuggingFace Transformers adapter for Granite 4.0 dense models on Spyre.

Granite 4.0 uses the ``granitemoehybrid`` model_type. Dense variants
(1B, Micro) have ``num_local_experts=1`` and no Mamba layers — they are
pure transformers that happen to use the MoE codebase.

Differences from Granite 3.3:
- Fused input_linear (gate+up) and output_linear MLP
- ``shared_intermediate_size`` config field
- Same multipliers (embedding, residual, attention, logits)

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("ibm-granite/granite-4.0-1b-base")
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-1b-base")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    split_fused_linear,
)


def _make_compiled_block(layer, res_mult, gate_proj, up_proj):
    """Compiled block for Granite 4.0 dense: split MLP, multipliers."""
    attn = layer.self_attn
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    down_proj = layer.shared_mlp.output_linear
    act_fn = layer.shared_mlp.activation

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
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

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

        h = residual + attn_out * res_mult

        # MLP: separate gate/up projections (split at prepare time)
        residual = h
        h = post_attn_ln(h)
        h = down_proj(act_fn(gate_proj(h)) * up_proj(h))
        h = residual + h * res_mult

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _run_backbone_forward(
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
    """Granite 4.0 backbone: embedding * multiplier, blocks, norm."""
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * model.config.embedding_multiplier

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

    h = model._spyre_compiled_norm(h)
    return h


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
    """Granite 4.0 causal-LM forward: backbone + head / scaling."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )
    logits = model.lm_head(h)
    logits = logits / model.config.logits_scaling
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 4.0 dense model in-place."""
    layer_types = set(model.config.layer_types)
    assert "mamba" not in layer_types, (
        "hf_granitemoehybrid adapter only supports pure-attention dense models "
        f"(layer_types={sorted(layer_types)}). "
        f"'{model.config._name_or_path}' is a Mamba-attention hybrid — "
        "Mamba SSM layers are not currently supported on Spyre."
    )

    model._spyre_rope = PrecomputedRotaryEmbedding(get_backbone(model).rotary_emb)
    pad_lm_head(model)

    res_mult = model.config.residual_multiplier

    # Split fused MLP weights and register as submodules so .to() moves them
    model._spyre_gate_projs = nn.ModuleList()
    model._spyre_up_projs = nn.ModuleList()
    for layer in get_backbone(model).layers:
        gate_proj, up_proj = split_fused_linear(layer.shared_mlp.input_linear.weight)
        model._spyre_gate_projs.append(gate_proj)
        model._spyre_up_projs.append(up_proj)

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, res_mult, gate, up)
        for layer, gate, up in zip(
            get_backbone(model).layers,
            model._spyre_gate_projs,
            model._spyre_up_projs,
        )
    ]
    model._spyre_compiled_norm = torch.compile(get_backbone(model).norm, dynamic=False)
