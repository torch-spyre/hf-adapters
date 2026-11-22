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

"""Shared post-norm decoder blocks and Spyre setup for OLMo 2 and OLMo 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer

from hf_adapters.hf_common import (
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    prepare_lm_head_for_spyre,
    prepare_rope_and_heads,
)

if TYPE_CHECKING:
    from transformers.models.olmo2.modeling_olmo2 import (
        Olmo2Attention,
        Olmo2Model,
    )
    from transformers.models.olmo3.modeling_olmo3 import (
        Olmo3Attention,
        Olmo3Model,
    )

CompiledBlock = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _make_compiled_block(layer: Olmo2DecoderLayer | Olmo3DecoderLayer) -> CompiledBlock:
    """Compile a post-norm block with Q/K RMSNorm on flattened projections."""
    attn: Olmo2Attention | Olmo3Attention = layer.self_attn
    mlp: nn.Module = layer.mlp
    post_attn_ln: nn.Module = layer.post_attention_layernorm
    post_ff_ln: nn.Module = layer.post_feedforward_layernorm
    q_norm: nn.Module = attn.q_norm
    k_norm: nn.Module = attn.k_norm
    v_head_dim: int = getattr(attn, "v_head_dim", attn.head_dim)

    def block_forward(
        hidden_states: torch.Tensor,
        selected_freqs: torch.Tensor,
        attn_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        bsz, seq_len, _ = hidden_states.shape

        # Q/K norm on flattened projections, then reshape.
        q: torch.Tensor = q_norm(attn.q_proj(hidden_states))
        k: torch.Tensor = k_norm(attn.k_proj(hidden_states))
        v: torch.Tensor = attn.v_proj(hidden_states)

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
            cache_index,
        )

        attn_out: torch.Tensor = F.scaled_dot_product_attention(
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

        # Post-norm: norm after attention/MLP, before each residual add.
        h: torch.Tensor = residual + post_attn_ln(attn_out)
        residual = h
        h = residual + post_ff_ln(mlp(h))

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def prepare_olmo_common(model: PreTrainedModel) -> None:
    """Prepare RoPE, the LM head, decoder blocks and final norm for OLMo 2/3.

    OLMo 3 validates its unpadded head dimension before calling this helper,
    then replaces the single RoPE with one rotary embedding per layer type.
    """
    prepare_rope_and_heads(model)
    prepare_lm_head_for_spyre(model)
    backbone: Olmo2Model | Olmo3Model = get_backbone(model)
    blocks: list[CompiledBlock] = []
    for layer in backbone.layers:
        if not isinstance(layer, (Olmo2DecoderLayer, Olmo3DecoderLayer)):
            raise TypeError("expected an OLMo 2 or OLMo 3 decoder layer")
        blocks.append(_make_compiled_block(layer))
    setattr(model, "_spyre_compiled_blocks", blocks)
    norm: Callable[[torch.Tensor], torch.Tensor] = torch.compile(
        backbone.norm, dynamic=False
    )
    setattr(model, "_spyre_compiled_norm", norm)
