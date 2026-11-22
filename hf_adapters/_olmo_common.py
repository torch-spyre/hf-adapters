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

"""Shared OLMo-family Spyre code.

- ``qk_norm_attention``: attention with Q/K RMSNorm over the flattened
  projections (OLMo 2, OLMo 3, OLMoE).
- ``require_unpadded_head_dim``: the stick-alignment guard that flattened Q/K
  RMSNorm needs, since head padding would change the norm's variance.
- ``prepare_olmo_common``: post-norm compiled blocks and Spyre setup for
  OLMo 2 and OLMo 3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    SpyreUnsupportedModelError,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    prepare_lm_head_for_spyre,
    prepare_rope_and_heads,
    text_config,
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


class QKNormAttention(Protocol):
    """Attention modules whose Q/K RMSNorm spans the flattened projections."""

    @property
    def q_proj(self) -> nn.Module: ...

    @property
    def k_proj(self) -> nn.Module: ...

    @property
    def v_proj(self) -> nn.Module: ...

    @property
    def o_proj(self) -> nn.Module: ...

    @property
    def q_norm(self) -> nn.Module: ...

    @property
    def k_norm(self) -> nn.Module: ...

    @property
    def head_dim(self) -> int: ...

    @property
    def scaling(self) -> float: ...


def qk_norm_attention(
    attn: QKNormAttention,
    hidden_states: torch.Tensor,
    selected_freqs: torch.Tensor,
    attn_mask: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K-normed RoPE attention with a KV-cache update, then ``o_proj``.

    Q/K RMSNorm runs on the flattened projections before the per-head reshape.
    Returns ``(attn_out, key_cache, value_cache)``; norms and residuals around
    attention are left to the caller (post-norm for OLMo 2/3, pre-norm for
    OLMoE).
    """
    bsz, seq_len, _ = hidden_states.shape

    q: torch.Tensor = attn.q_norm(attn.q_proj(hidden_states))
    k: torch.Tensor = attn.k_norm(attn.k_proj(hidden_states))
    v: torch.Tensor = attn.v_proj(hidden_states)

    q = q.view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

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
    return attn.o_proj(attn_out), key_cache, value_cache


def require_unpadded_head_dim(model: PreTrainedModel, family: str) -> int:
    """Return ``head_dim``, rejecting configs that would need head padding.

    ``prepare_rope_and_heads`` would zero-pad Q/K to a stick multiple, but
    OLMo's Q/K RMSNorm runs over the flattened projection, so padding changes
    its variance and silently corrupts attention.
    """
    cfg = text_config(model.config)
    head_dim: int = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    if head_dim % (2 * BLOCK_SIZE) != 0:
        raise SpyreUnsupportedModelError(
            f"{family} head_dim={head_dim} is not a multiple of {2 * BLOCK_SIZE}; "
            "head padding is not supported because Q/K RMSNorm spans the "
            "flattened projections."
        )
    return head_dim


def _make_compiled_block(layer: Olmo2DecoderLayer | Olmo3DecoderLayer) -> CompiledBlock:
    """Compile a post-norm block around the shared Q/K-norm attention."""
    attn: Olmo2Attention | Olmo3Attention = layer.self_attn
    mlp: nn.Module = layer.mlp
    post_attn_ln: nn.Module = layer.post_attention_layernorm
    post_ff_ln: nn.Module = layer.post_feedforward_layernorm

    def block_forward(
        hidden_states: torch.Tensor,
        selected_freqs: torch.Tensor,
        attn_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual: torch.Tensor = hidden_states
        attn_out: torch.Tensor
        attn_out, key_cache, value_cache = qk_norm_attention(
            attn,
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )

        # Post-norm: norm after attention/MLP, before each residual add.
        h: torch.Tensor = residual + post_attn_ln(attn_out)
        residual = h
        h = residual + post_ff_ln(mlp(h))

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def prepare_olmo_common(model: PreTrainedModel, family: str) -> None:
    """Prepare RoPE, the LM head, decoder blocks and final norm for OLMo 2/3.

    The head dimension must already be stick-aligned (see
    ``require_unpadded_head_dim``). OLMo 3 then replaces the single RoPE with
    one rotary embedding per layer type.
    """
    require_unpadded_head_dim(model, family)
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
