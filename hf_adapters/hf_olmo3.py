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
HuggingFace Transformers adapter for OLMo 3 models on Spyre.

Covers model_type ``olmo3``: the OLMo 3 7B family (``allenai/Olmo-3-7B-*``,
``allenai/Olmo-3-1025-7B``) and, size permitting, the 32B variants.

OLMo 3 is OLMo 2 plus interleaved sliding-window attention:

- **Block** — identical to OLMo 2: post-norm (``post_attention_layernorm`` /
  ``post_feedforward_layernorm`` before the residual add) and Q/K RMSNorm on the
  flattened projections. The shared OLMo 2/3 compiled block is reused; the
  attention mask is a block input, so the same block serves both layer types.
- **Layer types** — ``config.layer_types`` mixes ``sliding_attention`` and
  ``full_attention`` (3:1 on the 7B, ``sliding_window=4096``). Sliding layers
  get the causal mask narrowed to ``(q - sliding_window, q]`` with
  ``hf_common.add_causal_sliding_window_band``; full layers get it unchanged.
  The KV cache stays full-length for every layer, so no ring buffer is needed.
- **RoPE per layer type** — HF keeps one ``<layer_type>_inv_freq`` /
  ``<layer_type>_attention_scaling`` pair per layer type. On the 7B the sliding
  layers use default RoPE and the full layers YaRN (``factor=8``,
  ``attention_scaling≈1.208``), so one ``PrecomputedRotaryEmbedding`` is built
  per type (via ``InvFreqShim``, as in ``hf_gemma3``) and each block gets its
  own type's rotation.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from hf_adapters._olmo_common import CompiledBlock, prepare_olmo_common
from hf_adapters.hf_common import (
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    embed_text_tokens,
    get_backbone,
    run_lm_head,
    text_config,
)

_FULL_ATTENTION: str = "full_attention"
_SLIDING_ATTENTION: str = "sliding_attention"


@dataclass(frozen=True)
class _Olmo3State:
    """Per-model OLMo 3 metadata recorded by ``prepare_for_spyre``."""

    layer_types: tuple[str, ...]
    sliding_window: int | None


def _olmo3_state(model: nn.Module) -> _Olmo3State:
    state: object = getattr(model, "_spyre_olmo3", None)
    if not isinstance(state, _Olmo3State):
        raise RuntimeError("hf_olmo3.prepare_for_spyre(model) has not been called")
    return state


def _layer_masks(
    state: _Olmo3State,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    cache_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the additive mask for each layer type present in the model.

    ``attn_mask`` is the caller's causal + padding mask in KV-cache coordinates.
    Sliding layers additionally mask every key at least ``sliding_window``
    cache columns behind the query (HF's ``create_sliding_window_causal_mask``).
    """
    masks: dict[str, torch.Tensor] = {_FULL_ATTENTION: attn_mask}
    sliding_window: int | None = state.sliding_window
    if _SLIDING_ATTENTION not in state.layer_types or sliding_window is None:
        return masks

    cache_len: int = attn_mask.shape[-1]
    if cache_len <= sliding_window:
        # No query/key pair in a cache this short can be ``sliding_window``
        # columns apart, so the band would mask nothing. Skip the mask build
        # (and its host round-trip) rather than recomputing an identical mask.
        masks[_SLIDING_ATTENTION] = attn_mask
        return masks

    batch_size, seq_len = input_ids.shape
    # Query row j occupies cache column block_base + j, the same coordinate
    # system as attn_mask's key columns (see add_causal_sliding_window_band).
    block_base: int = int(cache_index[0])
    query_coords: torch.Tensor = (torch.arange(seq_len)[None, :] + block_base).expand(
        batch_size, seq_len
    )
    sliding_mask: torch.Tensor = add_causal_sliding_window_band(
        attn_mask, query_coords, sliding_window
    )
    masks[_SLIDING_ATTENTION] = sliding_mask
    return masks


def _run_backbone_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    cache_index: torch.Tensor,
) -> torch.Tensor:
    """OLMo 3 backbone: embedding, RoPE, per-layer-type masks, blocks, norm."""
    state: _Olmo3State = _olmo3_state(model)
    # The shared ``_spyre_*`` slots are read back from the model on every call
    # (not captured at prepare time) because hf_common and the CPU test helpers
    # rebind them, e.g. to unwrap torch.compile.
    ropes: dict[str, PrecomputedRotaryEmbedding] = getattr(model, "_spyre_rope")
    blocks: list[CompiledBlock] = getattr(model, "_spyre_compiled_blocks")
    norm: Callable[[torch.Tensor], torch.Tensor] = getattr(
        model, "_spyre_compiled_norm"
    )

    h: torch.Tensor = embed_text_tokens(model, input_ids)
    freqs: dict[str, torch.Tensor] = {
        layer_type: rope(h, position_ids) for layer_type, rope in ropes.items()
    }
    masks: dict[str, torch.Tensor] = _layer_masks(
        state, input_ids, attn_mask, cache_index
    )

    for i, compiled_block in enumerate(blocks):
        layer_type: str = state.layer_types[i]
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[layer_type],
            masks[layer_type],
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    return norm(h)


def _run_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    cache_index: torch.Tensor,
) -> torch.Tensor:
    """OLMo 3 causal-LM forward: backbone + LM head."""
    h: torch.Tensor = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    logits: torch.Tensor = run_lm_head(model, h)
    return logits


def prepare_for_spyre(model: PreTrainedModel) -> None:
    """Apply Spyre adaptations to an OLMo 3 model in-place."""
    cfg = text_config(model.config)
    backbone: nn.Module = get_backbone(model)
    layers: nn.Module = backbone.get_submodule("layers")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("expected the OLMo 3 backbone layers to be an nn.ModuleList")
    num_layers: int = len(layers)

    layer_types: tuple[str, ...] = tuple(
        getattr(cfg, "layer_types", None) or [_FULL_ATTENTION] * num_layers
    )
    if len(layer_types) != num_layers:
        raise ValueError(
            f"OLMo 3 config has {len(layer_types)} layer_types for "
            f"{num_layers} decoder layers"
        )
    unknown: set[str] = set(layer_types) - {_FULL_ATTENTION, _SLIDING_ATTENTION}
    if unknown:
        raise ValueError(f"OLMo 3 adapter does not support layer types {unknown}")
    sliding_window: int | None = getattr(cfg, "sliding_window", None)
    if _SLIDING_ATTENTION in layer_types and not sliding_window:
        raise ValueError(
            "OLMo 3 config has sliding_attention layers but no sliding_window"
        )

    # Prepare the shared post-norm blocks and heads (this also rejects
    # head_dims that would need padding), then replace the single rotary with
    # one per layer type.
    prepare_olmo_common(model, "OLMo 3")
    setattr(model, "_spyre_olmo3", _Olmo3State(layer_types, sliding_window))
    rotary: nn.Module = backbone.get_submodule("rotary_emb")
    ropes: dict[str, PrecomputedRotaryEmbedding] = {
        layer_type: PrecomputedRotaryEmbedding(
            InvFreqShim(
                getattr(rotary, f"{layer_type}_inv_freq"),
                getattr(rotary, f"{layer_type}_attention_scaling"),
            )
        )
        for layer_type in sorted(set(layer_types))
    }
    # The single rotary is a registered submodule; nn.Module refuses to
    # overwrite a child module with a dict, so unregister it first.
    delattr(model, "_spyre_rope")
    setattr(model, "_spyre_rope", ropes)
