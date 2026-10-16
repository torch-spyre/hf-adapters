# Copyright 2026 The Torch-Spyre Authors.
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

"""EmbeddingGemma 2: Spyre text/vision encoders with native HF media merging.

Requires Transformers with ``embedding_gemma2`` support. Audio, patch embedding,
media merging and final pooling run on CPU; transformer bodies run on Spyre.
Use bfloat16: the checkpoint's activations overflow float16.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import MethodType, SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from hf_adapters import hf_gemma4_vision
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_sliding_window_band,
    apply_rope_matmul,
    embed_text_tokens,
    get_backbone,
    to_default_device_layout,
)

_is_encoder_only = True
_preserve_native_forward = True


def _make_compiled_block(layer: Any) -> Callable[..., torch.Tensor]:
    attn: Any = layer.self_attn
    head_dim: int = attn.head_dim

    def block_forward(
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        h = layer.input_layernorm(hidden_states)
        q = attn.q_norm(attn.q_proj(h).view(bsz, seq_len, -1, head_dim))
        k = attn.k_norm(attn.k_proj(h).view(bsz, seq_len, -1, head_dim))
        v = attn.v_norm(attn.v_proj(h).view(bsz, seq_len, -1, head_dim))
        q = apply_rope_matmul(q.transpose(1, 2), freqs).contiguous()
        k = apply_rope_matmul(k.transpose(1, 2), freqs).contiguous()
        v = v.transpose(1, 2).contiguous()
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
            enable_gqa=True,
        )
        h = attn.o_proj(h.transpose(1, 2).reshape(bsz, seq_len, -1))
        h = hidden_states + layer.post_attention_layernorm(h)
        h = h + layer.post_feedforward_layernorm(
            layer.mlp(layer.pre_feedforward_layernorm(h))
        )
        return cast(
            torch.Tensor, layer.ple_block(h, per_layer_input) * layer.layer_scalar
        )

    return torch.compile(block_forward, dynamic=False)


def _run_embeds(
    backbone: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    h = inputs_embeds.cpu()
    bsz, seq_len, _ = h.shape
    if seq_len == 0:
        raise ValueError("Embedding inputs must contain at least one token")
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    if attention_mask is None:
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
    if attention_mask.shape != (bsz, seq_len):
        raise ValueError("attention_mask must match the input batch and sequence")
    valid = F.pad(attention_mask.cpu().bool(), (0, padded_len - seq_len))
    mask = torch.zeros((bsz, 1, padded_len, padded_len), dtype=h.dtype)
    mask = mask.masked_fill(~valid[:, None, None, :], -torch.inf)
    masks = {
        "full_attention": mask.to(DEVICE),
        "sliding_attention": add_sliding_window_band(
            mask, backbone.config.sliding_window
        ).to(DEVICE),
    }
    if position_ids is None:
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
    position_ids = F.pad(position_ids.cpu(), (0, padded_len - seq_len))
    h = F.pad(h, (0, 0, 0, padded_len - seq_len))
    # Compute PLE on CPU, avoiding a four-dimensional Spyre projection/slice.
    per_layer_inputs = backbone.ple(h)
    h = to_default_device_layout(h.to(DEVICE))
    freqs: dict[str, torch.Tensor] = {
        kind: rope(h, position_ids) for kind, rope in backbone._spyre_rope.items()
    }
    for i, block in enumerate(backbone._spyre_compiled_blocks):
        kind: str = backbone.config.layer_types[i]
        h = block(
            h,
            per_layer_inputs[:, :, i, :].contiguous().to(DEVICE),
            freqs[kind],
            masks[kind],
        )
        h = to_default_device_layout(h)
    h = backbone._spyre_compiled_norm(h)
    return cast(torch.Tensor, backbone.embedding_projection(h[:, :seq_len].cpu()))


def _text_forward(
    self: Any,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> BaseModelOutput:
    if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
        raise ValueError(
            "Attention and intermediate hidden-state outputs are unsupported"
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids and inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = embed_text_tokens(self, input_ids)
    return BaseModelOutput(
        last_hidden_state=_run_embeds(self, inputs_embeds, attention_mask, position_ids)  # type: ignore[arg-type]
    )


def _run_backbone_forward(
    model: Any,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shared encoder-prefill interface for text-only accuracy tests."""
    backbone: Any = get_backbone(model)
    return _run_embeds(
        backbone,
        embed_text_tokens(model, input_ids),
        torch.isfinite(attn_mask.cpu()[:, 0, 0]),
        position_ids,
    )


def _vision_forward(
    self: Any,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
    **kwargs: Any,
) -> BaseModelOutput:
    wrapper: Any = self._spyre_wrapper
    h = hf_gemma4_vision.prefill_vision_tower(wrapper, pixel_values, pixel_position_ids)
    return BaseModelOutput(last_hidden_state=h)


def prepare_for_spyre(model: Any) -> None:
    backbone: Any = get_backbone(model)
    if backbone.embed_tokens.weight.dtype == torch.float16:
        raise ValueError("EmbeddingGemma 2 requires bfloat16 or float32")
    prefix = "language_model." if backbone is not model else ""
    model._spyre_cpu_submodules = [
        prefix + name
        for name in ("embed_tokens", "ple", "embedding_projection", "rotary_emb")
    ]
    backbone._spyre_rope = {}
    for kind in set(backbone.config.layer_types):
        inv_freq = getattr(backbone.rotary_emb, f"{kind}_inv_freq")
        if inv_freq.numel() < BLOCK_SIZE:
            raise ValueError("EmbeddingGemma 2 RoPE requires head_dim >= 128")
        backbone._spyre_rope[kind] = PrecomputedRotaryEmbedding(
            InvFreqShim(
                inv_freq, getattr(backbone.rotary_emb, f"{kind}_attention_scaling")
            )
        )
        backbone._spyre_rope[kind].set_dtype(backbone.embed_tokens.weight.dtype)
    model._spyre_rope = backbone._spyre_rope
    backbone._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in backbone.layers
    ]
    backbone._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
    backbone.forward = MethodType(_text_forward, backbone)
    if getattr(model, "vision_tower", None) is not None:
        wrapper = SimpleNamespace(model=model)
        # The Gemma 4 vision implementation supports this tower's axial RoPE.
        hf_gemma4_vision.prepare_for_spyre(wrapper)
        model.vision_tower._spyre_wrapper = wrapper
        model.vision_tower.forward = MethodType(_vision_forward, model.vision_tower)
        model._spyre_cpu_submodules.extend(
            [
                "vision_tower.patch_embedder",
                "vision_tower.pooler",
                "vision_tower.encoder.rotary_emb",
                "embed_vision",
            ]
        )
    if getattr(model, "audio_tower", None) is not None:
        model._spyre_cpu_submodules.extend(["audio_tower", "embed_audio"])
