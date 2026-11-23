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

"""EmbeddingGemma 2 multimodal embeddings with the stock HF/ST interfaces.

The bidirectional text blocks and Gemma 4 vision blocks execute on Spyre.
Token lookup, media scattering, vision pooling, the stock audio tower and
SentenceTransformers pooling/normalization stay on CPU. Text uses projection-only
per-layer embeddings (PLE), unlike Gemma 4's token-identity PLE. Use bf16: fp16
cannot represent this checkpoint's activation range.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    EmbeddingGemma2Model,
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextModel,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2Attention,
    EmbeddingGemma2EncoderLayer,
    EmbeddingGemma2RMSNorm,
    EmbeddingGemma2TextPLE,
    EmbeddingGemma2TextPLEBlock,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

from hf_adapters import hf_gemma4_vision
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
    add_sliding_window_band,
    apply_rope_matmul,
    patch_cpu_inputs,
)
from hf_adapters.hf_gemma4 import (
    _gemma4_mlp_activation,
    _gemma4_rms_norm,
    _offset_zero_per_layer_input,
)

_is_encoder_only = True
# ST must retain the multimodal forward instead of installing text-only prefill.
_preserve_native_forward = True

CompiledBlock = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _rms_norm(hidden: torch.Tensor, norm: EmbeddingGemma2RMSNorm) -> torch.Tensor:
    """Reuse Gemma 4's reduction, with EmbeddingGemma's fp32 weight multiply."""
    weight: torch.Tensor | None = norm.weight.float() if norm.with_scale else None
    result: torch.Tensor = _gemma4_rms_norm(hidden.float(), weight, norm.eps)
    return result.to(hidden.dtype)


def _make_compiled_block(layer: EmbeddingGemma2EncoderLayer) -> CompiledBlock:
    attn: EmbeddingGemma2Attention = layer.self_attn
    head_dim: int = attn.head_dim
    q_heads: int = attn.q_proj.out_features // head_dim
    kv_heads: int = attn.k_proj.out_features // head_dim
    ple: EmbeddingGemma2TextPLEBlock = layer.ple_block

    def block_forward(
        hidden: torch.Tensor,
        per_layer_input: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        residual = hidden
        hidden = _rms_norm(hidden, layer.input_layernorm)
        q = attn.q_proj(hidden).view(batch, length, q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        q = apply_rope_matmul(_rms_norm(q, attn.q_norm), rope)
        k = apply_rope_matmul(_rms_norm(k, attn.k_norm), rope)
        v = _rms_norm(v, attn.v_norm)
        hidden = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=attn.scaling,
            enable_gqa=q_heads != kv_heads,
        )
        hidden = hidden.transpose(1, 2).reshape(batch, length, q_heads * head_dim)
        hidden = residual + _rms_norm(
            attn.o_proj(hidden), layer.post_attention_layernorm
        )
        residual = hidden
        hidden = _rms_norm(hidden, layer.pre_feedforward_layernorm)
        hidden = _gemma4_mlp_activation(
            layer.mlp.gate_proj(hidden)
        ) * layer.mlp.up_proj(hidden)
        hidden = residual + _rms_norm(
            layer.mlp.down_proj(hidden), layer.post_feedforward_layernorm
        )
        residual = hidden
        hidden = (
            _gemma4_mlp_activation(ple.per_layer_input_gate(hidden)) * per_layer_input
        )
        hidden = residual + _rms_norm(
            ple.per_layer_projection(hidden), ple.post_per_layer_input_norm
        )
        return hidden * layer.layer_scalar

    return torch.compile(block_forward, dynamic=False, fullgraph=True)


def _prepare_text(model: EmbeddingGemma2TextModel) -> None:
    config: EmbeddingGemma2TextConfig = model.config
    layers: list[EmbeddingGemma2EncoderLayer] = []
    for layer in model.layers:
        if not isinstance(layer, EmbeddingGemma2EncoderLayer):
            raise TypeError(
                f"Expected EmbeddingGemma2EncoderLayer, got {type(layer).__name__}"
            )
        layers.append(layer)
        head_dim: int = layer.self_attn.head_dim
        if head_dim % (2 * BLOCK_SIZE):
            raise SpyreUnsupportedModelError(
                f"EmbeddingGemma 2 text head_dim={head_dim} must be a multiple of {2 * BLOCK_SIZE}"
            )
    if config.hidden_activation != "gelu_pytorch_tanh":
        raise SpyreUnsupportedModelError("EmbeddingGemma 2 requires gelu_pytorch_tanh")
    ropes: dict[str, PrecomputedRotaryEmbedding] = {}
    for layer_type in model.unique_layer_types:
        ropes[layer_type] = PrecomputedRotaryEmbedding(
            InvFreqShim(
                getattr(model.rotary_emb, f"{layer_type}_inv_freq"),
                getattr(model.rotary_emb, f"{layer_type}_attention_scaling"),
            )
        )
        ropes[layer_type].set_dtype(model.dtype)
    setattr(model, "_spyre_rope", ropes)
    blocks: list[CompiledBlock] = [_make_compiled_block(layer) for layer in layers]
    setattr(model, "_spyre_compiled_blocks", blocks)

    def project_ple(hidden: torch.Tensor) -> torch.Tensor:
        ple: EmbeddingGemma2TextPLE = model.ple
        projected = (
            ple.per_layer_model_projection(hidden)
            * ple.per_layer_model_projection_scale
        )
        projected = projected.reshape(
            *hidden.shape[:-1],
            config.num_hidden_layers,
            config.hidden_size_per_layer_input,
        )
        return _rms_norm(projected, ple.per_layer_projection_norm)

    def final_norm(hidden: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden, model.norm)

    setattr(
        model,
        "_spyre_compiled_ple",
        torch.compile(project_ple, dynamic=False, fullgraph=True),
    )
    setattr(
        model,
        "_spyre_compiled_norm",
        torch.compile(final_norm, dynamic=False, fullgraph=True),
    )
    setattr(model, "forward", MethodType(_text_forward, model))


def _text_forward(
    self: EmbeddingGemma2TextModel,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> BaseModelOutput | tuple[torch.Tensor]:
    if kwargs.get("output_attentions", self.config.output_attentions) or kwargs.get(
        "output_hidden_states", self.config.output_hidden_states
    ):
        raise SpyreUnsupportedFeatureError(
            "EmbeddingGemma 2 does not return intermediate hidden states or attention weights on Spyre"
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    hidden: torch.Tensor
    if inputs_embeds is None:
        assert input_ids is not None
        hidden = self.embed_tokens(input_ids.cpu())
    else:
        hidden = inputs_embeds.cpu()
    batch, length, _ = hidden.shape
    if not length:
        raise ValueError("EmbeddingGemma 2 requires a nonempty sequence")
    pad = (-length) % BLOCK_SIZE
    padded_length = length + pad
    if attention_mask is not None and attention_mask.ndim != 2:
        raise SpyreUnsupportedFeatureError(
            "EmbeddingGemma 2 requires a 2D padding mask on Spyre"
        )
    valid = (
        torch.ones((batch, length), dtype=torch.bool)
        if attention_mask is None
        else attention_mask.cpu().bool()
    )
    valid = F.pad(valid, (0, pad), value=False)
    mask = torch.zeros((batch, 1, padded_length, padded_length), dtype=hidden.dtype)
    mask.masked_fill_(~valid[:, None, None, :], -torch.inf)
    # A padding query outside the sliding window of every real token must
    # still have a finite softmax. Only that query may attend to its own key.
    mask.diagonal(dim1=-2, dim2=-1).masked_fill_(~valid[:, None, :], 0.0)
    masks: dict[str, torch.Tensor] = {
        "full_attention": mask.to(DEVICE),
        "sliding_attention": add_sliding_window_band(
            mask, self.config.sliding_window
        ).to(DEVICE),
    }
    positions = (
        torch.arange(length)[None] if position_ids is None else position_ids.cpu()
    )
    positions = F.pad(positions, (0, pad))
    hidden = F.pad(hidden, (0, 0, 0, pad)).to(DEVICE)
    ropes: dict[str, PrecomputedRotaryEmbedding] = getattr(self, "_spyre_rope")
    selected: dict[str, torch.Tensor] = {}
    for layer_type, rope in ropes.items():
        rope.set_dtype(hidden.dtype)
        selected[layer_type] = rope(hidden, positions)
    project_ple: Callable[[torch.Tensor], torch.Tensor] = getattr(
        self, "_spyre_compiled_ple"
    )
    per_layer_inputs = project_ple(hidden)
    blocks: list[CompiledBlock] = getattr(self, "_spyre_compiled_blocks")
    assert self.config.layer_types is not None
    for index, block in enumerate(blocks):
        layer_type = self.config.layer_types[index]
        hidden = block(
            hidden,
            _offset_zero_per_layer_input(per_layer_inputs, index),
            selected[layer_type],
            masks[layer_type],
        ).clone()
    final_norm: Callable[[torch.Tensor], torch.Tensor] = getattr(
        self, "_spyre_compiled_norm"
    )
    hidden = self.embedding_projection(final_norm(hidden).cpu()[:, :length])
    return_dict: bool | None = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = self.config.return_dict
    if not return_dict:
        return (hidden,)
    return BaseModelOutput(last_hidden_state=hidden)


def prepare_for_spyre(model: EmbeddingGemma2Model | EmbeddingGemma2TextModel) -> None:
    """Patch only the execution bodies; retain HF's modality dispatch and outputs."""
    if not isinstance(model, (EmbeddingGemma2Model, EmbeddingGemma2TextModel)):
        raise TypeError(f"Expected EmbeddingGemma2Model, got {type(model).__name__}")
    if getattr(model, "_spyre_embeddinggemma2_prepared", False):
        return
    if model.dtype == torch.float16:
        raise SpyreUnsupportedModelError(
            "EmbeddingGemma 2 requires bfloat16 or float32; float16 overflows"
        )
    text: EmbeddingGemma2TextModel = (
        model.language_model if isinstance(model, EmbeddingGemma2Model) else model
    )
    _prepare_text(text)
    prefix = "language_model." if isinstance(model, EmbeddingGemma2Model) else ""
    cpu_modules: list[str] = [
        prefix + "embed_tokens",
        prefix + "rotary_emb",
        prefix + "embedding_projection",
    ]
    if isinstance(model, EmbeddingGemma2Model):
        if model.vision_tower is not None:
            hf_gemma4_vision.prepare_for_spyre(model, tower=model.vision_tower)

            def vision_forward(
                self: Gemma4VisionModel,
                pixel_values: torch.Tensor,
                pixel_position_ids: torch.Tensor,
                **kwargs: Any,
            ) -> BaseModelOutputWithPooling | tuple[torch.Tensor]:
                if kwargs.get("output_attentions", False) or kwargs.get(
                    "output_hidden_states", False
                ):
                    raise SpyreUnsupportedFeatureError(
                        "Gemma 4 vision does not return intermediate states on Spyre"
                    )
                hidden: torch.Tensor = hf_gemma4_vision.prefill_vision_tower(
                    model, pixel_values, pixel_position_ids, tower=self
                )
                return_dict: bool | None = kwargs.get("return_dict")
                if return_dict is None:
                    return_dict = self.config.return_dict
                if not return_dict:
                    return (hidden,)
                return BaseModelOutputWithPooling(last_hidden_state=hidden)

            setattr(
                model.vision_tower,
                "forward",
                MethodType(vision_forward, model.vision_tower),
            )
            cpu_modules.extend(
                [
                    "vision_tower.patch_embedder",
                    "vision_tower.pooler",
                    "vision_tower.encoder.rotary_emb",
                    "embed_vision",
                ]
            )
        if model.audio_tower is not None:
            cpu_modules.extend(["audio_tower", "embed_audio"])
        patch_cpu_inputs(model)
        for method_name in (
            "get_image_features",
            "get_video_features",
            "get_audio_features",
        ):
            patch_cpu_inputs(model, method_name=method_name)
    setattr(model, "_spyre_cpu_submodules", cpu_modules)
    # Opt in to ST device selection from a compute weight, since the first
    # parameter belongs to the CPU token embeddings. No parameter scan needed.
    setattr(
        model,
        "_spyre_device_parameter_name",
        prefix + "layers.0.self_attn.q_proj.weight",
    )
    setattr(model, "_spyre_embeddinggemma2_prepared", True)
