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

"""SigLIP dual-encoder text/image embeddings on Spyre (Transformers 5.15).

``google/siglip2-giant-opt-patch16-384`` uses ``SiglipModel``, with a fixed
384-pixel vision input, rather than the separate Siglip2 NaFlex architecture.
Both encoders use the padding and compiled blocks from ``hf_siglip_vision``.
Embeddings, final norms, text projection, and attention pooling stay on CPU,
as in the CLIP adapter, while the transformer blocks execute on Spyre.

The stock HF API is preserved: ``get_text_features`` / ``get_image_features``
return outputs with unnormalized ``pooler_output``; the joint ``forward``
returns normalized embeddings and scaled, biased image/text similarity logits.
Tokenize text with ``padding="max_length"`` (64 tokens for this checkpoint),
because SigLIP pools the final token, including tokenizer padding.

Usage::

    from hf_adapters import AutoSpyreModel
    from transformers import AutoProcessor

    path = "google/siglip2-giant-opt-patch16-384"
    model = AutoSpyreModel.from_pretrained(path)
    processor = AutoProcessor.from_pretrained(path)
    inputs = processor(text=["a photo of a cat"], images=image,
                       padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from types import MethodType
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import (
    SiglipConfig,
    SiglipModel,
    SiglipTextConfig,
    SiglipTextModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipVisionEmbeddings,
)

from hf_adapters import hf_siglip_vision
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    assert_spyre_dimensions,
    load_model_common,
    move_model_to_spyre,
    patch_layernorm,
)

_is_encoder_only = True


def validate_spyre_dimensions(config: SiglipConfig, model_name: str) -> None:
    """Validate both towers while allowing the intermediate dim we zero-pad."""
    for tower_name, tower_config in (
        ("text", config.text_config),
        ("vision", config.vision_config),
    ):
        cfg = copy.copy(cast(SiglipTextConfig | SiglipVisionConfig, tower_config))
        cfg.intermediate_size = (
            (cfg.intermediate_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        ) * BLOCK_SIZE
        assert_spyre_dimensions(cfg, f"{model_name} ({tower_name} tower)")


def _patch_encoder(encoder: SiglipEncoder) -> None:
    """Run prepared blocks with stick-aligned sequences and padding masks."""

    def encoder_forward(
        self: SiglipEncoder,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        if kwargs.get("output_attentions", False):
            raise ValueError("SigLIP Spyre blocks do not return attention weights")
        if kwargs.get("output_hidden_states", False):
            raise ValueError(
                "SigLIP Spyre blocks do not return intermediate hidden states"
            )
        hidden = inputs_embeds.to(DEVICE)
        seq_len = hidden.shape[1]
        pad_len = (-seq_len) % BLOCK_SIZE
        mask: torch.Tensor | None = (
            attention_mask.cpu() if attention_mask is not None else None
        )
        if pad_len:
            hidden = F.pad(hidden, (0, 0, 0, pad_len))
            if mask is None:
                mask = torch.ones((1, 1, 1, seq_len), dtype=torch.bool)
            # Mask only extra keys. Keeping padded queries able to attend to
            # real keys avoids all-masked softmax rows and resulting NaNs.
            fill_value: bool | float = (
                False if mask.dtype == torch.bool else float("-inf")
            )
            if mask.shape[-2] == seq_len:
                allowed: bool | float = True if mask.dtype == torch.bool else 0.0
                mask = F.pad(mask, (0, 0, 0, pad_len), value=allowed)
            # Pad keys last so even the added query rows cannot attend to
            # compiler-added keys (False for bool masks, -inf for additive).
            mask = F.pad(mask, (0, pad_len), value=fill_value)
        if mask is not None:
            mask = mask.to(DEVICE)
        hidden = hidden.clone()
        blocks = cast(list[Callable[..., torch.Tensor]], self._spyre_compiled_blocks)
        for block in blocks:
            hidden = block(hidden, mask)
            if hidden.device.type == "spyre":
                hidden = hidden.clone()
        return BaseModelOutput(
            last_hidden_state=cast(torch.FloatTensor, hidden.cpu()[:, :seq_len]),
        )

    encoder.forward = MethodType(encoder_forward, encoder)


def _patch_text_forward(text_model: SiglipTextModel) -> None:
    """Keep token lookups and HF mask construction on the host."""
    original_forward = type(text_model).forward

    def text_forward(
        self: SiglipTextModel,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        return cast(
            BaseModelOutputWithPooling,
            original_forward(
                self,
                input_ids=input_ids.cpu() if input_ids is not None else None,
                attention_mask=(
                    attention_mask.cpu() if attention_mask is not None else None
                ),
                position_ids=position_ids.cpu() if position_ids is not None else None,
                **kwargs,
            ),
        )

    text_model.forward = MethodType(text_forward, text_model)


def _patch_vision_forward(vision_model: SiglipVisionModel) -> None:
    """Reuse the CPU patch embedding while preserving HF interpolation/pooling."""
    original_forward = type(vision_model).forward
    embeddings = vision_model.embeddings
    original_embedding_forward = type(embeddings).forward
    patch_embed = cast(
        Callable[[torch.Tensor], torch.Tensor], vision_model._spyre_patch_embed
    )

    def embedding_forward(
        self: SiglipVisionEmbeddings,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if interpolate_pos_encoding:
            return cast(
                torch.Tensor,
                original_embedding_forward(
                    self,
                    cast(torch.FloatTensor, pixel_values.cpu()),
                    interpolate_pos_encoding=True,
                ),
            )
        expected_size = self.image_size
        if pixel_values.shape[-2:] != (expected_size, expected_size):
            raise ValueError(
                f"Expected {expected_size}x{expected_size} images; "
                "use interpolate_pos_encoding=True for other resolutions"
            )
        return patch_embed(pixel_values)

    def vision_forward(
        self: SiglipVisionModel,
        pixel_values: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        return cast(
            BaseModelOutputWithPooling,
            original_forward(
                self,
                pixel_values=pixel_values.cpu() if pixel_values is not None else None,
                interpolate_pos_encoding=interpolate_pos_encoding,
                **kwargs,
            ),
        )

    setattr(embeddings, "forward", MethodType(embedding_forward, embeddings))
    vision_model.forward = MethodType(vision_forward, vision_model)


def prepare_for_spyre(model: SiglipModel) -> None:
    """Prepare both towers in-place; retain HF projections and pooling."""
    if not isinstance(model, SiglipModel):
        raise ValueError(f"Expected SiglipModel, got {type(model).__name__}")
    if getattr(model, "_spyre_siglip_prepared", False):
        return
    # Match HF's fp32 norm reductions on device. Across the giant vision
    # tower's 40 blocks, fp16 fused reductions noticeably degrade embeddings.
    for tower in (model.text_model, model.vision_model):
        for layer in tower.encoder.layers:
            patch_layernorm(layer.layer_norm1, layer.layer_norm2)
    hf_siglip_vision.prepare_for_spyre(model.vision_model)
    model.vision_model.encoder._spyre_compiled_blocks = (
        model.vision_model._spyre_compiled_blocks
    )
    text_blocks, text_head_dim = hf_siglip_vision._prepare_encoder(
        model.text_model.encoder, cast(SiglipTextConfig, model.config.text_config)
    )
    model.text_model.encoder._spyre_compiled_blocks = text_blocks
    model.text_model.encoder._spyre_head_dim = text_head_dim
    _patch_encoder(model.text_model.encoder)
    _patch_encoder(model.vision_model.encoder)
    _patch_text_forward(model.text_model)
    _patch_vision_forward(model.vision_model)
    setattr(
        model,
        "_spyre_cpu_submodules",
        [
            "text_model.embeddings",
            "text_model.final_layer_norm",
            "text_model.head",
            "vision_model.embeddings",
            "vision_model.post_layernorm",
            "vision_model.head",
        ],
    )
    setattr(model, "_spyre_siglip_prepared", True)


def load_hf_model(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool | None = None,
) -> SiglipModel:
    """Load the stock HF dual encoder on CPU for reference or preparation."""
    model = SiglipModel.from_pretrained(
        model_path, dtype=dtype, device_map="cpu", trust_remote_code=trust_remote_code
    )
    model.eval()
    model.requires_grad_(False)
    return model


def load_model(model_path: str, dtype: torch.dtype = torch.float16) -> SiglipModel:
    """Load, prepare, and move the dual encoder using the shared loader."""
    import sys

    module = sys.modules[__name__]
    model: SiglipModel = load_model_common(model_path, module, dtype)
    move_model_to_spyre(model, module, dtype)
    return model
