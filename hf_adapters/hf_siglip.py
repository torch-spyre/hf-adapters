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
Both encoders share padding and execution through ``_vision_encoder``.
Embeddings, final norms, text projection, and attention pooling stay on CPU,
while the transformer blocks execute on Spyre.

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

from collections.abc import Callable
from types import MethodType

import torch
from transformers import (
    SiglipModel,
    SiglipTextConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings,
)

from hf_adapters import hf_siglip_vision
from hf_adapters._vision_encoder import patch_cpu_inputs, patch_encoder, prepare_encoder
from hf_adapters.hf_common import patch_layernorm

_is_encoder_only = True

# Validate both towers; prepare_encoder pads each tower's MLP intermediate size.
SPYRE_DIMENSION_CONFIGS: tuple[str, ...] = ("text_config", "vision_config")
SPYRE_PADDED_DIMENSIONS: tuple[str, ...] = ("intermediate_size",)


def _patch_vision_forward(vision_model: SiglipVisionModel) -> None:
    """Reuse the CPU patch embedding while preserving HF interpolation/pooling."""
    embeddings = vision_model.embeddings
    original_embedding_forward: Callable[..., torch.Tensor] = type(embeddings).forward
    patch_embed = hf_siglip_vision._make_patch_embed(vision_model)

    def embedding_forward(
        self: SiglipVisionEmbeddings,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if interpolate_pos_encoding:
            result: torch.Tensor = original_embedding_forward(
                self, pixel_values.cpu(), interpolate_pos_encoding=True
            )
            return result
        expected_size = self.image_size
        if pixel_values.shape[-2:] != (expected_size, expected_size):
            raise ValueError(
                f"Expected {expected_size}x{expected_size} images; "
                "use interpolate_pos_encoding=True for other resolutions"
            )
        return patch_embed(pixel_values)

    setattr(embeddings, "forward", MethodType(embedding_forward, embeddings))
    patch_cpu_inputs(vision_model, ("pixel_values",))


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
        config = tower.config
        if not isinstance(config, (SiglipTextConfig, SiglipVisionConfig)):
            raise TypeError(f"Unsupported SigLIP config: {type(config).__name__}")
        blocks, head_dim = prepare_encoder(tower.encoder, config)
        patch_encoder(tower.encoder, blocks)
        setattr(tower.encoder, "_spyre_head_dim", head_dim)
    patch_cpu_inputs(model.text_model, ("input_ids", "attention_mask", "position_ids"))
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
