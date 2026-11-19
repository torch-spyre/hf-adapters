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

"""Hugging Face CLIP text/vision encoders on Spyre.

Stock HF forwards retain causal text attention, EOS/CLS pooling, projections,
and similarity logits. Shared helpers compile and execute the encoder blocks;
embeddings, pooling, and projections remain on CPU.
"""

from __future__ import annotations

from transformers import (
    CLIPModel,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)

from hf_adapters.hf_common import (
    encoder_backbone_forward,
    patch_cpu_inputs,
    patch_pre_ln_encoder_forward,
    prepare_pre_ln_encoder,
)

_run_backbone_forward = encoder_backbone_forward
_is_encoder_only = True
SPYRE_DIMENSION_CONFIGS: tuple[str, ...] = ("text_config", "vision_config")
SPYRE_PADDED_DIMENSIONS: tuple[str, ...] = ("intermediate_size",)


def prepare_for_spyre(model) -> None:
    """Prepare full, projected, or bare CLIP towers without replacing HF pooling."""
    if getattr(model, "_spyre_clip_prepared", False):
        return
    cpu_submodules: list[str] = []
    towers: list[tuple[str, CLIPTextModel | CLIPVisionModel]]
    if isinstance(model, CLIPModel):
        towers = [
            ("text_model", model.text_model),
            ("vision_model", model.vision_model),
        ]
        cpu_submodules.extend(["text_projection", "visual_projection"])
    elif isinstance(model, CLIPTextModelWithProjection):
        towers = [("text_model", model.text_model)]
        cpu_submodules.append("text_projection")
    elif isinstance(model, CLIPVisionModelWithProjection):
        towers = [("vision_model", model.vision_model)]
        cpu_submodules.append("visual_projection")
    elif isinstance(model, (CLIPTextModel, CLIPVisionModel)):
        towers = [("", model)]
    else:
        raise ValueError(f"Unsupported CLIP model architecture: {type(model).__name__}")

    for prefix, tower in towers:
        config = tower.config
        if not isinstance(config, (CLIPTextConfig, CLIPVisionConfig)):
            raise TypeError(f"Unsupported CLIP tower config: {type(config).__name__}")
        blocks, head_dim = prepare_pre_ln_encoder(tower.encoder, config)
        patch_pre_ln_encoder_forward(tower.encoder, blocks)
        setattr(tower.encoder, "_spyre_head_dim", head_dim)
        if not prefix:
            setattr(model, "_spyre_compiled_blocks", blocks)
        modules: tuple[str, ...]
        if isinstance(tower, CLIPTextModel):
            modules = ("embeddings", "final_layer_norm")
            patch_cpu_inputs(tower, ("input_ids", "attention_mask", "position_ids"))
        else:
            modules = ("embeddings", "pre_layrnorm", "post_layernorm")
            patch_cpu_inputs(tower, ("pixel_values",))
        cpu_submodules.extend(
            f"{prefix}.{name}" if prefix else name for name in modules
        )
    setattr(model, "_spyre_cpu_submodules", cpu_submodules)
    setattr(model, "_spyre_clip_prepared", True)
