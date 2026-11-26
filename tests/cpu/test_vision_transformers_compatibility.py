# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline parity against the installed Transformers vision implementations."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import (
    Gemma4VisionConfig,
    Gemma4VisionModel,
    Granite4VisionConfig,
    Granite4VisionForConditionalGeneration,
    PixtralVisionConfig,
    PixtralVisionModel,
)

from hf_adapters import (
    hf_gemma4_vision,
    hf_granite_vision_mm,
    hf_pixtral_vision,
    hf_siglip_vision,
)


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


@pytest.mark.parametrize("image_sizes", [[(16, 24)], [(16, 24), (24, 16)]])
def test_pixtral_axial_rope_tower(
    monkeypatch: pytest.MonkeyPatch, image_sizes: list[tuple[int, int]]
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(771)
    model = PixtralVisionModel(
        PixtralVisionConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=4,
        )
    ).eval()
    pixels = torch.randn(len(image_sizes), 3, 24, 24)
    with torch.no_grad():
        expected = model(pixels, image_sizes=image_sizes).last_hidden_state
        hf_pixtral_vision.prepare_for_spyre(model)
        actual = hf_pixtral_vision.prefill_vision_tower(model, pixels, image_sizes)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("standardize", [False, True])
def test_gemma4_axial_rope_tower(
    monkeypatch: pytest.MonkeyPatch, standardize: bool
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(771)
    config = Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=32,
        standardize=standardize,
    )
    tower = Gemma4VisionModel(config).eval()
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=tower))
    coords = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    coords = torch.cat((coords, torch.full((4, 2), -1))).unsqueeze(0)
    pixels = torch.randn(1, 20, 12)
    with torch.no_grad():
        expected = tower(pixels, pixel_position_ids=coords).last_hidden_state
        hf_gemma4_vision.prepare_for_spyre(model)
        actual = hf_gemma4_vision.prefill_vision_tower(model, pixels, coords)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_granite_deepstack_packs_every_image(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(771)
    config = Granite4VisionConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "image_size": 8,
            "patch_size": 2,
        },
        text_config={
            "model_type": "granite4_vision_text",
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
        },
        qformer_config={
            "model_type": "blip_2_qformer",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
        },
        image_grid_pinpoints=[[8, 8]],
        vision_feature_select_strategy="full",
        downsample_rate="1/2",
        deepstack_layer_map=[[0, 0], [1, 1]],
        spatial_target_layers=[2, 3, 4, 5],
        projector_dropout=0.0,
    )
    model = Granite4VisionForConditionalGeneration(config).eval()
    pixels = torch.randn(batch_size, 2, 3, 8, 8)
    sizes = torch.tensor([[8, 8]] * batch_size)
    with torch.no_grad():
        expected = model.model.get_image_features(
            pixels, image_sizes=sizes
        ).deepstack_features
        hf_siglip_vision.prepare_for_spyre(model)
        actual = hf_granite_vision_mm._deepstack_features(model, pixels, sizes)
    assert len(actual) == 6
    for layer, features in expected:
        torch.testing.assert_close(
            actual[layer], torch.cat(features), atol=2e-5, rtol=2e-5
        )
