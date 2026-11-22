# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline parity with stock Transformers, including interleaved media."""

from typing import Any

import pytest
import torch
from transformers import (
    EmbeddingGemma2Config,
    EmbeddingGemma2Model,
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextModel,
    Gemma4AudioConfig,
    Gemma4VisionConfig,
)

from hf_adapters import hf_embeddinggemma2
from hf_adapters.hf_common import SpyreUnsupportedModelError


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


def _text_config() -> EmbeddingGemma2TextConfig:
    return EmbeddingGemma2TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        global_head_dim=256,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=64,
        embedding_dim=192,
        sliding_window=2,
        layer_types=["sliding_attention", "full_attention"],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_text_parity(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, use_embeds: bool
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    model = EmbeddingGemma2TextModel(_text_config()).eval().to(dtype)
    # Nontrivial norm scales expose the fp32 multiply required by this model.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.5, 1.5)
    ids = torch.randint(3, 100, (2, 9))
    mask = torch.tensor([[1] * 7 + [0, 0], [0, 0] + [1] * 7])
    positions = torch.arange(9)[None].expand(2, -1) + torch.tensor([[0], [3]])
    with torch.inference_mode():
        inputs: dict[str, torch.Tensor] = {
            "attention_mask": mask,
            "position_ids": positions,
        }
        inputs["inputs_embeds" if use_embeds else "input_ids"] = (
            model.embed_tokens(ids) if use_embeds else ids
        )
        expected = model(**inputs).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
        tuple_output = model(**inputs, return_dict=False)
    assert actual.shape == (2, 9, 192)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(tuple_output[0], actual)
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual.float(), expected.float(), dim=-1
        )
        assert cosine.min() > 0.999


@pytest.mark.parametrize("modality", ["image", "video"])
def test_interleaved_vision_parity(
    monkeypatch: pytest.MonkeyPatch, modality: str
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    vision = Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=32,
        standardize=False,
    )
    config = EmbeddingGemma2Config(
        text_config=_text_config(),
        vision_config=vision,
        audio_config=None,
        image_token_id=125,
        video_token_id=126,
        audio_token_id=127,
    )
    model = EmbeddingGemma2Model(config).eval()
    coords = torch.cartesian_prod(torch.arange(4), torch.arange(4))[None]
    pixels = torch.randn(1, 16, 12)
    token = config.image_token_id if modality == "image" else config.video_token_id
    inputs: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([[3, token, token, token, token, 4, 5]])
    }
    if modality == "image":
        inputs.update(pixel_values=pixels, image_position_ids=coords)
    else:
        inputs.update(
            pixel_values_videos=pixels,
            video_position_ids=coords,
            num_frames_per_video=torch.tensor([1]),
        )
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert "vision_tower.patch_embedder" in model._spyre_cpu_submodules


def test_reject_fp16() -> None:
    model = EmbeddingGemma2TextModel(_text_config()).half()
    with pytest.raises(SpyreUnsupportedModelError, match="float16 overflows"):
        hf_embeddinggemma2.prepare_for_spyre(model)


def test_audio_cpu_tower_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    audio = Gemma4AudioConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        output_proj_dims=64,
        subsampling_conv_channels=[128, 4],
    )
    model = EmbeddingGemma2Model(
        EmbeddingGemma2Config(
            text_config=_text_config(),
            audio_config=audio,
            vision_config=None,
            image_token_id=125,
            video_token_id=126,
            audio_token_id=127,
        )
    ).eval()
    features = torch.randn(1, 64, 128)
    feature_mask = torch.ones(1, 64, dtype=torch.bool)
    with torch.inference_mode():
        audio_output = model.get_audio_features(features, feature_mask)
        count = int(audio_output.attention_mask.sum())
        ids = torch.tensor([[3] + [127] * count + [4]])
        expected = model(
            input_ids=ids, input_features=features, input_features_mask=feature_mask
        ).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(
            input_ids=ids, input_features=features, input_features_mask=feature_mask
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert "audio_tower" in model._spyre_cpu_submodules


def test_fullgraph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_fn = torch.compile

    def trace(function: Any, **kwargs: Any) -> Any:
        return compile_fn(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", trace)
    model = EmbeddingGemma2TextModel(_text_config()).eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        expected = model(ids).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(ids).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.slow
@pytest.mark.model_harness("multimodal_embedding")
@pytest.mark.parametrize("model_path", ["google/embeddinggemma-2"])
def test_sentence_transformers_checkpoint(
    monkeypatch: pytest.MonkeyPatch, model_path: str
) -> None:
    """Public ST loading/processing/pooling against the full checkpoint."""
    import gc

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401
    from tests._multimodal_embedding_helpers import TEXT_PROMPTS, bounded_cpu_threads

    monkeypatch.setattr(torch, "compile", _eager)
    image = Image.new("RGB", (64, 64), color=(150, 80, 35))
    with bounded_cpu_threads(), torch.inference_mode():
        model = SentenceTransformer(
            model_path, device="cpu", model_kwargs={"dtype": torch.bfloat16}
        )
        expected = model.encode(
            TEXT_PROMPTS + [{"image": image}], convert_to_tensor=True
        )
        del model
        gc.collect()
        model = SentenceTransformer(model_path, backend="spyre", device="cpu")
        actual = model.encode(TEXT_PROMPTS + [{"image": image}], convert_to_tensor=True)
    assert actual.shape == (4, 768)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(actual.float(), expected.float())
    assert cosine.min() > 0.99
