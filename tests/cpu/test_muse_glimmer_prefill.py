# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Muse's public VLM API chunks embeddings without rerunning the vision tower."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from transformers import (
    GenerationConfig,
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
    PreTrainedModel,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from hf_adapters import hf_gemma3, hf_muse_glimmer_mm
from hf_adapters.auto_spyre_model import AutoSpyreModelForImageTextToText

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def _without_compile(function: _Callable, **kwargs: Any) -> _Callable:
    return function


@pytest.fixture
def muse_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[PreTrainedModel, PreTrainedModel]:
    torch.random.default_generator.manual_seed(729)
    monkeypatch.setattr(torch, "compile", _without_compile)
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 129,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "max_position_embeddings": 2048,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [10000.0, 0.0],
            "bos_token_id": 1,
            "eos_token_id": None,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "patch_temporal": 2,
            "merge_size": 2,
            "layer_types": ["window_attention", "full_attention"],
        },
        image_token_id=127,
        video_token_id=128,
        out_hidden_size=512,
        projector_hidden_size=128,
    )
    stock = MuseGlimmerForConditionalGeneration(config).eval()
    stock.save_pretrained(tmp_path)
    model = AutoSpyreModelForImageTextToText.from_pretrained(
        tmp_path, dtype=torch.float32
    )
    return stock, model


def _batch(length: int, with_images: bool) -> dict[str, torch.Tensor]:
    ids = torch.randint(3, 126, (2, length))
    mask = torch.ones_like(ids)
    # Mix right and left padding; row 1 has no images and fits inside the window.
    ids[0, -7:] = 0
    mask[0, -7:] = 0
    ids[1, :-31] = 0
    mask[1, :-31] = 0
    batch = {"input_ids": ids, "attention_mask": mask}
    if with_images:
        # Two images, including an image whose token span crosses a 64-row chunk.
        ids[0, 58:70] = 127
        batch["pixel_values"] = torch.randn(48, 24)
        batch["image_grid_thw"] = torch.tensor([[1, 4, 4], [1, 4, 8]])
    return batch


def _assert_outputs(
    actual: GenerateDecoderOnlyOutput, expected: GenerateDecoderOnlyOutput
) -> None:
    torch.testing.assert_close(actual.sequences, expected.sequences)
    assert len(actual.logits) == len(expected.logits)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-4)
    for got, want in zip(actual.scores, expected.scores):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-4)


def _stock_generate(
    stock: PreTrainedModel, batch: dict[str, torch.Tensor], options: dict[str, Any]
) -> GenerateDecoderOnlyOutput:
    # Stock decoder-only generate selects the last column, so its reference
    # must be left-padded even when the adapter caller supplied right padding.
    ids = torch.zeros_like(batch["input_ids"])
    mask = torch.zeros_like(batch["attention_mask"])
    for row in range(ids.shape[0]):
        tokens = batch["input_ids"][row, batch["attention_mask"][row].bool()]
        ids[row, -tokens.numel() :] = tokens
        mask[row, -tokens.numel() :] = 1
    reference_inputs = {**batch, "input_ids": ids, "attention_mask": mask}
    result: GenerateDecoderOnlyOutput = stock.generate(**reference_inputs, **options)
    # The adapter preserves the caller's original prefix in its returned result.
    result.sequences = torch.cat(
        (batch["input_ids"], result.sequences[:, ids.shape[1] :]), dim=1
    )
    return result


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize(
    ("chunk_options", "expected_chunks"),
    [
        ({}, [512, 512]),
        ({"prefill_chunk_size": 64}, [64] * 11),
        ({"prefill_chunk_size": None}, [704]),
    ],
)
@torch.no_grad()
def test_public_vlm_chunking_matches_stock(
    muse_models: tuple[PreTrainedModel, PreTrainedModel],
    monkeypatch: pytest.MonkeyPatch,
    with_images: bool,
    chunk_options: dict[str, Any],
    expected_chunks: list[int],
) -> None:
    stock, model = muse_models
    batch = _batch(648, with_images)  # 641 real tokens in the longer row.
    options: dict[str, Any] = {
        "max_new_tokens": 4,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_scores": True,
    }
    expected = _stock_generate(stock, batch, options)
    chunks: list[int] = []
    image_calls: list[int] = []
    head_rows: list[int] = []
    backbone = hf_gemma3._run_backbone_from_embeds
    vision = hf_muse_glimmer_mm._vision_features

    def record_backbone(
        prepared: torch.nn.Module, hidden: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if hidden.shape[1] > 1:
            chunks.append(hidden.shape[1])
        return backbone(prepared, hidden, *args, **kwargs)

    def record_vision(
        prepared: PreTrainedModel, pixels: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        image_calls.append(pixels.shape[0])
        return vision(prepared, pixels, grid)

    def record_head(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        head_rows.append(inputs[0].shape[1])

    monkeypatch.setattr(hf_gemma3, "_run_backbone_from_embeds", record_backbone)
    monkeypatch.setattr(hf_muse_glimmer_mm, "_vision_features", record_vision)
    handle = model.lm_head.register_forward_pre_hook(record_head)
    try:
        actual = model.generate(**batch, **options, **chunk_options)
    finally:
        handle.remove()
    _assert_outputs(actual, expected)
    assert chunks == expected_chunks
    assert image_calls == ([48] if with_images else [])
    assert head_rows == [1] * 4


@pytest.mark.parametrize("with_images", [False, True])
@torch.no_grad()
def test_prefill_policy_resets_between_requests(
    muse_models: tuple[PreTrainedModel, PreTrainedModel], with_images: bool
) -> None:
    stock, model = muse_models
    options: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_scores": True,
    }
    for length, padded, active_chunk in [
        (80, 128, None),
        (648, 1024, 512),
        (80, 128, None),
    ]:
        batch = _batch(length, with_images)
        expected = _stock_generate(stock, batch, options)
        actual = model.generate(**batch, **options)
        _assert_outputs(actual, expected)
        assert model._spyre_padded_prompt_len == padded
        assert model._spyre_active_prefill_chunk_size == active_chunk
        assert model._spyre_swa_mode == ("anchored" if active_chunk else None)


@torch.no_grad()
def test_explicit_one_shot_overrides_generation_config(
    muse_models: tuple[PreTrainedModel, PreTrainedModel],
) -> None:
    stock, model = muse_models
    batch = _batch(648, True)
    config: GenerationConfig = GenerationConfig.from_dict(
        {**model.generation_config.to_dict(), "prefill_chunk_size": 64}
    )
    options: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_scores": True,
    }
    expected = _stock_generate(stock, batch, options)
    for overrides, active_chunk in [({}, 64), ({"prefill_chunk_size": None}, None)]:
        actual = model.generate(
            **batch, generation_config=config, **options, **overrides
        )
        _assert_outputs(actual, expected)
        assert model._spyre_active_prefill_chunk_size == active_chunk


@torch.no_grad()
def test_chunked_images_decode_through_cache_roll(
    muse_models: tuple[PreTrainedModel, PreTrainedModel],
) -> None:
    stock, model = muse_models
    batch = _batch(648, True)
    options: dict[str, Any] = {
        "max_new_tokens": 70,  # Cross a full 64-row cache-roll interval.
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_scores": True,
    }
    expected = _stock_generate(stock, batch, options)
    actual = model.generate(**batch, prefill_chunk_size=64, **options)
    _assert_outputs(actual, expected)
