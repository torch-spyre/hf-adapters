# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Muse VLM generation and prefill agree with stock HF on a tiny checkpoint."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from hf_adapters import auto_spyre_model, hf_common, hf_muse_glimmer_mm

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def _without_compile(function: _Callable, **kwargs: Any) -> _Callable:
    return function


@pytest.fixture
def muse_model(monkeypatch: pytest.MonkeyPatch) -> MuseGlimmerForConditionalGeneration:
    # Seed CPU alone: torch.manual_seed also initializes Spyre on a device host.
    torch.random.default_generator.manual_seed(493)
    monkeypatch.setattr(torch, "compile", _without_compile)
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 129,  # Exercise LM-head vocabulary padding/cropping.
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "max_position_embeddings": 1024,
            "sliding_window": 32,
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
    return MuseGlimmerForConditionalGeneration(config).eval()


def _inputs(with_image: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ids = torch.randint(3, 126, (1, 73))  # Cross the sliding-attention boundary.
    image_inputs: dict[str, torch.Tensor] = {}
    if with_image:
        ids[:, 1:5] = 127
        image_inputs = {
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
    return ids, image_inputs


def test_vision_rope_preserves_stock_bfloat16_rounding() -> None:
    # Use the checkpoint's 96-wide heads, padded to 128 for Spyre.
    cfg = MuseGlimmerConfig(
        vision_config={"hidden_size": 192, "num_attention_heads": 2}
    ).vision_config
    rotary = MuseGlimmerVisionRotaryEmbedding(cfg)
    q = torch.randn(1, 7, 2, 96).to(torch.bfloat16)
    positions = torch.tensor([[0, 0], [0, 1], [1, 0], [3, 7], [9, 2], [5, 8], [20, 30]])
    cos, sin = rotary(q, (positions.flip(-1) + 1).unsqueeze(0))
    expected, _ = apply_rotary_pos_emb_vision(q, q, cos, sin)
    zeros = torch.zeros(1, 7, 2, 16, dtype=q.dtype)
    padded_q = torch.cat((q[..., :48], zeros, q[..., 48:], zeros), dim=-1)
    rope = hf_muse_glimmer_mm._vision_rope_matrices(
        rotary.inv_freq, positions, 128, q.dtype
    ).unsqueeze(0)
    rotated = hf_muse_glimmer_mm._apply_vision_rope(
        padded_q.transpose(1, 2), rope
    ).transpose(1, 2)
    actual = torch.cat((rotated[..., :48], rotated[..., 64:112]), dim=-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("with_image", [False, True])
@torch.no_grad()
def test_generate_matches_stock(
    muse_model: MuseGlimmerForConditionalGeneration, with_image: bool, tmp_path: Path
) -> None:
    ids, image_inputs = _inputs(with_image)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected: GenerateDecoderOnlyOutput = muse_model.generate(
        ids, attention_mask=torch.ones_like(ids), **image_inputs, **generation_kwargs
    )
    muse_model.save_pretrained(tmp_path)
    model = auto_spyre_model.AutoSpyreModelForImageTextToText.from_pretrained(
        tmp_path, dtype=torch.float32
    )
    projected_rows: list[int] = []

    def record_head(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        projected_rows.append(inputs[0].shape[1])

    handle = model.lm_head.register_forward_pre_hook(record_head)
    try:
        actual: GenerateDecoderOnlyOutput = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            **image_inputs,
            **generation_kwargs,
        )
    finally:
        handle.remove()

    torch.testing.assert_close(actual.sequences, expected.sequences)
    assert projected_rows == [1, 1, 1]
    assert len(actual.logits) == len(expected.logits) == 3
    for got, want in zip(actual.logits, expected.logits):
        assert got.shape == (1, 129)
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("logits_to_keep", [0, 1, 3])
@torch.no_grad()
def test_prefill_returns_requested_logits(
    muse_model: MuseGlimmerForConditionalGeneration, logits_to_keep: int
) -> None:
    ids, image_inputs = _inputs(with_image=True)
    expected: torch.Tensor = muse_model(
        ids, use_cache=False, logits_to_keep=logits_to_keep, **image_inputs
    ).logits
    hf_muse_glimmer_mm.prepare_for_spyre(muse_model)
    hf_common.set_rope_dtype(muse_model, torch.float32)
    cache_len = 128
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    keys, values = hf_common.allocate_kv_caches(muse_model, 1, cache_len, torch.float32)
    offsets = torch.tensor([0])
    mask: torch.Tensor = hf_common.build_prefill_mask(
        1, ids.shape[1], cache_len, offsets, dtype=torch.float32
    )
    # Omitting the option must preserve the full-logit default of the VLM protocol.
    options = {"logits_to_keep": logits_to_keep} if logits_to_keep else {}
    actual = hf_muse_glimmer_mm._prefill_forward(
        model=muse_model,
        input_ids=ids,
        position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
        attention_mask=mask,
        key_caches=keys,
        value_caches=values,
        cache_index=hf_common.make_cache_index(0, ids.shape[1], "cpu"),
        **image_inputs,
        **options,
    )
    assert actual.shape == (1, logits_to_keep or ids.shape[1], 129)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)


@torch.no_grad()
def test_mixed_image_batch_matches_stock(
    muse_model: MuseGlimmerForConditionalGeneration,
) -> None:
    # Unequal image grids cross vision windows and have different merge counts.
    # Row 0 has two images; row 1 is a shorter, left-padded text-only prompt.
    ids = torch.randint(3, 126, (2, 73))
    ids[0, 1:13] = 127
    ids[1, :19] = 0
    mask = torch.ones_like(ids)
    mask[1, :19] = 0
    inputs: dict[str, torch.Tensor] = {
        "pixel_values": torch.randn(48, 24),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 8]]),
    }
    options: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected: GenerateDecoderOnlyOutput = muse_model.generate(
        ids, attention_mask=mask, **inputs, **options
    )
    hf_muse_glimmer_mm.prepare_for_spyre(muse_model)
    hf_common.set_rope_dtype(muse_model, torch.float32)
    actual: GenerateDecoderOnlyOutput = auto_spyre_model._generate_image_text_to_text(
        hf_muse_glimmer_mm, muse_model, ids, attention_mask=mask, **inputs, **options
    )
    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("missing", ["pixel_values", "image_grid_thw"])
@torch.no_grad()
def test_image_inputs_must_be_paired(
    muse_model: MuseGlimmerForConditionalGeneration, missing: str
) -> None:
    ids, inputs = _inputs(with_image=True)
    del inputs[missing]
    hf_muse_glimmer_mm.prepare_for_spyre(muse_model)
    with pytest.raises(ValueError, match="must be supplied together"):
        auto_spyre_model._generate_image_text_to_text(
            hf_muse_glimmer_mm, muse_model, ids, max_new_tokens=1, **inputs
        )


@torch.no_grad()
def test_image_feature_count_must_match_placeholders(
    muse_model: MuseGlimmerForConditionalGeneration,
) -> None:
    ids, inputs = _inputs(with_image=True)
    ids[0, 1] = 3
    hf_muse_glimmer_mm.prepare_for_spyre(muse_model)
    with pytest.raises(
        ValueError, match="Image features and image tokens do not match"
    ):
        auto_spyre_model._generate_image_text_to_text(
            hf_muse_glimmer_mm, muse_model, ids, max_new_tokens=1, **inputs
        )
