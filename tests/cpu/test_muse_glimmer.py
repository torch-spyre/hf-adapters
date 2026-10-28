# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Muse generation agrees with stock through prefill/cache policy transitions."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest
import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    GenerationConfig,
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from hf_adapters import hf_common, hf_gemma3, hf_muse_glimmer

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def _without_compile(function: _Callable, **kwargs: Any) -> _Callable:
    return function


@pytest.fixture(autouse=True)
def eager_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", _without_compile)


def _model() -> MuseGlimmerForConditionalGeneration:
    torch.random.default_generator.manual_seed(729)
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
            "layer_rope_theta": [500000.0, 0.0],
            "bos_token_id": 1,
            "eos_token_id": None,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "layer_types": ["full_attention"],
        },
        image_token_id=127,
        video_token_id=128,
        out_hidden_size=512,
        projector_hidden_size=128,
    )
    model = MuseGlimmerForConditionalGeneration(config).eval().requires_grad_(False)
    # Nonzero centered weights exercise the fp32 (1 + weight) multiply too.
    for layer in model.model.language_model.layers:
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            layer.get_submodule(name).weight.normal_(std=0.05)
    return model


@pytest.mark.parametrize("padding_side", ["left", "right"])
@torch.no_grad()
def test_generate_matches_stock_across_cache_modes_and_roll(
    padding_side: str,
) -> None:
    model = _model()
    requests: list[tuple[torch.Tensor, torch.Tensor]] = []
    references: list[list[GenerateDecoderOnlyOutput]] = []
    options: dict[str, Any] = {
        "max_new_tokens": 70,  # Cross a 64-token anchored-cache roll.
        "do_sample": False,
        "eos_token_id": None,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    for lengths in ((13, 9), (641, 577), (37, 17)):
        ids = torch.zeros(2, max(lengths), dtype=torch.long)
        mask = torch.zeros_like(ids)
        expected: list[GenerateDecoderOnlyOutput] = []
        for row, length in enumerate(lengths):
            tokens = torch.randint(3, 126, (1, length))
            start = ids.shape[1] - length if padding_side == "left" else 0
            ids[row, start : start + length] = tokens[0]
            mask[row, start : start + length] = 1
            # Stock generation uses the last input column, so compare each
            # unpadded row rather than asking HF to generate from right padding.
            expected.append(
                cast(GenerateDecoderOnlyOutput, model.generate(tokens, **options))
            )
        requests.append((ids, mask))
        references.append(expected)

    norm_type = type(model.model.language_model.layers[0].input_layernorm)
    original_norm = norm_type.forward
    hf_muse_glimmer.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    assert norm_type.forward is original_norm

    for index, ((ids, mask), expected) in enumerate(zip(requests, references)):
        actual = cast(
            GenerateDecoderOnlyOutput,
            hf_muse_glimmer.generate(model, ids, mask, **options),
        )
        torch.testing.assert_close(actual.sequences[:, : ids.shape[1]], ids)
        assert len(actual.logits) == 70
        for row, reference in enumerate(expected):
            torch.testing.assert_close(
                actual.sequences[row, -70:], reference.sequences[0, -70:]
            )
            for got, want in zip(actual.logits, reference.logits):
                assert got.shape == (2, 129)
                torch.testing.assert_close(got[row], want[0], atol=2e-5, rtol=2e-4)
        assert model._spyre_swa_mode == ("anchored" if index == 1 else None)
        if index == 1:
            assert model._spyre_swa_state.capacity == 576


@pytest.mark.parametrize(
    "length,width,config_chunk,options,expected_chunk,expected_padding",
    [
        (13, 13, None, {}, None, 64),
        (13, 640, None, {}, None, 64),
        (513, 513, None, {}, 512, 1024),
        (513, 513, 256, {"prefill_chunk_size": 128}, 128, 640),
        (513, 513, 128, {}, 128, 640),
        (513, 513, 128, {"prefill_chunk_size": None}, None, 576),
    ],
)
@torch.no_grad()
def test_prefill_policy_respects_real_length_and_explicit_configuration(
    length: int,
    width: int,
    config_chunk: int | None,
    options: dict[str, Any],
    expected_chunk: int | None,
    expected_padding: int,
) -> None:
    model = _model()
    hf_muse_glimmer.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    config = GenerationConfig.from_model_config(model.config)
    if config_chunk is not None:
        config.prefill_chunk_size = config_chunk
    ids = torch.zeros(1, width, dtype=torch.long)
    ids[:, :length] = torch.randint(3, 126, (1, length))
    mask = torch.zeros_like(ids)
    mask[:, :length] = 1
    output = cast(
        torch.Tensor,
        hf_muse_glimmer.generate(
            model,
            ids,
            mask,
            generation_config=config,
            max_new_tokens=2,
            do_sample=False,
            **options,
        ),
    )
    assert output.shape == (1, width + 2)
    torch.testing.assert_close(output[:, :width], ids)
    assert model._spyre_padded_prompt_len == expected_padding
    assert model._spyre_active_prefill_chunk_size == expected_chunk
    assert getattr(config, "prefill_chunk_size", None) == config_chunk


@pytest.mark.parametrize("already_untied", [False, True])
@torch.no_grad()
def test_placeholder_input_rows_preserve_tied_output_head(already_untied: bool) -> None:
    model = _model()
    embedding = model.model.language_model.embed_tokens
    model.lm_head.weight = embedding.weight
    original_head = model.lm_head.weight.clone()
    ids = torch.tensor([[127, 3, 128, 4]])
    options: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected = cast(GenerateDecoderOnlyOutput, model.generate(ids, **options))
    if already_untied:
        # AutoSpyreModel unties before prepare; preserve its TP bookkeeping.
        hf_common.untie_embedding_and_lm_head(model)
    hf_muse_glimmer.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    assert model._spyre_lm_head_was_tied
    torch.testing.assert_close(model.lm_head.weight[:129], original_head)
    for token_id in (127, 128):
        torch.testing.assert_close(embedding.weight[token_id], embedding.weight[0])
    actual = cast(
        GenerateDecoderOnlyOutput, hf_muse_glimmer.generate(model, ids, **options)
    )
    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)


@torch.no_grad()
def test_shared_gemma_path_still_matches_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.random.default_generator.manual_seed(3)
    config = Gemma3TextConfig(
        vocab_size=129,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=1024,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = Gemma3ForCausalLM(config).eval().requires_grad_(False)
    ids = torch.randint(3, 126, (1, 193))
    options: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected = cast(GenerateDecoderOnlyOutput, model.generate(ids, **options))
    norm_type = type(model.model.layers[0].input_layernorm)
    monkeypatch.setattr(norm_type, "forward", norm_type.forward)
    hf_gemma3.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    actual: GenerateDecoderOnlyOutput = hf_common.generate(
        hf_gemma3._run_forward,
        model,
        ids,
        prefill_backbone_fn=hf_gemma3._run_backbone_forward,
        **options,
    )
    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)
