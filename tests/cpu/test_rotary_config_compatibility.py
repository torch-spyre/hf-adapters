# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE is read from Transformers' canonical rope_parameters mapping."""

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import (
    Phi3Config,
    Phi3ForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


@pytest.mark.parametrize("family", ["phi3", "qwen3_5"])
@pytest.mark.parametrize("rotary_fraction", [0.25, 0.75, 1.0])
@torch.no_grad()
def test_partial_rotary_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, rotary_fraction: float
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.random.default_generator.manual_seed(522)
    config_cls, model_cls = {
        "phi3": (Phi3Config, Phi3ForCausalLM),
        "qwen3_5": (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
    }[family]
    config = config_cls(
        vocab_size=131,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=256,
        original_max_position_embeddings=256,
        eos_token_id=None,
        pad_token_id=0,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": rotary_fraction,
        },
    )
    reference = model_cls(config).eval()
    ids = torch.tensor([[1, 3, 5, 7], [1, 4, 6, 8]])
    kwargs: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected = reference.generate(ids, **kwargs)
    reference.save_pretrained(tmp_path)
    loaded = AutoSpyreModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    actual = loaded.generate(ids, **kwargs)
    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits, strict=True):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)


class _PerTypeRotary(torch.nn.Module):
    """Rotary module with Gemma-style ``<layer_type>_inv_freq`` buffers."""

    def __init__(self, rope_type: str | dict[str, str]) -> None:
        super().__init__()
        self.rope_type: str | dict[str, str] = rope_type
        self.register_buffer("sliding_attention_inv_freq", torch.ones(64))
        self.register_buffer("full_attention_inv_freq", torch.full((64,), 0.5))
        self.sliding_attention_attention_scaling: float = 1.0
        self.full_attention_attention_scaling: float = 2.0


def test_build_layer_type_ropes_reads_per_type_buffers() -> None:
    from hf_adapters.hf_common import PrecomputedRotaryEmbedding, build_layer_type_ropes

    rotary = _PerTypeRotary(
        {"sliding_attention": "default", "full_attention": "proportional"}
    )
    ropes: dict[str, PrecomputedRotaryEmbedding] = build_layer_type_ropes(
        rotary,
        ["sliding_attention", "full_attention", "sliding_attention"],
        "Test",
    )
    assert set(ropes) == {"sliding_attention", "full_attention"}
    full: torch.nn.Module = ropes["full_attention"].original
    torch.testing.assert_close(full.inv_freq, rotary.full_attention_inv_freq)
    assert full.attention_scaling == 2.0


@pytest.mark.parametrize(
    "rope_type,expected",
    [
        (
            {"sliding_attention": "default", "full_attention": "dynamic"},
            "full_attention=dynamic",
        ),
        (
            {"sliding_attention": "longrope", "full_attention": "default"},
            "sliding_attention=longrope",
        ),
        ("dynamic", r"\(dynamic\)"),
    ],
)
def test_reject_dynamic_rope(rope_type: str | dict[str, str], expected: str) -> None:
    from hf_adapters.hf_common import SpyreUnsupportedModelError, build_layer_type_ropes

    with pytest.raises(SpyreUnsupportedModelError, match=expected):
        build_layer_type_ropes(
            _PerTypeRotary(rope_type), ["sliding_attention", "full_attention"], "Test"
        )
