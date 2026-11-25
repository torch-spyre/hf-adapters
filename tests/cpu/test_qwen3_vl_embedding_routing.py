# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embedding checkpoint routing must not claim the generative Qwen3-VL family."""

from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig, Qwen3VLConfig, Qwen3VLTextConfig

from hf_adapters import auto_spyre_model, hf_qwen3_vl_embedding
from hf_adapters.hf_common import SpyreNoAdapterError, SpyreUnsupportedModelError
from tests.cpu.test_qwen3_vl_embedding import _config

EMBEDDING_PATH = "Qwen/Qwen3-VL-Embedding-2B"


def _checkpoint_config() -> Qwen3VLConfig:
    config = _config()
    # Embedding and generative checkpoints declare this same architecture.
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    return config


@pytest.mark.parametrize(
    "auto_class",
    [auto_spyre_model.AutoSpyreModel, auto_spyre_model.AutoSpyreModelForCausalLM],
)
@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-2B-Thinking",
        "Qwen/Qwen3-VL-Reranker-2B",
        "Qwen/Qwen3-VL-Embedding-8B",
    ],
)
def test_unclaimed_checkpoints_fail_before_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
    auto_class: type[auto_spyre_model.AutoSpyreModel],
    model_path: str,
) -> None:
    monkeypatch.setattr(
        auto_spyre_model, "_load_model_config", Mock(return_value=_checkpoint_config())
    )
    load = Mock(side_effect=AssertionError("Unsupported checkpoint reached HF loading"))
    monkeypatch.setattr(auto_spyre_model, "load_model_common", load)
    with pytest.raises(SpyreNoAdapterError):
        auto_class.from_pretrained(model_path, dtype=torch.bfloat16)
    load.assert_not_called()


def test_verified_embedding_checkpoint_is_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto_spyre_model, "_load_model_config", Mock(return_value=_checkpoint_config())
    )
    assert (
        auto_spyre_model.resolve_adapter_module(EMBEDDING_PATH) is hf_qwen3_vl_embedding
    )


def test_embedding_checkpoint_rejects_generation_before_hf_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto_spyre_model, "_load_model_config", Mock(return_value=_checkpoint_config())
    )
    load = Mock(
        side_effect=AssertionError("Embedding checkpoint reached causal HF loading")
    )
    monkeypatch.setattr(auto_spyre_model, "load_model_common", load)
    with pytest.raises(SpyreUnsupportedModelError, match="encoder-only"):
        auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(EMBEDDING_PATH)
    load.assert_not_called()


@pytest.mark.parametrize(
    "mapping",
    [
        auto_spyre_model.IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING,
        auto_spyre_model.SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    ],
)
def test_checkpoint_override_does_not_bypass_task_mapping(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[type[PretrainedConfig], ModuleType],
) -> None:
    monkeypatch.setattr(
        auto_spyre_model, "_load_model_config", Mock(return_value=_checkpoint_config())
    )
    with pytest.raises(SpyreNoAdapterError):
        auto_spyre_model.resolve_adapter_module(EMBEDDING_PATH, mapping=mapping)


def test_bare_qwen3_vl_text_config_is_not_globally_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: object = _checkpoint_config().text_config
    assert isinstance(config, Qwen3VLTextConfig)
    config.architectures = ["Qwen3VLTextModel"]
    monkeypatch.setattr(
        auto_spyre_model, "_load_model_config", Mock(return_value=config)
    )
    with pytest.raises(SpyreNoAdapterError):
        auto_spyre_model.resolve_adapter_module("example/qwen3-vl-text")
