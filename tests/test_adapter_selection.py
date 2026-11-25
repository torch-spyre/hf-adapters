"""Offline adapter dispatch tests; run with ``pytest --noconftest``."""

from types import ModuleType
from unittest.mock import Mock

import pytest
from transformers import Gemma4Config, Gemma4TextConfig, GPT2Config, PretrainedConfig

from hf_adapters import auto_spyre_model as auto


@pytest.fixture(autouse=True)
def no_loading_or_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auto, "_load_model_config", Mock(side_effect=AssertionError("Config load"))
    )
    monkeypatch.setattr(
        auto,
        "assert_spyre_dimensions",
        Mock(side_effect=AssertionError("Dimension validation")),
    )


def test_config_mapping_and_unknown_configs() -> None:
    assert auto.select_adapter_module(GPT2Config()) is auto.hf_gpt2
    assert auto.select_adapter_module(PretrainedConfig()) is None


def test_custom_mapping_overrides_default() -> None:
    module = ModuleType("custom_adapter")
    mapping: dict[type[PretrainedConfig], ModuleType] = {GPT2Config: module}

    assert auto.select_adapter_module(GPT2Config(), mapping) is module
    assert auto.select_adapter_module(GPT2Config(), {}) is None


def test_first_known_architecture_wins_over_config_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ModuleType("first_adapter")
    second = ModuleType("second_adapter")
    monkeypatch.setitem(auto.ARCH_TO_ADAPTER_MODULE_MAPPING, "First", first)
    monkeypatch.setitem(auto.ARCH_TO_ADAPTER_MODULE_MAPPING, "Second", second)
    config = GPT2Config(architectures=["Unknown", "First", "Second"])

    assert auto.select_adapter_module(config, {}) is first


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_gemma4_dense_and_moe_dispatch(nested: bool, moe: bool) -> None:
    text_config = Gemma4TextConfig(enable_moe_block=moe)
    config: PretrainedConfig = (
        Gemma4Config(text_config=text_config) if nested else text_config
    )

    expected: ModuleType = auto.hf_gemma4_moe if moe else auto.hf_gemma4
    assert auto.select_adapter_module(config) is expected


def test_gemma4_variant_does_not_replace_architecture_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(auto.ARCH_TO_ADAPTER_MODULE_MAPPING, "Override", auto.hf_gemma4)
    config = Gemma4TextConfig(enable_moe_block=True, architectures=["Override"])

    assert auto.select_adapter_module(config) is auto.hf_gemma4


def test_gemma4_variant_does_not_replace_custom_mapping() -> None:
    module = ModuleType("custom_adapter")
    mapping: dict[type[PretrainedConfig], ModuleType] = {Gemma4TextConfig: module}

    assert (
        auto.select_adapter_module(Gemma4TextConfig(enable_moe_block=True), mapping)
        is module
    )


def test_resolver_loads_then_selects_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GPT2Config()
    module = ModuleType("custom_adapter")
    module.SPYRE_DIMENSION_CONFIGS = ("text_config",)
    module.SPYRE_PADDED_DIMENSIONS = ("head_dim",)
    mapping: dict[type[PretrainedConfig], ModuleType] = {GPT2Config: module}
    load = Mock(return_value=config)
    select = Mock(wraps=auto.select_adapter_module)
    validate = Mock()
    monkeypatch.setattr(auto, "_load_model_config", load)
    monkeypatch.setattr(auto, "select_adapter_module", select)
    monkeypatch.setattr(auto, "assert_spyre_dimensions", validate)

    assert (
        auto.resolve_adapter_module("org/model", mapping, trust_remote_code=True)
        is module
    )
    load.assert_called_once_with("org/model", trust_remote_code=True)
    select.assert_called_once_with(config, mapping)
    validate.assert_called_once_with(
        config,
        model_name="org/model",
        config_names=module.SPYRE_DIMENSION_CONFIGS,
        padded_dimensions=module.SPYRE_PADDED_DIMENSIONS,
    )


@pytest.mark.parametrize("config", [None, PretrainedConfig()])
def test_resolver_keeps_unsupported_config_errors(
    monkeypatch: pytest.MonkeyPatch, config: PretrainedConfig | None
) -> None:
    monkeypatch.setattr(auto, "_load_model_config", Mock(return_value=config))
    message: str = "Could not load config" if config is None else "is not supported"

    with pytest.raises(auto.SpyreNoAdapterError, match=message):
        auto.resolve_adapter_module("org/model")
