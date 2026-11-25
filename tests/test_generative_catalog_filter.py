"""Offline tests for generation discovery; run with ``pytest --noconftest``."""

import json
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import ModelInfo
from transformers import (
    AutoConfig,
    BertConfig,
    Gemma4TextConfig,
    GPT2Config,
    PretrainedConfig,
)

from utils import fetch_top_generative_models as generative
from utils import hf_model_catalog as catalog
from utils.fetch_top_embedding_models import keep as keep_embedding
from utils.hf_model_catalog import Ranking


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    from hf_adapters import _model_loading

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(_model_loading, "cached_file", Mock(return_value=None))

    def reject_request(
        self: httpx.Client, request: httpx.Request, **kwargs: object
    ) -> httpx.Response:
        raise AssertionError(f"Unexpected HTTP request: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", reject_request)


def _model(
    model_id: str = "org/model", model_type: str = "gpt2", *, has_modules: bool = False
) -> ModelInfo:
    siblings: list[dict[str, str]] = [{"rfilename": "model.safetensors"}]
    if has_modules:
        siblings.append({"rfilename": "modules.json"})
    return ModelInfo(
        id=model_id,
        config={"model_type": model_type},
        library_name="sentence-transformers",
        siblings=siblings,
        downloads=100,
        created_at="2026-01-01T00:00:00.000Z",
    )


@pytest.mark.parametrize(
    "model_type",
    [
        "bert",
        "camembert",
        "clip",
        "distilbert",
        "modernbert",
        "mpnet",
        "roberta",
        "siglip",
        "xlm-roberta",
    ],
)
def test_encoder_only_configs_are_rejected(
    monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    config: PretrainedConfig = AutoConfig.for_model(model_type)
    request = Mock(return_value=config)
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", request)

    # Hub metadata need not identify the encoder: use the loaded config and the
    # adapter flag, including BERT/RoBERTa configs that HF can load as causal LMs.
    assert generative.keep(_model(), "test-token") == (
        False,
        "encoder-only architecture does not support generation",
    )
    request.assert_called_once_with(
        "org/model", subfolder="", token="test-token", trust_remote_code=False
    )


@pytest.mark.parametrize("config", [GPT2Config(), PretrainedConfig()])
def test_decoder_and_unmapped_configs_reach_workers(
    monkeypatch: pytest.MonkeyPatch, config: PretrainedConfig
) -> None:
    monkeypatch.setattr(
        generative.AutoConfig, "from_pretrained", Mock(return_value=config)
    )

    assert generative.keep(_model(), False) == (True, "")


@pytest.mark.parametrize("encoder_override", [True, False])
def test_architecture_dispatch_takes_precedence_over_config_dispatch(
    monkeypatch: pytest.MonkeyPatch, encoder_override: bool
) -> None:
    from hf_adapters.auto_spyre_model import ARCH_TO_ADAPTER_MODULE_MAPPING

    module = ModuleType("test_adapter")
    module._is_encoder_only = encoder_override
    config: PretrainedConfig = GPT2Config() if encoder_override else BertConfig()
    config.architectures = ["TestArchitecture"]
    monkeypatch.setitem(ARCH_TO_ADAPTER_MODULE_MAPPING, "TestArchitecture", module)
    monkeypatch.setattr(
        generative.AutoConfig, "from_pretrained", Mock(return_value=config)
    )

    kept, reason = generative.keep(_model(), False)

    assert kept is not encoder_override
    assert bool(reason) is encoder_override


def test_embedding_scan_still_accepts_encoder_only_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generative.AutoConfig, "from_pretrained", Mock(return_value=BertConfig())
    )

    assert keep_embedding(_model(model_type="bert"), False) == (True, "")


def test_fetcher_uses_gemma4_variant_selected_by_shared_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hf_adapters import auto_spyre_model as auto

    config = Gemma4TextConfig(enable_moe_block=True)
    monkeypatch.setattr(
        generative.AutoConfig, "from_pretrained", Mock(return_value=config)
    )
    # A variant's encoder flag must be checked after shared selection, rather
    # than inspecting only the base adapter in the config-class mapping.
    monkeypatch.setattr(auto.hf_gemma4_moe, "_is_encoder_only", True, raising=False)

    assert generative.keep(_model(), False) == (
        False,
        "encoder-only architecture does not support generation",
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ValueError("Remote code required"), "requires trust_remote_code"),
        (OSError("Config missing"), "requires trust_remote_code"),
        (RuntimeError("Bad config"), "exception during generation config"),
    ],
)
def test_config_errors_remain_model_level_rejections(
    monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
) -> None:
    monkeypatch.setattr(
        generative.AutoConfig, "from_pretrained", Mock(side_effect=error)
    )

    assert generative.keep(_model(), False) == (False, reason)


@pytest.mark.parametrize("ranking", ["downloads", "trending_score"])
def test_ranked_fetch_excludes_encoders_before_counting_top_k(
    monkeypatch: pytest.MonkeyPatch, ranking: Ranking
) -> None:
    from hf_adapters import _model_loading

    layout_request = Mock(side_effect=AssertionError("Unexpected layout lookup"))
    monkeypatch.setattr(_model_loading, "cached_file", layout_request)
    models: list[ModelInfo] = [
        _model("org/encoder", "bert"),
        _model("org/decoder-first"),
        _model("org/decoder-second"),
        _model("org/decoder-third"),
    ]
    listing = Mock(return_value=models)
    monkeypatch.setattr(generative.HfApi, "list_models", listing)

    def load_config(model_id: str, **kwargs: object) -> PretrainedConfig:
        return BertConfig() if model_id == "org/encoder" else GPT2Config()

    request = Mock(side_effect=load_config)
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", request)

    rows: list[dict[str, object]] = generative.fetch_top_generative_models(
        limit=2, ranking=ranking
    )

    assert [row["model_id"] for row in rows] == [
        "org/decoder-first",
        "org/decoder-second",
    ]
    assert [row["rank"] for row in rows] == [1, 2]
    assert listing.call_args.kwargs["sort"] == ranking
    assert "siblings" in listing.call_args.kwargs["expand"]
    assert [row["config_class"] for row in rows] == ["GPT2Config", "GPT2Config"]
    # Enrichment reuses the config class discovered during filtering.
    assert request.call_count == len(models)
    # Native repositories need no modules.json probe across all candidates.
    layout_request.assert_not_called()


@pytest.mark.parametrize("missing_siblings", [False, True])
def test_generation_config_handles_unavailable_and_empty_sibling_metadata(
    monkeypatch: pytest.MonkeyPatch, missing_siblings: bool
) -> None:
    from hf_adapters import _model_loading

    model: ModelInfo = _model()
    model.siblings = None if missing_siblings else []
    layout_request = Mock(return_value="0_Transformer")
    monkeypatch.setattr(_model_loading, "model_subfolder", layout_request)
    config_request = Mock(return_value=GPT2Config())
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", config_request)

    assert generative._keep_generation_config(model, False) == (True, "")

    if missing_siblings:
        layout_request.assert_called_once_with("org/model")
    else:
        layout_request.assert_not_called()
    expected_subfolder: str = "0_Transformer" if missing_siblings else ""
    config_request.assert_called_once_with(
        "org/model", subfolder=expected_subfolder, token=False, trust_remote_code=False
    )


def test_catalog_fetches_only_config_types_missing_from_filter_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Mock(return_value="BertConfig")
    monkeypatch.setattr(catalog, "get_config_type", request)

    rows: list[dict[str, object]] = catalog.build_catalog(
        fetch_fn=lambda limit: [_model("org/known"), _model("org/missing", "bert")],
        filter_fn=lambda model: (True, ""),
        limit=2,
        output_csv=None,
        label="generative",
        token=False,
        known_config_types={"org/known": "GPT2Config"},
    )

    assert [row["config_class"] for row in rows] == ["GPT2Config", "BertConfig"]
    request.assert_called_once_with("org/missing", False)


def test_curated_catalog_still_enriches_without_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generative.HfApi,
        "model_info",
        Mock(return_value=_model("org/encoder", "bert")),
    )
    request = Mock(return_value=BertConfig())
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", request)

    rows: list[dict[str, object]] = generative.fetch_curated_generative_models_metadata(
        ["org/encoder"]
    )

    assert [row["model_id"] for row in rows] == ["org/encoder"]
    assert rows[0]["config_class"] == "BertConfig"
    request.assert_called_once_with("org/encoder", token=False, trust_remote_code=False)


def test_config_metadata_is_not_reused_across_catalog_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generative.HfApi, "list_models", Mock(side_effect=lambda **kwargs: [_model()])
    )
    request = Mock(side_effect=[GPT2Config(), PretrainedConfig()])
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", request)

    first: list[dict[str, object]] = generative.fetch_top_generative_models(limit=1)
    second: list[dict[str, object]] = generative.fetch_top_generative_models(limit=1)

    assert first[0]["config_class"] == "GPT2Config"
    assert second[0]["config_class"] == PretrainedConfig.__name__
    assert request.call_count == 2


@pytest.mark.parametrize("nested_encoder", [False, True])
def test_filter_and_loader_use_the_same_nested_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nested_encoder: bool
) -> None:
    from transformers.utils.hub import cached_file

    from hf_adapters import _model_loading
    from hf_adapters import auto_spyre_model as auto

    root_config: PretrainedConfig = GPT2Config() if nested_encoder else BertConfig()
    nested_config: PretrainedConfig = BertConfig() if nested_encoder else GPT2Config()
    root_config.save_pretrained(tmp_path)
    nested_config.save_pretrained(tmp_path / "0_Transformer")
    modules: list[dict[str, object]] = [
        {
            "idx": 0,
            "name": "0",
            "path": "0_Transformer",
            "type": "sentence_transformers.models.Transformer",
        }
    ]
    (tmp_path / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    monkeypatch.setattr(_model_loading, "cached_file", cached_file)

    expected: ModuleType = auto.hf_bert if nested_encoder else auto.hf_gpt2
    assert auto.resolve_adapter_module(str(tmp_path)) is expected

    request = Mock(wraps=AutoConfig.from_pretrained)
    monkeypatch.setattr(generative.AutoConfig, "from_pretrained", request)
    monkeypatch.setattr(
        generative.HfApi,
        "list_models",
        Mock(
            return_value=[
                _model(str(tmp_path), root_config.model_type, has_modules=True)
            ]
        ),
    )

    rows: list[dict[str, object]] = generative.fetch_top_generative_models(limit=1)

    if nested_encoder:
        assert rows == []
    else:
        assert [row["model_id"] for row in rows] == [str(tmp_path)]
        assert rows[0]["config_class"] == "GPT2Config"
    # The nested config also supplies enrichment metadata without another load.
    request.assert_called_once_with(
        str(tmp_path), subfolder="0_Transformer", token=False, trust_remote_code=False
    )
