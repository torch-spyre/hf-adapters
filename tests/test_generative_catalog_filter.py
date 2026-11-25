"""Offline tests for generation discovery; run with ``pytest --noconftest``."""

from types import ModuleType
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import ModelInfo
from transformers import AutoConfig, BertConfig, GPT2Config, PretrainedConfig

from utils import fetch_top_generative_models as generative
from utils.fetch_top_embedding_models import keep as keep_embedding
from utils.hf_model_catalog import Ranking


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_request(
        self: httpx.Client, request: httpx.Request, **kwargs: object
    ) -> httpx.Response:
        raise AssertionError(f"Unexpected HTTP request: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", reject_request)


def _model(model_id: str = "org/model", model_type: str = "gpt2") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        config={"model_type": model_type},
        library_name="sentence-transformers",
        siblings=[{"rfilename": "model.safetensors"}],
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
        "org/model", token="test-token", trust_remote_code=False
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
    # One config check per candidate, then the existing kept-row enrichment.
    assert request.call_count == len(models) + len(rows)
