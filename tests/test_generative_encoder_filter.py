"""Offline tests for issue #279; run with --noconftest."""

from unittest.mock import Mock

import pytest
from huggingface_hub import ModelInfo

from utils import fetch_top_generative_models as fetcher


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
def test_encoder_is_rejected_before_config_requests(
    monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    config_request = Mock(side_effect=AssertionError("must not load config"))
    monkeypatch.setattr(fetcher, "contains_remote_code", config_request)
    model = ModelInfo(
        id="org/encoder",
        config={"model_type": model_type},
        siblings=[{"rfilename": "model.safetensors"}],
    )

    assert fetcher.keep(model, False) == (False, "encoder-only architecture")
    config_request.assert_not_called()


@pytest.mark.parametrize("model_type", ["llama", "qwen3", "new_decoder", None])
def test_other_model_types_keep_existing_checks(
    monkeypatch: pytest.MonkeyPatch, model_type: str | None
) -> None:
    config_request = Mock(return_value=False)
    monkeypatch.setattr(fetcher, "contains_remote_code", config_request)
    model = ModelInfo(
        id="org/decoder",
        config={"model_type": model_type, "architectures": ["ModelForCausalLM"]},
        siblings=[{"rfilename": "model.safetensors"}],
    )

    assert fetcher.keep(model, False) == (True, "")
    config_request.assert_called_once_with(model)
