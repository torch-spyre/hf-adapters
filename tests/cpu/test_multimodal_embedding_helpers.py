# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the shared HF test contract without downloading checkpoints or Spyre."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    CLIPConfig,
    CLIPModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    SiglipConfig,
    SiglipImageProcessor,
    SiglipModel,
    SiglipProcessor,
)

from tests import _multimodal_embedding_helpers as helpers
from tests.spyre import test_e2e_multimodal_embed_compare_spyre as e2e


@pytest.fixture(params=["siglip", "clip"])
def encoder_and_processor(
    request: pytest.FixtureRequest,
) -> tuple[PreTrainedModel, SiglipProcessor]:
    # Two architectures/context lengths keep the harness independent of SigLIP.
    context_length = 8 if request.param == "siglip" else 12
    text_config: dict[str, int] = {
        "vocab_size": 4,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "max_position_embeddings": context_length,
        "eos_token_id": 3,
    }
    vision_config: dict[str, int] = {
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_size": 16,
        "patch_size": 4,
    }
    model: PreTrainedModel
    if request.param == "siglip":
        model = SiglipModel(
            SiglipConfig(text_config=text_config, vision_config=vision_config)
        )
    else:
        model = CLIPModel(
            CLIPConfig(
                text_config=text_config,
                vision_config=vision_config,
                projection_dim=24,
            )
        )
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "cat": 2, "dog": 3}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    )
    processor = SiglipProcessor(
        image_processor=SiglipImageProcessor(size={"height": 16, "width": 16}),
        tokenizer=fast_tokenizer,
    )
    return model.eval().requires_grad_(False), processor


def test_e2e_prepares_once_and_reuses_inputs(
    monkeypatch: pytest.MonkeyPatch,
    encoder_and_processor: tuple[PreTrainedModel, SiglipProcessor],
    tmp_path: Path,
) -> None:
    model, processor = encoder_and_processor
    context_length: int = model.config.text_config.max_position_embeddings
    case = helpers.EmbeddingCase(
        texts=["cat", "dog " * (context_length + 3)],
        images=[Image.new("RGB", (19, 21), "red"), Image.new("RGB", (17, 20), "blue")],
    )
    events: list[str] = []
    observed: list[torch.Tensor] = []
    original_get_text = model.get_text_features

    def mutating_get_text(**inputs: torch.Tensor) -> Any:
        events.append("encode")
        observed.append(inputs["input_ids"].clone())
        result: Any = original_get_text(**inputs)
        inputs["input_ids"].zero_()
        return result

    monkeypatch.setattr(model, "get_text_features", mutating_get_text)

    def load_encoder(
        model_path: str, *, spyre: bool, trust_remote_code: bool | None
    ) -> helpers.HuggingFaceEncoder:
        events.append("load_spyre" if spyre else "load_cpu")
        return helpers.HuggingFaceEncoder(model)

    processor_spy = Mock(wraps=processor)
    embedding_check = Mock(wraps=helpers._assert_embeddings)
    logit_check = Mock(wraps=helpers._assert_logits)
    # Random tiny checkpoints have no learned image/caption ranking. The ranking
    # assertion is exercised separately with controlled scores below.
    retrieval_check = Mock()
    cases: dict[str, helpers.EmbeddingCase] = {
        "text": helpers.EmbeddingCase(texts=case.texts),
        "scalar_image": helpers.EmbeddingCase(images=Image.new("RGB", (19, 21), "red")),
        "image_list": helpers.EmbeddingCase(images=[Image.new("RGB", (19, 21), "red")]),
        "batch": case,
    }
    monkeypatch.setattr(e2e, "load_encoder", load_encoder)
    monkeypatch.setattr(e2e, "embedding_cases", Mock(return_value=cases))
    monkeypatch.setattr(e2e, "_assert_embeddings", embedding_check)
    monkeypatch.setattr(e2e, "_assert_logits", logit_check)
    monkeypatch.setattr(e2e, "_assert_retrieval", retrieval_check)
    monkeypatch.setattr(
        e2e.AutoProcessor, "from_pretrained", Mock(return_value=processor_spy)
    )
    e2e.test_multimodal_embeddings_compare_spyre(str(tmp_path), False)
    assert events[0] == "load_cpu"
    assert events.count("load_cpu") == events.count("load_spyre") == 1
    spyre_load = events.index("load_spyre")
    assert set(events[1:spyre_load]) == {"encode"}
    assert events[spyre_load + 1 :] == events[1:spyre_load]
    assert (
        processor_spy.call_count == 5
    )  # Once per case/modality, shared across devices.
    assert embedding_check.call_count == len(cases)
    assert logit_check.call_count == 1
    assert retrieval_check.call_count == 2  # Standalone and joint features.
    assert observed[0].shape == (2, context_length)
    assert observed[0][0].tolist() == [2] + [0] * (context_length - 1)
    assert observed[0][1].tolist() == [3] * context_length
    for input_ids in observed[1:]:
        torch.testing.assert_close(observed[0], input_ids)
    actual, reference = logit_check.call_args.args
    # A broken scale/bias must fail even when the embeddings are identical.
    wrong_logits = copy.deepcopy(actual)
    wrong_logits.logits["logits_per_text"] += 5.0
    with pytest.raises(AssertionError):
        helpers._assert_logits(wrong_logits, reference)


def test_sentence_transformers_loader_preserves_native_feature_apis(
    monkeypatch: pytest.MonkeyPatch,
    encoder_and_processor: tuple[PreTrainedModel, SiglipProcessor],
) -> None:
    pytest.importorskip("sentence_transformers")
    from hf_adapters import st_backend

    model, processor = encoder_and_processor
    inputs = helpers.prepare_inputs(
        model,
        processor,
        helpers.EmbeddingCase(texts=["cat"], images=Image.new("RGB", (16, 16), "red")),
    )
    expected = helpers.HuggingFaceEncoder(model).encode(inputs)
    original_forward = model.forward
    monkeypatch.setattr(
        st_backend.AutoSpyreModel, "from_pretrained", Mock(return_value=model)
    )
    monkeypatch.setattr(
        st_backend,
        "resolve_adapter_module",
        Mock(
            side_effect=AssertionError(
                "Native feature APIs must not use a token-prefill driver"
            )
        ),
    )
    loaded = st_backend._spyre_load_model(
        SimpleNamespace(),
        "test/dual-encoder",
        "feature-extraction",
        model.config,
        "spyre",
        False,
    )
    assert loaded.forward == original_forward
    actual = helpers.HuggingFaceEncoder(loaded).encode(inputs)
    helpers._assert_embeddings(actual, expected)
    helpers._assert_logits(actual, expected)


@pytest.mark.parametrize("output", [None, torch.ones(1, 3), SimpleNamespace()])
def test_missing_pooled_outputs_are_blocking(output: Any) -> None:
    with pytest.raises(AssertionError):
        helpers.EmbeddingOutputs().add_features("image", output)


@pytest.mark.parametrize("field", ["embeddings", "logits"])
@pytest.mark.parametrize("problem", ["missing", "shape", "nan", "empty"])
def test_invalid_outputs_are_blocking(field: str, problem: str) -> None:
    reference = helpers.EmbeddingOutputs(
        embeddings={"image": torch.ones(1, 3)},
        logits={"logits_per_image": torch.ones(1, 3)},
    )
    actual = copy.deepcopy(reference)
    values: dict[str, torch.Tensor] = getattr(actual, field)
    key = next(iter(values))
    if problem == "missing":
        values.pop(key)
    elif problem == "shape":
        values[key] = torch.ones(3)
    elif problem == "nan":
        values[key][0, 0] = float("nan")
    else:
        values[key] = torch.ones(0, 3)
        getattr(reference, field)[key] = torch.ones(0, 3)
    with pytest.raises(AssertionError):
        helpers._assert_embeddings(actual, reference)


def test_inaccurate_embeddings_are_blocking() -> None:
    reference = helpers.EmbeddingOutputs(embeddings={"image": torch.ones(1, 3)})
    actual = helpers.EmbeddingOutputs(embeddings={"image": -torch.ones(1, 3)})
    with pytest.raises(AssertionError, match="cosine"):
        helpers._assert_embeddings(actual, reference)


def test_retrieval_allows_close_distractors_but_requires_correct_match() -> None:
    reference = torch.tensor([[0.1], [0.3], [0.102]])
    actual = torch.tensor([[0.102], [0.3], [0.1]])
    helpers._assert_retrieval(actual, reference, image_case="three_captions")
    actual[0, 0] = 0.4
    with pytest.raises(AssertionError):
        helpers._assert_retrieval(actual, reference, image_case="three_captions")


@pytest.mark.parametrize(
    ("subfolder", "module_type"),
    [
        ("", None),
        ("", "sentence_transformers.base.modules.Transformer"),
        ("0_CLIPModel", "sentence_transformers.models.CLIPModel"),
        ("custom_encoder", "sentence_transformers.models.Transformer"),
    ],
)
def test_clip_checkpoint_and_processor_subfolder(
    tmp_path: Path, subfolder: str, module_type: str | None
) -> None:
    from transformers import AutoProcessor

    config = CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 8,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 16,
            "patch_size": 4,
        },
        projection_dim=16,
    )
    model = CLIPModel(config)
    checkpoint = tmp_path / subfolder
    model.save_pretrained(checkpoint)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    processor = SiglipProcessor(
        image_processor=SiglipImageProcessor(size={"height": 16, "width": 16}),
        tokenizer=PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]"
        ),
    )
    processor.save_pretrained(checkpoint)
    if module_type is not None:
        (tmp_path / "modules.json").write_text(
            json.dumps(
                [
                    {"idx": 0, "name": "0", "path": subfolder, "type": module_type},
                    {
                        "idx": 1,
                        "name": "1",
                        "path": "1_Normalize",
                        "type": "sentence_transformers.models.Normalize",
                    },
                ]
            ),
            encoding="utf-8",
        )
    model_path = str(tmp_path)
    encoder = helpers.load_encoder(model_path, spyre=False)
    assert isinstance(encoder.model, CLIPModel)
    torch.testing.assert_close(
        encoder.model.text_projection.weight, model.text_projection.weight
    )
    assert isinstance(
        AutoProcessor.from_pretrained(
            model_path,
            subfolder=helpers.model_subfolder(model_path),
            local_files_only=True,
        ),
        SiglipProcessor,
    )


@pytest.mark.parametrize("has_metadata", [False, True])
def test_model_subfolder_uses_offline_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_metadata: bool
) -> None:
    from transformers.utils import hub

    revision = "a" * 40
    repository = tmp_path / "models--test--dual-encoder"
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    refs = repository / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision, encoding="utf-8")
    if has_metadata:
        (snapshot / "modules.json").write_text(
            json.dumps(
                [
                    {
                        "idx": 0,
                        "name": "0",
                        "path": "cached_encoder",
                        "type": "sentence_transformers.models.CLIPModel",
                    }
                ]
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub, "is_offline_mode", Mock(return_value=True))
    assert helpers.model_subfolder("test/dual-encoder") == (
        "cached_encoder" if has_metadata else ""
    )
