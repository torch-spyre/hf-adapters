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


def test_prepares_once_and_reuses_inputs(
    monkeypatch: pytest.MonkeyPatch,
    encoder_and_processor: tuple[PreTrainedModel, SiglipProcessor],
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
    monkeypatch.setattr(helpers, "load_encoder", load_encoder)
    monkeypatch.setattr(
        helpers.AutoProcessor, "from_pretrained", Mock(return_value=processor_spy)
    )
    run = helpers.prepare_model_cases("test/dual-encoder", {"batch": case})
    assert events[0] == "load_cpu" and events[-1] == "load_spyre"
    assert set(events[1:-1]) == {"encode"}
    actual, reference = run.compare("batch")
    first_run_events = events.copy()
    run.compare("batch")
    assert events == first_run_events  # All later checks reuse the first forward.
    assert processor_spy.call_count == 2  # Once per modality, not per device/forward.
    assert observed[0].shape == (2, context_length)
    assert observed[0][0].tolist() == [2] + [0] * (context_length - 1)
    assert observed[0][1].tolist() == [3] * context_length
    for input_ids in observed[1:]:
        torch.testing.assert_close(observed[0], input_ids)
    assert run.inputs["batch"].text is not None
    torch.testing.assert_close(run.inputs["batch"].text["input_ids"], observed[0])
    helpers._assert_embeddings(actual, reference)
    helpers._assert_hidden_states(actual, reference, modality="text")
    helpers._assert_hidden_states(actual, reference, modality="image")
    helpers._assert_logits(actual, reference)
    # A broken scale/bias must fail even when the embeddings are identical.
    wrong_logits = copy.deepcopy(actual)
    wrong_logits.logits["logits_per_text"] += 5.0
    with pytest.raises(AssertionError):
        helpers._assert_logits(wrong_logits, reference)


@pytest.mark.parametrize(
    "output", [None, torch.ones(1, 3), SimpleNamespace(pooler_output=torch.ones(1, 3))]
)
def test_missing_tower_outputs_are_blocking(output: Any) -> None:
    with pytest.raises(AssertionError) as error:
        helpers.EmbeddingOutputs().add_features("image", output)
    assert not isinstance(error.value, helpers.CosineAccuracyError)


@pytest.mark.parametrize("problem", ["missing", "shape", "nan"])
def test_structural_failures_precede_cosine_xfails(problem: str) -> None:
    reference = helpers.EmbeddingOutputs(
        embeddings={"text": torch.ones(1, 3), "image": torch.ones(1, 3)},
        hidden_states={"text": torch.ones(1, 4, 3), "image": torch.ones(1, 4, 3)},
    )
    actual = copy.deepcopy(reference)
    actual.embeddings["text"].neg_()  # Would otherwise trigger a cosine xfail first.
    if problem == "missing":
        actual.hidden_states.pop("image")
    elif problem == "shape":
        actual.hidden_states["image"] = torch.ones(1, 3, 3)
    else:
        actual.hidden_states["image"][0, 0, 0] = float("nan")
    with pytest.raises(AssertionError) as error:
        helpers._assert_embeddings(actual, reference)
    assert not isinstance(error.value, helpers.CosineAccuracyError)


def test_retrieval_allows_close_distractors_but_requires_correct_match() -> None:
    reference = torch.tensor([[0.1], [0.3], [0.102]])
    actual = torch.tensor([[0.102], [0.3], [0.1]])
    helpers._assert_retrieval(actual, reference, image_case="three_captions")
    actual[0, 0] = 0.4
    with pytest.raises(helpers.RetrievalAccuracyError):
        helpers._assert_retrieval(actual, reference, image_case="three_captions")
