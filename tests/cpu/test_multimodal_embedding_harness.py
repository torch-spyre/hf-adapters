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

"""Protect model-independent output handling and accuracy assertions."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from PIL import Image
from transformers import (
    CLIPConfig,
    CLIPModel,
    PreTrainedModel,
    SiglipConfig,
    SiglipModel,
)

from tests import _multimodal_embedding_helpers as helpers
from tests.model_registry import (
    CLIP_PATHS,
    MULTIMODAL_EMBED_PATHS,
    _merge_multimodal_models,
)


class TinyProcessor:
    def __call__(
        self,
        *,
        text: list[str] | None = None,
        images: helpers.ImageInput | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if text is not None:
            return {"input_ids": torch.tensor([[1, 4, 5, 2, 0]] * len(text))}
        count = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.ones(count, 3, 12, 12)}


@pytest.mark.parametrize("family", ["clip", "siglip"])
def test_hf_driver_accepts_different_dimensions_and_towers(family: str) -> None:
    text_config: dict[str, Any] = {
        "vocab_size": 16,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 8,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "projection_size": 24,
    }
    vision_config: dict[str, Any] = {
        "hidden_size": 24,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 12,
        "patch_size": 4,
    }
    model: PreTrainedModel
    if family == "clip":
        model = CLIPModel(
            CLIPConfig(
                text_config=text_config, vision_config=vision_config, projection_dim=7
            )
        )
    else:
        model = SiglipModel(
            SiglipConfig(text_config=text_config, vision_config=vision_config)
        )
    model.eval().requires_grad_(False)
    processor = TinyProcessor()
    encoder = helpers.HuggingFaceEncoder(
        model, processor, helpers.EmbeddingTestSettings()
    )
    captions = ["one", "two", "three"]
    images = [Image.new("RGB", (23, 31)), Image.new("RGB", (17, 19))]
    with torch.inference_mode():
        outputs = encoder.encode(captions, images)
        reference: Any = model(**processor(text=captions), **processor(images=images))
    dimension = 7 if family == "clip" else 24
    assert outputs.embeddings["text"].shape == (3, dimension)
    assert outputs.embeddings["image"].shape == (2, dimension)
    torch.testing.assert_close(outputs.embeddings["joint_text"], reference.text_embeds)
    torch.testing.assert_close(
        outputs.embeddings["joint_image"], reference.image_embeds
    )
    torch.testing.assert_close(
        outputs.hidden_states["joint_image_pool"],
        reference.vision_model_output.last_hidden_state,
    )
    assert helpers._cosine_scores(
        outputs.embeddings["text"], outputs.embeddings["image"]
    ).shape == (3, 2)


def test_tensor_features_and_required_hidden_states() -> None:
    outputs = helpers.EmbeddingOutputs()
    features = torch.randn(3, 11)
    outputs.add_features("text", features, require_hidden_states=False)
    torch.testing.assert_close(outputs.embeddings["text"], features)
    with pytest.raises(AssertionError, match="required last_hidden_state"):
        outputs.add_features("text", features, require_hidden_states=True)
    with pytest.raises(AssertionError, match="required tower output"):
        outputs.add_features("joint_image_pool", None, require_hidden_states=True)


def test_near_tied_distractors_can_swap_but_the_winner_cannot() -> None:
    reference = torch.tensor([[0.1000], [0.9], [0.1002]])
    actual = torch.tensor([[0.1003], [0.9], [0.1001]])
    helpers._assert_retrieval(
        actual, reference, image_case="three_captions", score_atol=0.005
    )
    with pytest.raises(AssertionError):
        helpers._assert_retrieval(
            torch.tensor([[0.91], [0.9], [0.1]]),
            reference,
            image_case="three_captions",
            score_atol=None,
        )


def test_logits_are_compared_to_reference_values() -> None:
    embedding = {"text": torch.ones(2, 7)}
    reference = helpers.EmbeddingOutputs(
        embedding, logits={"logits_per_text": torch.zeros(2, 2)}
    )
    actual = helpers.EmbeddingOutputs(
        embedding, logits={"logits_per_text": torch.full((2, 2), 4.0)}
    )
    with pytest.raises(AssertionError, match="logits_per_text"):
        helpers._assert_embeddings(actual, reference, helpers.EmbeddingTestSettings())


def test_registry_merge_rejects_key_collision() -> None:
    with pytest.raises(AssertionError, match="collisions"):
        _merge_multimodal_models(
            {"same": {"path": "first"}}, {"same": {"path": "second", "kind": "clip"}}
        )


def test_remote_code_registry_fallback_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool | None] = []

    def processor_loader(model_path: str, **kwargs: Any) -> TinyProcessor:
        observed.append(kwargs["trust_remote_code"])
        return TinyProcessor()

    def model_loader(model_path: str, **kwargs: Any) -> torch.nn.Module:
        observed.append(kwargs["trust_remote_code"])
        assert kwargs["dtype"] == torch.float32
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(helpers, "REMOTE_CODE_PATHS", frozenset({"custom/model"}))
    monkeypatch.setattr(helpers.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(helpers.AutoModel, "from_pretrained", model_loader)
    helpers.load_encoder("custom/model", helpers.EmbeddingTestSettings(), spyre=False)
    helpers.load_encoder(
        "custom/model",
        helpers.EmbeddingTestSettings(),
        spyre=False,
        trust_remote_code=False,
    )
    assert observed == [True, True, False, False]


def test_cpu_thread_limit_is_restored_after_failure() -> None:
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError), helpers.bounded_cpu_threads():
        assert torch.get_num_threads() <= 8
        raise RuntimeError("test")
    assert torch.get_num_threads() == before


def test_encoder_cannot_silently_omit_required_hidden_states() -> None:
    settings = helpers.EmbeddingTestSettings(
        backend="sentence_transformers", require_hidden_states=True
    )
    with pytest.raises(ValueError, match="does not expose hidden states"):
        helpers.load_encoder("new/model", settings, spyre=False)


def test_rankings_detect_swapped_distractors_with_same_top_match() -> None:
    reference = torch.tensor([[0.1], [0.9], [0.2]])
    actual = torch.tensor([[0.2], [0.9], [0.1]])
    with pytest.raises(AssertionError, match="well-separated ranking differs"):
        helpers._assert_retrieval(
            actual, reference, image_case="three_captions", score_atol=None
        )


def test_scalar_image_scores_normalize_arbitrary_embedding_width() -> None:
    text = torch.tensor([[1.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0]])
    image = torch.tensor([0, 7.0, 0, 0, 0])
    scores = helpers._cosine_scores(text, image)
    torch.testing.assert_close(scores, torch.tensor([[0.0], [1.0]]))


def test_hidden_state_drift_is_not_hidden_by_matching_embeddings() -> None:
    embeddings = {"image": torch.ones(1, 5)}
    reference = helpers.EmbeddingOutputs(embeddings, {"image": torch.ones(1, 3, 5)})
    actual = helpers.EmbeddingOutputs(embeddings, {"image": -torch.ones(1, 3, 5)})
    with pytest.raises(helpers.CosineAccuracyError, match="last_hidden_state"):
        helpers._assert_hidden_states(
            actual, reference, helpers.EmbeddingTestSettings()
        )


@pytest.mark.parametrize(
    "malformed", [torch.ones(1, 2, 5), torch.full((1, 3, 5), float("nan"))]
)
def test_vision_cosine_xfail_does_not_cover_invalid_outputs(
    malformed: torch.Tensor,
) -> None:
    reference = helpers.EmbeddingOutputs(hidden_states={"image": torch.ones(1, 3, 5)})
    actual = helpers.EmbeddingOutputs(hidden_states={"image": malformed})
    with pytest.raises(AssertionError) as failure:
        helpers._assert_hidden_states(
            actual, reference, helpers.EmbeddingTestSettings(), modality="image"
        )
    assert not isinstance(failure.value, helpers.CosineAccuracyError)


def test_registry_settings_scope_expected_failures_and_select_both_apis() -> None:
    assert set(CLIP_PATHS) <= set(MULTIMODAL_EMBED_PATHS)
    clip = helpers.settings_for_model("sentence-transformers/clip-ViT-B-32")
    siglip = helpers.settings_for_model("google/siglip2-giant-opt-patch16-384")
    other = helpers.settings_for_model("new-org/new-dual-encoder")
    assert clip.backend == "sentence_transformers"
    assert clip.embedding_cosine == 0.97
    assert clip.score_atol is None
    assert not clip.xfail_cases and not other.xfail_cases
    assert siglip.xfail_cases.keys() == {"uniform"}
    assert siglip.hidden_state_xfails.keys() == {"image"}
    assert not clip.hidden_state_xfails and not other.hidden_state_xfails
    assert siglip.require_hidden_states
    assert siglip.embedding_cosine == 0.99 and siglip.score_atol == 0.005
    assert other.backend == "huggingface"


def test_reference_finishes_before_device_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class RecordingEncoder:
        def __init__(self, spyre: bool) -> None:
            self.spyre = spyre

        def encode(
            self, texts: list[str] | None, images: helpers.ImageInput | None
        ) -> helpers.EmbeddingOutputs:
            events.append(f"encode:{self.spyre}")
            return helpers.EmbeddingOutputs({"text": torch.ones(3, 5)})

    def load(
        model_path: str,
        settings: helpers.EmbeddingTestSettings,
        *,
        spyre: bool,
        trust_remote_code: bool | None,
    ) -> RecordingEncoder:
        events.append(f"load:{spyre}")
        return RecordingEncoder(spyre)

    monkeypatch.setattr(helpers, "load_encoder", load)
    cases = {
        "first": helpers.EmbeddingCase(texts=["a", "b", "c"]),
        "second": helpers.EmbeddingCase(texts=["d"]),
    }
    prepared = helpers.prepare_model_cases(
        "example/model", helpers.EmbeddingTestSettings(), cases
    )
    prepared.compare("first")
    prepared.compare("second")
    prepared.compare("first")
    assert events == [
        "load:False",
        "encode:False",
        "encode:False",
        "load:True",
        "encode:True",
        "encode:True",
    ]


def test_loader_uses_dtype_policy_without_forcing_device_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2)
    calls: list[dict[str, Any]] = []

    class DtypeModel(torch.nn.Module):
        dtype = torch.bfloat16

    def dtype_policy(model_path: str, **kwargs: Any) -> torch.dtype:
        assert kwargs["target_device"] == "spyre"
        return torch.bfloat16

    def processor_loader(model_path: str, **kwargs: Any) -> TinyProcessor:
        return TinyProcessor()

    def reference_loader(model_path: str, **kwargs: Any) -> torch.nn.Module:
        assert kwargs["dtype"] == model.weight.dtype
        return model

    def device_loader(model_path: str, **kwargs: Any) -> DtypeModel:
        calls.append(kwargs)
        return DtypeModel()

    monkeypatch.setattr(helpers, "dtype_for_model_path", dtype_policy)
    monkeypatch.setattr(helpers.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(helpers.AutoModel, "from_pretrained", reference_loader)
    monkeypatch.setattr(helpers.AutoSpyreModel, "from_pretrained", device_loader)
    settings = helpers.EmbeddingTestSettings()
    helpers.load_encoder("new/model", settings, spyre=False)
    helpers.load_encoder("new/model", settings, spyre=True)
    assert len(calls) == 1 and "dtype" not in calls[0]
