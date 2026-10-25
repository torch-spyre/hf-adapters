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

"""Public API drivers for the shared multimodal embedding accuracy scenarios.

HF dual encoders expose get_text_features/get_image_features and a joint forward.
SentenceTransformers exposes encode for strings and PIL images. New checkpoints
using either interface only need registry settings, not new test functions.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModel, AutoProcessor, PreTrainedModel

from hf_adapters.auto_spyre_model import AutoSpyreModel, dtype_for_model_path
from tests._vision_helpers import _load_sample_image
from tests.model_registry import (
    MULTIMODAL_EMBEDDING_MODELS,
    REMOTE_CODE_PATHS,
    VISION_MODELS,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

ImageInput = Image.Image | list[Image.Image]


class CosineAccuracyError(AssertionError):
    """Finite, correctly shaped outputs miss a cosine accuracy threshold."""


@dataclass(frozen=True)
class EmbeddingTestSettings:
    backend: str = "huggingface"
    embedding_cosine: float = 0.99
    hidden_state_cosine: float = 0.99
    score_atol: float | None = 0.005
    ranking_atol: float = 0.01
    logit_atol: float = 1.0
    logit_rtol: float = 0.05
    hidden_state_xfails: dict[str, str] = field(default_factory=dict)
    require_hidden_states: bool = False
    joint_forward: bool = True
    text_processor_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"padding": True, "truncation": True}
    )
    xfail_cases: dict[str, str] = field(default_factory=dict)


def settings_for_model(model_path: str) -> EmbeddingTestSettings:
    """Use checkpoint metadata; off-registry HF paths use the same defaults."""
    for registry in (MULTIMODAL_EMBEDDING_MODELS, VISION_MODELS):
        for info in registry.values():
            if info["path"] == model_path:
                return EmbeddingTestSettings(**info.get("multimodal_test", {}))
    return EmbeddingTestSettings()


@dataclass
class EmbeddingOutputs:
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    hidden_states: dict[str, torch.Tensor] = field(default_factory=dict)
    logits: dict[str, torch.Tensor] = field(default_factory=dict)

    def add_features(
        self, name: str, output: Any, *, require_hidden_states: bool
    ) -> None:
        """Accept tensor features or HF outputs containing pooler_output."""
        if output is None and require_hidden_states:
            raise AssertionError(f"{name}: required tower output is missing")
        pooled: Any = (
            output
            if isinstance(output, torch.Tensor)
            else getattr(output, "pooler_output", None)
        )
        if not isinstance(pooled, torch.Tensor):
            raise TypeError(
                f"{name}: expected a tensor or an output with pooler_output"
            )
        self.embeddings[name] = pooled.detach().cpu()
        hidden: Any = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            self.hidden_states[name] = hidden.detach().cpu()
        elif require_hidden_states:
            raise AssertionError(f"{name}: required last_hidden_state is missing")


class EmbeddingEncoder(Protocol):
    def encode(
        self, texts: list[str] | None, images: ImageInput | None
    ) -> EmbeddingOutputs: ...


class HuggingFaceEncoder:
    def __init__(
        self, model: PreTrainedModel, processor: Any, settings: EmbeddingTestSettings
    ) -> None:
        self.model = model
        self.processor = processor
        self.settings = settings

    def encode(
        self, texts: list[str] | None, images: ImageInput | None
    ) -> EmbeddingOutputs:
        outputs = EmbeddingOutputs()
        if texts is not None:
            text_inputs: dict[str, Any] = dict(
                self.processor(
                    text=texts,
                    return_tensors="pt",
                    **self.settings.text_processor_kwargs,
                )
            )
            outputs.add_features(
                "text",
                self.model.get_text_features(**text_inputs),
                require_hidden_states=self.settings.require_hidden_states,
            )
        if images is not None:
            image_inputs: dict[str, Any] = dict(
                self.processor(images=images, return_tensors="pt")
            )
            outputs.add_features(
                "image",
                self.model.get_image_features(**image_inputs),
                require_hidden_states=self.settings.require_hidden_states,
            )
        if texts is not None and images is not None and self.settings.joint_forward:
            joint: Any = self.model(**text_inputs, **image_inputs)
            for modality in ("text", "image"):
                embedding: torch.Tensor = getattr(joint, f"{modality}_embeds")
                outputs.embeddings[f"joint_{modality}"] = embedding.detach().cpu()
                tower_name = "vision" if modality == "image" else "text"
                tower: Any = getattr(joint, f"{tower_name}_model_output", None)
                if tower is not None or self.settings.require_hidden_states:
                    outputs.add_features(
                        f"joint_{modality}_pool",
                        tower,
                        require_hidden_states=self.settings.require_hidden_states,
                    )
            for name in ("logits_per_text", "logits_per_image"):
                logits: Any = getattr(joint, name, None)
                if isinstance(logits, torch.Tensor):
                    outputs.logits[name] = logits.detach().cpu()
        return outputs


class SentenceTransformerEncoder:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def encode(
        self, texts: list[str] | None, images: ImageInput | None
    ) -> EmbeddingOutputs:
        outputs = EmbeddingOutputs()
        for name, inputs in (("text", texts), ("image", images)):
            if inputs is not None:
                outputs.add_features(
                    name,
                    self.model.encode(inputs, convert_to_tensor=True),
                    require_hidden_states=False,
                )
        return outputs


def load_encoder(
    model_path: str,
    settings: EmbeddingTestSettings,
    *,
    spyre: bool,
    trust_remote_code: bool | None = None,
) -> EmbeddingEncoder:
    """Use fp32 CPU references and the public Spyre loader's default dtype."""
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    if settings.backend == "sentence_transformers":
        if settings.require_hidden_states:
            raise ValueError("SentenceTransformer.encode does not expose hidden states")
        from sentence_transformers import SentenceTransformer

        if spyre:
            import hf_adapters.st_backend  # noqa: F401 — register the public backend

            st_model = SentenceTransformer(
                model_path, backend="spyre", trust_remote_code=trust_remote_code
            )
        else:
            st_model = SentenceTransformer(
                model_path,
                device="cpu",
                trust_remote_code=trust_remote_code,
                model_kwargs={"dtype": torch.float32},
            )
        return SentenceTransformerEncoder(st_model)
    if settings.backend != "huggingface":
        raise ValueError(
            f"Unsupported multimodal embedding test backend: {settings.backend}"
        )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model: PreTrainedModel
    if spyre:
        dtype = dtype_for_model_path(
            model_path, target_device="spyre", trust_remote_code=trust_remote_code
        )
        model = AutoSpyreModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        assert (
            model.dtype == dtype
        ), f"Default Spyre dtype: expected {dtype}, got {model.dtype}"
        print(f"Spyre model dtype: {model.dtype}")
    else:
        model = (
            AutoModel.from_pretrained(
                model_path,
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
            )
            .eval()
            .requires_grad_(False)
        )
    return HuggingFaceEncoder(model, processor, settings)


TEXT_PROMPTS: list[str] = [
    "Two dogs at the beach",
    "A cat in the snow",
    "A picture of London at night",
]


def _images_and_captions(image_case: str) -> tuple[ImageInput, list[str]]:
    if image_case == "three_captions":
        return _load_sample_image(), TEXT_PROMPTS
    if image_case == "uniform":
        # Source dimensions are arbitrary; each model's processor resizes them.
        return (
            [
                Image.new("RGB", (256, 256), color=color)
                for color in ((200, 40, 20), (20, 40, 200))
            ],
            ["a red square", "a blue square"],
        )
    if image_case != "cat_dog":
        raise ValueError(f"Unknown image case: {image_case}")
    dog_path = hf_hub_download(
        "huggingface/documentation-images", "image_process_dog.png", repo_type="dataset"
    )
    with Image.open(dog_path) as source:
        dog = source.convert("RGB")
    return [_load_sample_image(), dog], ["a photo of a cat", "a photo of a dog"]


def _assert_cosine(
    label: str, actual: torch.Tensor, expected: torch.Tensor, threshold: float
) -> None:
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    assert actual.numel() > 0, f"{label}: empty output"
    assert torch.isfinite(actual).all(), f"{label}: non-finite Spyre output"
    assert torch.isfinite(expected).all(), f"{label}: non-finite CPU reference"
    similarities = F.cosine_similarity(actual.float(), expected.float(), dim=-1)
    minimum = similarities.min().item()
    print(f"{label}: minimum cosine {minimum:.8f}")
    if minimum < threshold:
        raise CosineAccuracyError(f"{label}: cosine {minimum} below {threshold}")


def _assert_embeddings(
    actual: EmbeddingOutputs,
    reference: EmbeddingOutputs,
    settings: EmbeddingTestSettings,
) -> None:
    assert (
        actual.embeddings.keys() == reference.embeddings.keys()
    ), "Embedding fields differ"
    assert reference.embeddings, "No embeddings were produced"
    for name, expected in reference.embeddings.items():
        _assert_cosine(
            name, actual.embeddings[name], expected, settings.embedding_cosine
        )
    assert actual.logits.keys() == reference.logits.keys(), "Logit fields differ"
    for name, expected in reference.logits.items():
        assert actual.logits[name].shape == expected.shape, f"{name}: shape mismatch"
        assert torch.isfinite(
            actual.logits[name]
        ).all(), f"{name}: non-finite Spyre logits"

    for name, expected in reference.logits.items():
        torch.testing.assert_close(
            actual.logits[name].float(),
            expected.float(),
            atol=settings.logit_atol,
            rtol=settings.logit_rtol,
            msg=lambda message: f"{name}: {message}",
        )


def _assert_hidden_states(
    actual: EmbeddingOutputs,
    reference: EmbeddingOutputs,
    settings: EmbeddingTestSettings,
    *,
    modality: str | None = None,
) -> None:
    assert (
        actual.hidden_states.keys() == reference.hidden_states.keys()
    ), "Hidden-state fields differ"
    for name, expected in reference.hidden_states.items():
        if modality is not None and name not in (modality, f"joint_{modality}_pool"):
            continue
        _assert_cosine(
            f"{name}.last_hidden_state",
            actual.hidden_states[name],
            expected,
            settings.hidden_state_cosine,
        )


def _cosine_scores(text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Accept either a scalar-image vector or a batch, independently of width."""
    text_rows = text.float().reshape(-1, text.shape[-1])
    image_rows = image.float().reshape(-1, image.shape[-1])
    return F.normalize(text_rows, dim=-1) @ F.normalize(image_rows, dim=-1).T


def _assert_retrieval(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    image_case: str,
    score_atol: float | None,
    ranking_atol: float = 0.01,
) -> None:
    assert actual.shape == reference.shape, "Score shape mismatch"
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    print(f"CPU cosine scores: {reference.tolist()}")
    print(f"Spyre cosine scores: {actual.tolist()}")
    for dim, direction in ((0, "image-to-text"), (1, "text-to-image")):
        ref_rows = reference.T if dim == 0 else reference
        actual_rows = actual.T if dim == 0 else actual
        gaps = ref_rows.unsqueeze(-1) - ref_rows.unsqueeze(-2)
        actual_gaps = actual_rows.unsqueeze(-1) - actual_rows.unsqueeze(-2)
        separated = gaps > ranking_atol
        print(
            f"CPU {direction} ranking: {reference.argsort(dim=dim, descending=True).tolist()}"
        )
        print(
            f"Spyre {direction} ranking: {actual.argsort(dim=dim, descending=True).tolist()}"
        )
        assert (
            actual_gaps[separated] > 0
        ).all(), f"{direction}: well-separated ranking differs"
    # Correct retrieval is required even if the best candidates are close.
    if image_case == "three_captions":
        for label, scores in (("CPU", reference), ("Spyre", actual)):
            assert (
                scores[:, 0].argmax().item() == 1
            ), f"{label}: expected {TEXT_PROMPTS[1]!r} first"
    else:
        assert reference.shape[0] == reference.shape[1]
        correct_matches = torch.arange(reference.shape[0])
        for label, scores in (("CPU", reference), ("Spyre", actual)):
            for dim in (0, 1):
                assert torch.equal(
                    scores.argmax(dim=dim), correct_matches
                ), f"{label}: incorrect image/caption match"
    if score_atol is not None:
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=score_atol)


@dataclass(frozen=True)
class EmbeddingCase:
    texts: list[str] | None = None
    images: ImageInput | None = None


def embedding_cases() -> dict[str, EmbeddingCase]:
    image = _load_sample_image()
    cases = {
        "text": EmbeddingCase(texts=TEXT_PROMPTS),
        "scalar_image": EmbeddingCase(images=image),
        "image_list": EmbeddingCase(images=[image]),
    }
    for name in ("cat_dog", "three_captions", "uniform"):
        images, captions = _images_and_captions(name)
        cases[name] = EmbeddingCase(captions, images)
    return cases


@contextmanager
def bounded_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass
class PreparedEmbeddingModel:
    settings: EmbeddingTestSettings
    encoder: EmbeddingEncoder
    cases: dict[str, EmbeddingCase]
    reference: dict[str, EmbeddingOutputs]
    actual: dict[str, EmbeddingOutputs] = field(default_factory=dict)

    def compare(self, case_name: str) -> tuple[EmbeddingOutputs, EmbeddingOutputs]:
        if case_name not in self.actual:
            case = self.cases[case_name]
            with torch.inference_mode():
                self.actual[case_name] = self.encoder.encode(case.texts, case.images)
        return self.actual[case_name], self.reference[case_name]


def prepare_model_cases(
    model_path: str,
    settings: EmbeddingTestSettings,
    cases: dict[str, EmbeddingCase],
    *,
    trust_remote_code: bool | None = None,
) -> PreparedEmbeddingModel:
    reference_encoder = load_encoder(
        model_path, settings, spyre=False, trust_remote_code=trust_remote_code
    )
    with torch.inference_mode():
        reference = {
            name: reference_encoder.encode(case.texts, case.images)
            for name, case in cases.items()
        }
    del reference_encoder
    gc.collect()
    encoder = load_encoder(
        model_path, settings, spyre=True, trust_remote_code=trust_remote_code
    )
    return PreparedEmbeddingModel(settings, encoder, cases, reference)
