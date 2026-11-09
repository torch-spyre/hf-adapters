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

"""HF dual-encoder test helpers for pooled features, retrieval, and joint logits.

All checkpoints use fixed-length text from their config, fp32 CPU references,
and the public Spyre loader.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, PreTrainedModel

from hf_adapters._model_loading import model_subfolder
from hf_adapters.auto_spyre_model import AutoSpyreModel, dtype_for_model_path
from tests._vision_helpers import _load_sample_image
from tests.model_registry import REMOTE_CODE_PATHS

ImageInput = Image.Image | list[Image.Image]

EMBEDDING_COSINE = 0.99
SCORE_ATOL = 0.005
RANKING_ATOL = 0.01
LOGIT_ATOL = 1.0
LOGIT_RTOL = 0.05


def _cpu_tensor(output: Any, name: str) -> torch.Tensor:
    value: Any = getattr(output, name, None)
    assert isinstance(value, torch.Tensor), f"Required {name} tensor is missing"
    return value.detach().cpu()


@dataclass
class EmbeddingOutputs:
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    logits: dict[str, torch.Tensor] = field(default_factory=dict)

    def add_features(self, name: str, output: Any) -> None:
        """Read pooled features from standalone calls and joint tower outputs."""
        assert output is not None, f"{name}: required tower output is missing"
        self.embeddings[name] = _cpu_tensor(output, "pooler_output")


@dataclass(frozen=True)
class PreparedEmbeddingInputs:
    text: dict[str, torch.Tensor] | None = None
    image: dict[str, torch.Tensor] | None = None


def prepare_inputs(
    model: PreTrainedModel, processor: Any, case: EmbeddingCase
) -> PreparedEmbeddingInputs:
    """Tokenize/resize once; CPU and Spyre consume the same prepared values."""
    text: dict[str, torch.Tensor] | None = None
    image: dict[str, torch.Tensor] | None = None
    if case.texts is not None:
        text_config: Any = getattr(model.config, "text_config", None)
        context_length: Any = getattr(text_config, "max_position_embeddings", None)
        assert (
            isinstance(context_length, int) and context_length > 0
        ), "HF dual encoders must declare text_config.max_position_embeddings"
        text = dict(
            processor(
                text=case.texts,
                padding="max_length",
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
        )
    if case.images is not None:
        image = dict(processor(images=case.images, return_tensors="pt"))
    return PreparedEmbeddingInputs(text, image)


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Protect the shared reference inputs from in-place mutations in a forward.
    return {name: value.clone() for name, value in inputs.items()}


class HuggingFaceEncoder:
    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model

    def encode(self, inputs: PreparedEmbeddingInputs) -> EmbeddingOutputs:
        outputs = EmbeddingOutputs()
        if inputs.text is not None:
            outputs.add_features(
                "text", self.model.get_text_features(**_clone_inputs(inputs.text))
            )
        if inputs.image is not None:
            outputs.add_features(
                "image", self.model.get_image_features(**_clone_inputs(inputs.image))
            )
        if inputs.text is not None and inputs.image is not None:
            joint: Any = self.model(
                **_clone_inputs(inputs.text), **_clone_inputs(inputs.image)
            )
            for modality in ("text", "image"):
                outputs.embeddings[f"joint_{modality}"] = _cpu_tensor(
                    joint, f"{modality}_embeds"
                )
                tower_name = "vision" if modality == "image" else "text"
                outputs.add_features(
                    f"joint_{modality}_pool",
                    getattr(joint, f"{tower_name}_model_output", None),
                )
            for name in ("logits_per_text", "logits_per_image"):
                outputs.logits[name] = _cpu_tensor(joint, name)
        return outputs


def load_encoder(
    model_path: str,
    *,
    spyre: bool,
    trust_remote_code: bool | None = None,
) -> HuggingFaceEncoder:
    """Use fp32 CPU references and the public Spyre loader's default dtype."""
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
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
                subfolder=model_subfolder(model_path),
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
            )
            .eval()
            .requires_grad_(False)
        )
    return HuggingFaceEncoder(model)


TEXT_PROMPTS: list[str] = [
    "Two dogs at the beach",
    "A cat in the snow",
    "A picture of London at night",
]


def _assert_matching_outputs(
    actual: EmbeddingOutputs, reference: EmbeddingOutputs
) -> None:
    """Require matching fields and shapes, nonempty matrices, and finite values."""
    assert reference.embeddings, "No embeddings were produced"
    for label in ("embeddings", "logits"):
        actual_fields: dict[str, torch.Tensor] = getattr(actual, label)
        reference_fields: dict[str, torch.Tensor] = getattr(reference, label)
        assert (
            actual_fields.keys() == reference_fields.keys()
        ), f"{label}: fields differ"
        for name, expected in reference_fields.items():
            value = actual_fields[name]
            assert value.ndim == 2, f"{name}: expected a matrix"
            assert value.shape == expected.shape, f"{name}: shape mismatch"
            assert value.numel() > 0, f"{name}: empty output"
            assert torch.isfinite(value).all(), f"{name}: non-finite Spyre output"
            assert torch.isfinite(expected).all(), f"{name}: non-finite CPU reference"


def _assert_cosine(
    label: str, actual: torch.Tensor, expected: torch.Tensor, threshold: float
) -> None:
    similarities = F.cosine_similarity(actual.float(), expected.float(), dim=-1)
    minimum = similarities.min().item()
    print(f"{label}: minimum cosine {minimum:.8f}")
    if minimum < threshold:
        raise AssertionError(f"{label}: cosine {minimum} below {threshold}")


def _assert_embeddings(actual: EmbeddingOutputs, reference: EmbeddingOutputs) -> None:
    _assert_matching_outputs(actual, reference)
    for name, expected in reference.embeddings.items():
        _assert_cosine(name, actual.embeddings[name], expected, EMBEDDING_COSINE)


def _assert_logits(actual: EmbeddingOutputs, reference: EmbeddingOutputs) -> None:
    _assert_matching_outputs(actual, reference)
    assert reference.logits.keys() == {"logits_per_text", "logits_per_image"}
    for name, expected in reference.logits.items():
        torch.testing.assert_close(
            actual.logits[name].float(),
            expected.float(),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
            msg=lambda message: f"{name}: {message}",
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
        separated = gaps > RANKING_ATOL
        print(
            f"CPU {direction} ranking: {reference.argsort(dim=dim, descending=True).tolist()}"
        )
        print(
            f"Spyre {direction} ranking: {actual.argsort(dim=dim, descending=True).tolist()}"
        )
        if not (actual_gaps[separated] > 0).all():
            raise AssertionError(f"{direction}: well-separated ranking differs")
    # Correct retrieval is required even if the best candidates are close.
    if image_case == "three_captions":
        for label, scores in (("CPU", reference), ("Spyre", actual)):
            if scores[:, 0].argmax().item() != 1:
                raise AssertionError(f"{label}: expected {TEXT_PROMPTS[1]!r} first")
    else:
        assert reference.shape[0] == reference.shape[1]
        correct_matches = torch.arange(reference.shape[0])
        for label, scores in (("CPU", reference), ("Spyre", actual)):
            for dim in (0, 1):
                if not torch.equal(scores.argmax(dim=dim), correct_matches):
                    raise AssertionError(f"{label}: incorrect image/caption match")
    if not torch.allclose(actual, reference, rtol=0.0, atol=SCORE_ATOL):
        maximum_error = (actual - reference).abs().max().item()
        raise AssertionError(f"Cosine score error {maximum_error} exceeds {SCORE_ATOL}")


@dataclass(frozen=True)
class EmbeddingCase:
    texts: list[str] | None = None
    images: ImageInput | None = None


def embedding_cases() -> dict[str, EmbeddingCase]:
    """Single-image cases only.

    Two-image vision batches return non-finite outputs on Spyre, so the
    cat/dog and uniform-colour pairs are excluded until issue #747 is fixed:
    https://github.com/torch-spyre/hf-adapters/issues/747
    """
    image: Image.Image = _load_sample_image()
    return {
        "text": EmbeddingCase(texts=TEXT_PROMPTS),
        "scalar_image": EmbeddingCase(images=image),
        "image_list": EmbeddingCase(images=[image]),
        "three_captions": EmbeddingCase(TEXT_PROMPTS, image),
    }


@contextmanager
def bounded_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    try:
        yield
    finally:
        torch.set_num_threads(previous)
