"""Generic multimodal embedding accuracy and retrieval through the public ST API.

Covers every model registered in ``MULTIMODAL_EMBEDDING_MODELS``:

- **CPU vs Spyre**: encode captions, a sample image, and (when the registry
  declares ``supports_interleaved_media``) an interleaved caption + image input,
  then assert a per-input cosine threshold between CPU and ``backend="spyre"``.
- **Image-to-caption retrieval**: the sample image must rank the matching caption
  first, and CPU and Spyre must produce the same ranking.
- **Separate modality calls**: encode a PIL-image list, a single PIL image, then
  a padded text batch on the same model. Compare each output and the resulting
  retrieval ranking, retaining the original CLIP test's captions and API paths.
- **Interleaved retrieval**: the interleaved input must be closer to the sample
  image than to any unrelated caption, on both CPU and Spyre.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_multimodal_embed_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_multimodal_embed_compare_spyre.py \\
        --model-path google/embeddinggemma-2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

import hf_adapters.st_backend  # noqa: F401
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests._vision_helpers import SAMPLE_IMAGE, _load_sample_image
from tests.model_registry import MULTIMODAL_EMBED_PATHS, MULTIMODAL_EMBEDDING_MODELS

pytestmark = pytest.mark.model_harness("embedding")

COSINE_THRESHOLD = 0.99
CAPTIONS: tuple[str, ...] = (
    "A photo of a cat.",
    "Two dogs at the beach.",
    "A picture of London at night.",
)
MATCHING_CAPTION_INDEX = 0
# Preserve the original CLIP retrieval case as well as the mixed-input case.
SEPARATE_CAPTIONS: tuple[str, ...] = (
    "Two dogs at the beach",
    "A cat in the snow",
    "A picture of London at night",
)
SEPARATE_MATCHING_CAPTION_INDEX = 1
_MODEL_INFO_BY_PATH: dict[str, dict[str, Any]] = {
    cast(str, info["path"]): info for info in MULTIMODAL_EMBEDDING_MODELS.values()
}


@dataclass(frozen=True)
class MultimodalInputs:
    """Inputs and indices for one image-to-caption retrieval case."""

    image: Image.Image
    values: list[Any]
    labels: list[str]
    caption_indices: list[int]
    image_index: int
    interleaved_index: int | None


@dataclass(frozen=True)
class SeparateEmbeddings:
    """Outputs from repeated public-API calls on the same model instance."""

    image_batch: torch.Tensor
    single_image: torch.Tensor
    text_batch: torch.Tensor


def _encode_separate_modalities(
    model: SentenceTransformer, image: Image.Image
) -> SeparateEmbeddings:
    """Exercise PIL input forms, then switch towers with a multi-example batch."""
    image_batch: torch.Tensor = model.encode([image], convert_to_tensor=True).cpu()
    single_image: torch.Tensor = model.encode(image, convert_to_tensor=True).cpu()
    text_batch: torch.Tensor = model.encode(
        list(SEPARATE_CAPTIONS),
        batch_size=len(SEPARATE_CAPTIONS),
        convert_to_tensor=True,
    ).cpu()
    return SeparateEmbeddings(image_batch, single_image, text_batch)


def _build_inputs(model_path: str) -> MultimodalInputs:
    """Build a semantic retrieval case from registry-declared capabilities."""
    image = _load_sample_image()
    inputs: list[Any] = list(CAPTIONS)
    labels: list[str] = [f"caption: {caption}" for caption in CAPTIONS]
    caption_indices: list[int] = list(range(len(CAPTIONS)))
    image_index = len(inputs)
    inputs.append({"image": image})
    labels.append(f"image: {SAMPLE_IMAGE['filename']}")

    interleaved_index: int | None = None
    model_info: dict[str, Any] = _MODEL_INFO_BY_PATH.get(model_path, {})
    if model_info.get("supports_interleaved_media", False):
        placeholder: Any = model_info.get("interleaved_media_placeholder")
        if not isinstance(placeholder, str) or not placeholder:
            raise ValueError(
                f"{model_path} supports interleaved media but has no placeholder"
            )
        matching_caption: str = CAPTIONS[MATCHING_CAPTION_INDEX]
        interleaved_index = len(inputs)
        inputs.append({"text": f"{matching_caption} {placeholder}", "image": image})
        labels.append(f"interleaved: {matching_caption} + image")

    return MultimodalInputs(
        image=image,
        values=inputs,
        labels=labels,
        caption_indices=caption_indices,
        image_index=image_index,
        interleaved_index=interleaved_index,
    )


def _scores(
    embeddings: torch.Tensor, query_index: int, candidate_indices: list[int]
) -> torch.Tensor:
    """Cosine similarity between one query embedding and a list of candidates."""
    return F.cosine_similarity(
        embeddings[query_index].float().unsqueeze(0),
        embeddings[torch.tensor(candidate_indices)].float(),
        dim=-1,
    )


def _assert_embeddings_close(
    expected: torch.Tensor,
    actual: torch.Tensor,
    labels: list[str],
    title: str,
) -> None:
    """Check output shape and per-input CPU/Spyre embedding agreement."""
    assert actual.shape == expected.shape, (
        f"{title}: shape mismatch: CPU {tuple(expected.shape)} "
        f"vs Spyre {tuple(actual.shape)}"
    )
    assert actual.ndim == 2, f"{title}: expected a batch of embeddings"
    assert actual.shape[0] == len(labels), f"{title}: unexpected batch size"
    assert torch.isfinite(expected).all(), f"{title}: non-finite CPU embeddings"
    assert torch.isfinite(actual).all(), f"{title}: non-finite Spyre embeddings"
    cosine: torch.Tensor = F.cosine_similarity(expected.float(), actual.float(), dim=-1)
    print(f"\n## {title}: CPU vs Spyre\n")
    print("| Input | Cosine | Match |")
    print("|-------|--------|-------|")
    mismatches: list[str] = []
    for label, similarity in zip(labels, cosine.tolist(), strict=True):
        match: bool = similarity >= COSINE_THRESHOLD
        print(f"| {label!r} | {similarity:.6f} | {'OK' if match else 'FAIL'} |")
        if not match:
            mismatches.append(f"{label} ({similarity:.6f})")
    assert (
        not mismatches
    ), f"{title}: CPU/Spyre cosine < {COSINE_THRESHOLD} for: " + ", ".join(mismatches)


def _assert_caption_ranking(
    cpu_scores: torch.Tensor,
    spyre_scores: torch.Tensor,
    captions: tuple[str, ...],
    matching_index: int,
) -> None:
    """Require the full reference ranking and the semantically correct winner."""
    cpu_ranking: list[int] = cpu_scores.argsort(descending=True).tolist()
    spyre_ranking: list[int] = spyre_scores.argsort(descending=True).tolist()

    print("\n## Image-to-caption Retrieval\n")
    print("| Caption | CPU score | Spyre score |")
    print("|---------|-----------|-------------|")
    for index, caption in enumerate(captions):
        print(
            f"| {caption!r} | {cpu_scores[index].item():.4f} "
            f"| {spyre_scores[index].item():.4f} |"
        )
    print(f"\nCPU ranking:   {[captions[index] for index in cpu_ranking]}")
    print(f"Spyre ranking: {[captions[index] for index in spyre_ranking]}")

    assert cpu_ranking[0] == matching_index, (
        f"Expected {captions[matching_index]!r} to rank first for the cat "
        f"image on CPU, got {captions[cpu_ranking[0]]!r}"
    )
    assert spyre_ranking[0] == matching_index, (
        f"Expected {captions[matching_index]!r} to rank first for the cat "
        f"image on Spyre, got {captions[spyre_ranking[0]]!r}"
    )
    assert cpu_ranking == spyre_ranking, (
        "Image-to-caption ranking differs between CPU and Spyre.\n"
        f"  CPU:   {cpu_ranking}\n"
        f"  Spyre: {spyre_ranking}"
    )


def _assert_separate_embeddings(
    expected: SeparateEmbeddings, actual: SeparateEmbeddings
) -> None:
    """Cover the original text, image and cross-modal CLIP assertions."""
    for device, outputs in (("CPU", expected), ("Spyre", actual)):
        assert (
            outputs.single_image.ndim == 1
        ), f"{device}: encode(PIL_image) must return one unbatched embedding"
        width = outputs.single_image.shape[0]
        assert outputs.image_batch.shape == (
            1,
            width,
        ), f"{device}: encode([PIL_image]) must return a one-row batch"
        assert outputs.text_batch.shape == (
            len(SEPARATE_CAPTIONS),
            width,
        ), f"{device}: text batch must have one image-compatible embedding per caption"

    image_labels: list[str] = [SAMPLE_IMAGE["filename"]]
    _assert_embeddings_close(
        expected.image_batch, actual.image_batch, image_labels, "PIL Image Batch"
    )
    _assert_embeddings_close(
        expected.single_image.unsqueeze(0),
        actual.single_image.unsqueeze(0),
        image_labels,
        "Single PIL Image",
    )
    _assert_embeddings_close(
        expected.text_batch, actual.text_batch, list(SEPARATE_CAPTIONS), "Text Batch"
    )
    cpu_scores: torch.Tensor = F.cosine_similarity(
        expected.single_image.float().unsqueeze(0), expected.text_batch.float(), dim=-1
    )
    spyre_scores: torch.Tensor = F.cosine_similarity(
        actual.single_image.float().unsqueeze(0), actual.text_batch.float(), dim=-1
    )
    _assert_caption_ranking(
        cpu_scores, spyre_scores, SEPARATE_CAPTIONS, SEPARATE_MATCHING_CAPTION_INDEX
    )


@pytest.mark.parametrize(
    "model_path", MULTIMODAL_EMBED_PATHS, ids=MULTIMODAL_EMBED_PATHS
)
def test_multimodal_embedding_compare_spyre(model_path: str) -> None:
    """Compare registered multimodal inputs and retrieval results with CPU."""
    if model_path not in _MODEL_INFO_BY_PATH:
        pytest.skip(f"{model_path} does not declare image embedding support")

    inputs: MultimodalInputs = _build_inputs(model_path)
    cpu_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="cpu")
    cpu_model: SentenceTransformer = SentenceTransformer(
        model_path, device="cpu", model_kwargs={"dtype": cpu_dtype}
    )
    expected: torch.Tensor = cpu_model.encode(
        inputs.values, batch_size=1, convert_to_tensor=True
    ).cpu()
    expected_separate: SeparateEmbeddings = _encode_separate_modalities(
        cpu_model, inputs.image
    )
    del cpu_model

    spyre_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="spyre")
    spyre_model: SentenceTransformer = SentenceTransformer(
        model_path, backend="spyre", model_kwargs={"dtype": spyre_dtype}
    )
    actual: torch.Tensor = spyre_model.encode(
        inputs.values, batch_size=1, convert_to_tensor=True
    ).cpu()
    actual_separate: SeparateEmbeddings = _encode_separate_modalities(
        spyre_model, inputs.image
    )

    _assert_embeddings_close(expected, actual, inputs.labels, "Multimodal Embeddings")

    cpu_scores: torch.Tensor = _scores(
        expected, inputs.image_index, inputs.caption_indices
    )
    spyre_scores: torch.Tensor = _scores(
        actual, inputs.image_index, inputs.caption_indices
    )

    _assert_caption_ranking(cpu_scores, spyre_scores, CAPTIONS, MATCHING_CAPTION_INDEX)
    _assert_separate_embeddings(expected_separate, actual_separate)

    if inputs.interleaved_index is None:
        return

    # The interleaved input pairs the matching caption with the sample image, so
    # it must sit closer to that image than to any unrelated caption. This checks
    # the interleaved path is semantically right, not just CPU/Spyre-consistent.
    unrelated_indices: list[int] = [
        index for index in inputs.caption_indices if index != MATCHING_CAPTION_INDEX
    ]
    candidate_indices: list[int] = [inputs.image_index, *unrelated_indices]
    candidate_labels: list[str] = [inputs.labels[index] for index in candidate_indices]
    cpu_interleaved: torch.Tensor = _scores(
        expected, inputs.interleaved_index, candidate_indices
    )
    spyre_interleaved: torch.Tensor = _scores(
        actual, inputs.interleaved_index, candidate_indices
    )

    print("\n## Interleaved Retrieval\n")
    print("| Candidate | CPU score | Spyre score |")
    print("|-----------|-----------|-------------|")
    for index, label in enumerate(candidate_labels):
        print(
            f"| {label!r} | {cpu_interleaved[index].item():.4f} "
            f"| {spyre_interleaved[index].item():.4f} |"
        )

    for device, scores in (("CPU", cpu_interleaved), ("Spyre", spyre_interleaved)):
        top: int = int(scores.argmax().item())
        assert top == 0, (
            f"Expected the interleaved input to be closest to "
            f"{candidate_labels[0]!r} on {device}, got {candidate_labels[top]!r}"
        )
