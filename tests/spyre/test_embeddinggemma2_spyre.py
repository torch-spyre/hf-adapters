"""EmbeddingGemma 2 text/image accuracy and retrieval through the public ST API."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

import hf_adapters.st_backend  # noqa: F401

MODEL_PATH = "google/embeddinggemma-2"
pytestmark = pytest.mark.model_harness("embedding")


@pytest.fixture(scope="module")
def embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    # A generated image keeps this test independent of an external image host.
    image = Image.new("RGB", (224, 224), color=(30, 130, 210))
    inputs: list[Any] = [
        "task: search result | query: A blue image",
        "title: none | text: A solid blue picture.",
        "title: none | text: A dog running in the snow.",
        {"image": image},
        {"text": "A blue picture. <|image|>", "image": image},
    ]
    cpu = SentenceTransformer(
        MODEL_PATH, device="cpu", model_kwargs={"dtype": torch.bfloat16}
    )
    expected = cpu.encode(inputs, batch_size=1, convert_to_tensor=True)
    del cpu
    spyre = SentenceTransformer(
        MODEL_PATH, backend="spyre", model_kwargs={"dtype": torch.bfloat16}
    )
    actual = spyre.encode(inputs, batch_size=1, convert_to_tensor=True)
    return expected.cpu(), actual.cpu()


def test_embeddinggemma2_multimodal_accuracy(
    embeddings: tuple[torch.Tensor, torch.Tensor],
) -> None:
    expected, actual = embeddings
    assert actual.shape == expected.shape == (5, 768)
    assert torch.isfinite(actual).all()
    cosine = F.cosine_similarity(expected.float(), actual.float(), dim=-1)
    print(f"Text, image and interleaved cosine: {cosine.tolist()}")
    assert cosine.min().item() >= 0.99


def test_embeddinggemma2_retrieval_ranking(
    embeddings: tuple[torch.Tensor, torch.Tensor],
) -> None:
    expected, actual = embeddings
    cpu_scores = F.cosine_similarity(
        expected[:1].float(), expected[1:4].float(), dim=-1
    )
    spyre_scores = F.cosine_similarity(actual[:1].float(), actual[1:4].float(), dim=-1)
    assert torch.equal(
        cpu_scores.argsort(descending=True), spyre_scores.argsort(descending=True)
    )
