# Copyright 2025 The Torch-Spyre Authors.
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

"""
E2E CLIP accuracy: stock SentenceTransformers (CPU) vs ``backend="spyre"``.

Covers both towers of ``sentence-transformers/clip-ViT-B-32``:

- **Text tower**: encode a list of strings and compare per-token cosine similarity
  between CPU and Spyre outputs, then assert a sentence-embedding cosine threshold.
- **Vision tower**: encode PIL images downloaded from public URLs and compare
  the resulting image embeddings between CPU and Spyre.
- **Cross-modal**: verify the cosine-similarity ranking between one image embedding
  and a list of text embeddings is identical on CPU and Spyre.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_clip_compare_spyre.py
"""

from io import BytesIO
from typing import Any

import pytest
import torch
import torch.nn.functional as F

# Registers the "spyre" backend with sentence_transformers on import.
import hf_adapters.st_backend  # noqa: F401

pytestmark = pytest.mark.model_harness("embedding")

MODEL_PATH = "sentence-transformers/clip-ViT-B-32"

TEXT_PROMPTS = [
    "Two dogs in the snow",
    "A cat on a table",
    "A picture of London at night",
]

# Public-domain image URLs for the vision tower test.
# Using small, reliable JPEG images to minimise download time.
IMAGE_URLS = [
    # Labrador retriever (Wikimedia Commons, public domain)
    "https://upload.wikimedia.org/wikipedia/commons/3/34/Labrador_on_Quantock_%282175262184%29.jpg",
]

COSINE_THRESHOLD = 0.97


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(url: str):
    """Download an image from *url* and return a PIL Image (RGB)."""
    import requests
    from PIL import Image

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _load_models():
    """Load both the CPU and Spyre CLIP SentenceTransformer instances."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 70}")
    print(f"  {MODEL_PATH}")
    print(f"{'=' * 70}")

    print("  Loading stock SentenceTransformer on CPU ...")
    cpu_model = SentenceTransformer(MODEL_PATH, device="cpu")

    print("  Loading SentenceTransformer with backend='spyre' ...")
    spyre_model = SentenceTransformer(MODEL_PATH, backend="spyre")

    return cpu_model, spyre_model


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clip_text_compare_spyre() -> None:
    """Text tower: Spyre sentence embeddings are close to CPU reference."""
    cpu_model, spyre_model = _load_models()

    print(f"  Encoding {len(TEXT_PROMPTS)} text prompts on CPU ...")
    cpu_embs = cpu_model.encode(TEXT_PROMPTS, convert_to_tensor=True)

    print(f"  Encoding {len(TEXT_PROMPTS)} text prompts on Spyre ...")
    spyre_embs = spyre_model.encode(TEXT_PROMPTS, convert_to_tensor=True)

    assert (
        cpu_embs.shape == spyre_embs.shape
    ), f"Shape mismatch: CPU {tuple(cpu_embs.shape)} vs Spyre {tuple(spyre_embs.shape)}"

    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(TEXT_PROMPTS):
        cos = _cosine(cpu_embs[i], spyre_embs[i])
        rows.append({"prompt": prompt, "cosine": cos, "match": cos >= COSINE_THRESHOLD})

    print("\n## CLIP Text Tower: CPU vs Spyre\n")
    print("| Prompt | Cosine | Match |")
    print("|--------|--------|-------|")
    for r in rows:
        print(
            f"| {r['prompt']!r} | {r['cosine']:.6f} | {'OK' if r['match'] else 'FAIL'} |"
        )

    mismatches = [r for r in rows if not r["match"]]
    assert not mismatches, f"Text tower cosine < {COSINE_THRESHOLD} for: " + ", ".join(
        r["prompt"] for r in mismatches
    )


def test_clip_image_compare_spyre() -> None:
    """Vision tower: Spyre image embeddings are close to CPU reference."""
    cpu_model, spyre_model = _load_models()

    print(f"  Downloading {len(IMAGE_URLS)} image(s) ...")
    images = [_load_image(url) for url in IMAGE_URLS]

    print("  Encoding images on CPU ...")
    cpu_embs = cpu_model.encode(images, convert_to_tensor=True)

    print("  Encoding images on Spyre ...")
    spyre_embs = spyre_model.encode(images, convert_to_tensor=True)

    assert (
        cpu_embs.shape == spyre_embs.shape
    ), f"Shape mismatch: CPU {tuple(cpu_embs.shape)} vs Spyre {tuple(spyre_embs.shape)}"

    rows: list[dict[str, Any]] = []
    for i, url in enumerate(IMAGE_URLS):
        cos = _cosine(cpu_embs[i], spyre_embs[i])
        rows.append({"url": url, "cosine": cos, "match": cos >= COSINE_THRESHOLD})

    print("\n## CLIP Vision Tower: CPU vs Spyre\n")
    print("| Image | Cosine | Match |")
    print("|-------|--------|-------|")
    for r in rows:
        label = r["url"].split("/")[-1]
        print(f"| {label} | {r['cosine']:.6f} | {'OK' if r['match'] else 'FAIL'} |")

    mismatches = [r for r in rows if not r["match"]]
    assert (
        not mismatches
    ), f"Vision tower cosine < {COSINE_THRESHOLD} for: " + ", ".join(
        r["url"] for r in mismatches
    )


def test_clip_crossmodal_ranking_spyre() -> None:
    """Cross-modal: image–text cosine rankings are identical on CPU and Spyre.

    Encodes one dog image and three text descriptions. Asserts that the most
    similar text on Spyre is the same as on CPU, and that "Two dogs in the snow"
    ranks first (it describes a dog, matching the image content).
    """
    cpu_model, spyre_model = _load_models()

    print("  Downloading image ...")
    image = _load_image(IMAGE_URLS[0])

    print("  Encoding image + text on CPU ...")
    cpu_img = cpu_model.encode(image, convert_to_tensor=True)
    cpu_txt = cpu_model.encode(TEXT_PROMPTS, convert_to_tensor=True)

    print("  Encoding image + text on Spyre ...")
    spyre_img = spyre_model.encode(image, convert_to_tensor=True)
    spyre_txt = spyre_model.encode(TEXT_PROMPTS, convert_to_tensor=True)

    cpu_scores = F.cosine_similarity(
        cpu_img.float().unsqueeze(0), cpu_txt.float(), dim=1
    )
    spyre_scores = F.cosine_similarity(
        spyre_img.float().unsqueeze(0), spyre_txt.float(), dim=1
    )

    cpu_ranking = cpu_scores.argsort(descending=True).tolist()
    spyre_ranking = spyre_scores.argsort(descending=True).tolist()

    print("\n## CLIP Cross-modal Ranking\n")
    print("| Prompt | CPU score | Spyre score |")
    print("|--------|-----------|-------------|")
    for i, prompt in enumerate(TEXT_PROMPTS):
        print(f"| {prompt!r} | {cpu_scores[i]:.4f} | {spyre_scores[i]:.4f} |")
    print(f"\nCPU ranking:   {[TEXT_PROMPTS[i] for i in cpu_ranking]}")
    print(f"Spyre ranking: {[TEXT_PROMPTS[i] for i in spyre_ranking]}")

    assert cpu_ranking == spyre_ranking, (
        f"Cross-modal ranking differs between CPU and Spyre.\n"
        f"  CPU:   {cpu_ranking}\n"
        f"  Spyre: {spyre_ranking}"
    )
    assert cpu_ranking[0] == 0, (
        f"Expected 'Two dogs in the snow' to rank first for the dog image, "
        f"got: {TEXT_PROMPTS[cpu_ranking[0]]!r}"
    )
