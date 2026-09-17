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

Covers both towers of each registered CLIP checkpoint:

- **Text tower**: encode a list of strings and compare per-token cosine similarity
  between CPU and Spyre outputs, then assert a sentence-embedding cosine threshold.
- **Vision tower**: encode a PIL image downloaded from the Hugging Face Hub and
  compare the resulting image embeddings between CPU and Spyre.
- **Cross-modal**: verify the cosine-similarity ranking between one image embedding
  and a list of text embeddings is identical on CPU and Spyre.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_clip_compare_spyre.py
"""

from typing import Any

import pytest
import torch
import torch.nn.functional as F

# Registers the "spyre" backend with sentence_transformers on import.
import hf_adapters.st_backend  # noqa: F401
from tests._vision_helpers import _load_sample_image
from tests.model_registry import CLIP_PATHS

pytestmark = pytest.mark.model_harness("embedding")

TEXT_PROMPTS = [
    "Two dogs at the beach",
    "A cat in the snow",
    "A picture of London at night",
]

IMAGE_LABEL = "pipeline-cat-chonk.jpeg"

COSINE_THRESHOLD = 0.97


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_models(model_path: str):
    """Load both the CPU and Spyre CLIP SentenceTransformer instances for *model_path*."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    print("  Loading stock SentenceTransformer on CPU ...")
    cpu_model = SentenceTransformer(model_path, device="cpu")

    print("  Loading SentenceTransformer with backend='spyre' ...")
    spyre_model = SentenceTransformer(model_path, backend="spyre")

    return cpu_model, spyre_model


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_path", CLIP_PATHS, ids=CLIP_PATHS)
def test_clip_text_compare_spyre(model_path: str) -> None:
    """Text tower: Spyre sentence embeddings are close to CPU reference."""
    cpu_model, spyre_model = _load_models(model_path)

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


@pytest.mark.parametrize("model_path", CLIP_PATHS, ids=CLIP_PATHS)
def test_clip_image_compare_spyre(model_path: str) -> None:
    """Vision tower: Spyre image embeddings are close to CPU reference."""
    cpu_model, spyre_model = _load_models(model_path)

    print("  Downloading Hugging Face sample image ...")
    images = [_load_sample_image()]

    print("  Encoding images on CPU ...")
    cpu_embs = cpu_model.encode(images, convert_to_tensor=True)

    print("  Encoding images on Spyre ...")
    spyre_embs = spyre_model.encode(images, convert_to_tensor=True)

    assert (
        cpu_embs.shape == spyre_embs.shape
    ), f"Shape mismatch: CPU {tuple(cpu_embs.shape)} vs Spyre {tuple(spyre_embs.shape)}"

    rows: list[dict[str, Any]] = []
    for i, label in enumerate([IMAGE_LABEL]):
        cos = _cosine(cpu_embs[i], spyre_embs[i])
        rows.append({"label": label, "cosine": cos, "match": cos >= COSINE_THRESHOLD})

    print("\n## CLIP Vision Tower: CPU vs Spyre\n")
    print("| Image | Cosine | Match |")
    print("|-------|--------|-------|")
    for r in rows:
        print(
            f"| {r['label']} | {r['cosine']:.6f} | {'OK' if r['match'] else 'FAIL'} |"
        )

    mismatches = [r for r in rows if not r["match"]]
    assert (
        not mismatches
    ), f"Vision tower cosine < {COSINE_THRESHOLD} for: " + ", ".join(
        r["label"] for r in mismatches
    )


@pytest.mark.parametrize("model_path", CLIP_PATHS, ids=CLIP_PATHS)
def test_clip_crossmodal_ranking_spyre(model_path: str) -> None:
    """Cross-modal: image–text cosine rankings are identical on CPU and Spyre.

    Encodes one cat image and three text descriptions, then asserts that the
    ranking produced on Spyre is identical to the CPU reference.
    """
    cpu_model, spyre_model = _load_models(model_path)

    print("  Downloading image ...")
    image = _load_sample_image()

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
    assert cpu_ranking[0] == 1, (
        f"Expected '{TEXT_PROMPTS[1]}' to rank first for the cat image, "
        f"got: {TEXT_PROMPTS[cpu_ranking[0]]!r}"
    )
