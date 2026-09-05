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
CPU accuracy test for the CLIP adapter (``hf_clip.py``) via the ST backend.

Covers both towers of ``sentence-transformers/clip-ViT-B-32``:

- **Text tower**: encode a list of strings and assert per-sentence cosine
  similarity between the stock CPU model and the ``backend="spyre"`` model
  (with compiled blocks unwrapped and ``DEVICE`` patched to ``"cpu"`` by
  ``tests/conftest.py``) is above ``COS_SIM_THRESHOLD``.
- **Vision tower**: encode PIL images downloaded from public URLs and assert
  the same cosine threshold on the resulting image embeddings.
- **Cross-modal**: verify the image–text cosine ranking produced by the
  ``backend="spyre"`` path matches the stock CPU reference.

``DEVICE="cpu"`` patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest with no Spyre hardware requirement.

Usage::

    pytest -s -vvv tests/cpu/test_clip_cpu_accuracy.py
"""

import gc
from io import BytesIO

import pytest
import torch.nn.functional as F

from tests.cpu.conftest import _unwrap_compiled_blocks, cosine_per_row

pytestmark = pytest.mark.model_harness("embedding")

pytest.importorskip("sentence_transformers")
pytest.importorskip("PIL")

MODEL_PATH = "sentence-transformers/clip-ViT-B-32"

TEXT_PROMPTS = [
    "Two dogs in the snow",
    "A cat on a table",
    "A picture of London at night",
]

# Public-domain image URLs for the vision tower test.
IMAGE_URLS = [
    # Labrador retriever (Wikimedia Commons, public domain)
    "https://upload.wikimedia.org/wikipedia/commons/3/34/Labrador_on_Quantock_%282175262184%29.jpg",
]

COS_SIM_THRESHOLD: float = 0.999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(url: str):
    """Download *url* and return a PIL Image (RGB)."""
    import requests
    from PIL import Image

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _load_models():
    """Return (ref_model, spyre_model) both running on CPU."""
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401  (registers "spyre" backend with ST)

    ref_model = SentenceTransformer(MODEL_PATH, device="cpu")

    # backend="spyre" with DEVICE patched to "cpu" by conftest; unwrap compiled
    # blocks so the forward runs as plain Python on CPU.
    spyre_model = SentenceTransformer(MODEL_PATH, backend="spyre", device="cpu")
    _unwrap_compiled_blocks(spyre_model._first_module().model)

    return ref_model, spyre_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clip_text_cpu() -> None:
    """Text tower: ``backend='spyre'`` sentence embeddings match stock CPU."""
    ref_model, spyre_model = _load_models()

    ref_embs = ref_model.encode(TEXT_PROMPTS, convert_to_tensor=True)
    del ref_model
    gc.collect()

    spyre_embs = spyre_model.encode(TEXT_PROMPTS, convert_to_tensor=True)
    del spyre_model
    gc.collect()

    assert (
        ref_embs.shape == spyre_embs.shape
    ), f"Shape mismatch: ref {tuple(ref_embs.shape)} vs spyre {tuple(spyre_embs.shape)}"

    cos_sims = cosine_per_row(ref_embs, spyre_embs)
    min_sim = cos_sims.min().item()
    assert min_sim >= COS_SIM_THRESHOLD, (
        f"min cosine {min_sim:.6f} < threshold {COS_SIM_THRESHOLD}; "
        f"per-sentence: {cos_sims.tolist()}"
    )


def test_clip_image_cpu() -> None:
    """Vision tower: ``backend='spyre'`` image embeddings match stock CPU."""
    ref_model, spyre_model = _load_models()

    images = [_load_image(url) for url in IMAGE_URLS]

    ref_embs = ref_model.encode(images, convert_to_tensor=True)
    del ref_model
    gc.collect()

    spyre_embs = spyre_model.encode(images, convert_to_tensor=True)
    del spyre_model
    gc.collect()

    assert (
        ref_embs.shape == spyre_embs.shape
    ), f"Shape mismatch: ref {tuple(ref_embs.shape)} vs spyre {tuple(spyre_embs.shape)}"

    cos_sims = cosine_per_row(ref_embs, spyre_embs)
    min_sim = cos_sims.min().item()
    assert min_sim >= COS_SIM_THRESHOLD, (
        f"min cosine {min_sim:.6f} < threshold {COS_SIM_THRESHOLD}; "
        f"per-image: {cos_sims.tolist()}"
    )


def test_clip_crossmodal_ranking_cpu() -> None:
    """Cross-modal: image–text cosine rankings match between stock CPU and ``backend='spyre'``.

    Encodes one dog image and three text descriptions. Asserts that the ranking
    of texts by cosine similarity to the image is identical between the reference
    and the spyre-backend model, and that the dog-related caption ranks first.
    """
    ref_model, spyre_model = _load_models()

    image = _load_image(IMAGE_URLS[0])

    ref_img = ref_model.encode(image, convert_to_tensor=True)
    ref_txt = ref_model.encode(TEXT_PROMPTS, convert_to_tensor=True)
    del ref_model
    gc.collect()

    spyre_img = spyre_model.encode(image, convert_to_tensor=True)
    spyre_txt = spyre_model.encode(TEXT_PROMPTS, convert_to_tensor=True)
    del spyre_model
    gc.collect()

    ref_scores = F.cosine_similarity(
        ref_img.float().unsqueeze(0), ref_txt.float(), dim=1
    )
    spyre_scores = F.cosine_similarity(
        spyre_img.float().unsqueeze(0), spyre_txt.float(), dim=1
    )

    ref_ranking = ref_scores.argsort(descending=True).tolist()
    spyre_ranking = spyre_scores.argsort(descending=True).tolist()

    assert ref_ranking == spyre_ranking, (
        f"Cross-modal ranking differs between ref and spyre-backend.\n"
        f"  ref:   {ref_ranking}\n"
        f"  spyre: {spyre_ranking}"
    )
    assert ref_ranking[0] == 0, (
        f"Expected 'Two dogs in the snow' to rank first for the dog image, "
        f"got: {TEXT_PROMPTS[ref_ranking[0]]!r}"
    )
