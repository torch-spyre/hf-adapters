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

"""SentenceTransformers dual-encoder integration; HF accuracy has its own suite."""

import gc

import pytest
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Register the public SentenceTransformers backend.
import hf_adapters.st_backend  # noqa: F401
from tests._multimodal_embedding_helpers import TEXT_PROMPTS, bounded_cpu_threads
from tests._vision_helpers import _load_sample_image
from tests.model_registry import MULTIMODAL_EMBED_PATHS

pytestmark = pytest.mark.model_harness("multimodal_embedding")


@pytest.mark.parametrize(
    "model_path", MULTIMODAL_EMBED_PATHS, ids=MULTIMODAL_EMBED_PATHS
)
def test_sentence_transformers_multimodal_backend(model_path: str) -> None:
    """Preserve encode(), scalar/list handling, and exact caption ranking coverage."""
    image = _load_sample_image()
    with bounded_cpu_threads(), torch.inference_mode():
        cpu_model = SentenceTransformer(model_path, device="cpu")
        # Use each model's configured text length for both execution paths.
        context_length: int = (
            cpu_model[0].config.get_text_config().max_position_embeddings
        )
        text_kwargs = {
            "processing_kwargs": {
                "text": {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": context_length,
                }
            }
        }
        reference: dict[str, torch.Tensor] = {
            "text": cpu_model.encode(
                TEXT_PROMPTS, convert_to_tensor=True, **text_kwargs
            ).cpu(),
            "image": cpu_model.encode(image, convert_to_tensor=True).cpu(),
            "image_list": cpu_model.encode([image], convert_to_tensor=True).cpu(),
        }
        # Preparing Spyre may patch classes; finish the reference first.
        del cpu_model
        gc.collect()
        model = SentenceTransformer(model_path, backend="spyre")
        actual: dict[str, torch.Tensor] = {
            "text": model.encode(
                TEXT_PROMPTS, convert_to_tensor=True, **text_kwargs
            ).cpu(),
            "image": model.encode(image, convert_to_tensor=True).cpu(),
            "image_list": model.encode([image], convert_to_tensor=True).cpu(),
        }
        for name, expected in reference.items():
            value = actual[name]
            assert value.shape == expected.shape
            assert torch.isfinite(value).all()
            assert torch.isfinite(expected).all()
            similarities = F.cosine_similarity(value.float(), expected.float(), dim=-1)
            assert (similarities >= 0.97).all(), f"{name}: {similarities}"
        reference_scores = F.cosine_similarity(
            reference["image"].float().unsqueeze(0), reference["text"].float(), dim=1
        )
        actual_scores = F.cosine_similarity(
            actual["image"].float().unsqueeze(0), actual["text"].float(), dim=1
        )
        expected_ranking = reference_scores.argsort(descending=True)
        assert torch.equal(actual_scores.argsort(descending=True), expected_ranking)
        assert expected_ranking[0].item() == 1, "The cat caption must rank first"
