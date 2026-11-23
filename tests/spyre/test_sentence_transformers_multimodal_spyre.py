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

"""Multimodal embedding accuracy: stock SentenceTransformers (CPU) vs ``backend="spyre"``.

The single Spyre suite for ``MULTIMODAL_EMBEDDING_MODELS``. It exercises the
``hf_adapters.st_backend`` dual-encoder dispatch and ST's ``encode()`` routing
for text, a scalar image and an image list, checks the Spyre dtype, and
checks caption ranking and image/text cosine scores against the CPU reference.
"""

import gc
from contextlib import nullcontext
from itertools import chain
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Register the public SentenceTransformers backend.
import hf_adapters.st_backend  # noqa: F401
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests._multimodal_embedding_helpers import (
    TEXT_PROMPTS,
    bounded_cpu_threads,
    forbid_model_device_moves,
)
from tests._vision_helpers import _load_sample_image
from tests.model_registry import MULTIMODAL_EMBED_PATHS

pytestmark = pytest.mark.model_harness("multimodal_embedding")

# Same bound as the text-embedding suite (test_e2e_embed_compare_spyre.py).
COSINE_THRESHOLD: float = 0.99
# Absolute bound on image/text cosine scores vs the CPU reference.
SCORE_ATOL: float = 0.005


@pytest.mark.parametrize(
    "model_path", MULTIMODAL_EMBED_PATHS, ids=MULTIMODAL_EMBED_PATHS
)
def test_sentence_transformers_multimodal_backend(model_path: str) -> None:
    """Preserve encode(), scalar/list handling, and exact caption ranking coverage."""
    image = _load_sample_image()
    with bounded_cpu_threads(), torch.inference_mode():
        cpu_model = SentenceTransformer(model_path, device="cpu")
        # Preserve CLIP/SigLIP's fixed pooling lengths, while bounding models
        # with long contexts (EmbeddingGemma 2 advertises 262K RoPE positions).
        context_length: int = min(
            cpu_model.max_seq_length,
            cpu_model[0].config.get_text_config().max_position_embeddings,
            128,
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
        mixed_length_prompts: list[str] = []
        mixed_text_kwargs: dict[str, Any] = {
            "batch_size": 2,
            "processing_kwargs": {
                "text": {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": 1200,
                }
            },
        }
        if cpu_model[0].config.model_type == "embedding_gemma2":
            # The short row has padding queries farther than sliding_window
            # from every real key. They must not generate NaNs in later layers.
            mixed_length_prompts = ["The cat sits on the mat. " * 200, "A cat."]
            reference["mixed_length_text"] = cpu_model.encode(
                mixed_length_prompts, convert_to_tensor=True, **mixed_text_kwargs
            ).cpu()
        # Preparing Spyre may patch classes; finish the reference first.
        del cpu_model
        gc.collect()
        model = SentenceTransformer(model_path, backend="spyre")
        expected_dtype: torch.dtype = dtype_for_model_path(
            model_path, target_device="spyre"
        )
        spyre_dtype: torch.dtype = model[0].auto_model.dtype
        print(f"Spyre model dtype: {spyre_dtype}")
        assert (
            spyre_dtype == expected_dtype
        ), f"Spyre dtype: expected {expected_dtype}, got {spyre_dtype}"
        backbone: torch.nn.Module = model[0].auto_model
        preserve_placement = hasattr(backbone, "_spyre_device_parameter_name")
        if preserve_placement:
            # CPU helpers must be restored before the first encode, not just
            # afterward. Then guard against temporary round trips on every call.
            for name in backbone._spyre_cpu_submodules:
                module: torch.nn.Module = backbone.get_submodule(name)
                assert all(
                    tensor.device.type == "cpu"
                    for tensor in chain(module.parameters(), module.buffers())
                ), f"{name} was not restored to CPU during initialization"
        with forbid_model_device_moves(model) if preserve_placement else nullcontext():
            actual: dict[str, torch.Tensor] = {
                "text": model.encode(
                    TEXT_PROMPTS, convert_to_tensor=True, **text_kwargs
                ).cpu(),
                "image": model.encode(image, convert_to_tensor=True).cpu(),
                "image_list": model.encode([image], convert_to_tensor=True).cpu(),
            }
            if mixed_length_prompts:
                actual["mixed_length_text"] = model.encode(
                    mixed_length_prompts, convert_to_tensor=True, **mixed_text_kwargs
                ).cpu()
        if preserve_placement:
            print("No model parameter/buffer device transfers during repeated encodes.")
        for name, expected in reference.items():
            value = actual[name]
            assert value.shape == expected.shape
            assert torch.isfinite(value).all()
            assert torch.isfinite(expected).all()
            similarities = F.cosine_similarity(value.float(), expected.float(), dim=-1)
            print(f"{name}: minimum cosine {similarities.min().item():.6f}")
            assert (similarities >= COSINE_THRESHOLD).all(), f"{name}: {similarities}"
        reference_scores = F.cosine_similarity(
            reference["image"].float().unsqueeze(0), reference["text"].float(), dim=1
        )
        actual_scores = F.cosine_similarity(
            actual["image"].float().unsqueeze(0), actual["text"].float(), dim=1
        )
        print(f"CPU cosine scores: {reference_scores.tolist()}")
        print(f"Spyre cosine scores: {actual_scores.tolist()}")
        torch.testing.assert_close(
            actual_scores, reference_scores, rtol=0.0, atol=SCORE_ATOL
        )
        expected_ranking = reference_scores.argsort(descending=True)
        assert torch.equal(actual_scores.argsort(descending=True), expected_ranking)
        assert expected_ranking[0].item() == 1, "The cat caption must rank first"
