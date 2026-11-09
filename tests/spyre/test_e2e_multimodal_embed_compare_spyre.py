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

"""Compare multimodal embeddings, retrieval, and logits against an fp32 CPU reference."""

from __future__ import annotations

import gc

import pytest
import torch
import transformers
from transformers import AutoProcessor

from hf_adapters._model_loading import model_subfolder
from tests._multimodal_embedding_helpers import (
    _assert_embeddings,
    _assert_logits,
    _assert_retrieval,
    _cosine_scores,
    bounded_cpu_threads,
    embedding_cases,
    load_encoder,
    prepare_inputs,
)
from tests.model_registry import MULTIMODAL_EMBED_PATHS, REMOTE_CODE_PATHS

pytestmark = pytest.mark.model_harness("multimodal_embedding")


@pytest.mark.parametrize("model_path", MULTIMODAL_EMBED_PATHS)
def test_multimodal_embeddings_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    """Load each model once and check all scenarios through the public HF APIs."""
    print(f"Transformers: {transformers.__version__} ({transformers.__file__})")
    print("Embedding API: Hugging Face dual encoder; CPU reference: float32")
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS

    with bounded_cpu_threads():
        cpu_encoder = load_encoder(
            model_path, spyre=False, trust_remote_code=trust_remote_code
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            subfolder=model_subfolder(model_path),
            trust_remote_code=trust_remote_code,
        )
        inputs = {
            name: prepare_inputs(cpu_encoder.model, processor, case)
            for name, case in embedding_cases().items()
        }
        with torch.inference_mode():
            reference = {
                name: cpu_encoder.encode(prepared) for name, prepared in inputs.items()
            }
        # Finish every CPU reference before preparing Spyre, then release its model.
        del cpu_encoder, processor
        gc.collect()
        spyre_encoder = load_encoder(
            model_path, spyre=True, trust_remote_code=trust_remote_code
        )
        with torch.inference_mode():
            for name, prepared in inputs.items():
                print(f"Checking scenario: {name}")
                actual = spyre_encoder.encode(prepared)
                expected = reference[name]
                _assert_embeddings(actual, expected)
                if prepared.text is not None and prepared.image is not None:
                    _assert_logits(actual, expected)
                    for prefix in ("", "joint_"):
                        text_key, image_key = f"{prefix}text", f"{prefix}image"
                        print(f"Retrieval using {prefix or 'standalone_'}features")
                        _assert_retrieval(
                            _cosine_scores(
                                actual.embeddings[text_key],
                                actual.embeddings[image_key],
                            ),
                            _cosine_scores(
                                expected.embeddings[text_key],
                                expected.embeddings[image_key],
                            ),
                            image_case=name,
                        )
