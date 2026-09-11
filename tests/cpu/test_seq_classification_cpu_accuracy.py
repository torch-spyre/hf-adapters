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

"""CPU accuracy test for standard sequence-classification via the auto-loader.

One test case per registered sequence-classification model:

  test_auto_loader[<key>]
    Loads a stock HF reference on CPU, runs a forward to get ``[B, num_labels]``
    logits, then loads via ``AutoSpyreModelForSequenceClassification.from_pretrained``
    and calls ``model(**encoded, return_dict=True)``.  Asserts:
      - Output shape matches ``[B, num_labels]``
      - Per-label cosine similarity across the batch is >= threshold
      - Predicted class ids match exactly

Execution is delegated to ``_seq_classification_helpers``; this file owns only
the seq-classification-specific inputs and cosine assertions.

DEVICE is patched to ``"cpu"`` by ``tests/conftest.py``; torch.compile is
unwrapped by ``_unwrap_compiled_blocks`` so blocks run eagerly.
"""

import gc

import pytest
import torch
import torch.nn.functional as F
from _seq_classification_helpers import run_seq_classification_auto_loader_vs_ref

from tests.model_registry import SEQ_CLASSIFICATION_PATHS

pytestmark = pytest.mark.model_harness("seq_classification")

TEXTS: list[str] = [
    "Hello, my dog is cute.",
    "This movie was absolutely terrible.",
    "The weather is nice today.",
]

# fp16 encoder vs fp32 reference: per-label cosine should be very tight.
COSINE_THRESHOLD: float = 0.999


def _assert_seq_classification_logits(
    ref_logits: torch.Tensor,
    adapter_logits: torch.Tensor,
) -> None:
    assert (
        adapter_logits.shape == ref_logits.shape
    ), f"shape mismatch: adapter {adapter_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(adapter_logits).all()
    cos = F.cosine_similarity(adapter_logits, ref_logits, dim=-1)  # [B]
    assert (
        cos.min().item() >= COSINE_THRESHOLD
    ), f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}"
    ref_ids = ref_logits.argmax(dim=-1)
    adapter_ids = adapter_logits.argmax(dim=-1)
    assert torch.equal(
        adapter_ids, ref_ids
    ), f"predicted class mismatch: adapter {adapter_ids.tolist()} vs ref {ref_ids.tolist()}"


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_auto_loader(model_path: str) -> None:
    """AutoSpyreModelForSequenceClassification logits match HF CPU reference."""
    ref_logits, adapter_logits = run_seq_classification_auto_loader_vs_ref(
        model_path, TEXTS
    )
    gc.collect()
    _assert_seq_classification_logits(ref_logits, adapter_logits)
