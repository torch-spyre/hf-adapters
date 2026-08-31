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

"""CPU accuracy test for sequence-classification models used as rerankers.

One test case per registered reranker:

  test_auto_loader[<key>]
    Loads a stock HF reference on CPU, runs a forward to get logits, then loads
    via ``AutoSpyreModelForSequenceClassification.from_pretrained`` and calls
    ``model(**encoded, return_dict=True)``.  Asserts:
      - Output shape matches ``[B, num_labels]``
      - First-label logits are within an absolute tolerance of 0.05 of HF reference
      - Ranking order induced by the first-label logits is preserved

Execution is delegated to ``_seq_classification_helpers``; this file owns only
the reranker-specific inputs and ranking-order assertions.

DEVICE is patched to ``"cpu"`` by ``tests/conftest.py``; torch.compile is
unwrapped by ``_unwrap_compiled_blocks`` so blocks run eagerly.
"""

import gc

import pytest
import torch
from _seq_classification_helpers import run_seq_classification_auto_loader_vs_ref

from tests.model_registry import RERANKER_PATHS

pytestmark = pytest.mark.model_harness("reranker")

# Query-document pairs that cover a range of relevance (positive + negative)
# so ranking-order correctness is exercised in addition to absolute score match.
PAIRS: list[tuple[str, str]] = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "London is the capital of the United Kingdom."),
    ("How do transformers work?", "Transformers use self-attention mechanisms."),
    ("How do transformers work?", "A recipe for chocolate cake."),
]

# Absolute score tolerance — fp16 encoder output vs fp32 reference.
# The classification head is identical between both paths; differences come
# only from fp16 rounding in the backbone, which is tiny on CPU.
SCORE_ATOL: float = 0.05


def _assert_reranker_logits(
    ref_logits: torch.Tensor,
    adapter_logits: torch.Tensor,
) -> None:
    assert (
        adapter_logits.shape == ref_logits.shape
    ), f"logit shape mismatch: adapter {adapter_logits.shape} vs ref {ref_logits.shape}"
    ref_scores = ref_logits[:, 0]
    adapter_scores = adapter_logits[:, 0]
    max_abs_diff = (adapter_scores - ref_scores).abs().max().item()
    assert max_abs_diff <= SCORE_ATOL, (
        f"max absolute score difference {max_abs_diff:.4f} exceeds {SCORE_ATOL}.\n"
        f"  ref    = {ref_scores.tolist()}\n"
        f"  adapter= {adapter_scores.tolist()}"
    )
    ref_order = torch.argsort(ref_scores, descending=True).tolist()
    adapter_order = torch.argsort(adapter_scores, descending=True).tolist()
    assert (
        ref_order == adapter_order
    ), f"ranking order mismatch: ref {ref_order} vs adapter {adapter_order}"


@pytest.mark.parametrize("model_path", RERANKER_PATHS, ids=RERANKER_PATHS)
def test_auto_loader(model_path: str) -> None:
    """AutoSpyreModelForSequenceClassification logits match HF CPU reference."""
    ref_logits, adapter_logits = run_seq_classification_auto_loader_vs_ref(
        model_path, PAIRS
    )
    gc.collect()
    _assert_reranker_logits(ref_logits, adapter_logits)
