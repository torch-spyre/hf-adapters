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
E2E reranker accuracy: HF stock forward (CPU) vs adapter forward (Spyre).

Cross-encoder counterpart of test_e2e_embed_compare_spyre.py.

For each registered reranker, loads the model on CPU, runs a reference forward
to get scores, moves it to Spyre, runs the adapter forward, and asserts that the
raw logit scores are close and that ranking order is preserved.

Execution is delegated to ``_seq_classification_helpers.run_seq_classification_cpu_vs_spyre``;
this file owns only the reranker-specific inputs and ranking-order assertions.

Usage (on Spyre LPAR)::

    # All registered rerankers
    pytest -s -vvv tests/spyre/test_e2e_reranker_compare_spyre.py

    # Just BGE Reranker v2 M3
    pytest -s -vvv tests/spyre/test_e2e_reranker_compare_spyre.py -k bge_reranker
"""

import pytest
import torch

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)
from tests.model_registry import RERANKER_PATHS
from tests.spyre._seq_classification_helpers import run_seq_classification_cpu_vs_spyre

pytestmark = pytest.mark.model_harness("reranker")

# Pairs that span a range of relevance scores — ensures ranking order is
# meaningful, not just that all scores are close to zero.
PAIRS: list[tuple[str, str]] = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "London is the capital of the United Kingdom."),
    ("How do transformers work?", "Transformers use self-attention mechanisms."),
    ("How do transformers work?", "A recipe for chocolate cake with frosting."),
    (
        "What is RAG?",
        "Retrieval-Augmented Generation combines retrieval with generation.",
    ),
]

# Absolute score tolerance. Spyre fp16 introduces small numerical differences
# in the backbone hidden states; the classification head amplifies them slightly.
# 0.5 logit units is generous but the ranking order assertion is the primary check.
SCORE_ATOL: float = 0.5


@pytest.mark.parametrize("model_path", RERANKER_PATHS, ids=RERANKER_PATHS)
def test_e2e_reranker_compare_spyre(model_path: str) -> None:
    adapter = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    result = run_seq_classification_cpu_vs_spyre(model_path, adapter, PAIRS)

    ref_scores = result["ref_logits"][:, 0]
    spyre_scores = result["spyre_logits"][:, 0]

    abs_diffs = (spyre_scores - ref_scores).abs()
    max_diff = abs_diffs.max().item()
    ref_order = torch.argsort(ref_scores, descending=True).tolist()
    spyre_order = torch.argsort(spyre_scores, descending=True).tolist()

    print(f"\n## Reranker E2E: HF (CPU) vs Adapter (Spyre) — {model_path}\n")
    print("| Pair | HF Score | Spyre Score | Abs Diff | Match |")
    print("|------|----------|-------------|----------|-------|")
    for i, (hs, ss, d) in enumerate(
        zip(ref_scores.tolist(), spyre_scores.tolist(), abs_diffs.tolist())
    ):
        ok = "OK" if d <= SCORE_ATOL else "FAIL"
        print(f"| {i} | {hs:.4f} | {ss:.4f} | {d:.4f} | {ok} |")
    print(f"\nRanking order match: {'OK' if ref_order == spyre_order else 'MISMATCH'}")
    print(f"Max absolute diff: {max_diff:.4f}  (threshold: {SCORE_ATOL})")

    assert max_diff <= SCORE_ATOL, (
        f"Max absolute score diff {max_diff:.4f} exceeds {SCORE_ATOL}.\n"
        f"  HF scores    : {ref_scores.tolist()}\n"
        f"  Spyre scores : {spyre_scores.tolist()}"
    )
    assert ref_order == spyre_order, (
        f"Ranking order mismatch.\n"
        f"  HF order    : {ref_order}\n"
        f"  Spyre order : {spyre_order}\n"
        f"  HF scores   : {ref_scores.tolist()}\n"
        f"  Spyre scores: {spyre_scores.tolist()}"
    )
