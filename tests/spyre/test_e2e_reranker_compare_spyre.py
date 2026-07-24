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

Usage (on Spyre LPAR)::

    # All registered rerankers
    pytest -s -vvv tests/spyre/test_e2e_reranker_compare_spyre.py

    # Just BGE Reranker v2 M3
    pytest -s -vvv tests/spyre/test_e2e_reranker_compare_spyre.py -k bge_reranker
"""

import pytest
import torch
from conftest import load_ref_model
from model_registry import RERANKER_PATHS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
    torch_dtype_for_model_path,
)
from hf_adapters.hf_common import move_model_to_spyre, prefill_reranker

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


def _run_reranker_test(model_path: str) -> dict:
    """Full CPU-vs-Spyre comparison for one reranker model.

    Returns a result dict with raw scores and match flags for assertion and
    printing.
    """
    adapter = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    dtype = torch_dtype_for_model_path(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"  dtype: {dtype}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter,
        auto_model_cls=AutoModelForSequenceClassification,
    )

    encoded = tokenizer(
        PAIRS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", None)

    lengths = attention_mask.sum(dim=1).tolist()
    print(
        f"  Inputs: {len(PAIRS)} pairs, padded to {input_ids.shape[1]} tokens"
        f" (real lengths: {lengths})"
    )

    # --- HF reference on CPU ---
    print("  Running HF reference on CPU ...")
    with torch.no_grad():
        ref_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    ref_scores = ref_out.logits[:, 0].float()
    print(f"  HF scores: {ref_scores.tolist()}")

    # --- Adapter on Spyre ---
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    print("  Running adapter on Spyre ...")
    with torch.no_grad():
        spyre_scores = prefill_reranker(
            adapter._run_backbone_forward,
            model,
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
        ).float()
    print(f"  Spyre scores: {spyre_scores.tolist()}")

    abs_diffs = (spyre_scores - ref_scores).abs()
    max_diff = abs_diffs.max().item()
    ref_order = torch.argsort(ref_scores, descending=True).tolist()
    spyre_order = torch.argsort(spyre_scores, descending=True).tolist()
    rank_match = ref_order == spyre_order

    print(f"  Max absolute diff: {max_diff:.4f}  (threshold: {SCORE_ATOL})")
    print(f"  HF ranking:    {ref_order}")
    print(f"  Spyre ranking: {spyre_order}  {'OK' if rank_match else 'MISMATCH'}")

    return {
        "model_path": model_path,
        "ref_scores": ref_scores.tolist(),
        "spyre_scores": spyre_scores.tolist(),
        "abs_diffs": abs_diffs.tolist(),
        "max_diff": max_diff,
        "ref_order": ref_order,
        "spyre_order": spyre_order,
        "score_match": max_diff <= SCORE_ATOL,
        "rank_match": rank_match,
    }


def _print_result_table(result: dict) -> None:
    """Print a per-pair comparison table."""
    print(f"\n## Reranker E2E: HF (CPU) vs Adapter (Spyre) — {result['model_path']}\n")
    print("| Pair | HF Score | Spyre Score | Abs Diff | Match |")
    print("|------|----------|-------------|----------|-------|")
    for i, (hs, ss, d) in enumerate(
        zip(result["ref_scores"], result["spyre_scores"], result["abs_diffs"])
    ):
        ok = "OK" if d <= SCORE_ATOL else "FAIL"
        print(f"| {i} | {hs:.4f} | {ss:.4f} | {d:.4f} | {ok} |")
    print(f"\nRanking order match: {'OK' if result['rank_match'] else 'MISMATCH'}")
    print(f"Max absolute diff: {result['max_diff']:.4f}")


@pytest.mark.parametrize("model_path", RERANKER_PATHS, ids=RERANKER_PATHS)
def test_e2e_reranker_compare_spyre(model_path: str) -> None:
    result = _run_reranker_test(model_path)
    _print_result_table(result)

    assert result["score_match"], (
        f"Max absolute score diff {result['max_diff']:.4f} exceeds {SCORE_ATOL}.\n"
        f"  HF scores    : {result['ref_scores']}\n"
        f"  Spyre scores : {result['spyre_scores']}"
    )
    assert result["rank_match"], (
        f"Ranking order mismatch.\n"
        f"  HF order    : {result['ref_order']}\n"
        f"  Spyre order : {result['spyre_order']}\n"
        f"  HF scores   : {result['ref_scores']}\n"
        f"  Spyre scores: {result['spyre_scores']}"
    )
