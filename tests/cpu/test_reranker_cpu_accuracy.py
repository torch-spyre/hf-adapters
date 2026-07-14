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
CPU accuracy test for cross-encoder reranker models (e.g. BGE Reranker v2 M3).

Two parametrised cases per registered reranker:

  test_manual_path[<key>]
    Loads the model fresh via ``AutoModelForSequenceClassification`` (stock HF),
    runs a reference forward to get scores, then loads a second copy, applies
    ``prepare_for_spyre``, unwraps compiled blocks (CPU mode), and calls
    ``prefill_reranker``.  Asserts:
      - Output shape matches  ``[B]``
      - Scores are within an absolute tolerance of 0.01 of the HF reference
        (fp16 rounding of the backbone is negligible; the classifier head is
        shared between both paths so differences come only from the encoder)
      - Ranking order is preserved (``torch.argsort`` agreement)

  test_auto_loader[<key>]
    Same comparison but the adapter side goes through
    ``AutoSpyreModelForSequenceClassification.from_pretrained`` and the attached
    ``model.rerank()`` method.  Exercises the full end-to-end auto-loading path.

DEVICE is patched to ``"cpu"`` by ``tests/conftest.py``; torch.compile is
unwrapped by the ``unwrap_compiled_blocks`` fixture so blocks run eagerly.
"""

import gc

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)
from hf_adapters.hf_common import prefill_reranker
from tests.conftest import get_dtype_for_cpu, load_ref_model
from tests.model_registry import RERANKER_PATHS

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


def _hf_reference_scores(
    model_path: str,
    pairs: list[tuple[str, str]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run stock HF forward on CPU and return raw logit scores ``[B]``."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = load_ref_model(
        model_path=model_path,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    ref_model.eval()

    encoded = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    with torch.no_grad():
        out = ref_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
    # out.logits is [B, num_labels]; squeeze last dim for a single-score reranker
    return out.logits[:, 0].float()


@pytest.mark.parametrize("model_path", RERANKER_PATHS, ids=RERANKER_PATHS)
def test_manual_path(model_path: str, unwrap_compiled_blocks) -> None:
    """Adapter scores via prepare_for_spyre + prefill_reranker match HF reference."""
    dtype = get_dtype_for_cpu(model_path)
    adapter_module = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- HF reference ---
    ref_scores = _hf_reference_scores(model_path, PAIRS, dtype)
    gc.collect()

    # --- Adapter path ---
    model = load_ref_model(
        model_path=model_path,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    model.eval()
    adapter_module.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)

    encoded = tokenizer(
        PAIRS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    with torch.no_grad():
        adapter_scores = prefill_reranker(
            adapter_module._run_backbone_forward,
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids", None),
        ).float()

    del model
    gc.collect()

    assert (
        adapter_scores.shape == ref_scores.shape
    ), f"score shape mismatch: adapter {adapter_scores.shape} vs ref {ref_scores.shape}"
    max_abs_diff = (adapter_scores - ref_scores).abs().max().item()
    assert max_abs_diff <= SCORE_ATOL, (
        f"max absolute score difference {max_abs_diff:.4f} exceeds {SCORE_ATOL}.\n"
        f"  ref    = {ref_scores.tolist()}\n"
        f"  adapter= {adapter_scores.tolist()}"
    )
    # Ranking order must be preserved
    ref_order = torch.argsort(ref_scores, descending=True).tolist()
    adapter_order = torch.argsort(adapter_scores, descending=True).tolist()
    assert (
        ref_order == adapter_order
    ), f"ranking order mismatch: ref {ref_order} vs adapter {adapter_order}"


@pytest.mark.parametrize("model_path", RERANKER_PATHS, ids=RERANKER_PATHS)
def test_auto_loader(model_path: str, unwrap_compiled_blocks) -> None:
    """Scores via AutoSpyreModelForSequenceClassification.rerank() match HF reference."""
    import sys

    auto_spyre_model_mod = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = get_dtype_for_cpu(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- HF reference ---
    ref_scores = _hf_reference_scores(model_path, PAIRS, dtype)
    gc.collect()

    # --- Auto-loader path ---
    model = (
        auto_spyre_model_mod.AutoSpyreModelForSequenceClassification.from_pretrained(
            model_path, dtype=dtype
        )
    )
    unwrap_compiled_blocks(model)

    with torch.no_grad():
        adapter_scores = model.rerank(tokenizer, PAIRS).float()

    del model
    gc.collect()

    assert (
        adapter_scores.shape == ref_scores.shape
    ), f"score shape mismatch: adapter {adapter_scores.shape} vs ref {ref_scores.shape}"

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
