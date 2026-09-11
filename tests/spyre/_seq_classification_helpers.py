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
Shared helper for Spyre sequence-classification E2E tests.

Used by both ``test_e2e_reranker_compare_spyre.py`` and
``test_e2e_seq_classification_compare_spyre.py``.  Both tasks use the same
``AutoModelForSequenceClassification`` auto-class and the same
``prefill_sequence_classification`` call-site; they differ only in their input
texts and their assertions (ranking-order vs cosine similarity).
"""

from __future__ import annotations

import types

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import move_model_to_spyre, prefill_sequence_classification
from tests.conftest import load_ref_model


def run_seq_classification_cpu_vs_spyre(
    model_path: str,
    adapter: types.ModuleType,
    inputs: list[str] | list[tuple[str, str]],
) -> dict:
    """Load a seq-classification model, run a CPU reference forward, then run
    the adapter on Spyre via ``prefill_sequence_classification``.

    Args:
        model_path: HuggingFace model identifier.
        adapter: Resolved adapter module (provides ``_run_backbone_forward`` and
            ``prepare_for_spyre`` indirectly via ``move_model_to_spyre``).
        inputs: Tokenizer inputs — either a list of strings (single-sentence
            classification) or a list of ``(query, document)`` pairs (reranking).

    Returns:
        A dict with keys:
            ``ref_logits``   – ``[B, num_labels]`` float CPU tensor (HF reference).
            ``spyre_logits`` – ``[B, num_labels]`` float CPU tensor (adapter on Spyre).
            ``dtype``        – dtype used for the Spyre model.
    """
    dtype = dtype_for_model_path(model_path, target_device="spyre")

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
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", None)

    # --- HF reference on CPU ---
    print("  Running HF reference on CPU ...")
    with torch.no_grad():
        ref_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits.float()
    print(f"  HF logits shape: {ref_logits.shape}")

    # --- Adapter on Spyre ---
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    print("  Running adapter on Spyre ...")
    with torch.no_grad():
        spyre_logits = prefill_sequence_classification(
            adapter._run_backbone_forward,
            model,
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
        ).float()
    print(f"  Spyre logits shape: {spyre_logits.shape}")

    return {
        "ref_logits": ref_logits,
        "spyre_logits": spyre_logits,
        "dtype": dtype,
    }
