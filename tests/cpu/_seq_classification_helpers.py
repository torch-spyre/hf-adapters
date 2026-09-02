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
Shared helper for CPU sequence-classification accuracy tests.

Used by both ``test_reranker_cpu_accuracy.py`` and
``test_seq_classification_cpu_accuracy.py``.  Both tasks load via
``AutoSpyreModelForSequenceClassification`` and compare against a stock HF
reference; they differ only in their input texts/pairs and assertions.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tests.conftest import load_ref_model
from tests.cpu.conftest import _unwrap_compiled_blocks


def run_seq_classification_auto_loader_vs_ref(
    model_path: str,
    inputs: list[str] | list[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the auto-loader seq-classification path against a stock HF reference.

    Tokenizes ``inputs``, runs a stock HF reference forward, then loads via
    ``AutoSpyreModelForSequenceClassification.from_pretrained`` and calls
    ``model(**encoded, return_dict=True)``.

    Args:
        model_path: HuggingFace model identifier.
        inputs: Tokenizer inputs — a list of strings (single-sentence
            classification) or a list of ``(query, document)`` pairs (reranking).

    Returns:
        ``(ref_logits, adapter_logits)`` — both ``[B, num_labels]`` float CPU tensors.
    """
    auto_spyre_model_mod = sys.modules["hf_adapters.auto_spyre_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )

    # --- HF reference ---
    ref_model = load_ref_model(
        model_path=model_path,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    ref_model.eval()
    with torch.no_grad():
        ref_logits = ref_model(**encoded, return_dict=True).logits.float()
    del ref_model

    # --- Auto-loader path ---
    model = (
        auto_spyre_model_mod.AutoSpyreModelForSequenceClassification.from_pretrained(
            model_path
        )
    )
    _unwrap_compiled_blocks(model)

    with torch.no_grad():
        adapter_logits = model(**encoded, return_dict=True).logits.float()
    del model

    return ref_logits, adapter_logits
