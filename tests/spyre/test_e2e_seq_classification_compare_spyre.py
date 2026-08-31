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
E2E sequence-classification accuracy: HF stock forward (CPU) vs adapter (Spyre).

For each registered sequence-classification model, loads the model on CPU,
runs a reference forward to get ``[B, num_labels]`` logits, moves the same
model instance to Spyre, runs the adapter forward, and asserts that:

  - Output shape matches ``[B, num_labels]``
  - Per-sample cosine similarity over the label dimension is >= threshold
  - Predicted class ids match exactly between CPU and Spyre

Execution is delegated to ``_seq_classification_helpers.run_seq_classification_cpu_vs_spyre``;
this file owns only the seq-classification-specific inputs and cosine assertions.

Usage (on Spyre LPAR)::

    # All registered sequence-classification models
    pytest -s -vvv tests/spyre/test_e2e_seq_classification_compare_spyre.py

    # Just the SST-2 DistilBERT checkpoint
    pytest -s -vvv tests/spyre/test_e2e_seq_classification_compare_spyre.py \\
        -k distilbert
"""

import pytest
import torch
import torch.nn.functional as F
from _seq_classification_helpers import run_seq_classification_cpu_vs_spyre
from model_registry import SEQ_CLASSIFICATION_PATHS

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)

pytestmark = pytest.mark.model_harness("seq_classification")

TEXTS: list[str] = [
    "Hello, my dog is cute.",
    "This movie was absolutely terrible.",
    "The weather is nice today.",
]

# Spyre fp16 backbone vs CPU fp32: cosine over num_labels should be very tight.
COSINE_THRESHOLD: float = 0.99


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_e2e_seq_classification_compare_spyre(model_path: str) -> None:
    adapter = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    result = run_seq_classification_cpu_vs_spyre(model_path, adapter, TEXTS)

    ref_logits = result["ref_logits"]
    spyre_logits = result["spyre_logits"]

    ref_ids = ref_logits.argmax(dim=-1)
    spyre_ids = spyre_logits.argmax(dim=-1)
    cos = F.cosine_similarity(spyre_logits, ref_logits, dim=-1)  # [B]

    print("\n## Seq Classification: HF (CPU) vs Adapter (Spyre)\n")
    print("| Text | HF id | Spyre id | Cosine | Match |")
    print("|------|-------|----------|--------|-------|")
    for text, ref_id, spyre_id, c in zip(TEXTS, ref_ids, spyre_ids, cos):
        match = "Yes" if ref_id.item() == spyre_id.item() else "No"
        print(
            f"| {text} | {ref_id.item()} | {spyre_id.item()} | {c.item():.6f} | {match} |"
        )

    assert (
        spyre_logits.shape == ref_logits.shape
    ), f"shape mismatch: spyre {spyre_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(spyre_logits).all()
    assert cos.min().item() >= COSINE_THRESHOLD, (
        f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}\n"
        f"  HF logits    : {ref_logits.tolist()}\n"
        f"  Spyre logits : {spyre_logits.tolist()}"
    )
    assert torch.equal(spyre_ids, ref_ids), (
        f"predicted class mismatch.\n"
        f"  HF ids    : {ref_ids.tolist()}\n"
        f"  Spyre ids : {spyre_ids.tolist()}"
    )
