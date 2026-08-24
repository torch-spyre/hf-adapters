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
E2E token-classification accuracy: stock CPU logits vs Spyre encoder + CPU head.

Mirrors the structure of ``test_e2e_masked_lm_compare_spyre.py`` and
``test_e2e_question_answering_compare_spyre.py``: loads the HF model on CPU for
a reference forward, then loads via ``AutoSpyreModelForTokenClassification``
(encoder on Spyre, linear classifier head on CPU), and asserts:

  - Per-token logit cosine similarity ≥ 0.99 vs CPU reference (all real tokens).
  - Exact label-id match on every real token.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_token_classification_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_token_classification_compare_spyre.py -k bert-base-NER
"""

import gc

import pytest
import torch
import torch.nn.functional as F
from model_registry import TOKEN_CLASSIFICATION_PATHS
from transformers import AutoModelForTokenClassification, AutoTokenizer

from hf_adapters import AutoSpyreModelForTokenClassification
from hf_adapters.auto_spyre_model import dtype_for_model_path

pytestmark = pytest.mark.model_harness("token_classification")

# NER-targeted sentences covering a mix of entity types (PER, LOC, ORG).
# Kept consistent with the CPU accuracy test so results are directly comparable.
SENTENCES = [
    "John lives in New York and works at IBM.",
    "The Eiffel Tower is located in Paris, France.",
]
COSINE_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "model_path", TOKEN_CLASSIFICATION_PATHS, ids=TOKEN_CLASSIFICATION_PATHS
)
def test_e2e_token_classification_compare_spyre(model_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        SENTENCES,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    ref_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        dtype=dtype_for_model_path(model_path, target_device="cpu"),
        device_map="cpu",
    ).eval()
    with torch.no_grad():
        ref_logits = ref_model(**encoded, return_dict=True).logits.float()
    del ref_model
    gc.collect()

    model = AutoSpyreModelForTokenClassification.from_pretrained(
        model_path, dtype=dtype_for_model_path(model_path, target_device="spyre")
    )
    with torch.no_grad():
        logits = model(**encoded, return_dict=True).logits.float()

    assert logits.shape == ref_logits.shape
    assert torch.isfinite(logits).all()

    real_tokens = encoded["attention_mask"].bool()

    # Flatten real tokens across the batch; sentences may have different lengths,
    # so stack() would error — use boolean indexing instead.
    cosine = F.cosine_similarity(logits[real_tokens], ref_logits[real_tokens], dim=-1)

    ref_label_ids = ref_logits.argmax(dim=-1)
    label_ids = logits.argmax(dim=-1)

    id2label = model.config.id2label

    print("\n## Token-Classification Comparison: CPU vs Spyre\n")
    print("| Sentence | CPU Labels | Spyre Labels | Min Cosine | Match |")
    print("|----------|------------|--------------|-----------|-------|")
    for i, sentence in enumerate(SENTENCES):
        mask = real_tokens[i]
        ref_tags = [id2label[t] for t in ref_label_ids[i][mask].tolist()]
        tags = [id2label[t] for t in label_ids[i][mask].tolist()]
        per_seq_cosine = F.cosine_similarity(
            logits[i][mask], ref_logits[i][mask], dim=-1
        )
        min_cos = per_seq_cosine.min().item()
        match = "Yes" if tags == ref_tags else "No"
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][i][mask])
        print(
            f"| {sentence[:40]}... | {list(zip(tokens, ref_tags))} "
            f"| {list(zip(tokens, tags))} | {min_cos:.6f} | {match} |"
        )
    print(
        f"\nAll real-token cosine: mean={cosine.mean().item():.6f}, "
        f"min={cosine.min().item():.6f}"
    )

    assert cosine.min().item() >= COSINE_THRESHOLD
    assert torch.equal(label_ids[real_tokens], ref_label_ids[real_tokens])
