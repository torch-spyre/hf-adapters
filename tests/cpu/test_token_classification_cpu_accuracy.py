# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU accuracy for native ``AutoSpyreModelForTokenClassification`` forward."""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from tests.conftest import (
    get_dtype_for_cpu,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import TOKEN_CLASSIFICATION_PATHS

pytestmark = pytest.mark.model_harness("token_classification")

SENTENCES = [
    "John lives in New York and works at IBM.",
    "The Eiffel Tower is located in Paris, France.",
]
COSINE_THRESHOLD = 0.999


@pytest.mark.parametrize(
    "model_path", TOKEN_CLASSIFICATION_PATHS, ids=TOKEN_CLASSIFICATION_PATHS
)
def test_native_forward(model_path: str) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = get_dtype_for_cpu(model_path)
    adapter_module = resolve_adapter_module_for_test(
        model_path,
        mapping=auto_spyre_model.TOKEN_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        SENTENCES,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForTokenClassification,
    )
    with torch.no_grad():
        ref_logits = ref_model(**encoded, return_dict=True).logits.float()
    del ref_model
    gc.collect()

    model = auto_spyre_model.AutoSpyreModelForTokenClassification.from_pretrained(
        model_path, dtype=dtype
    )
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)
        logits = outputs.logits.float()
        tuple_outputs = model(**encoded, return_dict=False)
    del model
    gc.collect()

    assert logits.shape == ref_logits.shape
    assert torch.isfinite(logits).all()

    real_tokens = encoded["attention_mask"].bool()

    # Per-token cosine similarity over the label-logit dimension.
    # Flatten real tokens from [B, L] mask → 1-D index so each real position
    # is compared independently (sentences may have different lengths).
    cosine = F.cosine_similarity(logits[real_tokens], ref_logits[real_tokens], dim=-1)

    ref_label_ids = ref_logits.argmax(dim=-1)
    label_ids = logits.argmax(dim=-1)

    print("\n## Token-Classification CPU Comparison\n")
    print("| Sentence | HF Tags | Adapter Tags | Min Cosine | Match |")
    print("|----------|---------|--------------|-----------|-------|")
    for i, sentence in enumerate(SENTENCES):
        mask = real_tokens[i]
        ref_tags = ref_label_ids[i][mask].tolist()
        tags = label_ids[i][mask].tolist()
        per_seq_cosine = F.cosine_similarity(
            logits[i][mask], ref_logits[i][mask], dim=-1
        )
        min_cos = per_seq_cosine.min().item()
        match = "Yes" if tags == ref_tags else "No"
        print(f"| {sentence[:40]}... | {ref_tags} | {tags} | {min_cos:.6f} | {match} |")

    assert cosine.min().item() >= COSINE_THRESHOLD
    assert torch.equal(label_ids[real_tokens], ref_label_ids[real_tokens])

    # Verify tuple-output form matches dict form
    assert torch.equal(tuple_outputs[0], outputs.logits)
