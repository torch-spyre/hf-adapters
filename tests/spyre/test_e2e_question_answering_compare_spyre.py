# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""E2E extractive-QA accuracy: stock CPU vs Spyre encoder + CPU head."""

import gc

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from hf_adapters import AutoSpyreModelForQuestionAnswering
from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from tests.conftest import get_dtype_for_cpu
from tests.model_registry import QUESTION_ANSWERING_PATHS

QUESTIONS = ["Where does Ariel live?", "What color is the sky?"]
CONTEXTS = [
    "Ariel lives in Berlin and works on machine learning systems.",
    "On a clear day, the sky appears blue.",
]
COSINE_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "model_path", QUESTION_ANSWERING_PATHS, ids=QUESTION_ANSWERING_PATHS
)
def test_e2e_question_answering_compare_spyre(model_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        QUESTIONS,
        CONTEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )

    ref_model = AutoModelForQuestionAnswering.from_pretrained(
        model_path, dtype=get_dtype_for_cpu(model_path), device_map="cpu"
    ).eval()
    with torch.no_grad():
        ref_outputs = ref_model(**encoded, return_dict=True)
    del ref_model
    gc.collect()

    model = AutoSpyreModelForQuestionAnswering.from_pretrained(
        model_path, dtype=torch_dtype_for_model_path(model_path)
    )
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)

    for name in ("start_logits", "end_logits"):
        logits = getattr(outputs, name).float()
        ref_logits = getattr(ref_outputs, name).float()
        assert logits.shape == ref_logits.shape
        assert torch.isfinite(logits).all()
        cosine = F.cosine_similarity(logits, ref_logits, dim=-1)
        assert cosine.min().item() >= COSINE_THRESHOLD
        assert torch.equal(logits.argmax(dim=-1), ref_logits.argmax(dim=-1))
