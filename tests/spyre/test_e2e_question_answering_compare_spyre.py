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
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests.model_registry import QUESTION_ANSWERING_PATHS

pytestmark = pytest.mark.model_harness("question_answering")

QUESTIONS = ["Where does the engineer live?", "What color is the sky?"]
CONTEXTS = [
    "The engineer lives in Berlin and works on machine learning systems.",
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
        model_path,
        dtype=dtype_for_model_path(model_path, target_device="cpu"),
        device_map="cpu",
    ).eval()
    with torch.no_grad():
        ref_outputs = ref_model(**encoded, return_dict=True)
    del ref_model
    gc.collect()

    model = AutoSpyreModelForQuestionAnswering.from_pretrained(
        model_path, dtype=dtype_for_model_path(model_path, target_device="spyre")
    )
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)

    for name in ("start_logits", "end_logits"):
        logits = getattr(outputs, name).float()
        ref_logits = getattr(ref_outputs, name).float()
        assert logits.shape == ref_logits.shape
        assert torch.isfinite(logits).all()

    real_tokens = encoded["attention_mask"].bool()
    start_cosine = torch.stack(
        [
            F.cosine_similarity(
                outputs.start_logits[i, mask].float(),
                ref_outputs.start_logits[i, mask].float(),
                dim=0,
            )
            for i, mask in enumerate(real_tokens)
        ]
    )
    end_cosine = torch.stack(
        [
            F.cosine_similarity(
                outputs.end_logits[i, mask].float(),
                ref_outputs.end_logits[i, mask].float(),
                dim=0,
            )
            for i, mask in enumerate(real_tokens)
        ]
    )
    ref_start_ids = ref_outputs.start_logits.argmax(dim=-1)
    ref_end_ids = ref_outputs.end_logits.argmax(dim=-1)
    start_ids = outputs.start_logits.argmax(dim=-1)
    end_ids = outputs.end_logits.argmax(dim=-1)

    print("\n## Question-Answering Comparison: CPU vs Spyre\n")
    print(
        "| Question | CPU Answer | Spyre Answer | Start Cosine | End Cosine | Match |"
    )
    print(
        "|----------|------------|--------------|--------------|------------|-------|"
    )
    for i, question in enumerate(QUESTIONS):
        ref_answer = tokenizer.decode(
            encoded["input_ids"][i, ref_start_ids[i] : ref_end_ids[i] + 1],
            skip_special_tokens=True,
        ).strip()
        answer = tokenizer.decode(
            encoded["input_ids"][i, start_ids[i] : end_ids[i] + 1],
            skip_special_tokens=True,
        ).strip()
        match = "Yes" if answer == ref_answer else "No"
        print(
            f"| {question} | {ref_answer} | {answer} | "
            f"{start_cosine[i].item():.6f} | {end_cosine[i].item():.6f} | {match} |"
        )

    assert start_cosine.min().item() >= COSINE_THRESHOLD
    assert end_cosine.min().item() >= COSINE_THRESHOLD
    assert torch.equal(start_ids, ref_start_ids)
    assert torch.equal(end_ids, ref_end_ids)
