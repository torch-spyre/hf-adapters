# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU accuracy for native ``AutoSpyreModelForQuestionAnswering`` forward."""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from tests.conftest import (
    get_dtype_for_cpu,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import QUESTION_ANSWERING_PATHS

pytestmark = pytest.mark.model_harness("question_answering")

QUESTIONS = ["Where does the engineer live?", "What color is the sky?"]
CONTEXTS = [
    "The engineer lives in Berlin and works on machine learning systems.",
    "On a clear day, the sky appears blue.",
]
COSINE_THRESHOLD = 0.999


@pytest.mark.parametrize(
    "model_path", QUESTION_ANSWERING_PATHS, ids=QUESTION_ANSWERING_PATHS
)
def test_native_forward(model_path: str) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = get_dtype_for_cpu(model_path)
    adapter_module = resolve_adapter_module_for_test(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        QUESTIONS,
        CONTEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )

    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForQuestionAnswering,
    )
    with torch.no_grad():
        ref_outputs = ref_model(**encoded, return_dict=True)
    del ref_model
    gc.collect()

    model = auto_spyre_model.AutoSpyreModelForQuestionAnswering.from_pretrained(
        model_path, dtype=dtype
    )
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)
        tuple_outputs = model(**encoded, return_dict=False)
    del model
    gc.collect()

    for logits, ref_logits in (
        (outputs.start_logits, ref_outputs.start_logits),
        (outputs.end_logits, ref_outputs.end_logits),
    ):
        assert logits.shape == ref_logits.shape
        assert logits.device.type == "cpu"
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

    print("\n## Question-Answering CPU Comparison\n")
    print(
        "| Question | HF Answer | Adapter Answer | Start Cosine | End Cosine | Match |"
    )
    print(
        "|----------|-----------|----------------|--------------|------------|-------|"
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
    assert torch.equal(tuple_outputs[0], outputs.start_logits)
    assert torch.equal(tuple_outputs[1], outputs.end_logits)
