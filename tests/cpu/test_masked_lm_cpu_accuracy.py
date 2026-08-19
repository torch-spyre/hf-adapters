# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU accuracy for native ``AutoSpyreModelForMaskedLM`` forward."""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tests.conftest import (
    get_dtype_for_cpu,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import MASKED_LM_PATHS

pytestmark = pytest.mark.model_harness("masked_lm")

PROMPTS = [
    "The capital of France is {mask}.",
    "A language model can predict a {mask} token in a sentence.",
]
COSINE_THRESHOLD = 0.999


@pytest.mark.parametrize("model_path", MASKED_LM_PATHS, ids=MASKED_LM_PATHS)
def test_auto_loader(model_path: str) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = get_dtype_for_cpu(model_path)
    adapter_module = resolve_adapter_module_for_test(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer.mask_token is not None

    encoded = tokenizer(
        [prompt.format(mask=tokenizer.mask_token) for prompt in PROMPTS],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForMaskedLM,
    )
    with torch.no_grad():
        ref_logits = ref_model(**encoded, return_dict=True).logits.float()
    del ref_model
    gc.collect()

    model = auto_spyre_model.AutoSpyreModelForMaskedLM.from_pretrained(
        model_path, dtype=dtype
    )
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)
        logits = outputs.logits.float()
    del model
    gc.collect()

    assert logits.shape == ref_logits.shape
    assert torch.isfinite(logits).all()

    real_tokens = encoded["attention_mask"].bool()
    cosine = F.cosine_similarity(logits, ref_logits, dim=-1)[real_tokens]

    masked = encoded["input_ids"].eq(tokenizer.mask_token_id)
    ref_token_ids = ref_logits[masked].argmax(dim=-1)
    token_ids = logits[masked].argmax(dim=-1)
    mask_cosine = F.cosine_similarity(logits[masked], ref_logits[masked], dim=-1)

    print("\n## Masked-LM CPU Comparison\n")
    print("| Prompt | HF Token | Adapter Token | Mask Cosine | Match |")
    print("|--------|----------|---------------|-------------|-------|")
    for prompt, ref_token_id, token_id, mask_cos in zip(
        PROMPTS, ref_token_ids, token_ids, mask_cosine
    ):
        ref_token = tokenizer.decode(ref_token_id).strip()
        token = tokenizer.decode(token_id).strip()
        match = "Yes" if ref_token_id.item() == token_id.item() else "No"
        print(f"| {prompt} | {ref_token} | {token} | {mask_cos.item():.6f} | {match} |")
    print(
        f"\nReal-token cosine: mean={cosine.mean().item():.6f}, "
        f"min={cosine.min().item():.6f}"
    )

    assert cosine.min().item() >= COSINE_THRESHOLD
    assert torch.equal(token_ids, ref_token_ids)
