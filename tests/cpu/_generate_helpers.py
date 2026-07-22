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

"""CPU generate() helpers shared between test_generate_cpu and the weekly.

Kept in a non-``test_*`` module so importers (e.g. the weekly suite) don't
have to depend on a pytest-discovered file.
"""

import gc

import torch
from transformers import AutoTokenizer

from tests.conftest import load_ref_model, resolve_adapter_module_for_test

PROMPTS: list[str] = [
    "The capital of France is",
    "The chemical formula for water is",
]
MAX_NEW_TOKENS: int = 8


def hf_reference_outputs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Run HF native generate() on each prompt individually."""
    results: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_ids = out[0][encoded["input_ids"].shape[1] :]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


def simple_generate(model_path: str) -> None:
    """Load *model_path* on CPU and run HF ``generate()`` on the first prompt.

    Raises on any failure (adapter resolution, load, tokenizer, generate).
    The caller is responsible for classifying the exception.
    """
    adapter_mod = resolve_adapter_module_for_test(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(model_path, adapter_mod)
    try:
        hf_reference_outputs(model, tokenizer, PROMPTS[:1], MAX_NEW_TOKENS)
    finally:
        del model
        gc.collect()
