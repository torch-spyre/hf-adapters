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
Multi-batch generate() test: verify that ``hf_common.generate()`` produces
correct per-sequence outputs when called with batch_size > 1.

For each registered model, ``test_multibatch[<key>]`` runs the same prompts
through stock HF ``generate(do_sample=False)`` per-prompt, then through the
adapter's batched ``generate()``, and asserts the decoded text matches.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest.
"""

import gc

import pytest
import torch
from transformers import AutoTokenizer

from hf_adapters.auto_spyre_model import resolve_adapter_module
from tests.conftest import load_ref_model
from tests.model_registry import CAUSAL_PATHS

PROMPTS: list[str] = [
    "The capital of France is",
    "The chemical formula for water is",
]
MAX_NEW_TOKENS: int = 8


def _hf_reference_outputs(
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


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_multibatch(model_path: str, unwrap_compiled_blocks, hf_common_mod) -> None:
    adapter_mod = resolve_adapter_module(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # HF reference (per-prompt, BEFORE patching for cleanliness)
    model = load_ref_model(model_path, adapter_mod)
    model.eval()
    model.requires_grad_(False)
    hf_outputs = _hf_reference_outputs(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)
    del model
    gc.collect()

    # Adapter batched generate
    model = load_ref_model(model_path, adapter_mod)
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)
    adapter_outputs = hf_common_mod.generate(
        adapter_mod._run_forward,
        model,
        tokenizer,
        PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    del model
    gc.collect()

    for i, (prompt, hf_out, adapter_out) in enumerate(
        zip(PROMPTS, hf_outputs, adapter_outputs)
    ):
        assert (
            hf_out.strip() == adapter_out.strip()
        ), f"prompt[{i}] {prompt!r}: HF {hf_out!r} != adapter {adapter_out!r}"
