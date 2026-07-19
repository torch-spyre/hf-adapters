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
import sys

import pytest
from transformers import AutoTokenizer

from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.cpu._generate_helpers import (
    MAX_NEW_TOKENS,
    PROMPTS,
    hf_reference_outputs,
)
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_multibatch(model_path: str) -> None:
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    adapter_mod = resolve_adapter_module_for_test(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # HF reference (per-prompt, BEFORE patching for cleanliness)
    model = load_ref_model(model_path, adapter_mod)
    hf_outputs = hf_reference_outputs(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)
    del model
    gc.collect()

    # Adapter batched generate
    model = load_ref_model(model_path, adapter_mod)
    adapter_mod.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)
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
