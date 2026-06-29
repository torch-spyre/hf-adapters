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
E2E smoke test: load HF model on Spyre, generate tokens, verify non-trivial.

Verifies:
  - Model loads and moves to Spyre without error
  - generate() produces non-empty output
  - Generated tokens are not all-zero or all-same

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py
    pytest -s -vvv "tests/spyre/test_e2e_smoke_spyre.py::test_e2e_smoke_spyre[Qwen/Qwen3-0.6B]"
"""

import time

import pytest
from model_registry import CAUSAL_PATHS

from tests._helpers import torch_dtype_for_model_path


def _run_smoke(model_path):
    """Load model, generate 5 tokens, validate output. Returns a result dict."""
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM

    print(f"\n{'=' * 70}")
    print(f"  loading from {model_path}")
    print(f"{'=' * 70}")

    dtype = torch_dtype_for_model_path(model_path)
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")

    t0 = time.time()
    outputs = model.generate(
        tokenizer,
        [prompt],
        max_new_tokens=5,
        do_sample=False,
        timing=True,
    )
    gen_time = time.time() - t0

    output_text = outputs[0] if outputs else ""
    print(f"  Output: {output_text!r}")
    print(f"  Generate time: {gen_time:.1f}s")

    checks = {
        "non_empty": len(output_text.strip()) > 0,
        "not_all_spaces": output_text.strip() != "",
    }
    if output_text:
        gen_ids = tokenizer.encode(output_text, add_special_tokens=False)
        checks["has_tokens"] = len(gen_ids) > 0
        checks["not_all_zero"] = not all(t == 0 for t in gen_ids)
        checks["not_all_same"] = len(set(gen_ids)) > 1 or len(gen_ids) <= 1
        checks["token_ids"] = gen_ids
    else:
        checks["has_tokens"] = False
        checks["not_all_zero"] = False
        checks["not_all_same"] = False
        checks["token_ids"] = []

    passed = all(v for k, v in checks.items() if k != "token_ids")
    return {
        "model": model_path,
        "status": "PASS" if passed else "FAIL",
        "tokens": len(checks.get("token_ids", [])),
        "text": output_text[:50],
        "load_s": load_time,
        "gen_s": gen_time,
        "checks": checks,
    }


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_e2e_smoke_spyre(model_path):
    result = _run_smoke(model_path)
    print("\n## E2E Smoke Test Results\n")
    print("| Model | Status | Tokens | Generated Text | Load (s) | Gen (s) |")
    print("|-------|--------|--------|----------------|----------|---------|")
    print(
        f"| {result['model']} | {result['status']} | {result['tokens']} "
        f"| {result['text']!r} | {result['load_s']:.1f} | {result['gen_s']:.1f} |"
    )
    assert result["status"] == "PASS", result
