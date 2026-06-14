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

Usage (on Spyre pod):
    python3 test_e2e_smoke_spyre.py [qwen3|granite|granite2b|granite4|smollm3]

Verifies:
  - Model loads and moves to Spyre without error
  - generate() produces non-empty output
  - Generated tokens are not all-zero or all-same
"""

import sys
import time
import traceback

from model_registry import CAUSAL_LM_MODELS as MODEL_REGISTRY


def run_smoke(model_key):
    """Load model, generate 5 tokens, validate output."""
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM

    info = MODEL_REGISTRY[model_key]

    print(f"\n{'='*70}")
    print(f"  {info['name']}: loading from {info['path']}")
    print(f"{'='*70}")

    # Load model (downloads weights, prepares, moves to Spyre)
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")

    # Generate
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

    # Validate
    checks = {
        "non_empty": len(output_text.strip()) > 0,
        "not_all_spaces": output_text.strip() != "",
    }

    # Encode output to check token diversity
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
        "model": info["name"],
        "status": "PASS" if passed else "FAIL",
        "tokens": len(checks.get("token_ids", [])),
        "text": output_text[:50],
        "load_s": load_time,
        "gen_s": gen_time,
        "checks": checks,
    }


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["qwen3"]

    results = []
    for key in which:
        if key not in MODEL_REGISTRY:
            print(f"Unknown: {key}. Options: {list(MODEL_REGISTRY.keys())}")
            continue
        try:
            r = run_smoke(key)
            results.append(r)
        except Exception:
            print(f"\n!!! {MODEL_REGISTRY[key]['name']} FAILED:")
            traceback.print_exc()
            results.append(
                {
                    "model": MODEL_REGISTRY[key]["name"],
                    "status": "ERROR",
                    "tokens": 0,
                    "text": "",
                    "load_s": 0,
                    "gen_s": 0,
                }
            )

    # Summary table
    print("\n## E2E Smoke Test Results\n")
    print("| Model | Status | Tokens | Generated Text | Load (s) | Gen (s) |")
    print("|-------|--------|--------|----------------|----------|---------|")
    for r in results:
        print(
            f"| {r['model']} | {r['status']} | {r['tokens']} "
            f"| {r['text']!r} | {r['load_s']:.1f} | {r['gen_s']:.1f} |"
        )
