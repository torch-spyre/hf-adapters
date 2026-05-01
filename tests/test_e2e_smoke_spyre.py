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

import torch
import torch_spyre  # noqa: F401 — registers Spyre device

MODEL_REGISTRY = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_adapters.hf_qwen3",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "path": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "hf_adapters.hf_granite",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_adapters.hf_granite",
    },
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_adapters.hf_granitemoehybrid",
    },
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_adapters.hf_smollm3",
    },
    "llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_adapters.hf_llama",
    },
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_adapters.hf_phi3",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_adapters.hf_qwen2",
    },
    "mistral": {
        "name": "Mistral 7B v0.3",
        "path": "mistralai/Mistral-7B-v0.3",
        "adapter": "hf_adapters.hf_mistral",
    },
    "olmo": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_adapters.hf_olmo",
    },
    "olmo2": {
        "name": "OLMo2 1B",
        "path": "allenai/OLMo-2-0425-1B",
        "adapter": "hf_adapters.hf_olmo2",
    },
    "falcon3": {
        "name": "Falcon 3 1B",
        "path": "tiiuae/Falcon3-1B-Base",
        "adapter": "hf_adapters.hf_llama",
    },
    "deepseek-coder": {
        "name": "DeepSeek-Coder 1.3B",
        "path": "deepseek-ai/deepseek-coder-1.3b-base",
        "adapter": "hf_adapters.hf_llama",
    },
    # Ministral 3B is gated — requires HF auth. HF token configured on Spyre pod.
    # "ministral": {
    #     "name": "Ministral 3B",
    #     "path": "mistralai/Ministral-3B-Instruct",
    #     "adapter": "hf_adapters.hf_mistral",
    # },
    "yi": {
        "name": "Yi 1.5 6B",
        "path": "01-ai/Yi-1.5-6B",
        "adapter": "hf_adapters.hf_llama",
    },
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_adapters.hf_granite_vision",
    },
}


def run_smoke(model_key):
    """Load model, generate 5 tokens, validate output."""
    import importlib

    from transformers import AutoTokenizer

    info = MODEL_REGISTRY[model_key]
    adapter = importlib.import_module(info["adapter"])

    print(f"\n{'='*70}")
    print(f"  {info['name']}: loading from {info['path']}")
    print(f"{'='*70}")

    # Load model (downloads weights, prepares, moves to Spyre)
    t0 = time.time()
    model = adapter.load_model(info["path"])
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")

    # Generate
    t0 = time.time()
    outputs = adapter.generate(
        model, tokenizer, [prompt],
        max_new_tokens=5, do_sample=False, timing=True,
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
            results.append({
                "model": MODEL_REGISTRY[key]["name"],
                "status": "ERROR",
                "tokens": 0,
                "text": "",
                "load_s": 0,
                "gen_s": 0,
            })

    # Summary table
    print(f"\n## E2E Smoke Test Results\n")
    print(f"| Model | Status | Tokens | Generated Text | Load (s) | Gen (s) |")
    print(f"|-------|--------|--------|----------------|----------|---------|")
    for r in results:
        print(f"| {r['model']} | {r['status']} | {r['tokens']} "
              f"| {r['text']!r} | {r['load_s']:.1f} | {r['gen_s']:.1f} |")
