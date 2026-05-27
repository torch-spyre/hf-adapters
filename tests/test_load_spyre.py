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
Spyre loading test: every auto-class entry loads cleanly onto Spyre.

No forward pass — just verifies that ``AutoSpyreModelForCausalLM`` and
``AutoSpyreModel`` resolve, prepare, and move the model onto Spyre without
error. Causal-LM entries also check that a ``generate`` method is attached.

Usage (on Spyre pod):
    python3 tests/test_load_spyre.py                          # all models
    python3 tests/test_load_spyre.py qwen3 minilm             # subset
"""

import sys
import time
import traceback

# Import model registries and programmatically selected keys
from model_registry import (
    CAUSAL_KEYS,
    CAUSAL_LM_MODELS,
    EMBED_KEYS,
    EMBEDDING_MODELS,
)

# Build simplified path-only dicts for this standalone script
CAUSAL_LM_PATHS = {key: CAUSAL_LM_MODELS[key]["path"] for key in CAUSAL_KEYS}
EMBEDDING_PATHS = {key: EMBEDDING_MODELS[key]["path"] for key in EMBED_KEYS}


def load_causal_lm(key):
    import torch

    from hf_adapters import AutoSpyreModelForCausalLM

    path = CAUSAL_LM_PATHS[key]
    dtype = torch.float32 if key == "granite4" else torch.float16

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(path, dtype=dtype)
    load_s = time.time() - t0

    assert model is not None, f"{key}: from_pretrained returned None"
    assert callable(
        getattr(model, "generate", None)
    ), f"{key}: AutoSpyreModelForCausalLM did not attach generate()"
    return load_s


def load_embedding(key):
    import torch

    from hf_adapters import AutoSpyreModel

    path = EMBEDDING_PATHS[key]
    t0 = time.time()
    model = AutoSpyreModel.from_pretrained(path, dtype=torch.float16)
    load_s = time.time() - t0

    assert model is not None, f"{key}: from_pretrained returned None"
    return load_s


def run(key):
    if key in CAUSAL_LM_PATHS:
        kind = "causal-LM"
        path = CAUSAL_LM_PATHS[key]
        loader = load_causal_lm
    elif key in EMBEDDING_PATHS:
        kind = "embedding"
        path = EMBEDDING_PATHS[key]
        loader = load_embedding
    else:
        raise KeyError(key)

    print(f"\n{'=' * 70}")
    print(f"  [{kind}] {key}: loading {path}")
    print(f"{'=' * 70}")

    load_s = loader(key)
    print(f"  Load time: {load_s:.1f}s  -> PASS")
    return {"key": key, "kind": kind, "status": "PASS", "load_s": load_s}


if __name__ == "__main__":
    all_keys = list(CAUSAL_LM_PATHS.keys()) + list(EMBEDDING_PATHS.keys())
    which = sys.argv[1:] if len(sys.argv) > 1 else all_keys

    results = []
    for key in which:
        if key not in CAUSAL_LM_PATHS and key not in EMBEDDING_PATHS:
            print(f"Unknown: {key}. Options: {all_keys}")
            continue
        try:
            results.append(run(key))
        except Exception:
            print(f"\n!!! {key} FAILED:")
            traceback.print_exc()
            results.append({"key": key, "kind": "?", "status": "ERROR", "load_s": 0.0})

    print("\n## Spyre Load Test Results\n")
    print("| Key | Kind | Status | Load (s) |")
    print("|-----|------|--------|----------|")
    for r in results:
        print(f"| {r['key']} | {r['kind']} | {r['status']} | {r['load_s']:.1f} |")

    failed = [r for r in results if r["status"] != "PASS"]
    sys.exit(1 if failed else 0)
