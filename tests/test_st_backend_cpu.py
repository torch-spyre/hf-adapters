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

"""CPU accuracy test for the sentence-transformers ``backend="spyre"`` hook.

Loads each checkpoint twice:
  1. Stock ``SentenceTransformer`` on CPU (reference).
  2. ``SentenceTransformer(..., backend="spyre")`` via ``hf_adapters.st_backend``
     with ``DEVICE`` patched to ``"cpu"`` and compiled blocks unwrapped.

Compares per-sentence cosine similarity between the two sets of embeddings.
All similarities must exceed ``COS_SIM_THRESHOLD``.

Usage::

    python tests/test_st_backend_cpu.py
    python tests/test_st_backend_cpu.py qwen3_embed
"""

import gc
import importlib.util
import os
import sys

import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Patch DEVICE="cpu" in hf_common BEFORE any adapter import.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")

_common_path = os.path.join(ADAPTERS_DIR, "hf_common.py")
_common_spec = importlib.util.spec_from_file_location(
    "hf_adapters.hf_common", _common_path
)
_common_mod = importlib.util.module_from_spec(_common_spec)
sys.modules["hf_adapters.hf_common"] = _common_mod
_common_spec.loader.exec_module(_common_mod)
_common_mod.DEVICE = "cpu"

_pkg = type(sys)("hf_adapters")
_pkg.__path__ = [ADAPTERS_DIR]
sys.modules["hf_adapters"] = _pkg

import hf_adapters.st_backend  # noqa: F401, E402  (registers "spyre" backend with ST)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

COS_SIM_THRESHOLD = 0.999

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Spyre accelerates transformer inference at low latency.",
    "Embeddings are dense vector representations of text.",
    "Paris is the capital of France.",
]

TEST_MODELS = {
    "qwen3_embed": {
        "name": "Qwen3-Embedding-0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
    },
    "qwen2_embed": {
        "name": "GTE-Qwen2-1.5B",
        "path": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    },
    "e5_mistral": {
        "name": "E5-Mistral-7B",
        "path": "intfloat/e5-mistral-7b-instruct",
    },
    "bge_base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap_compiled_blocks(model):
    """Replace torch.compile wrappers with their original modules for CPU testing."""
    if hasattr(model, "_spyre_compiled_blocks"):
        model._spyre_compiled_blocks = [
            getattr(b, "_orig_mod", b) for b in model._spyre_compiled_blocks
        ]


def test_st_backend_cpu(model_key):
    from sentence_transformers import SentenceTransformer

    cfg = TEST_MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"  {cfg['name']}  ({cfg['path']})")
    print(f"{'='*70}")

    # --- Reference: stock ST on CPU ---
    print("  Loading reference (stock SentenceTransformer) ...")
    ref_model = SentenceTransformer(cfg["path"], device="cpu")
    ref_embeddings = ref_model.encode(TEST_SENTENCES, convert_to_tensor=True)  # [N, H]
    del ref_model
    gc.collect()

    # --- Ours: ST with backend="spyre" (DEVICE patched to cpu) ---
    print("  Loading spyre backend model ...")
    spyre_model = SentenceTransformer(cfg["path"], backend="spyre", device="cpu")
    _unwrap_compiled_blocks(spyre_model._first_module().model)
    spyre_embeddings = spyre_model.encode(
        TEST_SENTENCES, convert_to_tensor=True
    )  # [N, H]
    del spyre_model
    gc.collect()

    # Normalize both before cosine sim (ref may already be normalized, but be explicit)
    ref_norm = F.normalize(ref_embeddings.float(), dim=-1)
    spyre_norm = F.normalize(spyre_embeddings.float(), dim=-1)

    cos_sims = (ref_norm * spyre_norm).sum(dim=-1)  # [N]

    print(f"\n  {'Sentence':<55} {'cos_sim':>8}")
    print(f"  {'-'*55} {'-'*8}")
    all_pass = True
    for sent, sim in zip(TEST_SENTENCES, cos_sims.tolist()):
        ok = sim >= COS_SIM_THRESHOLD
        flag = "OK" if ok else "FAIL"
        print(f"  {sent[:54]:<55} {sim:>8.6f}  {flag}")
        if not ok:
            all_pass = False

    min_sim = cos_sims.min().item()
    result = "PASS" if all_pass else "FAIL"
    print(f"\n  Min cosine similarity: {min_sim:.6f}  (threshold: {COS_SIM_THRESHOLD})")
    print(f"  Result: {result}")
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = ["qwen3_embed"]

    results = {}
    for key in keys:
        if key not in TEST_MODELS:
            print(f"Unknown model key '{key}'. Available: {list(TEST_MODELS)}")
            sys.exit(1)
        try:
            results[key] = test_st_backend_cpu(key)
        except Exception:
            import traceback

            traceback.print_exc()
            results[key] = False

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for key, passed in results.items():
        name = TEST_MODELS[key]["name"]
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<40} {status}")
    print(f"{'='*70}")

    if not all(results.values()):
        sys.exit(1)
