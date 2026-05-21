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
CPU accuracy test for the embedding path.

Covers both decoder-backbone embedders (Qwen3-Embedding, etc.)
and encoder-only models (e.g., BERT-family).

Two test modes:

  Default (manual load): for each registered embedder,
    1. Load via stock ``AutoModel`` on CPU and run ``model.forward(...)`` to
       get a reference ``last_hidden_state``.
    2. Apply ``prepare_for_spyre``, unwrap compiled blocks, and call the
       appropriate prefill driver:
       - ``prefill_encoder`` for encoder-only adapters
         (adapter sets ``_is_encoder_only = True``).
       - ``prefill_embed`` for decoder-backbone embedders.
    3. Compare per-token cosine similarity and max abs diff.

  --auto-loader: loads via ``AutoSpyreModel.from_pretrained(...)``
    (single entry point that resolves the adapter from config). Reference
    forward still runs against a separately-loaded stock ``AutoModel``
    instance, since the adapter load patches RMSNorm globally and we
    want a clean reference.

Usage::

    python tests/test_embed_cpu_accuracy.py [qwen3-embed,bge-base,minilm]
    python tests/test_embed_cpu_accuracy.py --auto-loader [bge-base]
"""

import gc
import importlib
import importlib.util
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import adapter modules with DEVICE patched to "cpu" before any adapter loads.
# Mirrors the scaffolding in tests/test_adapter_cpu_accuracy.py.
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


def load_adapter(filename):
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Auto-loader support: pre-load all adapters, then auto_spyre_model
# ---------------------------------------------------------------------------

_AutoSpyreModel = None


def _get_auto_model_class():
    """Lazily pre-load every adapter and return AutoSpyreModel.

    Mirrors the auto-loader scaffolding in test_adapter_cpu_accuracy.py:
    each adapter must load against the DEVICE-patched hf_common, so we
    register the package, exec all adapter files into sys.modules, then
    import auto_spyre_model.
    """
    global _AutoSpyreModel
    if _AutoSpyreModel is not None:
        return _AutoSpyreModel

    adapter_files = [
        "hf_bert.py",
        "hf_granite.py",
        "hf_granite_vision.py",
        "hf_granitemoehybrid.py",
        "hf_llama.py",
        "hf_mistral.py",
        "hf_olmo.py",
        "hf_olmo2.py",
        "hf_phi3.py",
        "hf_qwen2.py",
        "hf_qwen3.py",
        "hf_smollm3.py",
    ]
    for name in adapter_files:
        mod = load_adapter(name)
        setattr(_pkg, name.replace(".py", ""), mod)

    auto_mod = load_adapter("auto_spyre_model.py")
    setattr(_pkg, "auto_spyre_model", auto_mod)
    _AutoSpyreModel = auto_mod.AutoSpyreModel
    return _AutoSpyreModel


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _unwrap_compiled_blocks(model):
    """Replace torch.compile-wrapped blocks with their CPU-runnable originals."""
    if not hasattr(model, "_spyre_compiled_blocks"):
        return
    unwrapped = []
    for cb in model._spyre_compiled_blocks:
        orig = getattr(cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None))
        unwrapped.append(orig if orig is not None else cb)
    model._spyre_compiled_blocks = unwrapped


def _per_token_cosine(a, b, attention_mask):
    """Mean cosine similarity over real (unmasked) tokens. ``a``, ``b``: [B, L, H]."""
    a32 = a.float()
    b32 = b.float()
    cos = F.cosine_similarity(a32, b32, dim=-1)  # [B, L]
    mask = attention_mask.bool()
    return cos[mask].mean().item(), cos[mask].min().item()


def _run_prefill(adapter_mod, model, input_ids, attention_mask):
    """Dispatch to prefill_encoder or prefill_embed based on adapter type."""
    if getattr(adapter_mod, "_is_encoder_only", False):
        return _common_mod.prefill_encoder(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )
    else:
        return _common_mod.prefill_embed(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def run_model_test(model_name, model_path, adapter_filename, dtype="float16"):
    from transformers import AutoModel, AutoTokenizer

    adapter_mod = load_adapter(adapter_filename)
    prepare_fn = adapter_mod.prepare_for_spyre

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    print(f"\n{'='*70}")
    print(f"  {model_name}: loading {model_path} ({dtype})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_path, dtype=torch_dtype, device_map="cpu")
    model.eval()
    model.requires_grad_(False)

    prompts = [
        "The capital of France is Paris.",
        "Sentence embeddings are useful.",
    ]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="right"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    print(f"  Inputs: {len(prompts)} prompts, padded to {input_ids.shape[1]} tokens")

    # --- HF reference ---
    # Run BEFORE prepare_for_spyre: the RMSNorm patch (decoder adapters) modifies
    # the class globally. Encoder adapters do not patch globally, but keeping the
    # same ordering discipline avoids surprises when mixing model types.
    print("  Running HF reference (AutoModel.forward) ...")
    with torch.no_grad():
        ref_out = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
    ref_hidden = ref_out.last_hidden_state  # [B, L, H]

    # --- Adapter ---
    print("  Preparing adapter ...")
    prepare_fn(model)
    _unwrap_compiled_blocks(model)

    prefill_label = (
        "prefill_encoder"
        if getattr(adapter_mod, "_is_encoder_only", False)
        else "prefill_embed"
    )
    print(f"  Running adapter ({prefill_label}) ...")
    with torch.no_grad():
        adapter_hidden, returned_mask = _run_prefill(
            adapter_mod, model, input_ids, attention_mask
        )
    del model
    gc.collect()

    # --- Compare ---
    assert (
        adapter_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: adapter {adapter_hidden.shape} vs ref {ref_hidden.shape}"
    mean_cos, min_cos = _per_token_cosine(adapter_hidden, ref_hidden, attention_mask)
    abs_diff = (adapter_hidden.float() - ref_hidden.float()).abs()
    real_mask = attention_mask.unsqueeze(-1).bool().expand_as(abs_diff)
    max_diff = abs_diff[real_mask].max().item()
    mean_diff = abs_diff[real_mask].mean().item()

    return {
        "model": model_name,
        "shape": tuple(adapter_hidden.shape),
        "mean_cos": mean_cos,
        "min_cos": min_cos,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


# ---------------------------------------------------------------------------
# Auto-loader test driver
# ---------------------------------------------------------------------------


def run_model_test_auto_loader(model_name, model_path, dtype="float16"):
    """Load via AutoSpyreModel.from_pretrained, compare prefill output vs HF."""
    from transformers import AutoModel, AutoTokenizer

    AutoSpyre = _get_auto_model_class()
    auto_mod = sys.modules["hf_adapters.auto_spyre_model"]

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    print(f"\n{'='*70}")
    print(f"  [AUTO] {model_name}: loading {model_path} ({dtype})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- HF reference ---
    # (load a clean instance; AutoSpyre load patches RMSNorm globally for decoders)
    print("  Loading HF reference (stock AutoModel) ...")
    ref_model = AutoModel.from_pretrained(
        model_path, dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    prompts = [
        "The capital of France is Paris.",
        "Sentence embeddings are useful.",
    ]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="right"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    print(f"  Inputs: {len(prompts)} prompts, padded to {input_ids.shape[1]} tokens")

    print("  Running HF reference (AutoModel.forward) ...")
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
    ref_hidden = ref_out.last_hidden_state
    del ref_model
    gc.collect()

    # --- Adapter via AutoSpyreModel ---
    print("  Loading via AutoSpyreModel.from_pretrained ...")
    model = AutoSpyre.from_pretrained(model_path, dtype=torch_dtype)
    _unwrap_compiled_blocks(model)

    adapter_module = auto_mod._resolve_adapter_module(model_path)

    prefill_label = (
        "prefill_encoder"
        if getattr(adapter_module, "_is_encoder_only", False)
        else "prefill_embed"
    )
    print(f"  Running adapter ({prefill_label}) ...")
    with torch.no_grad():
        adapter_hidden, _ = _run_prefill(
            adapter_module, model, input_ids, attention_mask
        )
    del model
    gc.collect()

    # --- Compare ---
    assert (
        adapter_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: adapter {adapter_hidden.shape} vs ref {ref_hidden.shape}"
    mean_cos, min_cos = _per_token_cosine(adapter_hidden, ref_hidden, attention_mask)
    abs_diff = (adapter_hidden.float() - ref_hidden.float()).abs()
    real_mask = attention_mask.unsqueeze(-1).bool().expand_as(abs_diff)
    max_diff = abs_diff[real_mask].max().item()
    mean_diff = abs_diff[real_mask].mean().item()

    return {
        "model": f"[AUTO] {model_name}",
        "shape": tuple(adapter_hidden.shape),
        "mean_cos": mean_cos,
        "min_cos": min_cos,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def print_results(result, threshold=0.9999):
    print(f"\n  {result['model']} Results")
    print(f"    shape:     {result['shape']}")
    print(f"    mean cos:  {result['mean_cos']:.6f}")
    print(f"    min cos:   {result['min_cos']:.6f}")
    print(f"    max diff:  {result['max_diff']:.4f}")
    print(f"    mean diff: {result['mean_diff']:.6f}")
    ok = result["min_cos"] >= threshold
    print(f"    status:    {'PASS' if ok else 'FAIL'} (threshold cos >= {threshold})")
    return ok


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # Decoder-backbone embedders
    "qwen3-embed": {
        "name": "Qwen3-Embedding 0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "adapter": "hf_qwen3.py",
    },
    # Encoder-only embedders
    "bge-base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
        "adapter": "hf_bert.py",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "hf_bert.py",
    },
}


def main():
    args = sys.argv[1:]
    use_auto_loader = False
    if "--auto-loader" in args:
        use_auto_loader = True
        args = [a for a in args if a != "--auto-loader"]

    selected = args if args else list(MODELS.keys())
    unknown = [s for s in selected if s not in MODELS]
    if unknown:
        print(f"Unknown model keys: {unknown}. Available: {list(MODELS.keys())}")
        sys.exit(2)

    all_ok = True
    for key in selected:
        info = MODELS[key]
        try:
            if use_auto_loader:
                result = run_model_test_auto_loader(
                    info["name"],
                    info["path"],
                    dtype=info.get("dtype", "float16"),
                )
            else:
                result = run_model_test(
                    info["name"],
                    info["path"],
                    info["adapter"],
                    dtype=info.get("dtype", "float16"),
                )
            ok = print_results(result)
            all_ok = all_ok and ok
        except Exception:
            traceback.print_exc()
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
