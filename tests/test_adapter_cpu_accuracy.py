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
CPU accuracy test: compare adapter forward passes against stock HF on CPU.

Tests prefill (full sequence) and 4 autoregressive decode steps, comparing
logits and greedy token selections at each step.

Usage:
    python tests/test_adapter_cpu_accuracy.py [granite|granite2b|qwen3|granite4|smollm3]
    python tests/test_adapter_cpu_accuracy.py --auto-loader [granite2b|llama|...]

The --auto-loader flag runs an end-to-end test through AutoSpyreModelForCausalLM,
comparing generate() output against HF native generation.

Requires: transformers, torch (2.x), sentencepiece
"""

import importlib
import importlib.util
import os
import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# Import adapter modules via importlib to patch DEVICE before loading.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")


def _load_adapter_module(filename):
    """Load an adapter .py file as a standalone module."""
    filepath = os.path.join(ADAPTERS_DIR, filename)
    mod_name = f"_adapter_{filename.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


# Step 1: load hf_common with DEVICE="cpu" and register it BEFORE any adapter
# imports, so that all subsequent imports (including auto_spyre_model and the
# adapter modules it pulls in) bind to this patched version.
_common_path = os.path.join(ADAPTERS_DIR, "hf_common.py")
_common_spec = importlib.util.spec_from_file_location(
    "hf_adapters.hf_common", _common_path
)
_common_mod = importlib.util.module_from_spec(_common_spec)
sys.modules["hf_adapters.hf_common"] = _common_mod
_common_spec.loader.exec_module(_common_mod)
_common_mod.DEVICE = "cpu"

# Step 2: register the hf_adapters package so cross-adapter imports resolve
# to our local directory rather than the installed package.
_pkg = type(sys)("hf_adapters")
_pkg.__path__ = [ADAPTERS_DIR]
sys.modules["hf_adapters"] = _pkg

# Step 3: now import auto_spyre_model — all adapter modules it loads will
# find the patched hf_common in sys.modules and bind DEVICE="cpu".
import hf_adapters.auto_spyre_model as _auto_spyre_model  # noqa: E402


def load_adapter(filename):
    """Load an adapter module by filename."""
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Auto-loader support: pre-load all adapters for AutoSpyreModelForCausalLM
# ---------------------------------------------------------------------------

# Extract adapter module names from CONFIG_TO_ADAPTER_MODULE_MAPPING

ADAPTER_FILES = []
for v in _auto_spyre_model.CONFIG_TO_ADAPTER_MODULE_MAPPING.values():
    ADAPTER_FILES.append(v.__name__.split(".")[-1])

_AutoSpyreModelForCausalLM = None


def _get_auto_model_class():
    """Lazily pre-load all adapters and return AutoSpyreModelForCausalLM."""
    global _AutoSpyreModelForCausalLM
    if _AutoSpyreModelForCausalLM is not None:
        return _AutoSpyreModelForCausalLM

    for name in ADAPTER_FILES:
        mod = load_adapter(f"{name}.py")
        setattr(_pkg, name, mod)

    auto_mod = load_adapter("auto_spyre_model.py")
    setattr(_pkg, "auto_spyre_model", auto_mod)
    _AutoSpyreModelForCausalLM = auto_mod.AutoSpyreModelForCausalLM
    return _AutoSpyreModelForCausalLM


# ---------------------------------------------------------------------------
# HF reference: token-by-token greedy with DynamicCache
# ---------------------------------------------------------------------------


def hf_greedy_steps(model, input_ids, num_decode=4):
    """Run stock HF model for prefill + N greedy decode steps.

    Returns list of dicts with logits/token for each step.
    Step 0 = prefill, steps 1..N = decode.
    """
    from transformers import DynamicCache

    results = []
    past = DynamicCache()
    ids = input_ids.clone()
    seq_len = ids.shape[1]

    for step in range(num_decode + 1):
        if step == 0:
            position_ids = torch.arange(seq_len).unsqueeze(0)
        else:
            position_ids = torch.tensor([[seq_len + step - 1]])

        with torch.no_grad():
            out = model(
                input_ids=ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )

        logits = out.logits
        last_logits = logits[0, -1, :].float()
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


# ---------------------------------------------------------------------------
# Adapter: prefill + decode using adapter _run_forward with KV cache
# ---------------------------------------------------------------------------


def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=4):
    """Run adapter forward for prefill + N greedy decode steps on CPU."""
    results = []
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    vocab_size = model.config.vocab_size

    # Match model dtype for KV caches and masks
    param_dtype = next(model.parameters()).dtype

    # Pre-allocate KV caches at full size
    max_cache_len = seq_len + num_decode
    key_caches = [
        torch.zeros(
            batch_size, num_kv_heads, max_cache_len, head_dim, dtype=param_dtype
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.zeros(
            batch_size, num_kv_heads, max_cache_len, v_head_dim, dtype=param_dtype
        )
        for _ in range(num_layers)
    ]

    # --- Prefill ---
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal_mask = torch.zeros((1, 1, seq_len, max_cache_len), dtype=param_dtype)
    for i in range(seq_len):
        causal_mask[:, :, i, i + 1 :] = -torch.inf

    with torch.no_grad():
        logits = run_forward_fn(
            model,
            input_ids,
            position_ids,
            causal_mask,
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=0,
        )

    last_logits = logits[0, -1, :].float()[:vocab_size]
    token = last_logits.argmax().item()
    results.append({"logits": last_logits, "token": token, "step": 0})

    cache_len = seq_len

    # --- Decode steps ---
    for step in range(1, num_decode + 1):
        next_ids = torch.tensor([[token]])
        next_pos = torch.tensor([[seq_len + step - 1]])
        decode_mask = torch.zeros((1, 1, 1, max_cache_len), dtype=param_dtype)
        decode_mask[:, :, :, cache_len + 1 :] = -torch.inf

        with torch.no_grad():
            logits = run_forward_fn(
                model,
                next_ids,
                next_pos,
                decode_mask,
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=cache_len,
            )

        last_logits = logits[0, -1, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        cache_len += 1

    return results


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def run_model_test(
    model_name,
    model_path,
    adapter_filename,
    num_decode=4,
    dtype="float16",
    load_fn=False,
):
    """Load model, run HF ref vs adapter, return comparison list."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_mod = load_adapter(adapter_filename)
    run_forward_fn = adapter_mod._run_forward
    prepare_fn = adapter_mod.prepare_for_spyre

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    print(f"\n{'='*70}")
    print(f"  {model_name}: loading {model_path} ({dtype})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if load_fn:
        model = adapter_mod.load_hf_model(model_path, torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
    model.eval()
    model.requires_grad_(False)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r}  ({input_ids.shape[1]} tokens)")

    # --- HF reference (BEFORE patching) ---
    print("  Running HF reference ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    # --- Adapter ---
    print("  Preparing adapter ...")
    prepare_fn(model)

    # Unwrap torch.compile for CPU (skip compilation overhead)
    if hasattr(model, "_spyre_compiled_blocks"):
        unwrapped = []
        for cb in model._spyre_compiled_blocks:
            orig = getattr(
                cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None)
            )
            unwrapped.append(orig if orig is not None else cb)
        model._spyre_compiled_blocks = unwrapped

    print("  Running adapter ...")
    adapter_results = adapter_greedy_steps(
        run_forward_fn,
        model,
        input_ids,
        num_decode=num_decode,
    )

    # --- Compare ---
    comparisons = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h_logits = hf_r["logits"]
        a_logits = ad_r["logits"]

        abs_diff = (h_logits - a_logits).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        h_top5 = h_logits.topk(5).indices.tolist()
        a_top5 = a_logits.topk(5).indices.tolist()
        top1_match = h_top5[0] == a_top5[0]
        top5_overlap = len(set(h_top5) & set(a_top5))

        step_label = "prefill" if step == 0 else f"decode-{step}"
        hf_tok_str = tokenizer.decode([hf_r["token"]])
        ad_tok_str = tokenizer.decode([ad_r["token"]])

        comparisons.append(
            {
                "step": step_label,
                "hf_token": hf_r["token"],
                "hf_tok_str": hf_tok_str,
                "adapter_token": ad_r["token"],
                "adapter_tok_str": ad_tok_str,
                "top1_match": top1_match,
                "top5_overlap": top5_overlap,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
        )

    return comparisons, tokenizer


def print_results_table(model_name, comparisons):
    """Print formatted results table. Returns True if all top-1 match."""
    print(f"\n  {model_name} Results")
    print(
        f"  {'Step':<12} {'HF Token':<20} {'Adapter Token':<20} "
        f"{'Top1':<6} {'Top5':<5} {'MaxDiff':<10} {'MeanDiff':<10}"
    )
    print(f"  {'-'*12} {'-'*20} {'-'*20} " f"{'-'*6} {'-'*5} {'-'*10} {'-'*10}")
    all_match = True
    for c in comparisons:
        match_str = "OK" if c["top1_match"] else "FAIL"
        if not c["top1_match"]:
            all_match = False
        hf_str = f"{c['hf_token']:>6} {c['hf_tok_str']!r:<12}"
        ad_str = f"{c['adapter_token']:>6} {c['adapter_tok_str']!r:<12}"
        print(
            f"  {c['step']:<12} {hf_str:<20} {ad_str:<20} "
            f"{match_str:<6} {c['top5_overlap']}/5  "
            f"{c['max_diff']:<10.4f} {c['mean_diff']:<10.6f}"
        )
    return all_match


# ---------------------------------------------------------------------------
# Auto-loader end-to-end test
# ---------------------------------------------------------------------------


def run_model_test_auto_loader(
    model_name, model_path, num_decode=4, dtype="float16", load_fn=False
):
    """Load via AutoSpyreModelForCausalLM, compare generate() output vs HF."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModel = _get_auto_model_class()
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    print(f"\n{'='*70}")
    print(f"  [AUTO] {model_name}: loading {model_path} ({dtype})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- Auto-loader path ---
    print("  Loading via AutoSpyreModelForCausalLM ...")
    model = AutoModel.from_pretrained(model_path, dtype=torch_dtype)

    # Unwrap torch.compile for CPU
    if hasattr(model, "_spyre_compiled_blocks"):
        unwrapped = []
        for cb in model._spyre_compiled_blocks:
            orig = getattr(
                cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None)
            )
            unwrapped.append(orig if orig is not None else cb)
        model._spyre_compiled_blocks = unwrapped

    prompt = "The capital of France is"
    print(f"  Prompt: {prompt!r}")
    print("  Running auto-loader generate ...")
    auto_outputs = model.generate(tokenizer, [prompt], max_new_tokens=num_decode)

    # --- HF reference ---
    print("  Loading HF reference model ...")
    if load_fn:
        adapter_mod = load_adapter(
            "hf_granite_vision.py"
            if "vision" in model_path.lower()
            else "hf_granite.py"
        )
        hf_model = adapter_mod.load_hf_model(model_path, torch_dtype)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="cpu"
        )
    hf_model.eval()
    hf_model.requires_grad_(False)

    encoded = tokenizer(prompt, return_tensors="pt")
    print("  Running HF generate ...")
    with torch.no_grad():
        hf_out = hf_model.generate(
            **encoded, max_new_tokens=num_decode, do_sample=False
        )
    hf_text = tokenizer.decode(
        hf_out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )

    match = auto_outputs[0].strip() == hf_text.strip()
    return {
        "auto_output": auto_outputs[0],
        "hf_output": hf_text,
        "match": match,
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "path": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_granitemoehybrid.py",
        "dtype": "float32",  # fp16 overflows on CPU due to multipliers
    },
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_smollm3.py",
    },
    "llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
    },
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_phi3.py",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
    },
    "mistral": {
        "name": "Mistral 7B v0.3",
        "path": "mistralai/Mistral-7B-v0.3",
        "adapter": "hf_mistral.py",
    },
    "olmo": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_olmo.py",
    },
    "olmo2": {
        "name": "OLMo2 1B",
        "path": "allenai/OLMo-2-0425-1B",
        "adapter": "hf_olmo2.py",
    },
    "falcon3": {
        "name": "Falcon 3 1B",
        "path": "tiiuae/Falcon3-1B-Base",
        "adapter": "hf_llama.py",
    },
    "deepseek-coder": {
        "name": "DeepSeek-Coder 1.3B",
        "path": "deepseek-ai/deepseek-coder-1.3b-base",
        "adapter": "hf_llama.py",
    },
    # Ministral 3B is gated — requires HF auth. Tested on Spyre pod only.
    # "ministral": {
    #     "name": "Ministral 3B",
    #     "path": "mistralai/Ministral-3B-Instruct",
    #     "adapter": "hf_mistral.py",
    # },
    "yi": {
        "name": "Yi 1.5 6B",
        "path": "01-ai/Yi-1.5-6B",
        "adapter": "hf_llama.py",
    },
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision.py",
        "load_fn": True,
    },
}


if __name__ == "__main__":
    args = sys.argv[1:]
    auto_loader_mode = "--auto-loader" in args
    if auto_loader_mode:
        args.remove("--auto-loader")

    which = args if args else list(MODELS.keys())

    if auto_loader_mode:
        # ---- Auto-loader end-to-end test ----
        all_results = {}
        for key in which:
            if key not in MODELS:
                print(f"Unknown model: {key}. Options: {list(MODELS.keys())}")
                continue
            m = MODELS[key]
            try:
                result = run_model_test_auto_loader(
                    m["name"],
                    m["path"],
                    num_decode=4,
                    dtype=m.get("dtype", "float16"),
                    load_fn=m.get("load_fn", False),
                )
                all_results[key] = result
                status = "PASS" if result["match"] else "FAIL"
                print(f"\n  {m['name']}: {status}")
                print(f"    Auto:  {result['auto_output']!r}")
                print(f"    HF:    {result['hf_output']!r}")
            except Exception as e:
                print(f"\n!!! {m['name']} FAILED:")
                traceback.print_exc()
                all_results[key] = {"error": str(e)}

        # Summary
        print(f"\n{'='*70}")
        print("  AUTO-LOADER SUMMARY")
        print(f"{'='*70}")
        for key in which:
            if key not in MODELS or key not in all_results:
                continue
            name = MODELS[key]["name"]
            res = all_results[key]
            if "error" in res:
                print(f"  {name:<22} ERROR: {res['error']}")
            else:
                status = "PASS" if res["match"] else "FAIL"
                print(f"  {name:<22} {status}")
        print(f"{'='*70}")

    else:
        # ---- Existing low-level logit comparison test ----
        all_results = {}
        for key in which:
            if key not in MODELS:
                print(f"Unknown model: {key}. Options: {list(MODELS.keys())}")
                continue
            m = MODELS[key]
            try:
                comps, _ = run_model_test(
                    m["name"],
                    m["path"],
                    m["adapter"],
                    num_decode=4,
                    dtype=m.get("dtype", "float16"),
                    load_fn=m.get("load_fn", False),
                )
                ok = print_results_table(m["name"], comps)
                all_results[key] = {"comparisons": comps, "all_match": ok}
            except Exception as e:
                print(f"\n!!! {m['name']} FAILED:")
                traceback.print_exc()
                all_results[key] = {"error": str(e)}

        # Summary
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        for key in which:
            if key not in MODELS or key not in all_results:
                continue
            name = MODELS[key]["name"]
            res = all_results[key]
            if "error" in res:
                print(f"  {name:<22} ERROR: {res['error']}")
            else:
                status = "PASS" if res["all_match"] else "FAIL"
                print(f"  {name:<22} {status}")
        print(f"{'='*70}")
