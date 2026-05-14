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
Multi-batch generate() test: verify that hf_common.generate() produces correct
per-sequence outputs when called with batch_size > 1.

Usage:
    python tests/test_multibatch_generate.py [granite2b|qwen3|smollm3|llama|...]
"""

import importlib
import importlib.util
import os
import sys
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")

# ---------------------------------------------------------------------------
# Bootstrap: load hf_common with DEVICE="cpu", then load adapter modules
# ---------------------------------------------------------------------------

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
sys.modules.setdefault("hf_adapters", _pkg)


def load_adapter(filename):
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# HF reference: run generate() independently for each prompt
# ---------------------------------------------------------------------------


def hf_reference_outputs(model, tokenizer, prompts, max_new_tokens):
    """Run HF native generate() on each prompt individually. Returns list of strings."""
    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_ids = out[0][encoded["input_ids"].shape[1] :]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


# ---------------------------------------------------------------------------
# Adapter: run hf_common.generate() on the full batch
# ---------------------------------------------------------------------------


def adapter_batch_outputs(adapter_mod, model, tokenizer, prompts, max_new_tokens):
    """Run hf_common.generate() with all prompts in one batch."""
    # Unwrap torch.compile so CPU tests don't trigger compilation
    if hasattr(model, "_spyre_compiled_blocks"):
        unwrapped = []
        for cb in model._spyre_compiled_blocks:
            orig = getattr(
                cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None)
            )
            unwrapped.append(orig if orig is not None else cb)
        model._spyre_compiled_blocks = unwrapped

    return _common_mod.generate(
        adapter_mod._run_forward,
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

# Two prompts that should produce clearly different continuations
PROMPTS = [
    "The capital of France is",
    "The chemical formula for water is",
]


def run_multibatch_test(
    model_name, model_path, adapter_filename, max_new_tokens=8, dtype="float16"
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_mod = load_adapter(adapter_filename)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    print(f"\n{'='*70}")
    print(f"  {model_name}: {model_path} ({dtype})")
    print("  Prompts:")
    for i, p in enumerate(PROMPTS):
        print(f"    [{i}] {p!r}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)

    # HF reference BEFORE patching (RMSNorm patch is global)
    print("  Running HF reference (per-prompt) ...")
    hf_outputs = hf_reference_outputs(model, tokenizer, PROMPTS, max_new_tokens)

    print("  Preparing adapter ...")
    adapter_mod.prepare_for_spyre(model)

    print("  Running adapter batch generate ...")
    adapter_outputs = adapter_batch_outputs(
        adapter_mod, model, tokenizer, PROMPTS, max_new_tokens
    )

    # Compare
    print(f"\n  Results (max_new_tokens={max_new_tokens}):")
    print(f"  {'Prompt':<40} {'HF output':<30} {'Adapter output':<30} {'Match'}")
    print(f"  {'-'*40} {'-'*30} {'-'*30} {'-'*5}")

    all_match = True
    for i, (prompt, hf_out, adapter_out) in enumerate(
        zip(PROMPTS, hf_outputs, adapter_outputs)
    ):
        match = hf_out.strip() == adapter_out.strip()
        if not match:
            all_match = False
        status = "OK" if match else "FAIL"
        print(f"  [{i}] {prompt:<38} {hf_out!r:<30} {adapter_out!r:<30} {status}")

    return all_match, hf_outputs, adapter_outputs


# ---------------------------------------------------------------------------
# Model registry (subset from test_adapter_cpu_accuracy.py)
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
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
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_granitemoehybrid.py",
        "dtype": "float32",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
    },
    "olmo": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_olmo.py",
    },
}

if __name__ == "__main__":
    args = sys.argv[1:]
    which = args if args else list(MODELS.keys())

    all_results = {}
    for key in which:
        if key not in MODELS:
            print(f"Unknown model: {key}. Options: {list(MODELS.keys())}")
            continue
        m = MODELS[key]
        try:
            ok, hf_outs, adapter_outs = run_multibatch_test(
                m["name"],
                m["path"],
                m["adapter"],
                max_new_tokens=8,
                dtype=m.get("dtype", "float16"),
            )
            all_results[key] = {"ok": ok, "hf": hf_outs, "adapter": adapter_outs}
        except Exception:
            print(f"\n!!! {m['name']} FAILED with exception:")
            traceback.print_exc()
            all_results[key] = {"error": True}

    print(f"\n{'='*70}")
    print("  MULTI-BATCH SUMMARY")
    print(f"{'='*70}")
    for key in which:
        if key not in MODELS or key not in all_results:
            continue
        name = MODELS[key]["name"]
        res = all_results[key]
        if res.get("error"):
            print(f"  {name:<22} ERROR")
        else:
            status = "PASS" if res["ok"] else "FAIL"
            print(f"  {name:<22} {status}")
    print(f"{'='*70}")
