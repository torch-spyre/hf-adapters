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
Spyre counterpart of ``test_generate_edge_cases.py``.

Same control-flow regimes (prefill / fill / expansion, block boundaries,
single-token prompts, ``max_new_tokens=0``, forced EOS at controlled offsets,
sampling determinism) but exercised on real Spyre hardware.

Differences from the CPU pytest version:
  - Script-style with ``__main__``; no conftest, no DEVICE patch, no
    ``unwrap_compiled_blocks``. The compiled blocks run on Spyre.
  - One model load per family — ``prepare_for_spyre`` + Spyre move is
    expensive, so the same prepared model is reused for every case.
  - HF reference outputs are captured on CPU *before* the Spyre move (the
    RMSNorm patch is global and would contaminate a second CPU forward).
  - Runs only a curated subset of the cases (each Spyre case takes minutes).

Shared case tables and helpers live in ``_generate_edge_case_helpers.py`` so
this script and the CPU pytest stay in sync.

Usage (on the Spyre pod)::

    python3 tests/test_generate_edge_cases_spyre.py [model_key ...]

Model keys come from ``tests/model_registry.py`` (e.g. ``qwen3``, ``granite2b``,
``llama``). Default is ``qwen3``.

Exit code is 0 only if every case passes.
"""

import gc
import sys
import time
import traceback
import warnings

import torch
from _generate_edge_case_helpers import (
    CASES as ALL_CASES,
)
from _generate_edge_case_helpers import (
    EOS_CASES as ALL_EOS_CASES,
)
from _generate_edge_case_helpers import (
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
    SPYRE_CASE_KEYS,
    SPYRE_EOS_CASE_KEYS,
    EosOverrideTokenizer,
    NoEosTokenizer,
    NoPadTokenizer,
    forced_eos_expected,
    greedy_token_ids,
    hf_reference_outputs,
    make_prompt_with_eos_inside,
    make_prompts,
    pick_forced_eos_id,
)
from model_registry import CAUSAL_LM_MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM

# Remove some repetitive warnings
warnings.filterwarnings("ignore", message=r".*is falling back to cpu.*")
warnings.filterwarnings(
    "ignore", message=r".*torch\.ops\.spyre\.overwrite is deprecated.*"
)

MODELS = CAUSAL_LM_MODELS

# Curated subset — each Spyre case takes minutes, so we drop redundant lengths
# and keep one representative per regime. The CPU pytest covers the full grid.
CASES = {k: ALL_CASES[k] for k in SPYRE_CASE_KEYS}
EOS_CASES = {k: ALL_EOS_CASES[k] for k in SPYRE_EOS_CASE_KEYS}


# ---------------------------------------------------------------------------
# One-model driver
# ---------------------------------------------------------------------------


def run_model(model_key):
    """Load one model, run every case, return a list of result rows."""
    info = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])

    # --- Reference: HF stock generate() on CPU, BEFORE patching ---
    # The RMSNorm patch and Spyre move are global, so we capture every
    # reference now and reuse them.
    print("  Capturing HF references on CPU ...")
    ref_dtype = torch.float32 if info.get("dtype") == "float32" else torch.float16
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=ref_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    case_refs = {}
    for case_id, (targets, max_new) in CASES.items():
        prompts = make_prompts(tokenizer, targets)
        case_refs[case_id] = (
            prompts,
            hf_reference_outputs(ref_model, tokenizer, prompts, max_new),
        )

    # For EOS cases, capture the per-prompt greedy token streams so we can
    # pick a forced eos_token_id and compute the expected truncated output.
    eos_refs = {}
    for case_id, (eos_offsets, max_new) in EOS_CASES.items():
        batch_size = len(eos_offsets)
        prompts = make_prompts(tokenizer, [5] * batch_size)
        per_prompt_ids = [
            greedy_token_ids(ref_model, tokenizer, p, max_new) for p in prompts
        ]
        eos_refs[case_id] = (prompts, per_prompt_ids)

    # Sampling-determinism reference: only need prompts, no HF reference (we
    # compare adapter-vs-adapter at fixed seeds).
    sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)

    # No-EOS reference: HF run with eos_token_id=None so it goes the full
    # max_new_tokens; the adapter side wraps the tokenizer with NoEosTokenizer.
    no_eos_prompts = make_prompts(tokenizer, [5, 12])
    no_eos_max_new = 64 + 7  # cross a block boundary (BLOCK_SIZE=64)
    no_eos_refs = []
    for prompt in no_eos_prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = ref_model.generate(
                **encoded,
                max_new_tokens=no_eos_max_new,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        new_ids = out[0][encoded["input_ids"].shape[1] :]
        no_eos_refs.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    # No-pad-token reference: same prompts as a normal mixed batch; the
    # adapter side wraps with NoPadTokenizer so generate() takes the
    # ``pad_token = eos_token`` fallback at the top of the function.
    no_pad_prompts = make_prompts(tokenizer, [5, 12])
    no_pad_max_new = 16
    no_pad_refs = hf_reference_outputs(
        ref_model, tokenizer, no_pad_prompts, no_pad_max_new
    )

    # EOS-inside-prompt reference: a single prompt with the model's eos_token_id
    # embedded mid-sequence; the adapter must NOT mistake it for an emission.
    eos_in_prompt_refs = None
    eos_in_prompt = None
    eos_in_prompt_max_new = 64 + 8
    if tokenizer.eos_token_id is not None:
        eos_in_prompt = make_prompt_with_eos_inside(
            tokenizer, tokenizer.eos_token_id, target_tokens=12
        )
        eos_in_prompt_refs = hf_reference_outputs(
            ref_model, tokenizer, [eos_in_prompt], eos_in_prompt_max_new
        )

    del ref_model
    gc.collect()

    # --- Load + prepare on Spyre once ---
    print("  Loading model on Spyre ...")
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
    print(f"  Spyre load+prepare: {time.time() - t0:.1f}s")

    rows = []

    # --- Greedy correctness cases ---
    print(f"  Running {len(CASES)} greedy correctness cases ...")
    for case_id, (targets, max_new) in CASES.items():
        prompts, hf_outputs = case_refs[case_id]
        try:
            t0 = time.time()
            spyre_outputs = model.generate(
                tokenizer, prompts, max_new_tokens=max_new, do_sample=False
            )
            elapsed = time.time() - t0
        except Exception:
            traceback.print_exc()
            print(f"    {case_id}: ERROR")
            rows.append(
                {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}
            )
            continue
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(hf_outputs, spyre_outputs))
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": case_id,
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"hf={hf_outputs!r} spyre={spyre_outputs!r}",
            }
        )

    # --- max_new_tokens=0 (locks in empty-output contract) ---
    print("  Running max_new_tokens=0 case ...")
    try:
        t0 = time.time()
        prompts = make_prompts(tokenizer, [5, 12])
        out = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
        elapsed = time.time() - t0
        ok = len(out) == len(prompts) and all(s == "" for s in out)
        print(f"    zero_new_tokens: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": "zero_new_tokens",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"got={out!r}",
            }
        )
    except Exception:
        traceback.print_exc()
        print("    zero_new_tokens: ERROR")
        rows.append(
            {
                "case": "zero_new_tokens",
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": "",
            }
        )

    # --- Forced EOS cases ---
    print(f"  Running {len(EOS_CASES)} forced EOS cases ...")
    for case_id, (eos_offsets, max_new) in EOS_CASES.items():
        prompts, per_prompt_ids = eos_refs[case_id]
        eos_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
        if eos_id is None:
            print(f"    forced_eos:{case_id}: SKIP")
            rows.append(
                {
                    "case": f"forced_eos:{case_id}",
                    "status": "SKIP",
                    "elapsed_s": 0.0,
                    "detail": "no clean shared eos token at requested offsets",
                }
            )
            continue
        expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)
        wrapped = EosOverrideTokenizer(tokenizer, eos_id)
        try:
            t0 = time.time()
            out = model.generate(
                wrapped, prompts, max_new_tokens=max_new, do_sample=False
            )
            elapsed = time.time() - t0
        except Exception:
            traceback.print_exc()
            print(f"    forced_eos:{case_id}: ERROR")
            rows.append(
                {
                    "case": f"forced_eos:{case_id}",
                    "status": "ERROR",
                    "elapsed_s": 0.0,
                    "detail": "",
                }
            )
            continue
        ok = all(e.strip() == g.strip() for e, g in zip(expected, out))
        print(f"    forced_eos:{case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": f"forced_eos:{case_id}",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"expected={expected!r} got={out!r}",
            }
        )

    # --- Sampling determinism (same seed -> equal; different seed -> differ) ---
    print("  Running sampling determinism case ...")
    try:
        sampling_kwargs = SAMPLING_KWARGS
        max_new = SAMPLING_MAX_NEW

        t0 = time.time()
        torch.manual_seed(1234)
        a1 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=max_new, **sampling_kwargs
        )
        torch.manual_seed(1234)
        a2 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=max_new, **sampling_kwargs
        )
        torch.manual_seed(9999)
        b = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=max_new, **sampling_kwargs
        )
        elapsed = time.time() - t0
        ok = a1 == a2 and a1 != b
        print(f"    sampling_determinism: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": "sampling_determinism",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"a1={a1!r} a2={a2!r} b={b!r}",
            }
        )
    except Exception:
        traceback.print_exc()
        print("    sampling_determinism: ERROR")
        rows.append(
            {
                "case": "sampling_determinism",
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": "",
            }
        )

    # --- eos_token_id is None: full-budget generation, no early stop ---
    print("  Running no-EOS full-budget case ...")
    try:
        wrapped = NoEosTokenizer(tokenizer)
        t0 = time.time()
        out = model.generate(
            wrapped, no_eos_prompts, max_new_tokens=no_eos_max_new, do_sample=False
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_eos_refs, out))
        print(
            f"    no_eos_runs_full_budget: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)"
        )
        rows.append(
            {
                "case": "no_eos_runs_full_budget",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"hf={no_eos_refs!r} spyre={out!r}",
            }
        )
    except Exception:
        traceback.print_exc()
        print("    no_eos_runs_full_budget: ERROR")
        rows.append(
            {
                "case": "no_eos_runs_full_budget",
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": "",
            }
        )

    # --- pad_token is None: ``pad_token = eos_token`` fallback ---
    print("  Running no-pad-token fallback case ...")
    try:
        wrapped = NoPadTokenizer(tokenizer)
        t0 = time.time()
        out = model.generate(
            wrapped, no_pad_prompts, max_new_tokens=no_pad_max_new, do_sample=False
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_pad_refs, out))
        print(f"    no_pad_token_fallback: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": "no_pad_token_fallback",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"hf={no_pad_refs!r} spyre={out!r}",
            }
        )
    except Exception:
        traceback.print_exc()
        print("    no_pad_token_fallback: ERROR")
        rows.append(
            {
                "case": "no_pad_token_fallback",
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": "",
            }
        )

    # --- top_k=0 sampling: skip the top-k filter branch ---
    print("  Running top_k=0 sampling case ...")
    try:
        kwargs = dict(do_sample=True, temperature=1.0, top_k=0)
        t0 = time.time()
        torch.manual_seed(2024)
        out1 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        torch.manual_seed(2024)
        out2 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        elapsed = time.time() - t0
        ok = out1 == out2 and all(s for s in out1)
        print(f"    sampling_top_k_zero: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        rows.append(
            {
                "case": "sampling_top_k_zero",
                "status": "PASS" if ok else "FAIL",
                "elapsed_s": elapsed,
                "detail": "" if ok else f"out1={out1!r} out2={out2!r}",
            }
        )
    except Exception:
        traceback.print_exc()
        print("    sampling_top_k_zero: ERROR")
        rows.append(
            {
                "case": "sampling_top_k_zero",
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": "",
            }
        )

    # --- EOS id inside the prompt: must NOT be mistaken for an emission ---
    print("  Running EOS-inside-prompt case ...")
    if eos_in_prompt is None:
        print("    eos_inside_prompt: SKIP")
        rows.append(
            {
                "case": "eos_inside_prompt",
                "status": "SKIP",
                "elapsed_s": 0.0,
                "detail": "tokenizer has no eos_token_id",
            }
        )
    else:
        try:
            t0 = time.time()
            out = model.generate(
                tokenizer,
                [eos_in_prompt],
                max_new_tokens=eos_in_prompt_max_new,
                do_sample=False,
            )
            elapsed = time.time() - t0
            ok = eos_in_prompt_refs[0].strip() == out[0].strip()
            print(f"    eos_inside_prompt: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
            rows.append(
                {
                    "case": "eos_inside_prompt",
                    "status": "PASS" if ok else "FAIL",
                    "elapsed_s": elapsed,
                    "detail": "" if ok else f"hf={eos_in_prompt_refs!r} spyre={out!r}",
                }
            )
        except Exception:
            traceback.print_exc()
            print("    eos_inside_prompt: ERROR")
            rows.append(
                {
                    "case": "eos_inside_prompt",
                    "status": "ERROR",
                    "elapsed_s": 0.0,
                    "detail": "",
                }
            )

    del model
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(model_to_rows):
    print("\n## Spyre generate() edge-case results\n")
    print("| Model | Case | Status | Time (s) | Detail |")
    print("|-------|------|--------|----------|--------|")
    for model_name, rows in model_to_rows.items():
        for r in rows:
            detail = r["detail"][:80] + "..." if len(r["detail"]) > 80 else r["detail"]
            print(
                f"| {model_name} | {r['case']} | {r['status']} "
                f"| {r['elapsed_s']:.1f} | {detail} |"
            )


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["qwen3"]

    model_to_rows = {}
    for key in which:
        if key not in MODELS:
            print(f"Unknown: {key}. Options: {list(MODELS.keys())}")
            continue
        try:
            model_to_rows[MODELS[key]["name"]] = run_model(key)
        except Exception:
            print(f"\n!!! {MODELS[key]['name']} FAILED to set up:")
            traceback.print_exc()
            model_to_rows[MODELS[key]["name"]] = [
                {
                    "case": "<setup>",
                    "status": "ERROR",
                    "elapsed_s": 0.0,
                    "detail": "",
                }
            ]

    print_summary(model_to_rows)

    all_rows = [r for rows in model_to_rows.values() for r in rows]
    n_pass = sum(1 for r in all_rows if r["status"] == "PASS")
    n_total = sum(1 for r in all_rows if r["status"] != "SKIP")
    print(f"\nResult: {n_pass}/{n_total} passed")
    sys.exit(0 if n_pass == n_total else 1)
