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
  - Each case runs in its own worker process (size-1 multiprocessing.Pool
    with maxtasksperchild=1) so VFIO DMA mappings are released between
    cases. The Spyre load + ``prepare_for_spyre`` is paid per case.
  - HF reference outputs are captured on CPU *before* the Spyre move in
    each worker (the RMSNorm patch is global, but each worker is fresh).
  - Runs only a curated subset of the cases (each Spyre case takes minutes).

Shared case tables and helpers live in ``_generate_edge_case_helpers.py`` so
this script and the CPU pytest stay in sync.

Usage (on the Spyre pod)::

    # Run all tests for default model (qwen3)
    python3 tests/test_generate_edge_cases_spyre.py

    # Run all tests for specific models
    python3 tests/test_generate_edge_cases_spyre.py qwen3 granite2b

    # Run only specific test case(s)
    python3 tests/test_generate_edge_cases_spyre.py qwen3 --case short_two_blocks_plus
    python3 tests/test_generate_edge_cases_spyre.py qwen3 --case short_two_blocks_plus single_token_prompt

Model keys come from ``tests/model_registry.py`` (e.g. ``qwen3``, ``granite2b``,
``llama``). Default is ``qwen3``.

Exit code is 0 only if every case passes.
"""

import argparse
import gc
import multiprocessing
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

# Names of the bespoke (non-greedy, non-forced-EOS) cases. Kept here so the
# CLI can validate --case against them and so each block can be gated.
SPECIAL_CASES = [
    "zero_new_tokens",
    "sampling_determinism",
    "no_eos_runs_full_budget",
    "no_pad_token_fallback",
    "sampling_top_k_zero",
    "eos_inside_prompt",
]

# Cases that don't need an HF reference forward (sampling-determinism compares
# adapter-vs-adapter at fixed seeds; zero_new_tokens has a constant expected
# output; sampling_top_k_zero only checks adapter-vs-adapter equality).
_CASES_WITHOUT_REF = {
    "zero_new_tokens",
    "sampling_determinism",
    "sampling_top_k_zero",
}


def all_case_names():
    """Every case name accepted by the --case filter."""
    return (
        list(CASES.keys())
        + [f"forced_eos:{k}" for k in EOS_CASES.keys()]
        + SPECIAL_CASES
    )


# ---------------------------------------------------------------------------
# Per-case runners. Each takes whatever it needs and returns a single row dict.
# Logic copied verbatim from the previous one-process driver.
# ---------------------------------------------------------------------------


def _run_greedy_case(model, tokenizer, ref_model, case_id):
    targets, max_new = CASES[case_id]
    prompts = make_prompts(tokenizer, targets)
    hf_outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new)
    try:
        t0 = time.time()
        spyre_outputs = model.generate(
            tokenizer, prompts, max_new_tokens=max_new, do_sample=False
        )
        elapsed = time.time() - t0
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}
    ok = all(hf.strip() == sp.strip() for hf, sp in zip(hf_outputs, spyre_outputs))
    print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
    return {
        "case": case_id,
        "status": "PASS" if ok else "FAIL",
        "elapsed_s": elapsed,
        "detail": "" if ok else f"hf={hf_outputs!r} spyre={spyre_outputs!r}",
    }


def _run_zero_new_tokens(model, tokenizer):
    case_id = "zero_new_tokens"
    print(f"  Running {case_id} case ...")
    try:
        t0 = time.time()
        prompts = make_prompts(tokenizer, [5, 12])
        out = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
        elapsed = time.time() - t0
        ok = len(out) == len(prompts) and all(s == "" for s in out)
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"got={out!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


def _run_eos_case(model, tokenizer, ref_model, case_id):
    eos_offsets, max_new = EOS_CASES[case_id]
    label = f"forced_eos:{case_id}"
    batch_size = len(eos_offsets)
    prompts = make_prompts(tokenizer, [5] * batch_size)
    per_prompt_ids = [
        greedy_token_ids(ref_model, tokenizer, p, max_new) for p in prompts
    ]
    eos_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
    if eos_id is None:
        print(f"    {label}: SKIP")
        return {
            "case": label,
            "status": "SKIP",
            "elapsed_s": 0.0,
            "detail": "no clean shared eos token at requested offsets",
        }
    expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)
    try:
        t0 = time.time()
        out = model.generate(
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
        )
        elapsed = time.time() - t0
    except Exception:
        traceback.print_exc()
        print(f"    {label}: ERROR")
        return {"case": label, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}
    ok = all(e.strip() == g.strip() for e, g in zip(expected, out))
    print(f"    {label}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
    return {
        "case": label,
        "status": "PASS" if ok else "FAIL",
        "elapsed_s": elapsed,
        "detail": "" if ok else f"expected={expected!r} got={out!r}",
    }


def _run_sampling_determinism(model, tokenizer):
    case_id = "sampling_determinism"
    print(f"  Running {case_id} case ...")
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
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
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"a1={a1!r} a2={a2!r} b={b!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


def _run_no_eos(model, tokenizer, ref_model):
    case_id = "no_eos_runs_full_budget"
    print(f"  Running {case_id} case ...")
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
    try:
        t0 = time.time()
        out = model.generate(
            tokenizer,
            no_eos_prompts,
            max_new_tokens=no_eos_max_new,
            do_sample=False,
            eos_token_id=None,
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_eos_refs, out))
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"hf={no_eos_refs!r} spyre={out!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


def _run_no_pad(model, tokenizer, ref_model):
    case_id = "no_pad_token_fallback"
    print(f"  Running {case_id} case ...")
    no_pad_prompts = make_prompts(tokenizer, [5, 12])
    no_pad_max_new = 16
    no_pad_refs = hf_reference_outputs(
        ref_model, tokenizer, no_pad_prompts, no_pad_max_new
    )
    try:
        wrapped = NoPadTokenizer(tokenizer)
        t0 = time.time()
        out = model.generate(
            wrapped, no_pad_prompts, max_new_tokens=no_pad_max_new, do_sample=False
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_pad_refs, out))
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"hf={no_pad_refs!r} spyre={out!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


def _run_top_k_zero(model, tokenizer):
    case_id = "sampling_top_k_zero"
    print(f"  Running {case_id} case ...")
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
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
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"out1={out1!r} out2={out2!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


def _run_eos_inside_prompt(model, tokenizer, ref_model):
    case_id = "eos_inside_prompt"
    print(f"  Running {case_id} case ...")
    if tokenizer.eos_token_id is None:
        print(f"    {case_id}: SKIP")
        return {
            "case": case_id,
            "status": "SKIP",
            "elapsed_s": 0.0,
            "detail": "tokenizer has no eos_token_id",
        }
    eos_in_prompt = make_prompt_with_eos_inside(
        tokenizer, tokenizer.eos_token_id, target_tokens=12
    )
    eos_in_prompt_max_new = 64 + 8
    eos_in_prompt_refs = hf_reference_outputs(
        ref_model, tokenizer, [eos_in_prompt], eos_in_prompt_max_new
    )
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
        print(f"    {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return {
            "case": case_id,
            "status": "PASS" if ok else "FAIL",
            "elapsed_s": elapsed,
            "detail": "" if ok else f"hf={eos_in_prompt_refs!r} spyre={out!r}",
        }
    except Exception:
        traceback.print_exc()
        print(f"    {case_id}: ERROR")
        return {"case": case_id, "status": "ERROR", "elapsed_s": 0.0, "detail": ""}


# ---------------------------------------------------------------------------
# Worker entry point: one case per process
# ---------------------------------------------------------------------------


def run_one_case(task):
    """Worker entry point. Loads a fresh tokenizer + (optionally) HF ref +
    Spyre model, runs ONE case, returns its row dict.

    Imported lazily so the parent process never imports hf_adapters or touches
    the Spyre device — only the worker does, and a fresh worker is spawned
    per case (Pool maxtasksperchild=1).
    """
    from hf_adapters import AutoSpyreModelForCausalLM

    model_key, case_name = task
    info = MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(info["path"])

    # CPU HF reference for cases that need it. Capture BEFORE the Spyre move
    # so the global RMSNorm patch doesn't contaminate it.
    ref_model = None
    if case_name not in _CASES_WITHOUT_REF:
        ref_dtype = torch.float32 if info.get("dtype") == "float32" else torch.float16
        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=ref_dtype, device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)

    # --- Load + prepare on Spyre ---
    print(f"  [{case_name}] Loading model on Spyre ...")
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
    print(f"  [{case_name}] Spyre load+prepare: {time.time() - t0:.1f}s")

    try:
        if case_name in CASES:
            row = _run_greedy_case(model, tokenizer, ref_model, case_name)
        elif case_name.startswith("forced_eos:"):
            row = _run_eos_case(model, tokenizer, ref_model, case_name.split(":", 1)[1])
        elif case_name == "zero_new_tokens":
            row = _run_zero_new_tokens(model, tokenizer)
        elif case_name == "sampling_determinism":
            row = _run_sampling_determinism(model, tokenizer)
        elif case_name == "no_eos_runs_full_budget":
            row = _run_no_eos(model, tokenizer, ref_model)
        elif case_name == "no_pad_token_fallback":
            row = _run_no_pad(model, tokenizer, ref_model)
        elif case_name == "sampling_top_k_zero":
            row = _run_top_k_zero(model, tokenizer)
        elif case_name == "eos_inside_prompt":
            row = _run_eos_inside_prompt(model, tokenizer, ref_model)
        else:
            row = {
                "case": case_name,
                "status": "ERROR",
                "elapsed_s": 0.0,
                "detail": f"unknown case: {case_name}",
            }
    finally:
        del model
        if ref_model is not None:
            del ref_model
        gc.collect()

    return row


# ---------------------------------------------------------------------------
# Parent driver: build task list, dispatch via size-1 process pool
# ---------------------------------------------------------------------------


def _build_task_list(case_filter):
    filter_set = set(case_filter) if case_filter else None

    def keep(name):
        return filter_set is None or name in filter_set

    tasks = []
    tasks.extend(k for k in CASES.keys() if keep(k))
    if keep("zero_new_tokens"):
        tasks.append("zero_new_tokens")
    tasks.extend(f"forced_eos:{k}" for k in EOS_CASES.keys() if keep(f"forced_eos:{k}"))
    for name in (
        "sampling_determinism",
        "no_eos_runs_full_budget",
        "no_pad_token_fallback",
        "sampling_top_k_zero",
        "eos_inside_prompt",
    ):
        if keep(name):
            tasks.append(name)
    return tasks


def run_model_via_pool(model_key, case_filter=None):
    """Run every selected case for ``model_key`` in its own worker process.

    Uses a size-1 multiprocessing.Pool with maxtasksperchild=1 so each case
    gets a brand-new worker process — VFIO DMA mappings are released between
    cases when the worker exits.
    """
    info = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'='*70}")

    tasks = _build_task_list(case_filter)
    if not tasks:
        print("  No cases selected.")
        return []

    print(f"  Dispatching {len(tasks)} cases, one per process ...")

    ctx = multiprocessing.get_context("spawn")
    rows = []
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        try:
            for row in pool.imap(run_one_case, [(model_key, t) for t in tasks]):
                rows.append(row)
        except Exception:
            traceback.print_exc()
            rows.append(
                {
                    "case": "<pool>",
                    "status": "ERROR",
                    "elapsed_s": 0.0,
                    "detail": "worker crashed",
                }
            )
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
    epilog_lines = [
        "Examples:",
        "  # Run all tests for qwen3 (default)",
        "  python3 tests/test_generate_edge_cases_spyre.py",
        "",
        "  # Run all tests for specific models",
        "  python3 tests/test_generate_edge_cases_spyre.py qwen3 granite2b",
        "",
        "  # Run only short_two_blocks_plus test for qwen3",
        "  python3 tests/test_generate_edge_cases_spyre.py qwen3 --case short_two_blocks_plus",
        "",
        "  # Run multiple specific cases (mix of greedy / forced_eos / special)",
        "  python3 tests/test_generate_edge_cases_spyre.py qwen3 \\",
        "      --case short_two_blocks_plus sampling_determinism forced_eos:eos_mid_block",
        "",
        "Available cases:",
        "  Greedy:",
        *[f"    {k}" for k in CASES.keys()],
        "  Forced-EOS:",
        *[f"    forced_eos:{k}" for k in EOS_CASES.keys()],
        "  Special:",
        *[f"    {k}" for k in SPECIAL_CASES],
    ]
    parser = argparse.ArgumentParser(
        description="Run Spyre generate() edge-case tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["qwen3"],
        help=f"Model keys to test. Options: {list(MODELS.keys())}. Default: qwen3",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        dest="cases",
        help="Run only specific test cases. See 'Available cases' below for the full list.",
    )

    args = parser.parse_args()
    which = args.models
    case_filter = args.cases

    if case_filter:
        print(f"Running filtered cases: {case_filter}")
        valid = set(all_case_names())
        invalid_cases = [c for c in case_filter if c not in valid]
        if invalid_cases:
            print(f"ERROR: Unknown case(s): {invalid_cases}")
            print(f"Available cases: {sorted(valid)}")
            sys.exit(1)

    model_to_rows = {}
    for key in which:
        if key not in MODELS:
            print(f"Unknown: {key}. Options: {list(MODELS.keys())}")
            continue
        try:
            model_to_rows[MODELS[key]["name"]] = run_model_via_pool(key, case_filter)
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
