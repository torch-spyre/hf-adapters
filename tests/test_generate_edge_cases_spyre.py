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

Each case *group* runs in its own Python subprocess: one child for all greedy
cases, one for all forced-EOS cases, and one per remaining one-off case. Long
runs accumulate VFIO DMA mappings the kernel only reclaims on process exit;
per-group isolation bounds the live mapping set without needing kernel-side
limits raised. The cost is one Spyre model load per group (instead of one per
case) — coarser batching keeps the wall-clock down while still resetting the
VFIO context often enough to avoid exhaustion.

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
import multiprocessing as mp
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
# Subprocess plumbing
# ---------------------------------------------------------------------------


def _ref_dtype(info):
    return torch.float32 if info.get("dtype") == "float32" else torch.float16


def _spawn(target, args, group_label, timeout=3600):
    """Run ``target(*args, q)`` in a fresh process; return its list of rows.

    Each child is a clean Python interpreter (start_method="spawn"), so the
    Spyre runtime initializes from scratch. On exit, the kernel reclaims every
    VFIO DMA mapping the child opened — which is the whole point.

    Children put a *list* of result rows on the queue (a group can produce
    several rows). Returns that list, or a single-element list with an ERROR
    row on timeout / no-result.
    """
    q = mp.Queue()
    p = mp.Process(target=target, args=(*args, q))
    t0 = time.time()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return [
            {
                "case": group_label,
                "status": "ERROR",
                "elapsed_s": time.time() - t0,
                "detail": f"timeout after {timeout}s",
            }
        ]
    if not q.empty():
        return q.get()
    return [
        {
            "case": group_label,
            "status": "ERROR",
            "elapsed_s": time.time() - t0,
            "detail": f"child exited with code {p.exitcode} and no result",
        }
    ]


def _row(label, ok, t0, detail=""):
    return {
        "case": label,
        "status": "PASS" if ok else "FAIL",
        "elapsed_s": time.time() - t0,
        "detail": "" if ok else detail,
    }


def _error_row(label, t0, exc):
    return {
        "case": label,
        "status": "ERROR",
        "elapsed_s": time.time() - t0,
        "detail": "".join(traceback.format_exception(exc)),
    }


# ---------------------------------------------------------------------------
# Per-group child workers
# ---------------------------------------------------------------------------
#
# Each worker:
#   1. Loads tokenizer.
#   2. Captures whatever HF references the group needs (CPU, before any Spyre
#      patching — the RMSNorm patch is global).
#   3. Loads the Spyre model once and runs every case in the group.
#   4. Puts a list of result rows on the queue.
#
# Workers must be top-level (picklable for spawn) and self-contained: the
# parent passes only the model_key, never live torch objects.


def _child_greedy_group(model_key, case_filter, q):
    """Run greedy CASES in one child (one Spyre model load).

    ``case_filter`` is an optional list of case IDs; when set, only those
    cases run. The parent already validated names are valid.
    """
    rows = []
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])

        cases_to_run = (
            {k: CASES[k] for k in case_filter} if case_filter else dict(CASES)
        )

        # HF references for every case — captured before any Spyre patching.
        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=_ref_dtype(info), device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        case_refs = {}
        for case_id, (targets, max_new) in cases_to_run.items():
            prompts = make_prompts(tokenizer, targets)
            case_refs[case_id] = (
                prompts,
                max_new,
                hf_reference_outputs(ref_model, tokenizer, prompts, max_new),
            )
        del ref_model
        gc.collect()

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])

        for case_id, (prompts, max_new, hf_outputs) in case_refs.items():
            t0 = time.time()
            try:
                spyre_outputs = model.generate(
                    tokenizer, prompts, max_new_tokens=max_new, do_sample=False
                )
                ok = all(
                    h.strip() == s.strip() for h, s in zip(hf_outputs, spyre_outputs)
                )
                rows.append(
                    _row(
                        case_id,
                        ok,
                        t0,
                        detail=f"hf={hf_outputs!r} spyre={spyre_outputs!r}",
                    )
                )
            except Exception as e:
                traceback.print_exc()
                rows.append(_error_row(case_id, t0, e))
    except Exception as e:
        traceback.print_exc()
        rows.append(_error_row("greedy_group:setup", time.time(), e))
    q.put(rows)


def _child_forced_eos_group(model_key, q):
    """Run every case in EOS_CASES in one child (one Spyre model load)."""
    rows = []
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])

        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=_ref_dtype(info), device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)

        # For each EOS case, capture greedy token streams and decide eos_id +
        # expected output now (CPU); the Spyre run reuses these.
        prepared = {}
        for case_id, (eos_offsets, max_new) in EOS_CASES.items():
            batch_size = len(eos_offsets)
            prompts = make_prompts(tokenizer, [5] * batch_size)
            per_prompt_ids = [
                greedy_token_ids(ref_model, tokenizer, p, max_new) for p in prompts
            ]
            eos_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
            if eos_id is None:
                prepared[case_id] = None  # signal SKIP
            else:
                expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)
                prepared[case_id] = (prompts, max_new, eos_id, expected)
        del ref_model
        gc.collect()

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])

        for case_id, prep in prepared.items():
            label = f"forced_eos:{case_id}"
            t0 = time.time()
            if prep is None:
                rows.append(
                    {
                        "case": label,
                        "status": "SKIP",
                        "elapsed_s": 0.0,
                        "detail": "no clean shared eos token at requested offsets",
                    }
                )
                continue
            prompts, max_new, eos_id, expected = prep
            try:
                out = model.generate(
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new,
                    do_sample=False,
                    eos_token_id=eos_id,
                )
                ok = all(e.strip() == g.strip() for e, g in zip(expected, out))
                rows.append(
                    _row(label, ok, t0, detail=f"expected={expected!r} got={out!r}")
                )
            except Exception as e:
                traceback.print_exc()
                rows.append(_error_row(label, t0, e))
    except Exception as e:
        traceback.print_exc()
        rows.append(_error_row("forced_eos_group:setup", time.time(), e))
    q.put(rows)


def _child_zero_new_tokens(model_key, q):
    label = "zero_new_tokens"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        prompts = make_prompts(tokenizer, [5, 12])

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        out = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
        ok = len(out) == len(prompts) and all(s == "" for s in out)
        q.put([_row(label, ok, t0, detail=f"got={out!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


def _child_sampling_determinism(model_key, q):
    label = "sampling_determinism"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        prompts = make_prompts(tokenizer, SAMPLING_TARGETS)

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        torch.manual_seed(1234)
        a1 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
        torch.manual_seed(1234)
        a2 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
        torch.manual_seed(9999)
        b = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **SAMPLING_KWARGS
        )
        ok = a1 == a2 and a1 != b
        q.put([_row(label, ok, t0, detail=f"a1={a1!r} a2={a2!r} b={b!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


def _child_no_eos_full_budget(model_key, q):
    label = "no_eos_runs_full_budget"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        prompts = make_prompts(tokenizer, [5, 12])
        max_new = 64 + 7  # cross a block boundary (BLOCK_SIZE=64)

        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=_ref_dtype(info), device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        hf_refs = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = ref_model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                )
            new_ids = out[0][encoded["input_ids"].shape[1] :]
            hf_refs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        del ref_model
        gc.collect()

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        out = model.generate(
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=None,
        )
        ok = all(h.strip() == s.strip() for h, s in zip(hf_refs, out))
        q.put([_row(label, ok, t0, detail=f"hf={hf_refs!r} spyre={out!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


def _child_no_pad_token(model_key, q):
    label = "no_pad_token_fallback"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        prompts = make_prompts(tokenizer, [5, 12])
        max_new = 16

        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=_ref_dtype(info), device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        hf_refs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new)
        del ref_model
        gc.collect()

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        wrapped = NoPadTokenizer(tokenizer)
        out = model.generate(wrapped, prompts, max_new_tokens=max_new, do_sample=False)
        ok = all(h.strip() == s.strip() for h, s in zip(hf_refs, out))
        q.put([_row(label, ok, t0, detail=f"hf={hf_refs!r} spyre={out!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


def _child_top_k_zero(model_key, q):
    label = "sampling_top_k_zero"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
        kwargs = dict(do_sample=True, temperature=1.0, top_k=0)

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        torch.manual_seed(2024)
        out1 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        torch.manual_seed(2024)
        out2 = model.generate(
            tokenizer, prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        ok = out1 == out2 and all(s for s in out1)
        q.put([_row(label, ok, t0, detail=f"out1={out1!r} out2={out2!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


def _child_eos_inside_prompt(model_key, q):
    label = "eos_inside_prompt"
    t0 = time.time()
    try:
        info = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(info["path"])
        if tokenizer.eos_token_id is None:
            q.put(
                [
                    {
                        "case": label,
                        "status": "SKIP",
                        "elapsed_s": 0.0,
                        "detail": "tokenizer has no eos_token_id",
                    }
                ]
            )
            return
        prompt = make_prompt_with_eos_inside(
            tokenizer, tokenizer.eos_token_id, target_tokens=12
        )
        max_new = 64 + 8

        ref_model = AutoModelForCausalLM.from_pretrained(
            info["path"], torch_dtype=_ref_dtype(info), device_map="cpu"
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        hf_refs = hf_reference_outputs(ref_model, tokenizer, [prompt], max_new)
        del ref_model
        gc.collect()

        model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
        out = model.generate(
            tokenizer, [prompt], max_new_tokens=max_new, do_sample=False
        )
        ok = hf_refs[0].strip() == out[0].strip()
        q.put([_row(label, ok, t0, detail=f"hf={hf_refs!r} spyre={out!r}")])
    except Exception as e:
        traceback.print_exc()
        q.put([_error_row(label, t0, e)])


# ---------------------------------------------------------------------------
# One-model driver — parent process
# ---------------------------------------------------------------------------


def run_model(model_key, case_filter=None):
    """Spawn one child per case group for ``model_key``; return result rows.

    If ``case_filter`` is given, only the greedy-cases child runs (with the
    filter applied inside it); the one-off cases are skipped. Matches the
    ``--case`` flag's pre-subprocess semantics.
    """
    info = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'='*70}")

    rows = []

    def _run_group(target, args, group_label):
        print(f"  [group {group_label}] launching child ...")
        group_rows = _spawn(target, args, group_label)
        for r in group_rows:
            print(f"    {r['case']}: {r['status']} ({r['elapsed_s']:.1f}s)")
        rows.extend(group_rows)

    if case_filter:
        # Validate against greedy CASES — that's the only group --case targets.
        invalid = [c for c in case_filter if c not in CASES]
        if invalid:
            print(f"  ERROR: unknown case(s) for --case filter: {invalid}")
            print(f"  Available cases: {list(CASES.keys())}")
            return rows
        _run_group(
            _child_greedy_group,
            (model_key, case_filter),
            f"greedy×{len(case_filter)}",
        )
        return rows

    _run_group(_child_greedy_group, (model_key, None), f"greedy×{len(CASES)}")
    _run_group(_child_zero_new_tokens, (model_key,), "zero_new_tokens")
    _run_group(_child_forced_eos_group, (model_key,), f"forced_eos×{len(EOS_CASES)}")
    _run_group(_child_sampling_determinism, (model_key,), "sampling_determinism")
    _run_group(_child_no_eos_full_budget, (model_key,), "no_eos_runs_full_budget")
    _run_group(_child_no_pad_token, (model_key,), "no_pad_token_fallback")
    _run_group(_child_top_k_zero, (model_key,), "sampling_top_k_zero")
    _run_group(_child_eos_inside_prompt, (model_key,), "eos_inside_prompt")

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
    # "spawn" gives each child a clean interpreter — no inherited torch_spyre
    # state from the parent. Required for the per-group VFIO reset.
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run Spyre generate() edge-case tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for qwen3 (default)
  python3 tests/test_generate_edge_cases_spyre.py

  # Run all tests for specific models
  python3 tests/test_generate_edge_cases_spyre.py qwen3 granite2b

  # Run only short_two_blocks_plus test for qwen3
  python3 tests/test_generate_edge_cases_spyre.py qwen3 --case short_two_blocks_plus

  # Run multiple specific cases
  python3 tests/test_generate_edge_cases_spyre.py qwen3 --case short_two_blocks_plus single_token_prompt
        """,
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
        help=f"Run only specific greedy cases. Options: {list(CASES.keys())}",
    )

    args = parser.parse_args()
    which = args.models
    case_filter = args.cases

    if case_filter:
        print(f"Running filtered cases: {case_filter}")
        invalid_cases = [c for c in case_filter if c not in CASES]
        if invalid_cases:
            print(f"ERROR: Unknown case(s): {invalid_cases}")
            print(f"Available cases: {list(CASES.keys())}")
            sys.exit(1)

    model_to_rows = {}
    for key in which:
        if key not in MODELS:
            print(f"Unknown: {key}. Options: {list(MODELS.keys())}")
            continue
        try:
            model_to_rows[MODELS[key]["name"]] = run_model(key, case_filter)
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
