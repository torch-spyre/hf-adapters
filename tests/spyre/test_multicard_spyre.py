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
Multicard tensor-parallel smoke test for hf-adapters on Spyre AIU cards.

Verifies that a model sharded across N cards via real tensor parallelism:
  - Loads correctly on every rank
  - Produces non-empty output containing the expected text
  - Reports TTFT and steady-state ITL (compile-spike outliers excluded)

This test must be launched with torchrun — one process per card.  Regular
pytest cannot drive multi-process TP because each rank needs its own OS
process with its own LOCAL_RANK and device assignment.

Requirements
------------
- transformers==5.15.x  (5.16 changed the tp_plan API; see pyproject.toml)

How to run
----------
First make sure transformers is at the right version::

    uv pip install "transformers==5.15.0"

SPYRE_DEVICES tells the Flex runtime which card indices to use per rank.
Index-to-physical-card mapping is handled internally by Flex and is not
independently verifiable from this script.

2-card example (valid indices are node-specific — check yours first)::

    export SPYRE_DEVICES=0,1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

4-card example::

    export SPYRE_DEVICES=0,1,2,3
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

pytest (single-card only — multi-card requires torchrun, see above)::

    pytest -s -vvv tests/spyre/test_multicard_spyre.py
"""

import contextlib
import io
import os
import re
import statistics
import time
import traceback
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.model_harness("causal")

# TODO(torch-spyre): Once torch.spyre.get_device_name(i) is implemented (it
# needs to call into Flex to map device index → PCI address), resolved_cards
# can be derived automatically without relying on AIU_WORLD_RANK_* env vars.

_DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"
_PROMPT = "The capital of France is"
_EXPECTED_SUBSTRING = "Paris"
_DEFAULT_MAX_NEW_TOKENS = 8
_DEFAULT_BATCH_SIZE = 1

# Compile-spike outlier filter: tokens whose latency exceeds this multiple of
# the median are excluded from the steady-state ITL figure.  This catches the
# second-token decode-graph compile spike without hardcoding a token index.
_OUTLIER_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _parse_timing(captured: str) -> tuple[float | None, float | None, list[float]]:
    """Return (ttft_ms, avg_decode_ms, per_token_list) from captured stdout."""
    ttft = None
    decode = None
    per_token: list[float] = []

    m = re.search(r"First-token latency:\s*([\d.]+)\s*ms", captured)
    if m:
        ttft = float(m.group(1))

    m = re.search(r"Avg next-token latency:\s*([\d.]+)\s*ms", captured)
    if m:
        decode = float(m.group(1))

    m = re.search(r"Per-token:\s*([\d.,\s]+)\s*ms", captured)
    if m:
        per_token = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]

    return ttft, decode, per_token


def _steady_state_itl(per_token: list[float]) -> float | None:
    """Mean of post-TTFT tokens after excluding compile-spike outliers.

    ``per_token[0]`` is TTFT and is always excluded.  Any remaining token
    whose latency exceeds ``_OUTLIER_THRESHOLD * median`` is also excluded.
    Returns None when fewer than 2 decode tokens are available.
    """
    if len(per_token) < 2:
        return None
    decode = per_token[1:]  # drop TTFT
    if len(decode) == 1:
        return decode[0]
    median = statistics.median(decode)
    if median <= 0:
        return statistics.mean(decode)
    steady = [v for v in decode if v <= _OUTLIER_THRESHOLD * median]
    return statistics.mean(steady) if steady else None


# ---------------------------------------------------------------------------
# Core smoke-test function (used by both the CLI script and pytest)
# ---------------------------------------------------------------------------


def run_multicard_smoke_test(
    model_path: str,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    dtype: "torch.dtype | None" = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Load model and generate tokens; return a diagnostics dict.

    Card selection is driven by SPYRE_DEVICES (primary) or AIU_IDS (fallback).
    Load and generation are wrapped in separate try/except blocks so a failure
    clearly identifies whether it occurred at load time or generate time, with
    the full traceback captured rather than swallowed.

    Args:
        model_path:     HuggingFace repo ID or local directory.
        max_new_tokens: Token generation budget.
        dtype:          Torch dtype passed to from_pretrained (None = model default).
        batch_size:     Number of identical prompts to batch together (default 1).

    Returns a dict with keys:
        model              - model path used
        spyre_devices_env  - value of SPYRE_DEVICES at call time (None if unset)
        resolved_cards     - PCI hint list derived from AIU_WORLD_RANK_N env vars
                             (informational only; not confirmed by the runtime)
        aiu_ids_env        - value of AIU_IDS at call time (None if unset)
        local_rank         - LOCAL_RANK of this process
        world_size         - WORLD_SIZE of this run
        batch_size         - batch size used
        status             - "PASS" | "FAIL" | "ERROR"
        load_s             - seconds spent in from_pretrained (None on load error)
        gen_s              - seconds spent in generate() (None on gen error)
        ttft_ms            - first-token latency in ms (or None)
        decode_ms          - avg next-token latency in ms (or None)
        steady_itl_ms      - steady-state ITL with outliers removed (or None)
        output             - generated text for sequence 0 (empty string on any error)
        error              - "PHASE FAILED\\n<traceback>" string, or None on PASS
    """
    # torch MUST be imported before torch_spyre.  torch_spyre registers itself
    # as a PyTorch backend at import time; importing it before torch triggers a
    # circular import error.
    import torch  # noqa: F401  — must precede any torch_spyre import
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM
    from tests.conftest import encode_generation_inputs

    aiu_ids_env = os.environ.get("AIU_IDS")
    spyre_devices_env = os.environ.get("SPYRE_DEVICES")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # If AIU_IDS has more addresses than WORLD_SIZE (e.g. all 4 cards set but
    # only 1 or 2 ranks launched), trim to WORLD_SIZE entries for display.
    if aiu_ids_env is not None:
        _ids = [c.strip() for c in aiu_ids_env.split(",") if c.strip()]
        if len(_ids) > world_size:
            aiu_ids_env = ",".join(_ids[:world_size])
            os.environ["AIU_IDS"] = aiu_ids_env

    # Attempt to map SPYRE_DEVICES indices to PCI addresses via AIU_WORLD_RANK_*
    # env vars set by the system login script.  This is a local hint only —
    # not confirmation of what Flex actually bound.  There is currently no API
    # to query the runtime for the real mapping; see TODO at top.
    resolved_cards: list[str] = []
    if spyre_devices_env is not None:
        for idx_str in spyre_devices_env.split(","):
            idx_str = idx_str.strip()
            if idx_str.isdigit():
                pci = os.environ.get(f"AIU_WORLD_RANK_{idx_str}")
                resolved_cards.append(pci if pci else f"index {idx_str} (unresolved)")

    print(f"\n{'=' * 70}")
    print(f"  Multicard smoke test  [rank {local_rank}/{world_size}]")
    print(f"  Model         : {model_path}")
    print(f"  SPYRE_DEVICES : {spyre_devices_env!r}")
    if resolved_cards:
        print(
            f"  PCI hint      : {', '.join(resolved_cards)}  (AIU_WORLD_RANK_* guess, unconfirmed)"
        )
    print(f"  AIU_IDS       : {aiu_ids_env!r}")
    print(f"  WORLD_SIZE    : {world_size}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  batch_size    : {batch_size}")
    print(f"  Datatype      : {dtype}")
    print(f"  Prompt        : {_PROMPT!r}")
    print(f"{'=' * 70}")

    result: dict[str, Any] = {
        "model": model_path,
        "spyre_devices_env": spyre_devices_env,
        "resolved_cards": resolved_cards,
        "aiu_ids_env": aiu_ids_env,
        "local_rank": local_rank,
        "world_size": world_size,
        "batch_size": batch_size,
        "status": "ERROR",
        "load_s": None,
        "gen_s": None,
        "ttft_ms": None,
        "decode_ms": None,
        "steady_itl_ms": None,
        "outputs": [],  # decoded text per sequence (populated after generation)
        "output": "",  # sequence 0 text, for backward-compat / single-card use
        "seq_checks": [],
        "error": None,
    }

    # ── Phase 1: model load ────────────────────────────────────────────────
    print(f"\n{'=' * 20} Loading Model...")

    model = None
    tokenizer = None
    load_t0 = time.time()
    try:
        tp = "auto" if world_size > 1 else None
        kwargs: dict[str, Any] = {"tp_plan": tp}
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = AutoSpyreModelForCausalLM.from_pretrained(model_path, **kwargs)
        result["load_s"] = time.time() - load_t0
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Load time  : {result['load_s']:.1f}s  [OK]")
    except Exception:
        result["load_s"] = time.time() - load_t0
        result["error"] = "LOAD FAILED\n" + traceback.format_exc()
        print(
            f"  [rank {local_rank}] Load FAILED (after {result['load_s']:.1f}s):\n{result['error']}"
        )
        return result

    # batch_size > 1: repeat the same prompt N times (tests the KV cache batch
    # scatter path without needing N different prompts).
    prompts = [_PROMPT] * batch_size
    encoded = encode_generation_inputs(tokenizer, prompts)

    actual_prompt_len = encoded["input_ids"].shape[1]
    print(
        f"  Input shape   : {list(encoded['input_ids'].shape)}  "
        f"(batch={batch_size}, tokens={actual_prompt_len})"
    )

    def _run_generate() -> tuple[list[str], str]:
        """Run one generate call; return (output_texts_per_seq, captured_stdout)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sequences = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                timing=True,
            )
        captured = buf.getvalue()
        print(captured, end="")
        output_texts = [
            tokenizer.decode(sequences[i, actual_prompt_len:], skip_special_tokens=True)
            for i in range(sequences.shape[0])
        ]
        return output_texts, captured

    # ── Phase 2: generation (warm up) ────────────────────────────────────────────────
    print(f"\n{'=' * 20} Warmup Model...")
    try:
        _run_generate()
    except Exception:
        result["error"] = "GENERATE FAILED\n" + traceback.format_exc()
        print(f"  [rank {local_rank}] Warmup FAILED:\n{result['error']}")
        return result

    # ── Phase 3: generation ────────────────────────────────────────────────
    print(f"\n{'=' * 20} Run Model...")
    gen_t0 = time.time()
    try:
        output_texts, captured = _run_generate()
        result["gen_s"] = time.time() - gen_t0

        ttft, decode, per_token = _parse_timing(captured)
        result["ttft_ms"] = ttft
        result["decode_ms"] = decode
        result["steady_itl_ms"] = _steady_state_itl(per_token)

        result["outputs"] = output_texts
        result["output"] = output_texts[0] if output_texts else ""

        for i, text in enumerate(output_texts):
            print(f"  Output[{i}]  : {text!r}")
        print(f"  Gen time   : {result['gen_s']:.1f}s  [OK]")
        if result["steady_itl_ms"] is not None:
            print(
                f"  Steady ITL : {result['steady_itl_ms']:.1f} ms  (outliers excluded)"
            )

        # Validate every sequence: non-empty, has tokens, not all-zero, not all-same.
        seq_checks: list[dict] = []
        for text in output_texts:
            c: dict[str, Any] = {
                "non_empty": len(text.strip()) > 0,
                "not_all_spaces": text.strip() != "",
            }
            if text:
                gen_ids = tokenizer.encode(text, add_special_tokens=False)
                c["has_tokens"] = len(gen_ids) > 0
                c["not_all_zero"] = not all(t == 0 for t in gen_ids)
                c["not_all_same"] = len(set(gen_ids)) > 1 or len(gen_ids) <= 1
            else:
                c["has_tokens"] = False
                c["not_all_zero"] = False
                c["not_all_same"] = False
            seq_checks.append(c)

        result["seq_checks"] = seq_checks
        all_pass = all(v for c in seq_checks for k, v in c.items())
        result["status"] = "PASS" if all_pass else "FAIL"
        if not all_pass:
            for i, c in enumerate(seq_checks):
                failed = [k for k, v in c.items() if not v]
                if failed:
                    print(f"  [seq {i}] FAIL checks: {failed}")
    except Exception:
        result["gen_s"] = time.time() - gen_t0
        result["error"] = "GENERATE FAILED\n" + traceback.format_exc()
        print(
            f"  [rank {local_rank}] Generate FAILED (after {result['gen_s']:.1f}s):\n{result['error']}"
        )

    return result


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------
# NOTE: This test runs in a single process (no torchrun).  It exercises the
# single-card code path (tp_plan=None, WORLD_SIZE=1).  Multi-card TP requires
# torchrun; use scripts/run_multicard_smoke.py for that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_path", [_DEFAULT_MODEL])
def test_multicard_smoke_single_card(model_path: str) -> None:
    """Single-card smoke test: load, generate, verify output passes all checks."""
    result = run_multicard_smoke_test(model_path)
    assert result["status"] == "PASS", (
        f"Smoke test failed with status {result['status']}.\n"
        f"Checks: {result.get('seq_checks')}\n"
        f"Error: {result['error']}"
    )
    assert any(
        _EXPECTED_SUBSTRING.lower() in text.lower()
        for text in result.get("outputs", [result["output"]])
    ), f"Expected {_EXPECTED_SUBSTRING!r} in outputs, got {result.get('outputs')!r}"


# ---------------------------------------------------------------------------
# Standalone entry point (called by scripts/run_multicard_smoke.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _model = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_MODEL
    _max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_MAX_NEW_TOKENS
    _result = run_multicard_smoke_test(_model, _max_new_tokens)
    print(f"\nFinal status: {_result['status']}")
    if _result["error"]:
        print(_result["error"])
        raise SystemExit(1)
