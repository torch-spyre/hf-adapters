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
Edge-case tests for ``hf_common.generate()``.

The decode loop has several control-flow regimes:
  * Prefill (i==0): logits sampled from ``[:, -1, :]``.
  * Filling (decode steps within a block): ``[:, -grab_idx, :]``.
  * Expansion (block boundary): cache grows by ``BLOCK_SIZE`` then
    ``[:, -BLOCK_SIZE, :]``.
  * Per-sequence ``finished`` mask + per-row ``num_generated`` block-walk on
    decode.

Each case here picks a prompt-length / max_new_tokens combination that drives
the loop into one or more of those regimes, then asserts the adapter's output
matches stock HF ``generate(do_sample=False)`` token-for-token.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest. Shared case tables and helpers (used by the Spyre
counterpart too) live in ``_generate_edge_case_helpers.py``.
"""

import gc
import types

import pytest
import torch
from model_registry import CAUSAL_PATHS
from transformers import AutoTokenizer

from hf_adapters.auto_spyre_model import resolve_adapter_module
from tests._generate_edge_case_helpers import (
    BLOCK_SIZE,
    CASES,
    EOS_CASES,
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
    NoPadTokenizer,
    forced_eos_expected,
    greedy_token_ids,
    hf_reference_outputs,
    make_prompt_with_eos_inside,
    make_prompts,
    pick_forced_eos_id,
)
from tests.conftest import load_ref_model

# All tests in this file are marked `slow` and dropped from PR runs except
# `test_generate_zero_new_tokens`, which is the lightest case and acts as the
# PR-eligible smoke test for this module. The full suite runs in the weekly cron.

# Hard-coded path for the single-model focused tests (determinism, sampling,
# forced EOS, etc.). Qwen3 0.6B is the smallest causal-LM in the registry.
QWEN3_PATH: str = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_adapter_generate(
    model_path: str,
    adapter_mod: types.ModuleType,
    hf_common_mod: types.ModuleType,
    unwrap_fn,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    **gen_kwargs,
) -> list[str]:
    """Load the adapter model, run batched generate, return decoded outputs.

    Extra ``gen_kwargs`` are forwarded to ``hf_common.generate`` (e.g.
    ``do_sample``, ``temperature``, ``top_k``).
    """
    model = load_ref_model(model_path, adapter_mod=adapter_mod)
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    unwrap_fn(model)
    gen_kwargs.setdefault("do_sample", False)
    outputs = hf_common_mod.generate(
        adapter_mod._run_forward,
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        **gen_kwargs,
    )
    del model
    gc.collect()
    return outputs


def _load_prepared_model(
    model_path: str, adapter_mod: types.ModuleType, unwrap_fn
) -> torch.nn.Module:
    """Load + prepare an adapter model once. Caller is responsible for ``del`` + gc."""
    model = load_ref_model(model_path, adapter_mod=adapter_mod)
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    unwrap_fn(model)
    return model


def _generate_only(
    model: torch.nn.Module,
    hf_common_mod: types.ModuleType,
    adapter_mod: types.ModuleType,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    **gen_kwargs,
) -> list[str]:
    """Run adapter generate against an already-prepared model (no reload)."""
    gen_kwargs.setdefault("do_sample", False)
    return hf_common_mod.generate(
        adapter_mod._run_forward,
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        **gen_kwargs,
    )


# Session-scoped cache for HF reference outputs. Stock HF generate() with
# do_sample=False is deterministic for a given (model_path, prompts,
# max_new_tokens), so multiple tests asking for the same reference can share
# one model load. We cache decoded strings, not the model — this sidesteps the
# RMSNorm-patching contamination concern (only the adapter model gets patched).
_HF_REF_CACHE: dict = {}


def _hf_reference_cached(
    model_path: str,
    adapter_mod: types.ModuleType,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    key = (model_path, tuple(prompts), max_new_tokens)
    if key in _HF_REF_CACHE:
        return _HF_REF_CACHE[key]
    ref_model = load_ref_model(model_path, adapter_mod=adapter_mod)
    ref_model.eval()
    ref_model.requires_grad_(False)
    outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new_tokens)
    del ref_model
    gc.collect()
    _HF_REF_CACHE[key] = outputs
    return outputs


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.parametrize("case_id", list(CASES.keys()), ids=list(CASES.keys()))
def test_generate_edge_case(
    model_path: str, case_id: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    targets, max_new_tokens = CASES[case_id]

    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, targets)

    # HF reference (cached across tests; deterministic do_sample=False)
    hf_outputs = _hf_reference_cached(
        model_path, adapter_mod, tokenizer, prompts, max_new_tokens
    )

    adapter_outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
    )

    assert len(adapter_outputs) == len(hf_outputs), (
        f"{case_id}: adapter returned {len(adapter_outputs)} outputs, "
        f"expected {len(hf_outputs)}"
    )
    for i, (hf_out, adapter_out) in enumerate(zip(hf_outputs, adapter_outputs)):
        assert hf_out.strip() == adapter_out.strip(), (
            f"{case_id} prompt[{i}] (target_tokens={targets[i]}, "
            f"max_new_tokens={max_new_tokens}):\n"
            f"  HF:      {hf_out!r}\n"
            f"  adapter: {adapter_out!r}"
        )


# ---------------------------------------------------------------------------
# Determinism: greedy generate() must be reproducible across calls
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_is_deterministic(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, [5, 12, 30])
    max_new_tokens = 8  # determinism is RNG-state, not block-expansion

    model = _load_prepared_model(model_path, adapter_mod, unwrap_compiled_blocks)
    try:
        out1 = _generate_only(
            model, hf_common_mod, adapter_mod, tokenizer, prompts, max_new_tokens
        )
        out2 = _generate_only(
            model, hf_common_mod, adapter_mod, tokenizer, prompts, max_new_tokens
        )
    finally:
        del model
        gc.collect()

    assert (
        out1 == out2
    ), f"non-deterministic greedy output:\n  run1: {out1}\n  run2: {out2}"


# ---------------------------------------------------------------------------
# Forced EOS: control where EOS lands so we exercise specific decode regimes
# ---------------------------------------------------------------------------
#
# The ``finished`` mask + ``num_generated`` block-walk in generate() has three
# subtle positions to cover:
#   (a) EOS within the first block (before any expansion step).
#   (b) EOS exactly on a block boundary (last token of first block).
#   (c) EOS in the second block (after at least one expansion).
#
# We can't rely on the model emitting EOS naturally at those positions, so we
# wrap the tokenizer and override ``eos_token_id`` to a token taken from the
# model's own greedy continuation at the desired offset.


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
@pytest.mark.parametrize("case_id", list(EOS_CASES.keys()), ids=list(EOS_CASES.keys()))
def test_generate_forced_eos(
    model_path: str, case_id: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    eos_offsets, max_new_tokens = EOS_CASES[case_id]
    batch_size = len(eos_offsets)

    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, [5] * batch_size)

    # Step 1: capture each prompt's natural greedy continuation.
    ref_model = load_ref_model(model_path, adapter_mod=adapter_mod)
    ref_model.eval()
    ref_model.requires_grad_(False)
    per_prompt_ids = [
        greedy_token_ids(ref_model, tokenizer, p, max_new_tokens) for p in prompts
    ]
    del ref_model
    gc.collect()

    eos_token_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
    if eos_token_id is None:
        pytest.skip(
            f"{case_id}: no shared token at offsets {eos_offsets} that is "
            "absent from earlier positions; cannot force a clean batched EOS"
        )

    # Build the expected per-row output: tokens up to (not including) the EOS.
    expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)

    # Step 2: run the adapter, forcing the stop token via the eos_token_id
    # kwarg (highest-precedence source in generate()'s HF-style resolution).
    adapter_outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        eos_token_id=eos_token_id,
    )

    assert len(adapter_outputs) == batch_size
    for b, (exp, got) in enumerate(zip(expected, adapter_outputs)):
        assert exp.strip() == got.strip(), (
            f"{case_id} row[{b}] (eos_offset={eos_offsets[b]}, "
            f"forced_eos_id={eos_token_id}):\n"
            f"  expected: {exp!r}\n"
            f"  got:      {got!r}"
        )


# ---------------------------------------------------------------------------
# max_new_tokens=0: loop must short-circuit and return empty strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_zero_new_tokens(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, [5, 12])

    outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens=0,
    )

    assert len(outputs) == len(prompts)
    for i, out in enumerate(outputs):
        assert out == "", f"prompt[{i}] expected empty output, got {out!r}"


# ---------------------------------------------------------------------------
# Sampling: identical seed -> identical outputs; different seed -> different
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_sampling_determinism(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
    max_new_tokens = SAMPLING_MAX_NEW

    sampling = SAMPLING_KWARGS

    model = _load_prepared_model(model_path, adapter_mod, unwrap_compiled_blocks)
    try:
        torch.manual_seed(1234)
        out_a1 = _generate_only(
            model,
            hf_common_mod,
            adapter_mod,
            tokenizer,
            prompts,
            max_new_tokens,
            **sampling,
        )

        torch.manual_seed(1234)
        out_a2 = _generate_only(
            model,
            hf_common_mod,
            adapter_mod,
            tokenizer,
            prompts,
            max_new_tokens,
            **sampling,
        )

        torch.manual_seed(9999)
        out_b = _generate_only(
            model,
            hf_common_mod,
            adapter_mod,
            tokenizer,
            prompts,
            max_new_tokens,
            **sampling,
        )
    finally:
        del model
        gc.collect()

    assert out_a1 == out_a2, (
        f"sampling not reproducible with the same seed:\n"
        f"  run1: {out_a1}\n  run2: {out_a2}"
    )
    assert out_a1 != out_b, (
        f"different seeds produced identical outputs (top_k=20 should diverge):\n"
        f"  seed=1234: {out_a1}\n  seed=9999: {out_b}"
    )


# ---------------------------------------------------------------------------
# eos_token_id=None: must run full max_new_tokens with no early stop
# ---------------------------------------------------------------------------
#
# Passing ``eos_token_id=None`` disables EOS stopping (matching stock HF): the
# resolved eos is None, the ``finished`` mask is never set, and the decode walk
# does not truncate. The adapter should emit exactly max_new_tokens per row.


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_no_eos_runs_full_budget(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, [5, 12])
    max_new_tokens = BLOCK_SIZE + 7  # cross a block boundary

    # HF reference with eos_token_id=None so HF also runs the full budget.
    ref_model = load_ref_model(model_path, adapter_mod=adapter_mod)
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = ref_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        new_ids = out[0][encoded["input_ids"].shape[1] :]
        assert len(new_ids) == max_new_tokens
        hf_outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    del ref_model
    gc.collect()

    # eos_token_id=None disables EOS stopping, same as the HF reference above.
    adapter_outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        eos_token_id=None,
    )

    assert len(adapter_outputs) == len(prompts)
    for i, (hf_out, ad_out) in enumerate(zip(hf_outputs, adapter_outputs)):
        assert (
            hf_out.strip() == ad_out.strip()
        ), f"no-eos prompt[{i}]:\n  HF:      {hf_out!r}\n  adapter: {ad_out!r}"


# ---------------------------------------------------------------------------
# pad_token is None: triggers ``pad_token = eos_token`` fallback (line 759)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_no_pad_token_fallback(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, [5, 12])  # mixed lengths -> left-padding
    max_new_tokens = 16

    hf_outputs = _hf_reference_cached(
        model_path, adapter_mod, tokenizer, prompts, max_new_tokens
    )

    wrapped = NoPadTokenizer(tokenizer)
    adapter_outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        wrapped,
        prompts,
        max_new_tokens,
    )

    for i, (hf_out, ad_out) in enumerate(zip(hf_outputs, adapter_outputs)):
        assert (
            hf_out.strip() == ad_out.strip()
        ), f"no-pad-token prompt[{i}]:\n  HF:      {hf_out!r}\n  adapter: {ad_out!r}"


# ---------------------------------------------------------------------------
# top_k=0 sampling: skips the top-k filter branch (line 931 of generate)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_sampling_top_k_zero(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
    max_new_tokens = SAMPLING_MAX_NEW

    sampling = dict(do_sample=True, temperature=1.0, top_k=0)

    model = _load_prepared_model(model_path, adapter_mod, unwrap_compiled_blocks)
    try:
        torch.manual_seed(2024)
        out1 = _generate_only(
            model,
            hf_common_mod,
            adapter_mod,
            tokenizer,
            prompts,
            max_new_tokens,
            **sampling,
        )
        torch.manual_seed(2024)
        out2 = _generate_only(
            model,
            hf_common_mod,
            adapter_mod,
            tokenizer,
            prompts,
            max_new_tokens,
            **sampling,
        )
    finally:
        del model
        gc.collect()

    assert len(out1) == len(prompts)
    for i, s in enumerate(out1):
        assert s, f"top_k=0 prompt[{i}] produced empty output"
    assert (
        out1 == out2
    ), f"top_k=0 sampling not reproducible:\n  run1: {out1}\n  run2: {out2}"


# ---------------------------------------------------------------------------
# EOS id appearing inside the prompt: must NOT trip the finished mask.
# ---------------------------------------------------------------------------
#
# generate() updates ``finished`` only on emitted next_tokens (line 949 of
# hf_common.generate), not on prompt content. A prompt containing the model's
# eos_token_id should still produce a normal greedy continuation matching HF.


@pytest.mark.slow
@pytest.mark.parametrize("model_path", [QWEN3_PATH], ids=[QWEN3_PATH])
def test_generate_eos_inside_prompt(
    model_path: str, unwrap_compiled_blocks, hf_common_mod
) -> None:
    adapter_mod = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        pytest.skip("tokenizer has no eos_token_id; case not applicable")
    prompts = [make_prompt_with_eos_inside(tokenizer, eos_id, target_tokens=12)]
    max_new_tokens = BLOCK_SIZE + 8  # cross a block boundary

    hf_outputs = _hf_reference_cached(
        model_path, adapter_mod, tokenizer, prompts, max_new_tokens
    )

    adapter_outputs = _run_adapter_generate(
        model_path,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
    )

    assert hf_outputs[0].strip() == adapter_outputs[0].strip(), (
        f"eos-in-prompt:\n  HF:      {hf_outputs[0]!r}\n"
        f"  adapter: {adapter_outputs[0]!r}"
    )
