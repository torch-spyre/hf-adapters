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

import pytest
import torch
from _generate_edge_case_helpers import (
    BLOCK_SIZE,
    CASES,
    EOS_CASES,
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
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
from model_registry import CAUSAL_LM_MODELS as MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch_dtype(info):
    return torch.float32 if info.get("dtype") == "float32" else torch.float16


def _run_adapter_generate(
    info,
    adapter_mod,
    hf_common_mod,
    unwrap_fn,
    tokenizer,
    prompts,
    max_new_tokens,
    **gen_kwargs,
):
    """Load the adapter model, run batched generate, return decoded outputs.

    Extra ``gen_kwargs`` are forwarded to ``hf_common.generate`` (e.g.
    ``do_sample``, ``temperature``, ``top_k``).
    """
    torch_dtype = _torch_dtype(info)
    model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
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


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
@pytest.mark.parametrize("case_id", list(CASES.keys()), ids=list(CASES.keys()))
def test_generate_edge_case(
    model_key, case_id, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    targets, max_new_tokens = CASES[case_id]

    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, targets)

    # HF reference (per-prompt, before patching to avoid contamination)
    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new_tokens)
    del ref_model
    gc.collect()

    adapter_outputs = _run_adapter_generate(
        info,
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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_is_deterministic(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12, 30])
    max_new_tokens = BLOCK_SIZE + 5  # cross a block boundary

    out1 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
    )
    out2 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
    )

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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
@pytest.mark.parametrize("case_id", list(EOS_CASES.keys()), ids=list(EOS_CASES.keys()))
def test_generate_forced_eos(
    model_key, case_id, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    eos_offsets, max_new_tokens = EOS_CASES[case_id]
    batch_size = len(eos_offsets)

    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5] * batch_size)

    # Step 1: capture each prompt's natural greedy continuation.
    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
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

    # Step 2: run the adapter with the override tokenizer.
    wrapped = EosOverrideTokenizer(tokenizer, eos_token_id)
    adapter_outputs = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        wrapped,
        prompts,
        max_new_tokens,
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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_zero_new_tokens(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])

    outputs = _run_adapter_generate(
        info,
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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_sampling_determinism(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
    max_new_tokens = SAMPLING_MAX_NEW

    sampling = SAMPLING_KWARGS

    torch.manual_seed(1234)
    out_a1 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        **sampling,
    )

    torch.manual_seed(1234)
    out_a2 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        **sampling,
    )

    torch.manual_seed(9999)
    out_b = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        **sampling,
    )

    assert out_a1 == out_a2, (
        f"sampling not reproducible with the same seed:\n"
        f"  run1: {out_a1}\n  run2: {out_a2}"
    )
    assert out_a1 != out_b, (
        f"different seeds produced identical outputs (top_k=20 should diverge):\n"
        f"  seed=1234: {out_a1}\n  seed=9999: {out_b}"
    )


# ---------------------------------------------------------------------------
# eos_token_id is None: must run full max_new_tokens with no early stop
# ---------------------------------------------------------------------------
#
# When the tokenizer exposes ``eos_token_id = None`` (line 841 of
# hf_common.generate), the ``finished`` mask is never set and the decode walk
# does not truncate. The adapter should emit exactly max_new_tokens per row.


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_no_eos_runs_full_budget(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])
    max_new_tokens = BLOCK_SIZE + 7  # cross a block boundary

    # HF reference with eos_token_id=None so HF also runs the full budget.
    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs = []
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

    wrapped = NoEosTokenizer(tokenizer)
    adapter_outputs = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        wrapped,
        prompts,
        max_new_tokens,
    )

    assert len(adapter_outputs) == len(prompts)
    for i, (hf_out, ad_out) in enumerate(zip(hf_outputs, adapter_outputs)):
        assert (
            hf_out.strip() == ad_out.strip()
        ), f"no-eos prompt[{i}]:\n  HF:      {hf_out!r}\n  adapter: {ad_out!r}"


# ---------------------------------------------------------------------------
# pad_token is None: triggers ``pad_token = eos_token`` fallback (line 759)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_no_pad_token_fallback(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, [5, 12])  # mixed lengths -> left-padding
    max_new_tokens = 16

    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new_tokens)
    del ref_model
    gc.collect()

    wrapped = NoPadTokenizer(tokenizer)
    adapter_outputs = _run_adapter_generate(
        info,
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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_sampling_top_k_zero(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
    max_new_tokens = SAMPLING_MAX_NEW

    sampling = dict(do_sample=True, temperature=1.0, top_k=0)

    torch.manual_seed(2024)
    out1 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        **sampling,
    )
    torch.manual_seed(2024)
    out2 = _run_adapter_generate(
        info,
        adapter_mod,
        hf_common_mod,
        unwrap_compiled_blocks,
        tokenizer,
        prompts,
        max_new_tokens,
        **sampling,
    )

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


@pytest.mark.parametrize("model_key", ["qwen3"], ids=["qwen3"])
def test_generate_eos_inside_prompt(
    model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        pytest.skip("tokenizer has no eos_token_id; case not applicable")
    prompts = [make_prompt_with_eos_inside(tokenizer, eos_id, target_tokens=12)]
    max_new_tokens = BLOCK_SIZE + 8  # cross a block boundary

    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new_tokens)
    del ref_model
    gc.collect()

    adapter_outputs = _run_adapter_generate(
        info,
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
