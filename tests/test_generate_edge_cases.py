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
this file is plain pytest.
"""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Registry: one representative per model family that exercises a distinct
# code path (RoPE shape / GQA layout / head-dim padding / tokenizer).
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
    "llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
    },
}

BLOCK_SIZE = 64  # mirrors hf_common.BLOCK_SIZE; kept local so case ids are stable

# Each case: (prompt_token_targets, max_new_tokens). prompt_token_targets is a
# list of approximate token lengths used to synthesise prompts via
# ``_make_prompt_of_length``. ``len(prompt_token_targets) == batch_size``.
CASES = {
    # --- batch=1: single-prompt control-flow regimes ---
    "single_token_prompt": ([1], 16),  # extreme left-padding (63 pads, 1 real)
    "short_one_token": ([5], 1),  # only the prefill arm runs
    "short_two_tokens": ([5], 2),  # prefill + first fill step
    "short_block_minus_one": ([5], BLOCK_SIZE - 1),  # last fill step of first block
    "short_exact_block": ([5], BLOCK_SIZE),  # last token is the expansion step
    "short_cross_block": ([5], BLOCK_SIZE + 1),  # first expansion + 1 fill
    "short_two_blocks_exact": ([5], 2 * BLOCK_SIZE),  # two complete blocks
    "short_two_blocks_plus": ([5], 2 * BLOCK_SIZE + 5),  # two expansions + partial
    "short_three_blocks": ([5], 3 * BLOCK_SIZE + 7),  # three expansions, long gen
    "medium_block_aligned": ([BLOCK_SIZE - 1], 16),  # prompt fills first block
    "prompt_exactly_block": ([BLOCK_SIZE], 16),  # prompt == BLOCK_SIZE (boundary)
    "prompt_block_plus_one": ([BLOCK_SIZE + 1], 16),  # prompt straddles blocks
    "long_multi_block": ([2 * BLOCK_SIZE], 16),  # prompt > one block
    "long_prompt_long_gen": ([2 * BLOCK_SIZE], 2 * BLOCK_SIZE + 3),  # both long
    # --- batch=3: per-row offsets, mixed-length scheduling ---
    "mixed_short": ([5, 12, 30], 16),
    "mixed_cross_block": ([5, 12, 30], BLOCK_SIZE + 5),
    "mixed_long_short": ([5, 12, 2 * BLOCK_SIZE], 32),
    "mixed_block_aligned": ([5, BLOCK_SIZE, BLOCK_SIZE - 1], 24),
    "mixed_with_single_token": ([1, 5, 30], 16),  # single-token row in batch
    # --- batch=2: EOS hit before max_new_tokens (per-sequence finished mask) ---
    # 64 is generous enough that an instruction-tuned model is likely to emit
    # EOS for at least one prompt within the budget. Even when neither does,
    # the per-prompt HF reference comparison still catches any divergence.
    "eos_within_budget": ([6, 8], 64),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch_dtype(info):
    return torch.float32 if info.get("dtype") == "float32" else torch.float16


def _make_prompt_of_length(tokenizer, target_tokens):
    """Build a prompt that tokenizes to ~target_tokens tokens.

    Repeats a base sentence until the tokenized length crosses the target,
    then truncates the id list back to exactly ``target_tokens`` and decodes.
    The tokenizer's BOS handling means re-encoding the decoded string can give
    a slightly different count, but it's close enough — the test only cares
    about which control-flow regime the length lands in, not exact lengths.
    """
    base = "The quick brown fox jumps over the lazy dog. "
    s = base
    while len(tokenizer(s, add_special_tokens=False)["input_ids"]) < target_tokens:
        s += base
    ids = tokenizer(s, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _make_prompts(tokenizer, targets):
    """Build a list of prompts, one per target token length."""
    return [_make_prompt_of_length(tokenizer, t) for t in targets]


def _hf_reference_outputs(model, tokenizer, prompts, max_new_tokens):
    """Run HF native generate() on each prompt individually."""
    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_ids = out[0][encoded["input_ids"].shape[1] :]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


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
    prompts = _make_prompts(tokenizer, targets)

    # HF reference (per-prompt, before patching to avoid contamination)
    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    hf_outputs = _hf_reference_outputs(ref_model, tokenizer, prompts, max_new_tokens)
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
    prompts = _make_prompts(tokenizer, [5, 12, 30])
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
# model's own greedy continuation at the desired offset. The token there is
# guaranteed (by determinism) to be argmax at that step, so generation stops
# exactly when expected.


class _EosOverrideTokenizer:
    """Thin proxy that forwards everything to a real tokenizer except
    ``eos_token_id``, which is replaced. ``__call__`` and ``decode`` are
    forwarded explicitly because ``hf_common.generate`` invokes them.
    """

    def __init__(self, base, eos_token_id):
        self._base = base
        self.eos_token_id = eos_token_id

    def __call__(self, *args, **kwargs):
        return self._base(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._base.decode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _greedy_token_ids(model, tokenizer, prompt, max_new_tokens):
    """Return the list of token IDs HF generate() emits greedily."""
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return out[0][encoded["input_ids"].shape[1] :].tolist()


# (eos_offset_per_prompt, max_new_tokens). EOS offset is 0-indexed: the token
# at that position becomes the EOS marker, so the decoded output should be the
# tokens up to (but excluding) that position.
EOS_CASES = {
    # batch=1 single-row regimes
    "eos_first_token": ([0], 16),  # finished.all() trips on step 0
    "eos_mid_block": ([10], 32),  # within first block, fill arm
    "eos_last_of_block": ([BLOCK_SIZE - 1], BLOCK_SIZE + 8),  # block boundary
    "eos_first_of_second_block": ([BLOCK_SIZE], BLOCK_SIZE + 16),  # expansion arm
    "eos_deep_in_second_block": ([BLOCK_SIZE + 20], 2 * BLOCK_SIZE),  # mid-block-2
    # batch=3 — per-row finished mask, EOS at different offsets per row
    "eos_staggered": ([3, BLOCK_SIZE - 1, BLOCK_SIZE + 5], 2 * BLOCK_SIZE),
    # batch=2 — one row finishes very early while the other generates well past
    # a block boundary; verifies the long row keeps producing correct tokens
    # after finished[short_row] = True.
    "eos_short_finishes_long_continues": ([2, BLOCK_SIZE + 10], 2 * BLOCK_SIZE),
}


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
    prompts = _make_prompts(tokenizer, [5] * batch_size)

    # Step 1: capture each prompt's natural greedy continuation.
    torch_dtype = _torch_dtype(info)
    ref_model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    per_prompt_ids = [
        _greedy_token_ids(ref_model, tokenizer, p, max_new_tokens) for p in prompts
    ]
    del ref_model
    gc.collect()

    # Pick the same forced-EOS token for the whole batch — generate() takes a
    # single eos_token_id. To make every row hit EOS at its own offset, choose
    # a token id that appears at offset[b] in row b *and not earlier*. If no
    # such token exists for some row (the natural sequence has the same token
    # earlier), skip the case rather than silently testing the wrong thing.
    candidate_ids = set(per_prompt_ids[0][eos_offsets[0] : eos_offsets[0] + 1])
    for b in range(1, batch_size):
        candidate_ids &= set(per_prompt_ids[b][eos_offsets[b] : eos_offsets[b] + 1])
    eos_token_id = None
    for cand in candidate_ids:
        ok = True
        for b in range(batch_size):
            if cand in per_prompt_ids[b][: eos_offsets[b]]:
                ok = False
                break
        if ok:
            eos_token_id = cand
            break
    if eos_token_id is None:
        pytest.skip(
            f"{case_id}: no shared token at offsets {eos_offsets} that is "
            "absent from earlier positions; cannot force a clean batched EOS"
        )

    # Build the expected per-row output: tokens up to (not including) the EOS.
    expected = [
        tokenizer.decode(per_prompt_ids[b][: eos_offsets[b]], skip_special_tokens=True)
        for b in range(batch_size)
    ]

    # Step 2: run the adapter with the override tokenizer.
    wrapped = _EosOverrideTokenizer(tokenizer, eos_token_id)
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
    prompts = _make_prompts(tokenizer, [5, 12])

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
    prompts = _make_prompts(tokenizer, [8, 16])
    max_new_tokens = BLOCK_SIZE + 4  # cross a block boundary under sampling

    sampling = dict(do_sample=True, temperature=1.0, top_k=20)

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
