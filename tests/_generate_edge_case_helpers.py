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

"""Shared helpers for ``test_generate_edge_cases_cpu.py`` (CPU pytest) and
``test_generate_edge_cases_spyre.py`` (Spyre script).

Pulls the model/device-agnostic pieces — prompt synthesis, HF reference
capture, forced-EOS tokenizer wrapper, and the case tables — into one place
so the two drivers can share them.
"""

import torch

BLOCK_SIZE = 64  # mirrors hf_common.BLOCK_SIZE; kept local so case ids are stable


# ---------------------------------------------------------------------------
# Case tables
# ---------------------------------------------------------------------------
#
# Each entry in ``CASES`` is ``(prompt_token_targets, max_new_tokens)``.
# ``prompt_token_targets`` is a list of approximate token lengths used to
# synthesise prompts; ``len(prompt_token_targets) == batch_size``.
#
# The Spyre script picks a curated subset by key (every Spyre case takes
# minutes); the CPU pytest runs the full grid. Keep keys stable so the
# Spyre subset stays in sync.

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
    "mixed_long_short": ([5, 12, BLOCK_SIZE], 32),
    "mixed_block_aligned": ([5, BLOCK_SIZE, BLOCK_SIZE - 1], 16),
    "mixed_with_single_token": ([1, 5, 30], 16),  # single-token row in batch
    # --- batch=2: EOS hit before max_new_tokens (per-sequence finished mask) ---
    # 64 is generous enough that an instruction-tuned model is likely to emit
    # EOS for at least one prompt within the budget. Even when neither does,
    # the per-prompt HF reference comparison still catches any divergence.
    "eos_within_budget": ([6, 8], BLOCK_SIZE + 4),
}


# Subset of CASES marked @pytest.mark.slow on the CPU lane: each is covered by
# a cheaper sibling regime (see issue #35), so on the default lane they are
# deselected. Run them with `--run-slow`.
SLOW_CPU_CASE_KEYS = {
    "short_three_blocks",  # 3rd expansion is the same path as 2nd (covered by short_two_blocks_plus)
    "short_two_blocks_exact",  # exact-block boundary == same path as +N (covered by short_two_blocks_plus)
    "long_prompt_long_gen",  # long-prompt covered by long_multi_block; long-gen by short_two_blocks_plus
    "mixed_cross_block",  # batch=3 cross-block covered by mixed_short + short_cross_block
}


# (eos_offset_per_prompt, max_new_tokens). EOS offset is 0-indexed: the token
# at that position becomes the EOS marker, so the decoded output should be the
# tokens up to (but excluding) that position.
EOS_CASES = {
    # batch=1 single-row regimes
    "eos_first_token": ([0], 16),  # finished.all() trips on step 0
    "eos_mid_block": ([10], 32),  # within first block, fill arm
    "eos_last_of_block": ([BLOCK_SIZE - 1], BLOCK_SIZE + 8),  # block boundary
    "eos_first_of_second_block": ([BLOCK_SIZE], BLOCK_SIZE + 16),  # expansion arm
    "eos_deep_in_second_block": ([BLOCK_SIZE + 20], BLOCK_SIZE + 24),  # mid-block-2
    # batch=3 — per-row finished mask, EOS at different offsets per row
    "eos_staggered": ([3, BLOCK_SIZE - 1, BLOCK_SIZE + 5], BLOCK_SIZE + 16),
    # batch=2 — one row finishes very early while the other generates well past
    # a block boundary; verifies the long row keeps producing correct tokens
    # after finished[short_row] = True.
    "eos_short_finishes_long_continues": ([2, BLOCK_SIZE + 10], BLOCK_SIZE + 16),
    # EOS on the very last allowed step: ``finished.all()`` and the
    # ``i == max_new_tokens - 1`` loop end coincide. max_new_tokens is set to
    # eos_offset + 1 so the EOS token is the last one emitted.
    "eos_on_last_step": ([7], 8),
}


# The Spyre side of these tests now lives as one file per case under
# tests/spyre/edge_cases/, parametrized over models. The CI matrix gives each
# (model, case) cell its own runner process — that's the process boundary
# replaced what test_generate_edge_cases_spyre.py used to do via a size-1
# multiprocessing.Pool.


# Sampling kwargs used by both the CPU pytest sampling-determinism test and the
# Spyre script. Kept here so the two stay in sync.
SAMPLING_KWARGS = dict(do_sample=True, temperature=1.0, top_k=20)
SAMPLING_MAX_NEW = (
    20  # short — sampling reproducibility is RNG-state, not block-expansion
)
SAMPLING_TARGETS = [8, 16]


# ---------------------------------------------------------------------------
# Prompt synthesis
# ---------------------------------------------------------------------------


def make_prompt_of_length(tokenizer, target_tokens):
    """Build a prompt that tokenizes to ~target_tokens tokens.

    Repeats a base sentence until the tokenized length crosses the target,
    then truncates the id list back to exactly ``target_tokens`` and decodes.
    The tokenizer's BOS handling means re-encoding the decoded string can give
    a slightly different count, but it's close enough — the caller only cares
    about which control-flow regime the length lands in, not exact lengths.
    """
    base = "The quick brown fox jumps over the lazy dog. "
    s = base
    while len(tokenizer(s, add_special_tokens=False)["input_ids"]) < target_tokens:
        s += base
    ids = tokenizer(s, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_prompts(tokenizer, targets):
    """Build a list of prompts, one per target token length."""
    return [make_prompt_of_length(tokenizer, t) for t in targets]


# ---------------------------------------------------------------------------
# HF reference capture (CPU; used by both CPU and Spyre drivers as ground truth)
# ---------------------------------------------------------------------------


def hf_reference_outputs(model, tokenizer, prompts, max_new_tokens):
    """Run stock HF ``generate(do_sample=False)`` on each prompt individually."""
    if max_new_tokens == 0:
        return ["" for _ in prompts]
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


def greedy_token_ids(model, tokenizer, prompt, max_new_tokens):
    """Return the list of token IDs HF ``generate(do_sample=False)`` emits."""
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0][encoded["input_ids"].shape[1] :].tolist()


# ---------------------------------------------------------------------------
# Forced EOS — pick a clean stop token to pass via generate(eos_token_id=...)
# ---------------------------------------------------------------------------
#


def pick_forced_eos_id(per_prompt_ids, eos_offsets):
    """Pick a token id that lands at offset[b] in row b for every row, and is
    absent from earlier positions.

    ``generate()`` takes a single ``eos_token_id`` for the whole batch, so to
    make every row hit EOS at its own offset we need a token shared across
    rows at the requested offsets and not present earlier in any row. Returns
    ``None`` if no such token exists — caller should skip the case rather
    than silently testing the wrong thing.
    """
    if any(off >= len(ids) for ids, off in zip(per_prompt_ids, eos_offsets)):
        return None
    candidates = {per_prompt_ids[0][eos_offsets[0]]}
    for b in range(1, len(per_prompt_ids)):
        candidates &= {per_prompt_ids[b][eos_offsets[b]]}
    for cand in candidates:
        if all(
            cand not in per_prompt_ids[b][: eos_offsets[b]]
            for b in range(len(per_prompt_ids))
        ):
            return cand
    return None


def forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer):
    """Decode the per-row tokens up to (not including) each row's EOS offset."""
    return [
        tokenizer.decode(per_prompt_ids[b][: eos_offsets[b]], skip_special_tokens=True)
        for b in range(len(eos_offsets))
    ]


# ---------------------------------------------------------------------------
# Tokenizer overrides for branch coverage in ``hf_common.generate``
# ---------------------------------------------------------------------------


class _DelegatingTokenizer:
    """Forwards ``__call__``/``decode`` and unknown attribute reads to a base
    tokenizer. Subclasses override specific attributes to exercise branches in
    ``hf_common.generate`` (e.g. no pad token).
    """

    def __init__(self, base):
        self._base = base

    def __call__(self, *args, **kwargs):
        return self._base(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._base.decode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


class NoPadTokenizer(_DelegatingTokenizer):
    """Hides ``pad_token`` so ``generate()`` exercises the
    ``pad_token = eos_token`` fallback at the top of the function. Reading
    ``pad_token`` returns ``None``; writes are stored on the base tokenizer
    so ``generate()``'s assignment takes effect.
    """

    @property
    def pad_token(self):
        return None

    @pad_token.setter
    def pad_token(self, value):
        # generate() does ``tokenizer.pad_token = tokenizer.eos_token`` when
        # pad_token is None — let that assignment land on the real tokenizer
        # so subsequent ``tokenizer(... padding=True ...)`` calls work.
        self._base.pad_token = value


# ---------------------------------------------------------------------------
# In-prompt EOS — verify generate() does NOT mistake an EOS id appearing in
# the prompt for an emission and stop early.
# ---------------------------------------------------------------------------


def make_prompt_with_eos_inside(tokenizer, eos_token_id, target_tokens=10):
    """Build a prompt whose token ids include ``eos_token_id`` somewhere in
    the middle. The ``finished`` mask is only updated on emitted tokens (line
    949 of hf_common.generate), so generation should proceed normally.

    Returns the decoded prompt string. The prompt is ~target_tokens long with
    the EOS id placed at position ``target_tokens // 2``.
    """
    base_ids = tokenizer(
        "The quick brown fox jumps over the lazy dog.",
        add_special_tokens=False,
    )["input_ids"][: target_tokens - 1]
    mid = max(1, len(base_ids) // 2)
    ids = base_ids[:mid] + [eos_token_id] + base_ids[mid:]
    return tokenizer.decode(ids, skip_special_tokens=False)
