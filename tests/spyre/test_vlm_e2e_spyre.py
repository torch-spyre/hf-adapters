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
End-to-end Spyre test for multimodal (image→text) VLM adapters.

For each registered ``kind="vlm"`` adapter in ``VISION_MODELS``, loads the full
VLM, prepares it for Spyre (both towers compiled), moves it to Spyre, and
compares the adapter (Spyre) against stock HF (CPU) **token-by-token over prefill
+ N decode steps**. The hard assertion is a per-step logit-vector cosine floor;
per-step top-1 agreement is printed for visibility but not asserted (see below).
This is the same spirit as ``test_e2e_token_compare_spyre`` for causal LMs, which
likewise compares per-step logits — but it asserts the cosine rather than top-1.

  test_vlm_generate_spyre[<key>]
    Builds image+prompt with the processor, runs stock greedy on CPU capturing
    its per-step logits and token ids, then drives the adapter on Spyre
    **teacher-forced on stock's token sequence** (prefill via the adapter's
    deepstack ``_prefill_forward``; decode via the adapter's own block-decode
    mechanics, but feeding stock's chosen token back each step instead of the
    adapter's argmax). At each step the adapter's full logit vector must stay
    within ``MIN_COSINE`` of stock's. A coherent free-run caption is also printed
    as a human-eyeball diagnostic.

Why cosine, not top-1? Both paths run the same fp16 dtype, so any divergence is
Spyre's native accumulation + the adapter's op decomposition (matmul-RoPE, padded
SDPA) versus CPU's fp16 — not a precision tier. Teacher-forcing
(same prefix both sides) removes greedy-fork amplification, so the cosine cleanly
measures per-step logit fidelity and exercises the full decode path (KV-cache,
masks, RoPE) that a prefill-only check would skip; a real decode bug tanks it. But
top-1 alone is too brittle here: when the next-token distribution has two near-tied
candidates (within fp16 rounding), the tiny substrate difference can flip the
argmax while cosine stays ~0.9999 — benign rounding on a numerically arbitrary
winner, not a regression. Open-ended caption prompts hit such near-ties readily
(the sharp text prompts ``test_e2e_token_compare_spyre`` uses do not, which is why
it can assert top-1). So we assert the cosine and report top-1.

Parametrized off ``VISION_MODELS``; selects ``kind="vlm"`` entries.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py
    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm
"""

import gc
import math
import types
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _vision_helpers import (
    build_vlm_batch,
    stock_vlm_generate,
    stock_vlm_greedy_steps,
)
from conftest import load_hf_vlm
from model_registry import VISION_PATHS

from hf_adapters.auto_spyre_model import (
    IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    allocate_kv_caches,
    build_expansion_mask,
    get_model_dtype,
    move_to_spyre_with_layout,
    pad_and_position,
)
from tests.conftest import torch_dtype_for_model_path

MAX_NEW_TOKENS = 16
# Decode steps to verify token-by-token (prefill + this many decode steps). Kept
# modest like the causal lane (test_e2e_token_compare_spyre uses 4).
NUM_COMPARE_STEPS = 5
# Per-step logit-vector agreement floor (teacher-forced, so each step is a clean
# same-prefix comparison). This is the hard assertion — a real decode bug tanks
# the cosine. Generous vs measured agreement (~0.9999) so benign fp16 substrate
# rounding never trips it, while a genuine regression (cosine << 0.999) does.
MIN_COSINE = 0.999
PROMPT = "Briefly describe this image."


def _adapter_generate(
    adapter: types.ModuleType,
    model: nn.Module,
    processor: Any,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> list[str]:
    """Drive an adapter's multimodal ``generate`` from a processor batch."""
    return adapter.generate(
        model,
        processor,
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def _adapter_teacher_forced_steps(
    adapter: types.ModuleType,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    forced_tokens: list[int],
) -> list[torch.Tensor]:
    """Per-step adapter logits on Spyre, teacher-forced on ``forced_tokens``.

    Replicates the adapter ``generate`` block-decode bookkeeping (KV caches,
    per-step ``decode_pos`` / fill vs expansion masks, the right-aligned
    ``result`` block-walk), but feeds ``forced_tokens[i]`` back at each decode
    step instead of the adapter's own argmax — so the adapter sees the *same*
    prefix as stock and the comparison is free of greedy-fork drift. Step 0 is
    the deepstack prefill (the same ``_prefill_forward`` that ``generate`` /
    ``prefill_logits`` use).

    Returns ``[logits_step0, logits_step1, ...]`` (CPU, ``[vocab]`` each) of
    length ``len(forced_tokens)`` — prefill plus one per forced decode token.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    model_d_type = get_model_dtype(model)
    backbone = adapter.get_backbone(model)
    emb_mult = backbone.embedding_multiplier

    batch_size, prompt_length = input_ids.shape
    actual_prompt_lengths = attention_mask.sum(dim=1)
    n_steps = len(forced_tokens)

    max_cache_len = (
        math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
        + math.ceil((n_steps + 1) / BLOCK_SIZE) * BLOCK_SIZE
    )
    padded_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
        input_ids, actual_prompt_lengths
    )
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, model_d_type
    )

    result = padded_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    fill_mask_device = None
    per_step_logits = []

    def embed_ids(ids):
        return backbone.embed_tokens(ids) * emb_mult

    def _write_token(tok_id):
        nonlocal result, tokens_in_block
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        if tokens_in_block == 0:
            result = torch.nn.functional.pad(result, (0, BLOCK_SIZE))
        grab = (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        result[:, -grab] = tok_id

    # --- Step 0: deepstack prefill ---
    logits = adapter._prefill_forward(
        model,
        padded_ids,
        padded_len,
        prompt_offsets,
        position_ids,
        pixel_values,
        image_sizes,
        key_caches,
        value_caches,
        max_cache_len,
    )
    per_step_logits.append(logits.to("cpu")[0, -1, :].float())
    decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
    for j in range(BLOCK_SIZE):
        decode_pos[0, j] = actual_prompt_lengths[0].item() + j - BLOCK_SIZE
    _write_token(forced_tokens[0])

    # --- Decode steps teacher-forced on stock's tokens ---
    for i in range(1, n_steps):
        is_filling = tokens_in_block > 0
        next_input = result[:, -BLOCK_SIZE:].to(DEVICE)
        next_embeds = embed_ids(next_input)
        if is_filling:
            fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
            logits = adapter._logits_from_embeds(
                model,
                next_embeds,
                decode_pos.to(DEVICE),
                fill_mask_device,
                key_caches,
                value_caches,
                is_filling=True,
                token_index=tokens_in_block,
                cache_position=fill_pos,
            )
            grab_idx = BLOCK_SIZE - tokens_in_block
            step_logits = logits.to("cpu")[0, -grab_idx, :].float()
        else:
            current_cache_len += BLOCK_SIZE
            decode_pos = decode_pos + BLOCK_SIZE
            exp_mask = build_expansion_mask(
                batch_size,
                BLOCK_SIZE,
                max_cache_len,
                current_cache_len,
                prompt_offsets,
                dtype=model_d_type,
            )
            logits = adapter._logits_from_embeds(
                model,
                next_embeds,
                decode_pos.to(DEVICE),
                exp_mask.to(DEVICE),
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=current_cache_len - BLOCK_SIZE,
            )
            step_logits = logits.to("cpu")[0, -BLOCK_SIZE, :].float()
            fill_mask_device = exp_mask.to(DEVICE)
        per_step_logits.append(step_logits)
        _write_token(forced_tokens[i])

    return per_step_logits


@pytest.mark.parametrize("model_path", VISION_PATHS, ids=VISION_PATHS)
def test_vlm_generate_spyre(model_path: str) -> None:
    adapter = resolve_adapter_module(
        model_path, mapping=IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )
    dtype = torch_dtype_for_model_path(model_path)

    processor, batch = build_vlm_batch(model_path, PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    tokenizer = processor.tokenizer

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    # --- Stock CPU reference (run first, before prepare_for_spyre patches RMSNorm).
    # Capture stock's per-step greedy logits + token ids (the forcing sequence and
    # the per-step top-1 reference), plus its free-run caption for an eyeball. ---
    print("  Running stock CPU reference (per-step greedy) ...")
    ref_logits, ref_tokens = stock_vlm_greedy_steps(
        model_path, batch, dtype, NUM_COMPARE_STEPS
    )
    ref_text = stock_vlm_generate(model_path, processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()

    # --- Adapter on Spyre ---
    print("  Loading model for Spyre ...")
    model = load_hf_vlm(model_path, dtype, adapter_mod=adapter)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    move_to_spyre_with_layout(model, dtype)

    # Per-step adapter logits on Spyre, teacher-forced on stock's tokens (so the
    # comparison is free of greedy-fork drift while still exercising decode).
    print(f"  Running adapter prefill + {len(ref_tokens) - 1} forced decode steps ...")
    with torch.no_grad():
        sp_logits = _adapter_teacher_forced_steps(adapter, model, batch, ref_tokens)

    # Free-run caption — human-eyeball diagnostic + non-degeneracy check only.
    print("  Running adapter free-run generate (diagnostic) ...")
    with torch.no_grad():
        adapter_text = _adapter_generate(
            adapter, model, processor, batch, MAX_NEW_TOKENS
        )
    del model
    gc.collect()

    # --- Per-step logit cosine (the assertion) + top-1 agreement (reported) ---
    # Teacher-forced, so every step is a clean same-prefix comparison. The cosine
    # floor is the hard check: a real decode bug (KV-cache, mask, RoPE) tanks the
    # whole logit vector. Top-1 agreement is printed for visibility but NOT
    # asserted — a single near-tie (two candidates within fp16 rounding) can flip
    # the argmax while cosine stays ~0.9999, which is benign substrate rounding,
    # not a regression.
    print("  step      | stock top1          | spyre top1          | match | cosine")
    low_cosine = []
    n_top1_match = 0
    for step, (rl, sl) in enumerate(zip(ref_logits, sp_logits)):
        n = min(rl.shape[-1], sl.shape[-1])
        rl, sl = rl[:n], sl[:n]
        r_top1 = int(rl.argmax())
        s_top1 = int(sl.argmax())
        cosine = F.cosine_similarity(rl, sl, dim=-1).item()
        ok = r_top1 == s_top1
        n_top1_match += ok
        label = "prefill" if step == 0 else f"decode-{step}"
        print(
            f"  {label:>9} | {r_top1:>6} {tokenizer.decode([r_top1])!r:>11} "
            f"| {s_top1:>6} {tokenizer.decode([s_top1])!r:>11} "
            f"| {'OK' if ok else 'tie?':>5} | {cosine:.6f}"
        )
        if cosine < MIN_COSINE:
            low_cosine.append((label, cosine))
    print(f"  top-1 agreement: {n_top1_match}/{len(ref_logits)} steps (reported)")
    print(f"  stock:   {ref_text!r}")
    print(f"  adapter free-run: {adapter_text[0]!r}")

    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
    assert not low_cosine, (
        f"adapter logits drifted below cosine {MIN_COSINE} (teacher-forced) at: "
        + ", ".join(f"{lab}: {c:.6f}" for lab, c in low_cosine)
    )
