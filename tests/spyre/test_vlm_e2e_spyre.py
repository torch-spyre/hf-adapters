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
+ N decode steps**, asserting at every step both top-1 agreement and a logit-vector
cosine floor — the same criterion as ``test_e2e_token_compare_spyre`` for causal
LMs (which asserts top-1 and reports the logit diffs).

  test_vlm_generate_spyre[<key>]
    Builds image+prompt with the processor, runs stock greedy on CPU capturing
    its per-step logits and token ids, then drives the adapter on Spyre
    **teacher-forced on stock's token sequence** (prefill via the adapter's
    deepstack ``_prefill_forward``; decode via the adapter's own block-decode
    mechanics, but feeding stock's chosen token back each step instead of the
    adapter's argmax). At each step the adapter's top-1 must equal stock's top-1
    AND its full logit vector must stay within ``MIN_COSINE`` of stock's (the
    cosine catches logit degradation that hasn't yet flipped the argmax). A
    coherent free-run caption is also printed as a human-eyeball diagnostic.

Why teacher-forced (vs each side free-running its own greedy)? Both paths run the
same fp16 dtype, so any divergence is Spyre's native accumulation + the adapter's
op decomposition (``x*x*x`` gelu-tanh, matmul-RoPE, padded SDPA) versus CPU's
fp16 — not a precision tier. When the next-token distribution has two near-tied
candidates, that small difference can flip which one wins; an own-greedy run then
forks at that step and the two sequences diverge for the rest of the generation
(often into an equally-valid caption), even though no step is actually wrong.
Feeding both sides the **same prefix** removes that fork amplification, so a
per-step top-1 mismatch signals a real decode bug (KV-cache, mask, RoPE) rather
than benign drift — while still exercising the full decode path a prefill-only
check would skip. (Open-ended caption prompts hit near-ties more readily than the
sharp text prompts ``test_e2e_token_compare_spyre`` uses, which is why this lane
forces the prefix rather than comparing two independent greedy runs.)

Parametrized off ``VISION_MODELS``; selects ``kind="vlm"`` entries.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py
    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm
"""

import gc
import importlib
import math

import pytest
import torch
import torch.nn.functional as F
from _helpers import torch_dtype_for
from _vision_helpers import (
    build_vlm_batch,
    stock_vlm_generate,
    stock_vlm_greedy_steps,
)
from model_registry import VISION_MODELS

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _model_dtype,
    _move_to_spyre_with_layout,
    allocate_kv_caches,
    build_expansion_mask,
)

MAX_NEW_TOKENS = 16
# Decode steps to verify token-by-token (prefill + this many decode steps). Kept
# modest like the causal lane (test_e2e_token_compare_spyre uses 4).
NUM_COMPARE_STEPS = 5
# Per-step logit-vector agreement floor (teacher-forced, so each step is a clean
# same-prefix comparison). Asserted alongside top-1 to catch logit degradation
# that hasn't yet flipped the argmax. Generous: fp16 substrate drift is small.
MIN_COSINE = 0.99
PROMPT = "Briefly describe this image."

MODELS = {k: v for k, v in VISION_MODELS.items() if v.get("kind") == "vlm"}


def _adapter_generate(adapter, model, processor, batch, max_new_tokens):
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


def _adapter_teacher_forced_steps(adapter, model, batch, forced_tokens):
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

    model_dtype = _model_dtype(model)
    backbone = adapter.get_backbone(model)
    emb_mult = backbone.embedding_multiplier

    batch_size, prompt_length = input_ids.shape
    actual_prompt_lengths = attention_mask.sum(dim=1)
    n_steps = len(forced_tokens)

    max_cache_len = (
        math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
        + math.ceil((n_steps + 1) / BLOCK_SIZE) * BLOCK_SIZE
    )
    padded_ids, padded_len, prompt_offsets, position_ids = adapter._pad_and_position(
        input_ids, actual_prompt_lengths
    )
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, model_dtype
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
                dtype=model_dtype,
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


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_vlm_generate_spyre(model_key):
    info = MODELS[model_key]
    adapter_module_name = info["adapter"].replace(".py", "")
    adapter = importlib.import_module(f"hf_adapters.{adapter_module_name}")
    dtype = torch_dtype_for(info)

    processor, batch = build_vlm_batch(info["path"], PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    tokenizer = processor.tokenizer

    print(f"\n{'=' * 70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'=' * 70}")

    # --- Stock CPU reference (run first, before prepare_for_spyre patches RMSNorm).
    # Capture stock's per-step greedy logits + token ids (the forcing sequence and
    # the per-step top-1 reference), plus its free-run caption for an eyeball. ---
    print("  Running stock CPU reference (per-step greedy) ...")
    ref_logits, ref_tokens = stock_vlm_greedy_steps(
        info["path"], batch, dtype, NUM_COMPARE_STEPS
    )
    ref_text = stock_vlm_generate(info["path"], processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()
    print(f"  stock:   {ref_text!r}")

    # --- Adapter on Spyre ---
    print("  Loading model for Spyre ...")
    model = adapter.load_hf_model(info["path"], dtype)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, dtype)

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

    # --- Per-step top-1 agreement + logit cosine (the assertions) ---
    print("  step      | stock top1          | spyre top1          | match | cosine")
    mismatches = []
    low_cosine = []
    for step, (rl, sl) in enumerate(zip(ref_logits, sp_logits)):
        n = min(rl.shape[-1], sl.shape[-1])
        rl, sl = rl[:n], sl[:n]
        r_top1 = int(rl.argmax())
        s_top1 = int(sl.argmax())
        cosine = F.cosine_similarity(rl, sl, dim=-1).item()
        ok = r_top1 == s_top1
        label = "prefill" if step == 0 else f"decode-{step}"
        print(
            f"  {label:>9} | {r_top1:>6} {tokenizer.decode([r_top1])!r:>11} "
            f"| {s_top1:>6} {tokenizer.decode([s_top1])!r:>11} "
            f"| {'OK' if ok else 'FAIL':>5} | {cosine:.6f}"
        )
        if not ok:
            mismatches.append((label, r_top1, s_top1))
        if cosine < MIN_COSINE:
            low_cosine.append((label, cosine))
    print(f"  adapter free-run: {adapter_text[0]!r}")

    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
    assert (
        not mismatches
    ), "adapter top-1 diverged from stock (teacher-forced) at: " + ", ".join(
        f"{lab}: stock {rt} {tokenizer.decode([rt])!r} vs "
        f"spyre {st} {tokenizer.decode([st])!r}"
        for lab, rt, st in mismatches
    )
    assert not low_cosine, (
        f"adapter logits drifted below cosine {MIN_COSINE} (teacher-forced) at: "
        + ", ".join(f"{lab}: {c:.6f}" for lab, c in low_cosine)
    )
