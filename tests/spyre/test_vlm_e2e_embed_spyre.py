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
End-to-end VLM last_hidden_state ("multimodal embedding") accuracy test.

Sister of ``test_vlm_e2e_spyre.py``: same teacher-forced prefill + N-decode-step
scaffold, but compares the text decoder's post-final-norm hidden state (what
``model.lm_head`` reads) instead of the post-``lm_head`` logits. Sister of
``test_e2e_embed_compare_spyre.py``: same cosine-based per-token / pooled
comparison and markdown reporting, but for the fused image+text decoder
representation rather than an encoder-only text embedder.

  test_vlm_embed_spyre[<model_path>]
    Builds image+prompt with the processor, captures per-step last-layer hidden
    state + greedy token ids from stock HF (CPU), then drives the adapter on
    Spyre teacher-forced on stock's token sequence and compares the adapter's
    last_hidden_state against stock's at each step:

      - step 0 (prefill): per-token cosine over real (attention-mask=1) tokens,
        plus pooled cosine — mirrors ``test_e2e_embed_compare_spyre``.
      - step k>0 (decode): single-vector cosine at the just-decoded position.

Why last_hidden_state, not logits? Hidden state is one step upstream of the LM
head (``model.lm_head(h) / config.text_config.logits_scaling`` — see
``hf_granite_vision_mm._logits_from_embeds``), so it isolates decoder
correctness (KV cache, mask, RoPE, deepstack injection) from the LM head. In
practice cosine here lands ≥ 0.9999; we hold the assertion at 0.999 to match
the logit test's ``MIN_COSINE`` and absorb benign fp16 substrate rounding.

Padding-side note: ``build_vlm_batch`` sets ``padding_side='left'`` and this
test runs batch size 1, so tokenizer padding is a no-op on the tensor content.
Stock's ``.generate(**batch, ...)`` and the adapter's teacher-forced loop
consume the same ``input_ids``/``attention_mask`` — the two forwards are
directly comparable.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_vlm_e2e_embed_spyre.py
    pytest -s -vvv tests/spyre/test_vlm_e2e_embed_spyre.py -k granite_vision
"""

import gc
import math
import types
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _vision_helpers import build_vlm_batch, stock_vlm_greedy_hidden_steps
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

NUM_COMPARE_STEPS = 5
# Same floor as the logit test (``test_vlm_e2e_spyre.MIN_COSINE``). Hidden state
# is un-amplified by ``lm_head``, so measured agreement is tighter (≥0.9999);
# holding the assert at 0.999 keeps a robust floor across substrates.
MIN_COSINE = 0.999
PROMPT = "Briefly describe this image."


def _adapter_teacher_forced_hidden_steps(
    adapter: types.ModuleType,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    forced_tokens: list[int],
) -> tuple[list[torch.Tensor], int, int]:
    """Per-step adapter last_hidden_state on Spyre, teacher-forced.

    Replicates the adapter ``generate`` block-decode bookkeeping (KV caches,
    per-step ``decode_pos`` / fill vs expansion masks, right-aligned ``result``
    block-walk), but feeds ``forced_tokens[i]`` back at each decode step
    instead of the adapter's own argmax. Step 0 is the deepstack prefill via
    ``adapter._prefill_hidden`` (mirror of ``_prefill_forward`` that returns
    hidden state instead of logits); decode steps use
    ``adapter._hidden_from_embeds`` (mirror of ``_logits_from_embeds``).

    Returns
      - ``hidden_steps``: list of ``len(forced_tokens)`` fp32 CPU tensors.
          step 0  : ``[padded_len, H]`` — the whole left-padded prefill
                    hidden state for the single batch row; caller slices
                    ``[prompt_offset : prompt_offset + actual_len, :]`` to
                    align with stock's ``[prompt_len, H]`` reference.
          step k>0: ``[H]``            — hidden state at the just-decoded token
                    position (same ``grab_idx`` / ``-BLOCK_SIZE`` slot the
                    logit test uses for the corresponding logit vector).
      - ``padded_len``, ``prompt_offset``: alignment metadata for step 0.
    """
    input_ids: torch.Tensor = batch["input_ids"]
    attention_mask: torch.Tensor = batch["attention_mask"]
    pixel_values: torch.Tensor = batch["pixel_values"]
    image_sizes: torch.Tensor = batch["image_sizes"]

    model_d_type: torch.dtype = get_model_dtype(model)
    backbone = adapter.get_backbone(model)
    emb_mult: float = backbone.embedding_multiplier

    batch_size, prompt_length = input_ids.shape
    actual_prompt_lengths: torch.Tensor = attention_mask.sum(dim=1)
    n_steps: int = len(forced_tokens)

    max_cache_len: int = (
        math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
        + math.ceil((n_steps + 1) / BLOCK_SIZE) * BLOCK_SIZE
    )
    padded_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
        input_ids, actual_prompt_lengths
    )
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, model_d_type
    )

    result: torch.Tensor = padded_ids.clone()
    current_cache_len: int = padded_len
    tokens_in_block: int = BLOCK_SIZE - 1
    fill_mask_device: torch.Tensor | None = None
    hidden_steps: list[torch.Tensor] = []

    def embed_ids(ids: torch.Tensor) -> torch.Tensor:
        return backbone.embed_tokens(ids) * emb_mult

    def _write_token(tok_id: int) -> None:
        nonlocal result, tokens_in_block
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        if tokens_in_block == 0:
            result = torch.nn.functional.pad(result, (0, BLOCK_SIZE))
        grab: int = (
            (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        )
        result[:, -grab] = tok_id

    # --- Step 0: deepstack prefill (hidden state, pre-lm_head) ---
    prefill_hidden: torch.Tensor = adapter._prefill_hidden(
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
    # [padded_len, H] fp32 on CPU for the single batch row; caller does the
    # left-pad slice + attention-mask filtering.
    hidden_steps.append(prefill_hidden.to("cpu")[0].float())
    decode_pos: torch.Tensor = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
    for j in range(BLOCK_SIZE):
        decode_pos[0, j] = actual_prompt_lengths[0].item() + j - BLOCK_SIZE
    _write_token(forced_tokens[0])

    # --- Decode steps teacher-forced on stock's tokens ---
    for i in range(1, n_steps):
        is_filling: bool = tokens_in_block > 0
        next_input: torch.Tensor = result[:, -BLOCK_SIZE:].to(DEVICE)
        next_embeds: torch.Tensor = embed_ids(next_input)
        if is_filling:
            fill_pos: int = current_cache_len - BLOCK_SIZE + tokens_in_block
            hidden: torch.Tensor = adapter._hidden_from_embeds(
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
            grab_idx: int = BLOCK_SIZE - tokens_in_block
            step_hidden: torch.Tensor = hidden.to("cpu")[0, -grab_idx, :].float()
        else:
            current_cache_len += BLOCK_SIZE
            decode_pos = decode_pos + BLOCK_SIZE
            exp_mask: torch.Tensor = build_expansion_mask(
                batch_size,
                BLOCK_SIZE,
                max_cache_len,
                current_cache_len,
                prompt_offsets,
                dtype=model_d_type,
            )
            hidden = adapter._hidden_from_embeds(
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
            step_hidden = hidden.to("cpu")[0, -BLOCK_SIZE, :].float()
            fill_mask_device = exp_mask.to(DEVICE)
        hidden_steps.append(step_hidden)
        _write_token(forced_tokens[i])

    prompt_offset: int = int(prompt_offsets[0].item())
    return hidden_steps, padded_len, prompt_offset


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool ``[L, H]`` over real (mask=1) tokens → ``[H]``."""
    m: torch.Tensor = mask.unsqueeze(-1).float()
    summed: torch.Tensor = (hidden.float() * m).sum(dim=0)
    counts: torch.Tensor = m.sum(dim=0).clamp(min=1)
    return summed / counts


def _compare_prefill(
    ref: torch.Tensor,
    sp: torch.Tensor,
    mask: torch.Tensor,
    step_label: str,
    input_ids: torch.Tensor | None = None,
    image_token_id: int | None = None,
    top_k_outliers: int = 8,
) -> dict[str, Any]:
    """Per-token cosine + pooled cosine over real tokens (step 0 / prefill).

    When ``input_ids`` and ``image_token_id`` are provided, also prints the
    worst-``top_k_outliers`` per-token cosines with their absolute position
    in the sequence and whether they land on an ``<image>`` slot — the shape
    of the divergence (image-token vs boundary vs uniform) tells us which
    subsystem drifted (deepstack injection vs alignment vs decoder).
    """
    assert (
        ref.shape == sp.shape
    ), f"prefill shape mismatch: ref {tuple(ref.shape)} vs adapter {tuple(sp.shape)}"
    r32: torch.Tensor = ref.float()
    s32: torch.Tensor = sp.float()
    per_tok_cos: torch.Tensor = F.cosine_similarity(r32, s32, dim=-1)  # [L]
    abs_diff: torch.Tensor = (r32 - s32).abs()

    pooled_r: torch.Tensor = _mean_pool(ref, mask)
    pooled_s: torch.Tensor = _mean_pool(sp, mask)
    pooled_cos: float = F.cosine_similarity(
        pooled_r.unsqueeze(0), pooled_s.unsqueeze(0), dim=-1
    ).item()

    m_bool: torch.Tensor = mask.bool()
    cos_real: torch.Tensor = per_tok_cos[m_bool]
    diff_real: torch.Tensor = abs_diff[m_bool]
    min_cos: float = cos_real.min().item()

    if input_ids is not None and image_token_id is not None:
        # Per-token diagnostic: worst-K real positions. Report absolute position
        # in the sequence (0-indexed), token id, whether it's an <image> slot,
        # and the per-token cosine + max abs diff at that row.
        real_positions: torch.Tensor = m_bool.nonzero(as_tuple=True)[0]
        cos_worst_idx: torch.Tensor = torch.argsort(cos_real)[:top_k_outliers]
        n_image_tokens: int = int((input_ids == image_token_id).sum().item())
        print(
            f"  [diag] real tokens: {int(m_bool.sum().item())}, "
            f"of which <image> slots: {n_image_tokens}"
        )
        print(f"  [diag] worst {min(top_k_outliers, len(cos_real))} per-token cosines:")
        print("  [diag] rank | seq_pos | token_id | is_image | cosine   | max_diff")
        for rank, k in enumerate(cos_worst_idx.tolist()):
            seq_pos: int = int(real_positions[k].item())
            tok_id: int = int(input_ids[seq_pos].item())
            is_img: bool = tok_id == image_token_id
            row_max_diff: float = abs_diff[seq_pos].max().item()
            print(
                f"  [diag] {rank:>4} | {seq_pos:>7} | {tok_id:>8} "
                f"| {str(is_img):>8} | {cos_real[k].item():.6f} | {row_max_diff:.4f}"
            )

    return {
        "step": step_label,
        "n_real": int(m_bool.sum().item()),
        "mean_cos": cos_real.mean().item(),
        "min_cos": min_cos,
        "pooled_cos": pooled_cos,
        "max_diff": diff_real.max().item(),
        "mean_diff": diff_real.mean().item(),
        "hf_nan": bool(ref.isnan().any().item()),
        "spyre_nan": bool(sp.isnan().any().item()),
        "match": min_cos >= MIN_COSINE,
    }


def _compare_decode(
    ref: torch.Tensor,
    sp: torch.Tensor,
    step_label: str,
) -> dict[str, Any]:
    """Single-vector cosine at one decoded token position (step k>0)."""
    assert ref.shape == sp.shape, (
        f"{step_label} shape mismatch: ref {tuple(ref.shape)} vs adapter "
        f"{tuple(sp.shape)}"
    )
    r32: torch.Tensor = ref.float()
    s32: torch.Tensor = sp.float()
    cos: float = F.cosine_similarity(r32.unsqueeze(0), s32.unsqueeze(0), dim=-1).item()
    abs_diff: torch.Tensor = (r32 - s32).abs()
    return {
        "step": step_label,
        "n_real": 1,
        "mean_cos": cos,
        "min_cos": cos,
        "pooled_cos": cos,
        "max_diff": abs_diff.max().item(),
        "mean_diff": abs_diff.mean().item(),
        "hf_nan": bool(ref.isnan().any().item()),
        "spyre_nan": bool(sp.isnan().any().item()),
        "match": cos >= MIN_COSINE,
    }


def _print_table(model_path: str, rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per step."""
    print("\n## E2E VLM Hidden-State Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(
        "| Model | Step | Real Len | Mean Cos | Min Cos | Pooled Cos "
        "| Max Diff | Mean Diff | HF NaN | Spyre NaN | Match |"
    )
    print(
        "|-------|------|----------|----------|---------|------------"
        "|----------|-----------|--------|-----------|-------|"
    )
    for r in rows:
        match = "OK" if r["match"] else "FAIL"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {model_path} | {r['step']} | {r['n_real']} "
            f"| {r['mean_cos']:.6f} | {r['min_cos']:.6f} | {r['pooled_cos']:.6f} "
            f"| {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {hn} | {sn} | {match} |"
        )


@pytest.mark.parametrize("model_path", VISION_PATHS, ids=VISION_PATHS)
def test_vlm_embed_spyre(model_path: str) -> None:
    adapter = resolve_adapter_module(
        model_path, mapping=IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )
    dtype: torch.dtype = torch_dtype_for_model_path(model_path)

    processor, batch = build_vlm_batch(model_path, PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    # --- Stock CPU reference (before prepare_for_spyre patches RMSNorm).
    # Capture per-step last-layer hidden state + greedy token ids + the prefill
    # attention mask (real vs tokenizer-pad positions). ---
    print("  Running stock CPU reference (per-step greedy hidden state) ...")
    ref_hidden_steps, ref_tokens, prefill_amask = stock_vlm_greedy_hidden_steps(
        model_path, batch, dtype, NUM_COMPARE_STEPS
    )
    gc.collect()

    # --- Adapter on Spyre ---
    print("  Loading model for Spyre ...")
    model: nn.Module = load_hf_vlm(model_path, dtype, adapter_mod=adapter)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    move_to_spyre_with_layout(model, dtype)

    print(f"  Running adapter prefill + {len(ref_tokens) - 1} forced decode steps ...")
    with torch.no_grad():
        sp_hidden_steps, padded_len, prompt_offset = (
            _adapter_teacher_forced_hidden_steps(adapter, model, batch, ref_tokens)
        )
    del model
    gc.collect()

    # --- Compare per step ---
    # Prefill: align adapter's left-padded `[padded_len, H]` to stock's
    # `[prompt_len, H]`. `prompt_offsets` is `padded_len - actual_length` and
    # real tokens live at `[prompt_offset : prompt_offset + actual_length]`.
    actual_prompt_len: int = int(batch["attention_mask"].sum(dim=1)[0].item())
    ref_prefill: torch.Tensor = ref_hidden_steps[0]
    sp_prefill_real: torch.Tensor = sp_hidden_steps[0][
        prompt_offset : prompt_offset + actual_prompt_len, :
    ]
    # ref_prefill is [prompt_len, H] from stock. When the tokenizer produced no
    # pad (bs=1, single row), prompt_len == actual_prompt_len; when it did,
    # `prefill_amask` marks the real positions and we still compare over the
    # full prompt_len rows.
    if ref_prefill.shape[0] != sp_prefill_real.shape[0]:
        # Batch had tokenizer pads: ref covers the full prompt_len; adapter's
        # aligned slice only covers real tokens. Clip ref down to the real
        # positions using the mask so shapes match.
        ref_prefill = ref_prefill[prefill_amask]
        prefill_input_ids: torch.Tensor = batch["input_ids"][0][prefill_amask]
        prefill_mask_use = torch.ones(ref_prefill.shape[0], dtype=torch.bool)
    else:
        prefill_input_ids = batch["input_ids"][0]
        prefill_mask_use = prefill_amask

    # For the diagnostic: read image_token_id off the checkpoint config (static).
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    image_token_id: int = int(cfg.image_token_id)

    rows: list[dict[str, Any]] = []
    rows.append(
        _compare_prefill(
            ref_prefill,
            sp_prefill_real,
            prefill_mask_use,
            "prefill",
            input_ids=prefill_input_ids,
            image_token_id=image_token_id,
        )
    )
    for k in range(1, len(ref_hidden_steps)):
        rows.append(
            _compare_decode(ref_hidden_steps[k], sp_hidden_steps[k], f"decode-{k}")
        )

    _print_table(model_path, rows)
    n_match: int = sum(1 for r in rows if r["match"])
    print(f"\nPer-step min-cosine >= {MIN_COSINE}: {n_match}/{len(rows)} steps")
    mismatches: list[dict[str, Any]] = [r for r in rows if not r["match"]]
    assert not mismatches, mismatches
