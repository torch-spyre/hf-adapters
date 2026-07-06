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
E2E token-level comparison: HF stock forward (CPU) vs adapter forward (Spyre).

For each model, runs prefill + 4 greedy decode steps on both CPU (stock HF)
and Spyre (adapter), comparing logits and greedy tokens at each step.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py
    pytest -s -vvv "tests/spyre/test_e2e_token_compare_spyre.py::test_e2e_token_compare_spyre[Qwen/Qwen3-0.6B]"
"""

import math
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_registry import CAUSAL_PATHS

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    get_model_dtype,
    move_to_spyre_with_layout,
    untie_embedding_and_lm_head,
)
from tests.conftest import load_ref_model, torch_dtype_for_model_path


def hf_greedy_steps(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Run stock HF model for prefill + N decode steps on CPU."""
    from transformers import DynamicCache

    results = []
    past = DynamicCache()
    ids = input_ids.clone()
    seq_len = ids.shape[1]

    for step in range(num_decode + 1):
        if step == 0:
            position_ids = torch.arange(seq_len).unsqueeze(0)
        else:
            position_ids = torch.tensor([[seq_len + step - 1]])

        with torch.no_grad():
            out = model(
                input_ids=ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )

        logits = out.logits[0, -1, :].float()
        token = logits.argmax().item()
        results.append({"logits": logits, "token": token, "step": step})
        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


def adapter_greedy_steps(
    run_forward_fn: Callable,
    model: nn.Module,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Run adapter forward on Spyre for prefill + N decode steps."""
    from hf_adapters.hf_common import (
        allocate_kv_caches,
        build_expansion_mask,
        build_prefill_mask,
    )

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    cfg = model.config
    vocab_size = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size

    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(seq_len)

    max_cache_len = (
        padded_len + math.ceil(num_decode / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE
    )

    dtype = get_model_dtype(model)

    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, dtype
    )

    results = []

    prefill_mask = build_prefill_mask(
        batch_size, padded_len, max_cache_len, prompt_offset, dtype=dtype
    )

    with torch.no_grad():
        logits = run_forward_fn(
            model,
            padded_ids.to(DEVICE),
            position_ids.to(DEVICE),
            prefill_mask.to(DEVICE),
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=0,
        )
    logits_cpu = logits.to("cpu")[0, -1, :].float()[:vocab_size]
    token = logits_cpu.argmax().item()
    results.append({"logits": logits_cpu, "token": token, "step": 0})

    result = padded_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
    for j in range(BLOCK_SIZE):
        decode_pos[:, j] = seq_len + j - BLOCK_SIZE
    fill_mask_device = None

    if tokens_in_block == BLOCK_SIZE - 1:
        result = F.pad(result, (0, BLOCK_SIZE))
    tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
    grab_idx = BLOCK_SIZE if tokens_in_block == 0 else BLOCK_SIZE - tokens_in_block
    result[:, -grab_idx] = token

    for step in range(1, num_decode + 1):
        is_filling = tokens_in_block > 0
        next_input = result[:, -BLOCK_SIZE:].to(DEVICE)

        if is_filling:
            fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
            with torch.no_grad():
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),
                    fill_mask_device,
                    key_caches,
                    value_caches,
                    is_filling=True,
                    token_index=tokens_in_block,
                    cache_position=fill_pos,
                )
            logits_cpu = logits.to("cpu")
            grab_logit = BLOCK_SIZE - tokens_in_block
            last_logits = logits_cpu[0, -grab_logit, :].float()[:vocab_size]
        else:
            current_cache_len += BLOCK_SIZE
            decode_pos = decode_pos + BLOCK_SIZE
            exp_mask = build_expansion_mask(
                batch_size,
                BLOCK_SIZE,
                max_cache_len,
                current_cache_len,
                prompt_offset,
                dtype=dtype,
            )
            with torch.no_grad():
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),
                    exp_mask.to(DEVICE),
                    key_caches,
                    value_caches,
                    is_filling=False,
                    token_index=0,
                    cache_position=current_cache_len - BLOCK_SIZE,
                )
            logits_cpu = logits.to("cpu")
            last_logits = logits_cpu[0, -BLOCK_SIZE, :].float()[:vocab_size]
            fill_mask_device = exp_mask.to(DEVICE)

        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        if tokens_in_block == BLOCK_SIZE - 1:
            result = F.pad(result, (0, BLOCK_SIZE))
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        grab_idx = BLOCK_SIZE if tokens_in_block == 0 else BLOCK_SIZE - tokens_in_block
        result[:, -grab_idx] = token

    return results


def _compare_results(
    hf_results: list[dict[str, Any]],
    adapter_results: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
) -> list[dict[str, Any]]:
    """Compare HF vs adapter results, return comparison rows."""
    rows = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h_logits = hf_r["logits"]
        a_logits = ad_r["logits"]

        min_vocab = min(h_logits.shape[0], a_logits.shape[0])
        h = h_logits[:min_vocab]
        a = a_logits[:min_vocab]

        diff = (h - a).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        h_top1 = h.argmax().item()
        a_top1 = a.argmax().item()
        match = h_top1 == a_top1

        step_label = "prefill" if step == 0 else f"decode-{step}"
        h_str = tokenizer.decode([hf_r["token"]])
        a_str = tokenizer.decode([ad_r["token"]])
        rows.append(
            {
                "model": model_name,
                "step": step_label,
                "hf_token": hf_r["token"],
                "hf_str": h_str,
                "spyre_token": ad_r["token"],
                "spyre_str": a_str,
                "top1_match": match,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "hf_nan": h_logits.isnan().any().item(),
                "spyre_nan": a_logits.isnan().any().item(),
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per step."""
    print("\n## E2E Token Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(
        "| Model | Step | HF Token | Spyre Token | Match "
        "| Max Diff | Mean Diff | HF NaN | Spyre NaN |"
    )
    print(
        "|-------|------|----------|-------------|-------"
        "|----------|-----------|--------|-----------|"
    )
    for r in rows:
        match = "OK" if r["top1_match"] else "FAIL"
        hf_col = f"{r['hf_token']:>5} {r['hf_str']!r}"
        sp_col = f"{r['spyre_token']:>5} {r['spyre_str']!r}"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {r['model']} | {r['step']} | {hf_col} | {sp_col} "
            f"| {match} | {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {hn} | {sn} |"
        )


def _run_model_test(model_path: str, num_decode: int = 4) -> list[dict[str, Any]]:
    """Full comparison for one model. Returns the list of comparison rows."""
    from transformers import AutoTokenizer

    from hf_adapters.auto_spyre_model import resolve_adapter_module

    adapter = resolve_adapter_module(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(model_path=model_path, adapter_mod=adapter)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    print("  Preparing adapter ...")
    untie_embedding_and_lm_head(model)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    # Use bfloat16 on Spyre when the registry requests it; otherwise float16.
    # (Spyre does not support float32, so float32 registry entries still use float16.)
    spyre_dtype = torch_dtype_for_model_path(model_path)
    move_to_spyre_with_layout(model, spyre_dtype)
    print("  Running adapter on Spyre ...")
    adapter_results = adapter_greedy_steps(
        adapter._run_forward,
        model,
        input_ids,
        num_decode=num_decode,
    )

    return _compare_results(hf_results, adapter_results, tokenizer, model_path)


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_e2e_token_compare_spyre(model_path: str) -> None:
    rows = _run_model_test(model_path)
    _print_table(rows)
    n_match = sum(1 for r in rows if r["top1_match"])
    print(f"\nTop-1 agreement: {n_match}/{len(rows)} steps")
    mismatches = [r for r in rows if not r["top1_match"]]
    assert not mismatches, mismatches
