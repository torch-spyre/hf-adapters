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
    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k qwen3
"""

import math
from typing import Any, Callable

import pytest
import torch
from transformers import PreTrainedModel

from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    encode_prompts,
    generation_cache_len,
    get_model_dtype,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.model_registry import (
    CAUSAL_PATHS,
    NON_BLOCKING_CAUSAL_MODELS,
    xfail_non_blocking,
)

pytestmark = pytest.mark.model_harness("causal")


def hf_greedy_steps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Run stock HF model for prefill + N decode steps on CPU."""
    from transformers import DynamicCache

    results = []
    past = DynamicCache(config=model.config)
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
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Run adapter forward on Spyre for prefill + N decode steps."""
    from hf_adapters.hf_common import (
        allocate_kv_caches,
        build_decode_mask,
        build_prefill_mask,
        make_cache_index,
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

    max_cache_len = generation_cache_len(seq_len, num_decode + 1)
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
            cache_index=make_cache_index(0, padded_len, DEVICE),
        )
    logits_cpu = logits.to("cpu")[0, -1, :].float()[:vocab_size]
    token = logits_cpu.argmax().item()
    results.append({"logits": logits_cpu, "token": token, "step": 0})

    # Single-token decode, mirroring hf_common.generate: each step feeds the token
    # the previous step produced and writes exactly one cache slot, so generated
    # tokens are contiguous from padded_len. (This harness used to reproduce the
    # FMS block walk — BLOCK_SIZE tokens in, one slot written — but generate() no
    # longer does that, and a per-step logits comparison is only meaningful if it
    # exercises the same path production uses.)
    result = torch.cat([padded_ids, torch.tensor([[token]])], dim=1)
    current_cache_len = padded_len

    for step in range(1, num_decode + 1):
        next_input = result[:, -1:].to(DEVICE)
        decode_pos = torch.full(
            (batch_size, 1), current_cache_len - prompt_offset, dtype=torch.long
        )
        decode_mask = build_decode_mask(
            batch_size, max_cache_len, current_cache_len, prompt_offset, dtype=dtype
        )
        with torch.no_grad():
            logits = run_forward_fn(
                model,
                next_input,
                decode_pos.to(DEVICE),
                decode_mask.to(DEVICE),
                key_caches,
                value_caches,
                cache_index=make_cache_index(current_cache_len, 1, DEVICE),
            )
        last_logits = logits.to("cpu")[0, -1, :].float()[:vocab_size]
        current_cache_len += 1

        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        result = torch.cat([result, torch.tensor([[token]])], dim=1)

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

    adapter = resolve_adapter_module_for_test(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(model_path=model_path, adapter_mod=adapter)

    prompt = "The capital of France is"
    # Tokenize following the model's canonical scheme (chat template for
    # instruct models, plain post-processing for base models). The same IDs feed
    # the HF reference and Spyre adapter, keeping the comparison symmetric.
    encoded = encode_prompts(tokenizer, prompt)
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    # Use bf16/fp16 dtype, requested by the registry or based on the model config.
    # (Spyre does not support float32, so float32 entries will use fp16.)
    spyre_dtype = dtype_for_model_path(model_path, target_device="spyre")
    move_model_to_spyre(model=model, module=adapter, dtype=spyre_dtype)
    print("  Running adapter on Spyre ...")
    adapter_results = adapter_greedy_steps(
        adapter._run_forward,
        model,
        input_ids,
        num_decode=num_decode,
    )

    return _compare_results(hf_results, adapter_results, tokenizer, model_path)


def token_compare_spyre(
    model_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _run_model_test(model_path)
    mismatches = [r for r in rows if not r["top1_match"]]
    return mismatches, rows


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(CAUSAL_PATHS, table=NON_BLOCKING_CAUSAL_MODELS)
)
def test_e2e_token_compare_spyre(model_path: str) -> None:
    mismatches, rows = token_compare_spyre(model_path)
    _print_table(rows)
    n_match = sum(1 for r in rows if r["top1_match"])
    print(f"\nTop-1 agreement: {n_match}/{len(rows)} steps")
    assert not mismatches, mismatches
