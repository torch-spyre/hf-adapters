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

"""E2E token-level comparison: HF stock forward (CPU) vs adapter forward (Spyre).

For each model, runs prefill + 4 greedy decode steps on both CPU (stock HF)
and Spyre (adapter), then asserts:
  - No NaN in either side's logits.
  - Logit max-abs-diff at prefill stays under a per-model threshold.
  - Top-1 greedy token match across the 5 steps meets a per-model min ratio.

The CPU reference forward MUST run BEFORE prepare_for_spyre() — the RMSNorm
patch is global and would taint the reference if the order were swapped.

Usage (on Spyre pod):
    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k qwen3
"""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F
from _helpers import torch_dtype_for
from model_registry import CAUSAL_LM_MODELS

DEVICE = "spyre"

# Per-model assertion thresholds. Default fallback applies for any key not
# listed: max_diff < 1.0, top1 match >= 3/5 of the 5 prefill+decode steps.
DEFAULT_THRESHOLDS = {"max_diff": 1.0, "top1_min": 3 / 5}
THRESHOLDS = {
    "qwen3": {"max_diff": 0.5, "top1_min": 4 / 5},
    "granite": {"max_diff": 0.5, "top1_min": 4 / 5},
    "granite2b": {"max_diff": 0.5, "top1_min": 4 / 5},
    "granite4": {"max_diff": 0.8, "top1_min": 3 / 5},
    "smollm3": {"max_diff": 0.5, "top1_min": 4 / 5},
    "tiny_llama": {"max_diff": 0.5, "top1_min": 4 / 5},
    "qwen2": {"max_diff": 0.5, "top1_min": 4 / 5},
    "ministral": {"max_diff": 0.5, "top1_min": 4 / 5},
    "olmo": {"max_diff": 0.5, "top1_min": 4 / 5},
    "olmo2": {"max_diff": 0.5, "top1_min": 4 / 5},
    "falcon3": {"max_diff": 0.5, "top1_min": 4 / 5},
    "deepseek-coder": {"max_diff": 0.5, "top1_min": 4 / 5},
}


def _hf_greedy_steps(model, input_ids, num_decode=4):
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


def _adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=4):
    """Run adapter forward on Spyre for prefill + N decode steps."""
    from hf_adapters.hf_common import (
        BLOCK_SIZE,
        _model_dtype,
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
    dtype = _model_dtype(model)
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, dtype, device=DEVICE
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


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_token_compare(model_key):
    """CPU stock HF vs Spyre adapter: logits and greedy tokens across 5 steps."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf_adapters.hf_common import (
        _move_to_spyre_with_layout,
        _untie_embedding_and_lm_head,
    )

    info = CAUSAL_LM_MODELS[model_key]
    thresholds = THRESHOLDS.get(model_key, DEFAULT_THRESHOLDS)
    print(f"\n  {info['name']}: {info['path']}")
    print(f"  Thresholds: {thresholds}")

    adapter_module_name = info["adapter"].replace(".py", "")
    adapter = importlib.import_module(f"hf_adapters.{adapter_module_name}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    dtype = torch_dtype_for(info)
    model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=dtype, device_map="cpu"
    )
    model.eval()
    model.requires_grad_(False)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # CPU reference MUST run before prepare_for_spyre — RMSNorm patching is global.
    print("  Running HF reference on CPU ...")
    hf_results = _hf_greedy_steps(model, input_ids, num_decode=4)

    print("  Preparing adapter and moving to Spyre ...")
    _untie_embedding_and_lm_head(model)
    adapter.prepare_for_spyre(model)
    _move_to_spyre_with_layout(model, dtype)

    print("  Running adapter on Spyre ...")
    adapter_results = _adapter_greedy_steps(
        adapter._run_forward, model, input_ids, num_decode=4
    )

    # --- Compare ---
    matches = 0
    prefill_max_diff = None
    failures = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h = hf_r["logits"]
        a = ad_r["logits"]
        min_vocab = min(h.shape[0], a.shape[0])
        diff = (h[:min_vocab] - a[:min_vocab]).abs()
        max_diff = diff.max().item()
        match = h[:min_vocab].argmax().item() == a[:min_vocab].argmax().item()
        if match:
            matches += 1
        if step == 0:
            prefill_max_diff = max_diff

        hf_nan = bool(h.isnan().any().item())
        sp_nan = bool(a.isnan().any().item())
        step_label = "prefill" if step == 0 else f"decode-{step}"
        print(
            f"  step={step_label:9s} max_diff={max_diff:.4f} "
            f"hf={hf_r['token']!r}({hf_r['token']}) "
            f"sp={ad_r['token']!r}({ad_r['token']}) "
            f"match={'OK' if match else 'FAIL'}"
        )
        if hf_nan:
            failures.append(f"step {step_label}: HF logits contain NaN")
        if sp_nan:
            failures.append(f"step {step_label}: Spyre logits contain NaN")

    total = len(hf_results)
    top1_ratio = matches / total

    if failures:
        pytest.fail(f"{model_key}: " + "; ".join(failures))

    assert prefill_max_diff is not None and prefill_max_diff < thresholds["max_diff"], (
        f"{model_key}: prefill logit max-abs-diff {prefill_max_diff:.4f} "
        f">= threshold {thresholds['max_diff']}"
    )
    assert top1_ratio >= thresholds["top1_min"], (
        f"{model_key}: top-1 token match {matches}/{total} "
        f"({top1_ratio:.2f}) < threshold {thresholds['top1_min']:.2f}"
    )
