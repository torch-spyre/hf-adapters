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
CPU accuracy test: compare adapter forward passes against stock HF on CPU.

For each registered causal-LM, two parametrized test cases run:

  test_manual_path[<key>]
    Token-by-token greedy comparison: prefill + 4 decode steps. Loads HF
    once for the reference, then a fresh copy + ``prepare_for_spyre`` for
    the adapter. Asserts the same top-1 token at every step.

  test_auto_loader[<key>]
    End-to-end ``AutoSpyreModelForCausalLM.from_pretrained`` +
    ``model.generate(...)``, compared to ``hf_model.generate(do_sample=False)``.
    Asserts the decoded text matches.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest.
"""

import gc

import pytest
import torch
from transformers import AutoTokenizer

from hf_adapters.auto_spyre_model import resolve_adapter_module
from tests.conftest import load_ref_model, torch_dtype_for_model_path
from tests.model_registry import CAUSAL_PATHS

PROMPT = "The capital of France is"
NUM_DECODE = 4


def hf_greedy_steps(model, input_ids, num_decode=NUM_DECODE):
    """Run stock HF model for prefill + N greedy decode steps with DynamicCache.

    Returns a list of dicts with ``logits``, ``token``, ``step``.
    Step 0 is prefill; steps 1..N are decode.
    """
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

        last_logits = out.logits[0, -1, :].float()
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=NUM_DECODE):
    """Run adapter forward for prefill + N greedy decode steps on CPU."""
    from hf_adapters.hf_common import allocate_kv_caches

    results = []
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # vocab_size lives on the text config for multimodal-wrapped causal LMs
    # (e.g. Gemma 4's composite Gemma4UnifiedConfig); fall back to it.
    cfg = model.config
    vocab_size = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size

    param_dtype = next(model.parameters()).dtype
    max_cache_len = seq_len + num_decode

    # Per-layer KV-cache shapes (honors model._spyre_kv_shapes for
    # heterogeneous architectures like Gemma 4; uniform otherwise).
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, param_dtype, device="cpu"
    )

    # --- Prefill ---
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal_mask = torch.zeros((1, 1, seq_len, max_cache_len), dtype=param_dtype)
    for i in range(seq_len):
        causal_mask[:, :, i, i + 1 :] = -torch.inf

    with torch.no_grad():
        logits = run_forward_fn(
            model,
            input_ids,
            position_ids,
            causal_mask,
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=0,
        )

    last_logits = logits[0, -1, :].float()[:vocab_size]
    token = last_logits.argmax().item()
    results.append({"logits": last_logits, "token": token, "step": 0})

    cache_len = seq_len

    # --- Decode steps ---
    for step in range(1, num_decode + 1):
        next_ids = torch.tensor([[token]])
        next_pos = torch.tensor([[seq_len + step - 1]])
        decode_mask = torch.zeros((1, 1, 1, max_cache_len), dtype=param_dtype)
        decode_mask[:, :, :, cache_len + 1 :] = -torch.inf

        with torch.no_grad():
            logits = run_forward_fn(
                model,
                next_ids,
                next_pos,
                decode_mask,
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=cache_len,
            )

        last_logits = logits[0, -1, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        cache_len += 1

    return results


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_manual_path(model_path, unwrap_compiled_blocks, set_rope_dtype):
    adapter_mod = resolve_adapter_module(model_path)
    torch_dtype = torch_dtype_for_model_path(model_path)

    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

    # Phase 1: HF reference
    model = load_ref_model(model_path, adapter_mod)
    hf_results = hf_greedy_steps(model, input_ids, num_decode=NUM_DECODE)
    del model
    gc.collect()

    # Phase 2: adapter (fresh load — prepare_for_spyre is destructive)
    model = load_ref_model(model_path, adapter_mod=adapter_mod)
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    # Manual path skips load_model_common; propagate the chosen dtype to the
    # RoPE freq cache like the production move does (needed for bf16 models).
    set_rope_dtype(model, torch_dtype)
    unwrap_compiled_blocks(model)
    adapter_results = adapter_greedy_steps(
        adapter_mod._run_forward, model, input_ids, num_decode=NUM_DECODE
    )
    del model
    gc.collect()

    for hf_r, ad_r in zip(hf_results, adapter_results):
        step_label = "prefill" if hf_r["step"] == 0 else f"decode-{hf_r['step']}"
        assert hf_r["token"] == ad_r["token"], (
            f"{step_label}: HF token {hf_r['token']!r} "
            f"({tokenizer.decode([hf_r['token']])!r}) != "
            f"adapter token {ad_r['token']!r} "
            f"({tokenizer.decode([ad_r['token']])!r})"
        )


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_auto_loader(model_path, auto_spyre_model, unwrap_compiled_blocks):
    torch_dtype = torch_dtype_for_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Phase 1: auto-loader generate
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        model_path, dtype=torch_dtype
    )
    unwrap_compiled_blocks(model)
    auto_outputs = model.generate(
        tokenizer, [PROMPT], max_new_tokens=NUM_DECODE, do_sample=False
    )
    del model
    gc.collect()

    # Phase 2: HF reference (fresh)
    adapter_mod = resolve_adapter_module(model_path)
    hf_model = load_ref_model(model_path, adapter_mod)
    hf_model.eval()
    hf_model.requires_grad_(False)
    encoded = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model.generate(
            **encoded, max_new_tokens=NUM_DECODE, do_sample=False
        )
    hf_text = tokenizer.decode(
        hf_out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    del hf_model
    gc.collect()

    assert (
        auto_outputs[0].strip() == hf_text.strip()
    ), f"auto-loader output {auto_outputs[0]!r} != HF reference {hf_text!r}"
