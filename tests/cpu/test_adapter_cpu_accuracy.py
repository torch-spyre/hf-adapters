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
import sys

import pytest
import torch
from transformers import AutoTokenizer

from hf_adapters.hf_common import encode_prompts
from tests.conftest import (
    encode_generation_inputs,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import CAUSAL_PATHS

pytestmark = pytest.mark.model_harness("causal")

PROMPT = "The capital of France is"
NUM_DECODE = 4


def hf_greedy_steps(model, input_ids, num_decode=NUM_DECODE):
    """Run stock HF model for prefill + N greedy decode steps with DynamicCache.

    Returns a list of dicts with ``logits``, ``token``, ``step``.
    Step 0 is prefill; steps 1..N are decode.
    """
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

        last_logits = out.logits[0, -1, :].float()
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=NUM_DECODE):
    """Run adapter forward for prefill + N greedy decode steps on CPU."""
    from hf_adapters.hf_common import allocate_kv_caches, make_cache_index

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
            cache_index=make_cache_index(0, seq_len),
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
                cache_index=make_cache_index(cache_len, 1),
            )

        last_logits = logits[0, -1, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        cache_len += 1

    return results


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
def test_auto_loader(model_path):
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Phase 1: auto-loader generate
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(model_path)
    _unwrap_compiled_blocks(model)
    encoded = encode_generation_inputs(tokenizer, [PROMPT])
    auto_sequences = model.generate(
        **encoded,
        max_new_tokens=NUM_DECODE,
        do_sample=False,
    )
    auto_outputs = tokenizer.batch_decode(
        auto_sequences[:, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    del model
    gc.collect()

    # Phase 2: HF reference (fresh)
    adapter_mod = resolve_adapter_module_for_test(model_path)
    hf_model = load_ref_model(model_path, adapter_mod)
    encoded = encode_prompts(tokenizer, PROMPT)
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


@pytest.fixture
def gemma_moe_compiler(monkeypatch):
    """Only compiler hints/config are stubbed; import the actual adapter."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from hf_adapters import hf_gemma4_moe as moe

    config = SimpleNamespace(
        sencores=32,
        ignore_work_division_hints=False,
        ignore_wsr_hints=False,
        indexed_selection_consumer_layout=False,
        layout_solver="greedy",
        co_optimizing_lx_planning=False,
        ktir_emitter=False,
        lx_planning=True,
    )
    monkeypatch.setitem(
        sys.modules, "torch_spyre._inductor", SimpleNamespace(config=config)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_spyre._inductor.propagate_hints",
        SimpleNamespace(spyre_hint=lambda **kw: nullcontext()),
    )
    return moe, config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "gate_blocks,down_blocks",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_gemma_decode_block_composition(
    gemma_moe_compiler, monkeypatch, dtype, gate_blocks, down_blocks
):
    """Prove dispatch engages both changes, including the 768-column down tail."""
    from torch.utils._python_dispatch import TorchDispatchMode

    moe, _ = gemma_moe_compiler
    monkeypatch.setattr(moe, "_DECODE_GATE_UP_K_PANEL", 704 if gate_blocks else None)
    monkeypatch.setattr(moe, "_DECODE_DOWN_OUTPUT_PANEL", 1024 if down_blocks else None)

    def make(*shape):
        return torch.empty(shape, device="meta", dtype=dtype)

    x, gate, up, down = (
        make(1, 2816),
        make(128, 2816, 704),
        make(128, 2816, 704),
        make(128, 704, 2816),
    )
    monkeypatch.setattr(moe, "_router_probs", lambda *args: make(1, 8))
    monkeypatch.setattr(
        moe,
        "_topk",
        lambda *args: (make(1, 8), torch.empty(1, 8, device="meta", dtype=torch.int64)),
    )
    selected, products = [], []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.index.Tensor and args[0].ndim == 3:
                selected.append(tuple(args[0].shape))
            if func == torch.ops.aten.bmm.default:
                products.append(tuple(args[1].shape))
            return func(*args, **(kwargs or {}))

    with Record():
        result = moe._compiled_moe_loop_region(
            x, x, None, None, None, make(128, 64), gate, up, down, 8, 32, 64, 1e-6
        )
    expected_gate = [(8, 704, 704)] * 8 if gate_blocks else [(8, 2816, 704)] * 2
    expected_down = (
        [(8, 704, w) for w in (1024, 1024, 768)] if down_blocks else [(8, 704, 2816)]
    )
    assert products == expected_gate + expected_down
    assert sorted(selected) == sorted((128, *shape[1:]) for shape in products)
    assert result.shape == (1, 2816)


def test_gemma_schedule_fallbacks(gemma_moe_compiler, monkeypatch):
    moe, config = gemma_moe_compiler
    assert moe._decode_route_schedule_enabled(1, 8)
    assert not moe._decode_route_schedule_enabled(2, 8)
    assert not moe._decode_route_schedule_enabled(1, 7)
    del config.indexed_selection_consumer_layout
    assert not moe._decode_route_schedule_enabled(1, 8)
    for chooser, width in (
        (moe._decode_down_panel, 1024),
        (moe._decode_gate_up_panel, 704),
    ):
        assert chooser(2816, 704, (torch.bfloat16,) * 4, True) == width
        for hidden, dtypes, eligible in (
            (2816, (torch.bfloat16,) * 4, False),
            (1408, (torch.bfloat16,) * 4, True),
            (2816, (torch.float16,) * 3 + (torch.bfloat16,), True),
            (2816, (torch.float32,) * 4, True),
        ):
            assert chooser(hidden, 704, dtypes, eligible) is None

    # Missing compiler support declines automatically but an explicit request
    # must fail before attention can mutate the cache.
    assert moe._prefill_expert_config() == {"allow_all_ops_in_lx_planning": True}
    monkeypatch.setattr(moe, "_PREFILL_EXPERT_DIVISIONS", True)
    with pytest.raises(RuntimeError, match="reader-compatible"):
        moe.Gemma4MoEBlock.forward(
            None,
            torch.empty(1, 512, 2816, device="meta"),
            None,
            None,
            None,
            None,
            None,
            None,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_prefill_schedule_selection(gemma_moe_compiler, monkeypatch, dtype):
    moe, config = gemma_moe_compiler
    config.consumer_compatible_input_staging = False
    config.read_copy_elision = False
    config.lx_planner_relayout = False
    assert moe._prefill_expert_config()["consumer_compatible_input_staging"]
    tensors = [
        torch.empty(shape, dtype=dtype, device="meta")
        for shape in ((512, 2816), (128, 2816, 704), (128, 2816, 704), (128, 704, 2816))
    ]
    assert moe._validate_prefill_expert_inputs(*tensors)
    tensors[-1] = tensors[-1].to(torch.float32)
    assert not moe._validate_prefill_expert_inputs(*tensors)
    monkeypatch.setattr(moe, "_PREFILL_EXPERT_DIVISIONS", True)
    with pytest.raises(ValueError, match="matching"):
        moe._validate_prefill_expert_inputs(*tensors)


def test_gemma_block_addresses_and_order(gemma_moe_compiler):
    """Small exact integers isolate movement/order from device rounding."""
    moe, _ = gemma_moe_compiler
    ids = torch.tensor([[0, 2, 2, 0]])
    x = torch.arange(20).reshape(4, 1, 5).double() % 3
    gate = torch.arange(3 * 5 * 7).reshape(3, 5, 7).double() % 5
    up, down = gate + 1, gate.transpose(1, 2).contiguous()
    g, u = moe._decode_gate_up_blocks(x, gate, up, ids, 2)
    assert torch.equal(g, torch.bmm(x, gate[ids].reshape(4, 5, 7)))
    assert torch.equal(u, torch.bmm(x, up[ids].reshape(4, 5, 7)))
    actual = moe._decode_down_output_blocks(g, down, ids, 2)
    assert torch.equal(actual, torch.bmm(g, down[ids].reshape(4, 7, 5)))
