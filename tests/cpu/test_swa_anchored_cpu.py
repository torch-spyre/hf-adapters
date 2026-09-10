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

"""CPU integration test for the anchored compact buffer, across a shift.

Drives 70 decode steps through one Gemma 4 sliding layer twice: once against a
full-length cache behind a band mask (what the adapter does today), once against a
192-row anchored compact buffer that rolls at token 64. The outputs must agree at
every step, which is what says the compaction, the stick offsets, the shift and the
runtime-mask updates and valid-start erosion all line up.

W=128 rather than Gemma 4's 1024 so the test crosses a shift in 70 steps; the
arithmetic is identical.
"""

import copy
import types

import torch
from _swa_helpers import identity_freqs, make_sliding_attention

import hf_adapters.hf_gemma4 as hf_gemma4
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    add_causal_sliding_window_band,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)
from hf_adapters.hf_gemma4 import _run_blocks_over_embeds
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    allocate_swa_caches,
    anchored_step,
    compact_after_prefill,
    roll_compact_buffer,
)

WINDOW = 128
PROMPT = 256
STEPS = 70  # crosses the shift at token 64
FULL_CAPACITY = 384  # >= PROMPT + STEPS, stick-aligned
HEAD_DIM = 64
Q_HEADS = 4
KV_HEADS = 2


def _band_mask(seqlen, block_base):
    """What _build_layer_masks builds for a sliding layer today.

    Dispatches the way ``generate`` does: ``build_decode_mask`` for a one-token
    step at a non-zero cache position, ``build_prefill_mask`` otherwise.
    ``build_prefill_mask`` masks every column past the query's *relative* row
    index, so at a non-zero ``block_base`` it allows only column 0 — and the band
    is additive, so it cannot widen that back. Using it for decode masks every
    column and fails the comparison for a reason unrelated to the buffer.
    """
    if seqlen == 1 and block_base > 0:
        mask = build_decode_mask(1, FULL_CAPACITY, block_base, 0, dtype=torch.float32)
    else:
        mask = build_prefill_mask(1, seqlen, FULL_CAPACITY, 0, dtype=torch.float32)
    coords = torch.arange(seqlen)[None, :] + block_base
    return add_causal_sliding_window_band(mask, coords, WINDOW)


def test_anchored_decode_matches_the_full_cache_band_path():
    torch.manual_seed(11)
    band = make_sliding_attention(Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"

    band_k = torch.zeros(1, KV_HEADS, FULL_CAPACITY, HEAD_DIM)
    band_v = torch.zeros(1, KV_HEADS, FULL_CAPACITY, HEAD_DIM)
    # Prefill buffer for a sliding layer: max(sliding_capacity(128), PROMPT).
    op_k = torch.zeros(1, KV_HEADS, PROMPT, HEAD_DIM)
    op_v = torch.zeros(1, KV_HEADS, PROMPT, HEAD_DIM)

    hidden = torch.randn(1, PROMPT, Q_HEADS * HEAD_DIM)
    freqs = identity_freqs(1, PROMPT, HEAD_DIM)
    index = make_cache_index(0, PROMPT)

    band_out, band_k, band_v = band(
        hidden, freqs, _band_mask(PROMPT, 0), band_k, band_v, index
    )
    op_out, op_k, op_v = op(
        hidden,
        freqs,
        _band_mask(PROMPT, 0)[..., :PROMPT],
        op_k,
        op_v,
        index,
    )
    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)

    state = SlidingWindowCache.after_prefill(WINDOW, PROMPT, [0])
    op_k, op_v = compact_after_prefill(op_k, op_v, state, PROMPT)
    assert op_k.shape[2] == 192

    shifts = 0
    for step_index in range(STEPS):
        slot = PROMPT + step_index
        token = torch.randn(1, 1, Q_HEADS * HEAD_DIM)
        token_freqs = identity_freqs(1, 1, HEAD_DIM)

        expected, band_k, band_v = band(
            token,
            token_freqs,
            _band_mask(1, slot),
            band_k,
            band_v,
            make_cache_index(slot, 1),
        )

        step = anchored_step(state, "cpu", torch.float32)
        shifts += int(step.do_shift)
        if step.do_shift:
            op_k, op_v = roll_compact_buffer(op_k, op_v)
        actual, op_k, op_v = op(
            token,
            token_freqs,
            step.attention_mask,
            op_k,
            op_v,
            step.cache_index,
        )
        state.advance()

        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5, msg=f"step {step_index}"
        )

    assert shifts == 1, "70 steps must cross exactly one 64-row shift"
    assert op_k.shape[2] == 192, "the compact buffer must never grow"


def test_anchored_decode_keeps_one_fixed_tensor_signature():
    """All positions change mask contents without changing graph metadata."""
    window, prompt = 1024, 4096
    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    capacity = state.capacity
    visible_ranges = set()
    for _ in range(200):
        step = anchored_step(state, "cpu", torch.float16)
        assert isinstance(
            step.cache_index, torch.Tensor
        ), "write position must be a tensor"
        assert step.attention_mask.shape == (1, 1, 1, capacity)
        assert step.attention_mask.dtype == torch.float16
        visible = torch.where(step.attention_mask[0, 0, 0] == 0)[0]
        visible_ranges.add((visible[0].item(), visible[-1].item()))
        state.advance()
    assert len(visible_ranges) == BLOCK_SIZE
    assert {end for _, end in visible_ranges} == set(range(state.anchor, capacity))


def test_anchored_shift_at_the_shipped_geometry():
    """The 1088-row, 1024-anchor roll Gemma 4 actually runs, crossed once.

    W=1024 means anchor 1024, so 64 writes fill rows [1024, 1088) and the 65th
    step triggers the roll. Small head_dim and head counts keep this quick; what
    is under test is the bookkeeping at the real capacity, not the arithmetic
    intensity.
    """
    window, prompt, steps = 1024, 1024, 65
    capacity, full_capacity = 1088, 1152
    head_dim, q_heads, kv_heads = 32, 2, 1

    def band_mask(seqlen, block_base):
        if seqlen == 1 and block_base > 0:
            mask = build_decode_mask(
                1, full_capacity, block_base, 0, dtype=torch.float32
            )
        else:
            mask = build_prefill_mask(1, seqlen, full_capacity, 0, dtype=torch.float32)
        coords = torch.arange(seqlen)[None, :] + block_base
        return add_causal_sliding_window_band(mask, coords, window)

    torch.manual_seed(21)
    band = make_sliding_attention(q_heads, kv_heads, head_dim, window, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"

    band_k = torch.zeros(1, kv_heads, full_capacity, head_dim)
    band_v = torch.zeros(1, kv_heads, full_capacity, head_dim)
    op_k = torch.zeros(1, kv_heads, prompt, head_dim)
    op_v = torch.zeros(1, kv_heads, prompt, head_dim)

    hidden = torch.randn(1, prompt, q_heads * head_dim)
    freqs = identity_freqs(1, prompt, head_dim)
    index = make_cache_index(0, prompt)
    _, band_k, band_v = band(hidden, freqs, band_mask(prompt, 0), band_k, band_v, index)
    _, op_k, op_v = op(
        hidden,
        freqs,
        band_mask(prompt, 0)[..., :prompt],
        op_k,
        op_v,
        index,
    )

    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    assert state.capacity == capacity and state.anchor == 1024
    assert state.valid_start == [0], "a prompt of exactly anchor rows leaves no gap"
    op_k, op_v = compact_after_prefill(op_k, op_v, state, prompt)
    assert op_k.shape[2] == capacity

    shifts = 0
    for step_index in range(steps):
        slot = prompt + step_index
        token = torch.randn(1, 1, q_heads * head_dim)
        token_freqs = identity_freqs(1, 1, head_dim)
        expected, band_k, band_v = band(
            token,
            token_freqs,
            band_mask(1, slot),
            band_k,
            band_v,
            make_cache_index(slot, 1),
        )
        step = anchored_step(state, "cpu", torch.float32)
        shifts += int(step.do_shift)
        if step.do_shift:
            op_k, op_v = roll_compact_buffer(op_k, op_v)
        actual, op_k, op_v = op(
            token,
            token_freqs,
            step.attention_mask,
            op_k,
            op_v,
            step.cache_index,
        )
        state.advance()
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5, msg=f"step {step_index}"
        )

    assert shifts == 1, f"65 steps at anchor 1024 must roll exactly once, got {shifts}"
    assert op_k.shape[2] == capacity, "the compact buffer must never grow"


def test_chunked_prefill_compacts_only_on_first_decode_and_reuses_shared_kv(
    monkeypatch,
):
    """Exercise the driver boundary that receives prefill views, then owner caches."""
    window = 64
    prompt_len = 256
    hidden_size = 8
    layer_types = ["sliding_attention", "sliding_attention"]
    producer_of = [None, 0]
    config = types.SimpleNamespace(
        layer_types=layer_types,
        sliding_window=window,
        enable_moe_block=False,
    )
    backbone = types.SimpleNamespace(
        layers=[
            types.SimpleNamespace(layer_scalar=torch.tensor(1.0)),
            types.SimpleNamespace(layer_scalar=torch.tensor(1.0)),
        ],
        norm=types.SimpleNamespace(weight=None, with_scale=False, eps=1e-6),
    )

    class ProducerBlock:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            hidden,
            freqs,
            mask,
            key_cache,
            value_cache,
            cache_index,
            layer_scalar,
            per_layer_input,
            query_row_mask,
        ):
            self.calls.append(
                {
                    "capacity": key_cache.shape[2],
                    "cache_index": cache_index.clone(),
                    "attention_mask": mask.clone(),
                    "before": key_cache.clone(),
                }
            )
            marks = (cache_index + 1).to(key_cache.dtype).view(1, 1, -1, 1)
            rows = marks.expand(1, key_cache.shape[1], -1, key_cache.shape[3])
            # Return fresh objects, as a compiled mutation may, to prove the
            # driver's consumer aliases are rebound after every call.
            key_cache = key_cache.index_copy(2, cache_index, rows)
            value_cache = value_cache.index_copy(2, cache_index, rows)
            return hidden, key_cache, value_cache

    class SharedBlock:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            hidden,
            freqs,
            mask,
            key_cache,
            value_cache,
            layer_scalar,
            per_layer_input,
            query_row_mask,
        ):
            self.calls.append((key_cache, value_cache, mask.clone()))
            return hidden

    producer = ProducerBlock()
    consumer = SharedBlock()
    model = types.SimpleNamespace(
        config=config,
        model=backbone,
        _spyre_rope={"sliding_attention": lambda hidden, positions: positions},
        _spyre_compiled_blocks=[producer, consumer],
        _spyre_producer_of=producer_of,
        _spyre_swa_mode="anchored",
        _spyre_padded_prompt_len=prompt_len,
        _spyre_prompt_offsets=torch.tensor([0]),
        _spyre_kv_shapes=[(1, hidden_size, hidden_size)] * 2,
    )
    monkeypatch.setattr(
        hf_gemma4,
        "_compiled_gemma4_rms_norm",
        lambda hidden, weight, eps: hidden,
    )
    keys, values = allocate_swa_caches(model, 1, prompt_len + 64, torch.float32, "cpu")
    assert keys[1] is keys[0]
    assert keys[0].shape[2] == prompt_len

    for chunk_start in (0, 128):
        hf_gemma4._run_blocks_over_embeds(
            model,
            torch.zeros(1, 128, hidden_size),
            torch.arange(chunk_start, chunk_start + 128).view(1, -1),
            torch.zeros(1, 1, 128, prompt_len),
            keys,
            values,
            make_cache_index(chunk_start, 128),
        )
        assert model._spyre_swa_state is None
        assert keys[0].shape[2] == prompt_len
        assert keys[1] is keys[0] and values[1] is values[0]

    hf_gemma4._run_blocks_over_embeds(
        model,
        torch.zeros(1, 1, hidden_size),
        torch.tensor([[prompt_len]]),
        torch.zeros(1, 1, 1, prompt_len + 64),
        keys,
        values,
        make_cache_index(prompt_len, 1),
    )

    decode_call = producer.calls[-1]
    assert decode_call["capacity"] == 128
    assert decode_call["cache_index"].tolist() == [64]
    assert decode_call["attention_mask"].shape == (1, 1, 1, 128)
    assert torch.all(decode_call["attention_mask"][..., :1] < 0)
    assert torch.all(decode_call["attention_mask"][..., 1:65] == 0)
    assert torch.all(decode_call["attention_mask"][..., 65:] < 0)
    assert decode_call["before"][0, 0, :64, 0].tolist() == list(
        range(prompt_len - 64 + 1, prompt_len + 1)
    )
    shared_key, shared_value, shared_mask = consumer.calls[-1]
    assert shared_key is keys[0] and shared_value is values[0]
    assert keys[1] is keys[0] and values[1] is values[0]
    assert torch.equal(shared_mask, decode_call["attention_mask"])
    assert model._spyre_swa_state.write_row == 65


def test_caller_supplied_masks_use_the_sliding_op_path(monkeypatch):
    """The VLM's mixed mask reaches an anchored sliding block unchanged."""

    class RecordingBlock:
        def __init__(self):
            self.mask = None

        def __call__(
            self,
            hidden,
            freqs,
            mask,
            key_cache,
            value_cache,
            cache_index,
            layer_scalar,
            per_layer_input,
            query_row_mask,
        ):
            self.mask = mask
            return hidden, key_cache, value_cache

    block = RecordingBlock()
    config = types.SimpleNamespace(
        layer_types=["sliding_attention"],
        sliding_window=64,
        enable_moe_block=False,
    )
    backbone = types.SimpleNamespace(
        layers=[types.SimpleNamespace(layer_scalar=torch.tensor(1.0))],
        norm=types.SimpleNamespace(weight=None, with_scale=False, eps=1e-6),
    )
    model = types.SimpleNamespace(
        config=config,
        model=backbone,
        _spyre_rope={"sliding_attention": lambda hidden, positions: positions},
        _spyre_compiled_blocks=[block],
        _spyre_producer_of=[None],
        _spyre_swa_mode="anchored",
        _spyre_swa_is_causal=False,
    )
    monkeypatch.setattr(
        hf_gemma4,
        "_compiled_gemma4_rms_norm",
        lambda hidden, weight, eps: hidden,
    )

    hidden = torch.zeros(1, 64, 8)
    sliding_mask = torch.zeros(1, 1, 64, 64)
    sliding_mask[..., 0, 63] = 7
    caches = [torch.zeros(1, 1, 64, 8)]
    _run_blocks_over_embeds(
        model,
        hidden,
        torch.arange(64).view(1, -1),
        None,
        caches.copy(),
        caches.copy(),
        make_cache_index(0, 64),
        masks={
            "full_attention": torch.full_like(sliding_mask, -1),
            "sliding_attention": sliding_mask,
        },
    )

    assert block.mask is sliding_mask


def test_moe_sliding_layer_uses_the_compact_cache_index(monkeypatch):
    """MoE attention must write to the anchored row, not the global position."""

    class RecordingMoEBlock:
        def __init__(self):
            self.cache_index = None

        def __call__(
            self,
            hidden,
            freqs,
            mask,
            key_cache,
            value_cache,
            cache_index,
            layer_scalar,
        ):
            self.cache_index = cache_index
            return hidden, key_cache, value_cache

    window, prompt_len = 64, 256
    state = SlidingWindowCache.after_prefill(window, prompt_len, [0])
    block = RecordingMoEBlock()
    config = types.SimpleNamespace(
        layer_types=["sliding_attention"],
        sliding_window=window,
        enable_moe_block=True,
    )
    backbone = types.SimpleNamespace(
        layers=[types.SimpleNamespace(layer_scalar=torch.tensor(1.0))],
        norm=types.SimpleNamespace(weight=None, with_scale=False, eps=1e-6),
    )
    model = types.SimpleNamespace(
        config=config,
        model=backbone,
        _spyre_rope={"sliding_attention": lambda hidden, positions: positions},
        _spyre_compiled_blocks=[block],
        _spyre_producer_of=[None],
        _spyre_swa_mode="anchored",
        _spyre_swa_state=state,
        _spyre_padded_prompt_len=prompt_len,
        _spyre_prompt_offsets=torch.tensor([0]),
    )
    monkeypatch.setattr(
        hf_gemma4,
        "_compiled_gemma4_rms_norm",
        lambda hidden, weight, eps: hidden,
    )

    capacity = state.capacity
    keys = [torch.zeros(1, 1, capacity, 8)]
    values = [torch.zeros(1, 1, capacity, 8)]
    _run_blocks_over_embeds(
        model,
        torch.zeros(1, 1, 8),
        torch.tensor([[prompt_len]]),
        torch.zeros(1, 1, 1, prompt_len + 64),
        keys,
        values,
        make_cache_index(prompt_len, 1),
    )

    assert block.cache_index.tolist() == [state.anchor]
