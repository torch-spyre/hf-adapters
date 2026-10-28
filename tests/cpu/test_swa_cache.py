# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""CPU tests for the anchored compact KV buffer's bookkeeping.

The invariant under test, at the start of every 64-token stick period:

  * physical rows ``[0, anchor)`` hold the most recent ``anchor`` tokens, all real
  * rows ``[anchor, capacity)`` are empty and take the next 64 writes

The single-row decode cursor traverses the trailing stick while its position is
carried by a fixed-shape runtime mask, so all 64 rows reuse one decode graph.
``buffer_origin`` stays zero and ``anchor`` is ``capacity - 64``.
"""

import types

import pytest
import torch

from hf_adapters.hf_common import allocate_kv_caches
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    allocate_swa_caches,
    compact_after_prefill,
    compact_sliding_buffers,
    prefill_ring_step,
    roll_sliding_buffers,
    sliding_capacity,
    sliding_prefill_capacity,
)

WINDOW = 1024
CAPACITY = 1088  # sliding_capacity(1024)
ANCHOR = 1024


def test_after_prefill_anchors_the_write_row():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    assert state.capacity == CAPACITY
    assert state.anchor == ANCHOR
    assert state.write_row == ANCHOR
    assert state.stick_offset() == 0
    # A prompt longer than the buffer fills rows [0, anchor) with real tokens.
    assert state.valid_start == [0]


def test_after_prefill_short_prompt_leaves_a_masked_front():
    # 100 prompt tokens cannot fill 1024 rows: the unwritten front is zeros, and
    # only valid_start can keep them out of the window.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0])
    assert state.valid_start == [ANCHOR - 100]
    assert state.write_row == ANCHOR


def test_after_prefill_counts_left_padding_that_survives_compaction():
    # 100 padded columns of which 17 are pad: 83 real tokens land last, so the
    # threshold is the unwritten front plus the pad that came with them.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[17])
    assert state.valid_start == [ANCHOR - 100 + 17]


def test_after_prefill_drops_left_padding_that_falls_off_the_front():
    # A prompt longer than the buffer: the pad columns are the oldest tokens and
    # compaction discards them, so nothing needs masking.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[17])
    assert state.valid_start == [0]


def test_after_prefill_is_per_sequence():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0, 17])
    assert state.valid_start == [ANCHOR - 100, ANCHOR - 100 + 17]


def test_the_write_stick_holds_64_tokens_then_asks_for_a_shift():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    rows = []
    for _ in range(64):
        assert not state.needs_shift()
        rows.append(state.write_row)
        state.advance()
    assert rows == list(range(ANCHOR, CAPACITY))
    assert state.needs_shift()


def test_shift_returns_to_the_anchor_and_erodes_valid_start():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0])
    assert state.valid_start == [ANCHOR - 100]
    for _ in range(64):
        state.advance()
    state.shift()
    assert state.write_row == ANCHOR
    assert state.stick_offset() == 0
    # 64 of the unwritten front rows fell off, so the threshold drops by 64.
    assert state.valid_start == [ANCHOR - 100 - 64]


def test_valid_start_never_goes_negative():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=1020, offsets=[0])
    assert state.valid_start == [4]
    state.shift()
    assert state.valid_start == [0]


def test_stick_offset_tracks_the_position_within_the_stick():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    for expected in range(64):
        assert state.stick_offset() == expected
        state.advance()


def test_compaction_keeps_the_newest_rows_right_aligned_at_the_anchor():
    """The prefill/decode boundary: the tail of a big buffer into a compact one."""
    prompt_len = 2048
    big_k = torch.zeros(1, 2, prompt_len, 8)
    big_v = torch.zeros(1, 2, prompt_len, 8)
    # Row r carries the value r, so provenance is checkable.
    marks = torch.arange(prompt_len, dtype=torch.float32).view(1, 1, prompt_len, 1)
    big_k += marks
    big_v += marks + 0.5

    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, offsets=[0])
    compact_k, compact_v = compact_after_prefill(big_k, big_v, state, prompt_len)

    assert compact_k.shape == (1, 2, CAPACITY, 8)
    # Rows [0, anchor) hold prompt rows [prompt_len - anchor, prompt_len).
    assert compact_k[0, 0, 0, 0].item() == prompt_len - ANCHOR
    assert compact_k[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1
    assert compact_v[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1 + 0.5
    # The write stick starts empty -- the op reads it, so it must be zero, not junk.
    assert not compact_k[:, :, ANCHOR:, :].any()
    assert not compact_v[:, :, ANCHOR:, :].any()


def test_compaction_right_aligns_a_short_prompt():
    prompt_len = 100
    big_k = torch.zeros(1, 2, prompt_len, 8)
    big_v = torch.zeros(1, 2, prompt_len, 8)
    marks = torch.arange(prompt_len, dtype=torch.float32).view(1, 1, prompt_len, 1)
    big_k += marks
    big_v += marks

    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, offsets=[0])
    compact_k, _ = compact_after_prefill(big_k, big_v, state, prompt_len)

    # Real rows end at the anchor; everything before valid_start stays zero.
    assert compact_k[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1
    assert compact_k[0, 0, ANCHOR - prompt_len, 0].item() == 0.0
    assert not compact_k[:, :, : state.valid_start[0], :].any()


def test_capacity_matches_sliding_capacity():
    """One source of truth for the allocation size."""
    for window in (512, 1024, 100):
        state = SlidingWindowCache.after_prefill(window, 4096, offsets=[0])
        assert state.capacity == sliding_capacity(window)
        assert state.anchor == state.capacity - 64


def test_capacity_uses_the_exact_staggered_width_before_rounding():
    # W=65 plus a 64-row query block spans exactly 128 columns, not 129.
    assert sliding_capacity(65) == 128


def test_prefill_capacity_is_chunk_aligned_and_covers_the_window():
    assert sliding_prefill_capacity(1024, 512) == 1536
    assert sliding_prefill_capacity(512, 512) == 1024
    assert sliding_prefill_capacity(100, 64) == 192


def test_prefill_ring_step_maps_physical_rows_to_logical_tokens():
    first = prefill_ring_step(0, 512, 1536)
    assert torch.equal(first.cache_index, torch.arange(512))
    assert torch.equal(first.key_cache_coords[:512], torch.arange(512))
    assert torch.all(first.key_cache_coords[512:] < 0)

    wrapped = prefill_ring_step(1536, 512, 1536)
    assert torch.equal(wrapped.cache_index, torch.arange(512))
    assert torch.equal(wrapped.key_cache_coords[:512], torch.arange(1536, 2048))
    assert torch.equal(wrapped.key_cache_coords[512:], torch.arange(512, 1536))


def test_prefill_ring_step_rejects_unaligned_chunks():
    with pytest.raises(ValueError, match="block_start must be aligned"):
        prefill_ring_step(64, 512, 1536)


def test_swa_allocator_uses_prompt_capacity_only_for_sliding_layers():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=WINDOW,
        ),
        _spyre_kv_shapes=[(2, 8, 8), (4, 16, 16)],
        _spyre_padded_prompt_len=2048,
    )
    keys, values = allocate_swa_caches(model, 1, 4096, torch.float32, "cpu")
    assert [cache.shape for cache in keys] == [(1, 2, 2048, 8), (1, 4, 4096, 16)]
    assert [cache.shape for cache in values] == [(1, 2, 2048, 8), (1, 4, 4096, 16)]
    assert all(not cache.any() for cache in [*keys, *values])


def test_swa_allocator_bounds_chunked_prefill_capacity():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=WINDOW,
        ),
        _spyre_kv_shapes=[(2, 8, 8), (4, 16, 16)],
        _spyre_padded_prompt_len=8192,
        _spyre_active_prefill_chunk_size=512,
    )
    keys, values = allocate_swa_caches(model, 1, 8704, torch.float32, "cpu")
    assert [cache.shape for cache in keys] == [(1, 2, 1536, 8), (1, 4, 8704, 16)]
    assert [cache.shape for cache in values] == [
        (1, 2, 1536, 8),
        (1, 4, 8704, 16),
    ]


def test_compaction_reads_the_logical_tail_from_a_wrapped_prefill_ring():
    prompt_len = 8192
    prefill_capacity = sliding_prefill_capacity(WINDOW, 512)
    ring_k = torch.zeros(1, 2, prefill_capacity, 8)
    ring_v = torch.zeros_like(ring_k)
    for block_start in range(0, prompt_len, 512):
        step = prefill_ring_step(block_start, 512, prefill_capacity)
        marks = torch.arange(block_start, block_start + 512, dtype=torch.float32).view(
            1, 1, 512, 1
        )
        ring_k.index_copy_(2, step.cache_index, marks.expand(1, 2, 512, 8))
        ring_v.index_copy_(2, step.cache_index, (marks + 0.5).expand(1, 2, 512, 8))

    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, offsets=[0])
    compact_k, compact_v = compact_after_prefill(ring_k, ring_v, state, prompt_len)
    expected = torch.arange(prompt_len - ANCHOR, prompt_len, dtype=torch.float32)
    assert torch.equal(compact_k[0, 0, :ANCHOR, 0], expected)
    assert torch.equal(compact_v[0, 0, :ANCHOR, 0], expected + 0.5)


def test_common_allocator_hook_dispatches_to_swa_allocator():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=WINDOW,
        ),
        _spyre_kv_shapes=[(2, 8, 8), (2, 8, 8)],
        _spyre_padded_prompt_len=512,
        _spyre_cache_allocator=allocate_swa_caches,
    )
    keys, _ = allocate_kv_caches(model, 1, 4096, torch.float32, device="cpu")
    assert [cache.shape[2] for cache in keys] == [CAPACITY, 4096]


def test_swa_allocator_aliases_kv_shared_layers_to_their_producers():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=WINDOW,
        ),
        _spyre_kv_shapes=[(2, 8, 8)] * 4,
        _spyre_producer_of=[None, None, 0, 1],
        _spyre_padded_prompt_len=2048,
    )
    keys, values = allocate_swa_caches(model, 1, 4096, torch.float32, "cpu")

    assert keys[2] is keys[0] and values[2] is values[0]
    assert keys[3] is keys[1] and values[3] is values[1]
    assert keys[0].shape[2] == 2048
    assert keys[1].shape[2] == 4096


def test_compaction_and_roll_rebind_kv_shared_layers():
    layer_types = ["sliding_attention", "sliding_attention"]
    producer_of = [None, 0]
    prompt_len = 2048
    key = torch.zeros(1, 2, prompt_len, 8)
    value = torch.zeros_like(key)
    marks = torch.arange(prompt_len, dtype=torch.float32).view(1, 1, -1, 1)
    key += marks
    value += marks
    keys = [key, key]
    values = [value, value]
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, [0])

    compact_sliding_buffers(
        layer_types,
        keys,
        values,
        state,
        prompt_len,
        producer_of=producer_of,
    )
    assert keys[1] is keys[0] and values[1] is values[0]
    assert keys[0].shape[2] == CAPACITY

    old_key = keys[0]
    roll_sliding_buffers(layer_types, keys, values, producer_of=producer_of)
    assert keys[0] is not old_key
    assert keys[1] is keys[0] and values[1] is values[0]
