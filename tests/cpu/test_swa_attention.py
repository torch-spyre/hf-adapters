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

"""CPU tests for adapter-side sliding-window mask and ring-cache handling."""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import add_causal_sliding_window_band, build_prefill_mask
from hf_adapters.swa_attention import (
    fit_attention_mask,
    prefill_ring_step,
    sliding_window_attention,
)


def _attention_mask(query, key_cache, window_size, offset):
    """What the Gemma adapters pass: causal + left-pad mask, then a band."""
    batch, _, seqlen_q, _ = query.shape
    capacity = key_cache.size(2)
    mask = build_prefill_mask(batch, seqlen_q, capacity, offset, dtype=query.dtype)
    coords = torch.arange(seqlen_q)[None, :].expand(batch, seqlen_q)
    return add_causal_sliding_window_band(mask, coords, window_size)


def _inputs(batch=1, q_heads=4, kv_heads=2, seqlen_q=128, capacity=256, head_dim=32):
    """Query plus a full-length cache whose rows past ``seqlen_q`` stay zero."""
    torch.manual_seed(0)
    query = torch.randn(batch, q_heads, seqlen_q, head_dim)
    key_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    value_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    key_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    value_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    return query, key_cache, value_cache


def test_attention_mask_is_fitted_to_the_physical_cache():
    mask = torch.zeros(2, 1, 4, 128, dtype=torch.float16)
    expanded = fit_attention_mask(mask, 192)
    cropped = fit_attention_mask(mask, 64)

    assert expanded.shape == (2, 1, 4, 192)
    assert torch.equal(expanded[..., :128], mask)
    assert torch.all(expanded[..., 128:] < 0)
    assert torch.equal(cropped, mask[..., :64])


def test_causal_band_can_be_remapped_to_prefill_ring_order():
    query_coords = torch.arange(8, 12)[None, :]
    base = build_prefill_mask(
        1,
        4,
        16,
        0,
        dtype=torch.float32,
        query_start=8,
    )
    logical = add_causal_sliding_window_band(base, query_coords, 4)
    key_cache_coords = torch.tensor([8, 9, 10, 11, 4, 5, 6, 7])
    physical = add_causal_sliding_window_band(
        base,
        query_coords,
        4,
        key_cache_coords=key_cache_coords,
    )

    torch.testing.assert_close(
        physical,
        logical.index_select(-1, key_cache_coords),
    )


def test_causal_band_masks_unwritten_prefill_ring_rows():
    query_coords = torch.arange(4)[None, :]
    base = build_prefill_mask(1, 4, 16, 0, dtype=torch.float32)
    key_cache_coords = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1])
    physical = add_causal_sliding_window_band(
        base,
        query_coords,
        4,
        key_cache_coords=key_cache_coords,
    )

    assert torch.all(physical[..., 4:] < 0)


def test_prefill_ring_attention_matches_logical_cache_order():
    torch.manual_seed(11)
    batch, query_heads, kv_heads, head_dim = 1, 4, 2, 32
    block_start, query_length, logical_capacity = 8, 4, 16
    ring_capacity, window = 8, 4
    query = torch.randn(batch, query_heads, query_length, head_dim)
    logical_key = torch.randn(batch, kv_heads, logical_capacity, head_dim)
    logical_value = torch.randn(batch, kv_heads, logical_capacity, head_dim)
    step = prefill_ring_step(block_start, query_length, ring_capacity)

    ring_key = torch.zeros(batch, kv_heads, ring_capacity, head_dim)
    ring_value = torch.zeros_like(ring_key)
    valid = step.key_cache_coords >= 0
    ring_key[..., valid, :] = logical_key[..., step.key_cache_coords[valid], :]
    ring_value[..., valid, :] = logical_value[..., step.key_cache_coords[valid], :]

    query_coords = torch.arange(block_start, block_start + query_length)[None, :]
    base_mask = build_prefill_mask(
        batch,
        query_length,
        logical_capacity,
        0,
        dtype=query.dtype,
        query_start=block_start,
    )
    logical_mask = add_causal_sliding_window_band(base_mask, query_coords, window)
    ring_mask = add_causal_sliding_window_band(
        base_mask,
        query_coords,
        window,
        key_cache_coords=step.key_cache_coords,
    )
    expected = F.scaled_dot_product_attention(
        query,
        logical_key,
        logical_value,
        attn_mask=logical_mask,
        enable_gqa=True,
    )
    actual = sliding_window_attention(
        query,
        ring_key,
        ring_value,
        ring_mask,
        window_size=window,
        is_causal=True,
        scale=None,
    )

    torch.testing.assert_close(actual, expected)


def test_explicit_scale_is_honored():
    """Gemma 4 attends unscaled (scaling == 1.0), so scale must reach SDPA."""
    query, key_cache, value_cache = _inputs()
    mask = _attention_mask(query, key_cache, 64, 0)
    unscaled = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        mask,
        window_size=64,
        is_causal=True,
        scale=1.0,
    )
    default = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        mask,
        window_size=64,
        is_causal=True,
        scale=None,
    )
    assert not torch.allclose(unscaled, default)
