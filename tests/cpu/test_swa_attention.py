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

"""CPU tests for the sliding-window attention dispatcher.

``spyre::sliding_window_attention`` exists only on the spyre device, so this lane
exercises the CPU reference branch — and pins it to the band-masked SDPA the Gemma
adapters compute today, which is the equivalence the whole replacement rests on.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import add_causal_sliding_window_band, build_prefill_mask
from hf_adapters.swa_attention import (
    fit_attention_mask,
    physical_cache_capacity,
    sliding_capacity,
    sliding_window_attention,
)


def _attention_mask(query, key_cache, window_size, offset):
    """What the Gemma adapters pass: causal + left-pad mask, then a band."""
    batch, _, seqlen_q, _ = query.shape
    capacity = key_cache.size(2)
    mask = build_prefill_mask(batch, seqlen_q, capacity, offset, dtype=query.dtype)
    coords = torch.arange(seqlen_q)[None, :].expand(batch, seqlen_q)
    return add_causal_sliding_window_band(mask, coords, window_size)


def _band_masked_attention(query, key_cache, value_cache, window_size, offset):
    mask = _attention_mask(query, key_cache, window_size, offset)
    return F.scaled_dot_product_attention(
        query, key_cache, value_cache, attn_mask=mask, enable_gqa=True
    )


def _inputs(batch=1, q_heads=4, kv_heads=2, seqlen_q=128, capacity=256, head_dim=32):
    """Query plus a full-length cache whose rows past ``seqlen_q`` stay zero."""
    torch.manual_seed(0)
    query = torch.randn(batch, q_heads, seqlen_q, head_dim)
    key_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    value_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    key_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    value_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    return query, key_cache, value_cache


def test_sliding_capacity_is_window_plus_one_stick():
    # Gemma 4 and Gemma 3, the two models this lands for.
    assert sliding_capacity(1024) == 1088
    assert sliding_capacity(512) == 576


def test_sliding_capacity_rounds_up_to_a_stick():
    # A window that is not a whole number of sticks still gets a stick-aligned
    # allocation: rejection_reason refuses a capacity that is not.
    assert sliding_capacity(100) == 192
    assert sliding_capacity(1) == 64  # one row plus 63 rows of query stagger
    assert sliding_capacity(100) % 64 == 0


def test_attention_mask_is_fitted_to_the_physical_cache():
    mask = torch.zeros(2, 1, 4, 128, dtype=torch.float16)
    expanded = fit_attention_mask(mask, 192)
    cropped = fit_attention_mask(mask, 64)

    assert expanded.shape == (2, 1, 4, 192)
    assert torch.equal(expanded[..., :128], mask)
    assert torch.all(expanded[..., 128:] < 0)
    assert torch.equal(cropped, mask[..., :64])


def test_physical_cache_capacity_uses_logical_width_on_cpu():
    cache = torch.zeros(1, 2, 576, 64)
    assert physical_cache_capacity(cache[:, :, :512, :]) == 512
    assert physical_cache_capacity(cache) == 576


def test_reference_matches_the_band_masked_path():
    """The equivalence the replacement rests on, at phase-1 geometry."""
    query, key_cache, value_cache = _inputs()
    expected = _band_masked_attention(query, key_cache, value_cache, 64, 0)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        _attention_mask(query, key_cache, 64, 0),
        window_size=64,
        is_causal=True,
        scale=None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_reference_honors_valid_start_like_left_padding():
    """valid_start must reproduce build_prefill_mask's left-pad columns.

    Compared only over rows ``>= 17``. A row below the threshold has its ENTIRE
    window excluded — row ``i``'s window is ``(i - 64, i]``, and every one of those
    columns is below 17 — so it is a fully-masked row, and the two paths legitimately
    disagree there: this reference fills uniformly with ``-inf`` and so spreads weight
    over every column, while the band path's mask mixes ``_mask_fill_value`` (on the
    pad columns) with ``-inf`` (out of band) and so spreads weight over the pad
    columns only. Neither answer means anything — those query rows ARE padding, and
    their outputs are discarded — so the assertion covers the rows whose attention is
    defined, plus finiteness everywhere, which is the property that actually matters
    (a non-finite value would travel into the next layer's KV cache).
    """
    query, key_cache, value_cache = _inputs()
    expected = _band_masked_attention(query, key_cache, value_cache, 64, 17)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        _attention_mask(query, key_cache, 64, 17),
        window_size=64,
        is_causal=True,
        scale=None,
    )
    torch.testing.assert_close(
        actual[:, :, 17:], expected[:, :, 17:], rtol=1e-5, atol=1e-6
    )
    assert torch.isfinite(actual).all(), "fully-masked rows must stay finite"


def test_reference_honors_per_sequence_valid_start():
    """A ragged batch: each entry gets its own threshold.

    Per-entry row slicing for the same reason as the test above: entry 1's threshold
    of 40 makes its rows ``[0, 40)`` fully masked, while entry 0's threshold of 0
    makes none of its rows fully masked.
    """
    query, key_cache, value_cache = _inputs(batch=2)
    offsets = (0, 40)
    masks = [
        _attention_mask(query[b : b + 1], key_cache[b : b + 1], 64, offset)
        for b, offset in enumerate(offsets)
    ]
    per_entry = [
        F.scaled_dot_product_attention(
            query[b : b + 1],
            key_cache[b : b + 1],
            value_cache[b : b + 1],
            attn_mask=masks[b],
            enable_gqa=True,
        )
        for b in range(len(offsets))
    ]
    expected = torch.cat(per_entry, dim=0)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        torch.cat(masks),
        window_size=64,
        is_causal=True,
        scale=None,
    )
    for b, offset in enumerate(offsets):
        torch.testing.assert_close(
            actual[b, :, offset:], expected[b, :, offset:], rtol=1e-5, atol=1e-6
        )
    assert torch.isfinite(actual).all(), "fully-masked rows must stay finite"


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
