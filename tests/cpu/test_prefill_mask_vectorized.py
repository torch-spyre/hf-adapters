# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""Causal and padding boundaries for vectorized and advancing prefill masks."""

import pytest
import torch

from hf_adapters.hf_common import (
    _ChunkedPrefillMaskBuilder,
    _mask_fill_value,
    build_prefill_mask,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "length,capacity,start,offsets",
    [(4, 16, 0, [0, 3]), (4, 16, 8, [1, 2]), (1984, 2048, 0, [0])],
)
def test_prefill_mask_matches_causal_and_padding_boundaries(
    dtype, length, capacity, start, offsets
):
    mask = build_prefill_mask(
        len(offsets),
        length,
        capacity,
        torch.tensor(offsets),
        dtype=dtype,
        query_start=start,
    )
    expected = torch.zeros_like(mask)
    fill = _mask_fill_value(dtype)
    for batch, offset in enumerate(offsets):
        expected[batch, :, :, :offset] = fill
    for row in range(length):
        expected[:, :, row, start + row + 1 :] = fill
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)


def test_multiple_chunks_preserve_padding_and_advance_causal_frontier():
    offsets = torch.tensor([0, 3])
    builder = _ChunkedPrefillMaskBuilder(2, 4, 16, offsets, device="cpu")
    for start in (0, 4, 8, 12):
        actual = builder.build(start)
        expected = build_prefill_mask(2, 4, 16, offsets, query_start=start)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
