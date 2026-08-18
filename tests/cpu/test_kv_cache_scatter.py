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

"""CPU tests for the indirect-scatter KV cache write.

The cache keeps its ``[B, n_kv, max_cache_len, head_dim]`` logical shape; only
the Spyre *device* layout is pinned (cache-position dim at device position 0) so
the scatter addresses correctly. These tests cover the shape/index contract and
equivalence with the previous slice-assignment write, without needing a device.

Device-specific concerns (the layout pin itself, one-binary-per-position) are
covered by the Spyre suites.
"""

import pytest
import torch

from hf_adapters.hf_common import allocate_kv_caches, kv_cache_update


class _FakeConfig:
    def __init__(self, num_layers, num_kv_heads, head_dim, hidden_size=512):
        self.num_hidden_layers = num_layers
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = max(1, hidden_size // head_dim)


class _FakeModel:
    """Minimal stand-in for the parts of a model allocate_kv_caches reads."""

    def __init__(self, num_layers=2, num_kv_heads=8, head_dim=128, kv_shapes=None):
        self.config = _FakeConfig(num_layers, num_kv_heads, head_dim)
        if kv_shapes is not None:
            self._spyre_kv_shapes = kv_shapes


def test_allocate_keeps_attention_shapes():
    """Shapes are unchanged by the redesign: [B, n_kv, L, hd]."""
    model = _FakeModel(num_layers=3, num_kv_heads=8, head_dim=128)
    keys, values = allocate_kv_caches(model, 4, 576, torch.float32, device="cpu")
    assert len(keys) == len(values) == 3
    for k, v in zip(keys, values):
        assert k.shape == (4, 8, 576, 128), k.shape
        assert v.shape == (4, 8, 576, 128), v.shape
        assert not k.any() and not v.any(), "caches must start zeroed"


def test_allocate_honors_heterogeneous_kv_shapes():
    """Gemma-4 style: per-layer (n_kv, head_dim, v_head_dim) differ."""
    shapes = [(8, 128, 128), (4, 256, 256), (8, 128, 64)]
    model = _FakeModel(kv_shapes=shapes)
    keys, values = allocate_kv_caches(model, 2, 128, torch.float32, device="cpu")
    assert [tuple(k.shape) for k in keys] == [
        (2, 8, 128, 128),
        (2, 4, 128, 256),
        (2, 8, 128, 128),
    ]
    # v_head_dim drives the value cache's last dim.
    assert [tuple(v.shape) for v in values] == [
        (2, 8, 128, 128),
        (2, 4, 128, 256),
        (2, 8, 128, 64),
    ]


def _roundtrip_case(batch, n_kv, n_write, head_dim, cache_len=576, start=100):
    """Scatter distinct values, then read them back at the written positions."""
    cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    # Distinct value per (b, h, row, d) so any axis mix-up is visible.
    k = torch.arange(batch * n_kv * n_write * head_dim, dtype=torch.float32).reshape(
        batch, n_kv, n_write, head_dim
    )
    v = k * -1.0
    idx = torch.arange(start, start + n_write, dtype=torch.long)

    key_cache, value_cache = kv_cache_update(k, v, cache, cache.clone(), idx)

    assert key_cache.shape == (batch, n_kv, cache_len, head_dim)
    torch.testing.assert_close(key_cache[:, :, idx, :], k)
    torch.testing.assert_close(value_cache[:, :, idx, :], v)


def test_scatter_roundtrip_single_row_decode():
    _roundtrip_case(batch=1, n_kv=8, n_write=1, head_dim=128)


def test_scatter_roundtrip_block_write():
    _roundtrip_case(batch=1, n_kv=8, n_write=64, head_dim=128, start=192)


def test_scatter_roundtrip_batched():
    """B>1 is where a batch/head mix-up would show up."""
    _roundtrip_case(batch=4, n_kv=8, n_write=1, head_dim=128)
    _roundtrip_case(batch=4, n_kv=8, n_write=64, head_dim=128, start=256)


def test_scatter_roundtrip_mqa_and_padded_head():
    _roundtrip_case(batch=2, n_kv=1, n_write=4, head_dim=128)
    _roundtrip_case(batch=4, n_kv=32, n_write=64, head_dim=256, start=0)


def test_scatter_leaves_other_positions_untouched():
    batch, n_kv, cache_len, head_dim = 2, 8, 128, 64
    cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    k = torch.ones(batch, n_kv, 4, head_dim)
    idx = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    key_cache, _ = kv_cache_update(k, k, cache, cache.clone(), idx)

    written = key_cache.ne(0).any(-1).any(0).any(0).nonzero().flatten().tolist()
    assert written == [10, 11, 12, 13], written


def test_scatter_at_non_contiguous_positions():
    """cache_index need not be contiguous - only distinct."""
    batch, n_kv, cache_len, head_dim = 1, 4, 256, 64
    cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    k = torch.arange(batch * n_kv * 3 * head_dim, dtype=torch.float32).reshape(
        batch, n_kv, 3, head_dim
    )
    idx = torch.tensor([5, 200, 77], dtype=torch.long)

    key_cache, _ = kv_cache_update(k, k, cache, cache.clone(), idx)
    torch.testing.assert_close(key_cache[:, :, idx, :], k)


def test_sequential_writes_accumulate():
    """Successive decode steps must not clobber earlier positions."""
    batch, n_kv, cache_len, head_dim = 1, 4, 128, 64
    key_cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    value_cache = torch.zeros(batch, n_kv, cache_len, head_dim)

    for step in range(8):
        k = torch.full((batch, n_kv, 1, head_dim), float(step + 1))
        idx = torch.tensor([step], dtype=torch.long)
        key_cache, value_cache = kv_cache_update(k, k, key_cache, value_cache, idx)

    for pos in range(8):
        assert torch.all(key_cache[:, :, pos, :] == float(pos + 1)), f"pos {pos} lost"


def test_matches_slice_assignment_reference():
    """The scatter must equal what the old slice-assignment write produced."""
    batch, n_kv, cache_len, head_dim, n = 2, 4, 128, 64, 16
    pos = 32

    k = torch.randn(batch, n_kv, n, head_dim)
    v = torch.randn(batch, n_kv, n, head_dim)

    old_key = torch.zeros(batch, n_kv, cache_len, head_dim)
    old_value = torch.zeros(batch, n_kv, cache_len, head_dim)
    old_key[:, :, pos : pos + n, :] = k
    old_value[:, :, pos : pos + n, :] = v

    cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    idx = torch.arange(pos, pos + n, dtype=torch.long)
    new_key, new_value = kv_cache_update(k, v, cache, cache.clone(), idx)

    torch.testing.assert_close(new_key, old_key)
    torch.testing.assert_close(new_value, old_value)


def test_cache_index_length_must_match_k():
    """``cache_index`` has exactly one entry per position in ``k``/``v``.

    Every caller now decodes a single token per step, so there is no longer a
    "compute more than you store" case. An earlier API carried a ``source_index``
    to select a subset of ``k``'s positions; that gather turned a rank-6 RoPE
    output into rank-4 and tripped torch-spyre#3732, and it is gone. Pin the
    invariant so a mismatched pair fails loudly here rather than mis-writing the
    cache.
    """
    batch, n_kv, cache_len, head_dim = 2, 4, 128, 64
    cache = torch.zeros(batch, n_kv, cache_len, head_dim)

    # Matched: 1 position, 1 destination.
    k1 = torch.randn(batch, n_kv, 1, head_dim)
    kv_cache_update(k1, k1, cache, cache.clone(), torch.tensor([77], dtype=torch.long))
    torch.testing.assert_close(cache[:, :, 77, :], k1[:, :, 0, :])

    # Mismatched: 64 positions offered, 1 destination -> index_copy_ must reject.
    k64 = torch.randn(batch, n_kv, 64, head_dim)
    with pytest.raises(IndexError):
        kv_cache_update(
            k64,
            k64,
            torch.zeros(batch, n_kv, cache_len, head_dim),
            torch.zeros(batch, n_kv, cache_len, head_dim),
            torch.tensor([77], dtype=torch.long),
        )


def test_write_is_in_place():
    """The write must mutate the caller's cache, not return a copy.

    Out-of-place ``index_copy`` would allocate a whole cache per layer per step,
    and on Spyre the copy comes back with the DEFAULT device layout instead of the
    pinned one — so the next step's scatter would silently address the wrong rows
    (torch-spyre#3705). Guard the semantics here, where it is cheap to check.
    """
    batch, n_kv, cache_len, head_dim = 1, 2, 16, 8
    key_cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    value_cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    k = torch.ones(batch, n_kv, 1, head_dim)
    idx = torch.tensor([3], dtype=torch.long)

    out_key, out_value = kv_cache_update(k, k, key_cache, value_cache, idx)

    # Same objects back, and the originals carry the write.
    assert out_key is key_cache
    assert out_value is value_cache
    assert torch.all(key_cache[:, :, 3, :] == 1.0)
    assert torch.all(value_cache[:, :, 3, :] == 1.0)


def test_single_position_write_matches_slice_assignment():
    """The decode-step case: writing one position via ``cache_index`` matches a
    direct slice assignment at that position."""
    batch, n_kv, cache_len, head_dim = 2, 4, 128, 64
    pos = 77
    k = torch.randn(batch, n_kv, 1, head_dim)

    old = torch.zeros(batch, n_kv, cache_len, head_dim)
    old[:, :, pos : pos + 1, :] = k

    cache = torch.zeros(batch, n_kv, cache_len, head_dim)
    new_key, _ = kv_cache_update(
        k, k, cache, cache.clone(), torch.tensor([pos], dtype=torch.long)
    )
    torch.testing.assert_close(new_key, old)
