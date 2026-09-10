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

"""CPU wiring tests for Gemma 4's sliding layer on the SWA op path.

On CPU the dispatcher takes its reference branch, so what these check is the
*wiring*: that the coordinates the adapter derives, the mask suppression, and the
KV write all line up with the band-masked path they replace. The numerics of the
op itself are a device question (tests/spyre/test_swa_layer_ab_spyre.py).
"""

import copy

import torch
import torch.nn as nn
from _swa_helpers import HeadRMSNorm, identity_freqs, make_sliding_attention

from hf_adapters.hf_common import (
    add_causal_sliding_window_band,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)
from hf_adapters.hf_gemma4 import Gemma4SharedBlock

WINDOW = 64
CAPACITY = 256
PREFILL = 128
HEAD_DIM = 64


def _band_mask(batch, seqlen, block_base, dtype=torch.float32):
    """Exactly what _build_layer_masks builds for a sliding layer today.

    Deviation from the brief: the brief's version called ``build_prefill_mask``
    unconditionally. That builder assumes the block starts at cache column 0
    (it masks every column past the query's *relative* row index), which is
    only correct for a fresh prefill. ``generate`` (hf_common.py) dispatches to
    ``build_decode_mask`` instead for a one-token decode step at a non-zero
    ``block_base`` -- that mask allows every real column up to ``block_base``,
    not just column 0. Since the band is combined additively (``mask + band``,
    not a max/OR), an over-restrictive base cannot be widened back by the band;
    with the brief's version every column in the decode test's row is masked
    (finite fill or -inf) and the comparison fails for a reason unrelated to
    the adapter's wiring. Verified against ``op_out`` before making this
    change: the fixed helper matches the op path, the original did not.
    """
    if seqlen == 1 and block_base > 0:
        mask = build_decode_mask(batch, CAPACITY, block_base, 0, dtype=dtype)
    else:
        mask = build_prefill_mask(batch, seqlen, CAPACITY, 0, dtype=dtype)
    coords = (torch.arange(seqlen)[None, :] + block_base).expand(batch, seqlen)
    return add_causal_sliding_window_band(mask, coords, WINDOW)


def _caches(batch=1, num_kv_heads=2):
    return (
        torch.zeros(batch, num_kv_heads, CAPACITY, HEAD_DIM),
        torch.zeros(batch, num_kv_heads, CAPACITY, HEAD_DIM),
    )


def test_phase1_prefill_matches_the_band_masked_path():
    torch.manual_seed(1)
    hidden = torch.randn(1, PREFILL, 4 * HEAD_DIM)
    freqs = identity_freqs(1, PREFILL, HEAD_DIM)
    index = make_cache_index(0, PREFILL)

    band = make_sliding_attention(window_size=WINDOW, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    band_out, band_k, band_v = band(
        hidden, freqs, _band_mask(1, PREFILL, 0), *_caches(), index
    )
    op_out, op_k, op_v = op(hidden, freqs, _band_mask(1, PREFILL, 0), *_caches(), index)

    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(op_k, band_k)
    torch.testing.assert_close(op_v, band_v)


def test_phase1_decode_matches_the_band_masked_path():
    """One token at cache slot 128, reading the 64 columns behind it."""
    torch.manual_seed(2)
    key_cache, value_cache = _caches()
    key_cache[:, :, :PREFILL, :] = torch.randn(1, 2, PREFILL, HEAD_DIM)
    value_cache[:, :, :PREFILL, :] = torch.randn(1, 2, PREFILL, HEAD_DIM)

    hidden = torch.randn(1, 1, 4 * HEAD_DIM)
    freqs = identity_freqs(1, 1, HEAD_DIM)
    index = make_cache_index(PREFILL, 1)

    band = make_sliding_attention(window_size=WINDOW, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    band_out, _, _ = band(
        hidden,
        freqs,
        _band_mask(1, 1, PREFILL),
        key_cache.clone(),
        value_cache.clone(),
        index,
    )
    op_out, _, _ = op(
        hidden,
        freqs,
        _band_mask(1, 1, PREFILL),
        key_cache.clone(),
        value_cache.clone(),
        index,
    )

    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)


def test_global_layer_still_uses_the_mask():
    """is_sliding=False must ignore swa_mode entirely."""
    torch.manual_seed(3)
    hidden = torch.randn(1, PREFILL, 4 * HEAD_DIM)
    freqs = identity_freqs(1, PREFILL, HEAD_DIM)
    index = make_cache_index(0, PREFILL)
    mask = build_prefill_mask(1, PREFILL, CAPACITY, 0, dtype=torch.float32)

    module = make_sliding_attention(swa_mode="phase1")
    module.is_sliding = False

    out, _, _ = module(hidden, freqs, mask, *_caches(), index)
    assert torch.isfinite(out).all()


def test_kv_sharing_sliding_layer_matches_the_band_masked_path():
    """A Q-only consumer must use its producer's cache through the SWA op path."""
    torch.manual_seed(4)
    hidden_size = 4 * HEAD_DIM
    qk_gain = HEAD_DIM**-0.25

    class FakeSharedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.self_attn.q_norm = HeadRMSNorm(HEAD_DIM, weight_init=qk_gain)
            self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.self_attn.scaling = 1.0
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.pre_feedforward_layernorm = nn.Identity()
            self.post_feedforward_layernorm = nn.Identity()
            self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)
            self.register_buffer("layer_scalar", torch.tensor(1.0))

    layer = FakeSharedLayer()
    band = Gemma4SharedBlock(
        layer,
        num_q_heads=4,
        head_dim=HEAD_DIM,
        has_ple=False,
        is_sliding=True,
        window_size=WINDOW,
        swa_mode=None,
    ).eval()
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"

    hidden = torch.randn(1, PREFILL, hidden_size)
    freqs = identity_freqs(1, PREFILL, HEAD_DIM)
    key_cache, value_cache = _caches()
    key_cache.normal_()
    value_cache.normal_()

    band_out = band(
        hidden,
        freqs,
        _band_mask(1, PREFILL, 0),
        key_cache,
        value_cache,
        layer.layer_scalar,
    )
    op_out = op(
        hidden,
        freqs,
        _band_mask(1, PREFILL, 0),
        key_cache,
        value_cache,
        layer.layer_scalar,
    )

    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)
