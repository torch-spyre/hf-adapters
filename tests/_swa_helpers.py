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

"""Builders shared by the CPU and Spyre sliding-window attention tests.

A real Gemma 4 checkpoint is 12B parameters and its device token-compare is
currently red for unrelated reasons, so the SWA replacement is tested through one
``Gemma4Attention`` with random weights at Gemma 4's shapes.
"""

import types

import torch
import torch.nn as nn


class HeadRMSNorm(nn.Module):
    """Per-head RMSNorm over the last dim, standing in for Gemma4RMSNorm.

    ``weight_init`` sets the (constant) gain. It defaults to 1.0, but the Q/K
    norms in ``make_sliding_attention`` pass a sub-unit gain to stand in for the
    real model's learned norms — see there for why.
    """

    def __init__(self, head_dim, eps=1e-6, weight_init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((head_dim,), float(weight_init)))
        self.eps = eps

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FakeKVModel:
    """Minimal stand-in for what ``allocate_kv_caches`` reads off a model.

    Used so the tests get caches with the pinned device layout that
    ``kv_cache_update``'s indirect scatter requires, rather than a bare
    ``torch.zeros`` that would silently write to the wrong rows.
    """

    def __init__(self, kv_shapes):
        self._spyre_kv_shapes = kv_shapes


def identity_freqs(batch, seqlen, head_dim, dtype=torch.float32):
    """``selected_freqs`` that make ``apply_rope_matmul`` a no-op.

    Shape ``[B, L, 2, 2, D/2]`` holding the 2x2 identity, so RoPE cannot
    contribute a difference to an A/B comparison of the attention itself.
    """
    half = head_dim // 2
    freqs = torch.zeros(batch, seqlen, 2, 2, half, dtype=dtype)
    freqs[:, :, 0, 0, :] = 1.0
    freqs[:, :, 1, 1, :] = 1.0
    return freqs


def make_sliding_attention(
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=64,
    window_size=64,
    swa_mode=None,
    dtype=torch.float32,
    seed=0,
):
    """A ``Gemma4Attention`` wired as a sliding layer, with random weights.

    Gemma 4's sliding layers keep a separate V (``is_kv_eq_v`` is False, which is
    a global-layer property), carry per-head RMSNorm on Q/K/V, and attend
    **unscaled** (``scaling == 1.0``).

    The Q/K norms use a ``head_dim ** -0.25`` gain rather than the identity. With
    unit gain and iid-normal random weights, ``q . k`` at ``scaling == 1.0`` has
    std ``sqrt(head_dim)`` — 16 at ``head_dim == 256`` — so the softmax is nearly
    one-hot and its argmax flips under fp16 rounding, an artifact of *random*
    weights the real model never sees (its learned norms and correlated
    activations keep scores moderate). The ``-0.25`` gain makes score std
    ``sqrt(head_dim) * (head_dim ** -0.25) ** 2 == 1`` for any ``head_dim``,
    standing in for those learned norms. It leaves V — and therefore the output
    scale the tolerances are measured against — at unit gain, and keeps
    ``scaling == 1.0`` so the op is still exercised in the production regime.
    """
    from hf_adapters.hf_gemma4 import Gemma4Attention

    torch.manual_seed(seed)
    hidden = num_q_heads * head_dim
    qk_gain = head_dim**-0.25
    attn = types.SimpleNamespace(
        q_proj=nn.Linear(hidden, num_q_heads * head_dim, bias=False),
        k_proj=nn.Linear(hidden, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(hidden, num_kv_heads * head_dim, bias=False),
        o_proj=nn.Linear(num_q_heads * head_dim, hidden, bias=False),
        q_norm=HeadRMSNorm(head_dim, weight_init=qk_gain),
        k_norm=HeadRMSNorm(head_dim, weight_init=qk_gain),
        v_norm=HeadRMSNorm(head_dim),
        scaling=1.0,
    )
    module = Gemma4Attention(
        attn,
        num_q_heads,
        num_kv_heads,
        head_dim,
        False,
        is_sliding=True,
        window_size=window_size,
        swa_mode=swa_mode,
    )
    return module.to(dtype).eval()
