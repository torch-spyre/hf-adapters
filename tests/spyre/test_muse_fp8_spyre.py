# Copyright 2026 The Torch-Spyre Authors.
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

"""Compile the Muse Glimmer fp8 groups at real widths and check them against
the CPU emulation (random weights). Run this before any fp8 model run: it is
the numerics gate for the fused, per-row-scaled layout.

Usage::

    pytest -s -vvv tests/spyre/test_muse_fp8_spyre.py
"""

import pytest
import torch
import torch.nn as nn
import torch_spyre  # noqa: F401

from hf_adapters import _muse_fp8 as fp8
from hf_adapters.hf_common import DEVICE

pytestmark = pytest.mark.requires_spyre

# Muse Glimmer 30B widths: hidden 6656; q|k|v|gate = 4096 + 256 + 256 + 4096;
# o 4096 -> 6656; mlp gate|up 2 x 19968; down 19968 -> 6656; and the TP 2 / 4
# per-rank shards. The CPU emulation of these codes with fp16 activations is
# within 0.0002 of its fp32 run at every width, 0.037-0.038 from the exact
# 16-bit product; the device bound (0.03) is against the fp32 emulation.
SHAPES = {
    "qkvg": ([4096, 256, 256, 4096], 6656),
    "o": ([6656], 4096),
    "gate_up": ([19968, 19968], 6656),
    "down": ([6656], 19968),
    # TP 2 shards: half the heads / intermediate per rank.
    "tp2_qkvg": ([2048, 128, 128, 2048], 6656),
    "tp2_o": ([6656], 2048),
    "tp2_gate_up": ([9984, 9984], 6656),
    "tp2_down": ([6656], 9984),
    # TP 4 shards: 8 query heads, one (replicated) KV head, a quarter of the MLP.
    "tp4_qkvg": ([1024, 128, 128, 1024], 6656),
    "tp4_o": ([6656], 1024),
    "tp4_gate_up": ([4992, 4992], 6656),
    "tp4_down": ([6656], 4992),
    # DFlash assistant: q|k|v = 4096 + 1024 + 1024 (8 KV heads); fc tap block.
    "draft_qkv": ([4096, 1024, 1024], 6656),
    # Its TP 2 / 4 shards (o, gate|up, down match the target's shard widths).
    "tp2_draft_qkv": ([2048, 512, 512], 6656),
    "tp4_draft_qkv": ([1024, 256, 256], 6656),
    "draft_fc_block": ([6656], 6656),
}


def _linears(widths, k, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in widths:
        linear = nn.Linear(k, n, bias=False)
        w = torch.randn(n, k, generator=g) / k**0.5
        linear.weight = nn.Parameter(w.half(), requires_grad=False)
        out.append(linear)
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("group", list(SHAPES))
def test_fp8_group_spyre(group, tokens):
    torch._dynamo.reset()
    widths, k = SHAPES[group]
    host_w, host_s = fp8._fuse(_linears(widths, k, 1), "cpu", torch.float32)
    dev_w, dev_s = fp8._fuse(_linears(widths, k, 1), DEVICE, torch.float16)
    splits = tuple(widths) if len(widths) > 1 else None
    x = torch.randn(1, tokens, k, generator=torch.Generator().manual_seed(2)).half()
    expected = fp8.fp8_matmul(x.float(), host_w, host_s, splits)
    with torch.no_grad():
        actual = torch.compile(
            lambda x: fp8.fp8_matmul(x, dev_w, dev_s, splits),
            fullgraph=True,
            dynamic=False,
        )(x.to(DEVICE))
    if splits is None:
        actual, expected = [actual], [expected]
    for a, e in zip(actual, expected):
        a = a.to("cpu").float()
        assert a.shape == e.shape
        assert torch.isfinite(a).all()
        # Same codes; the device differs in fp16 accumulation and activation rounding.
        assert _rel(a, e) < 0.03, _rel(a, e)


# One rank's shard of the block (before the o / down all-reduce) at TP 1 / 2 / 4:
# (query heads, KV heads, MLP intermediate). TP 4 replicates Muse's 2 KV heads.
BLOCK_SHARDS = {1: (32, 2, 19968), 2: (16, 1, 9984), 4: (8, 1, 4992)}


@pytest.mark.parametrize("tp", list(BLOCK_SHARDS))
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize("tokens", [1, 16])
def test_fp8_decoder_block_spyre(tokens, use_rope, tp):
    """One real-width Muse decoder layer (or one TP rank's shard of it) in its
    fp8 block (``_make_block``) against the same block and codes on the CPU
    (random weights). Run before any fp8 model run on top of the per-group
    gate above."""
    from transformers import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerTextDecoderLayer,
    )

    from hf_adapters.hf_common import (
        allocate_kv_cache_tensor,
        make_cache_index,
        to_default_device_layout,
    )

    torch._dynamo.reset()
    torch.manual_seed(3)
    heads, kv_heads, inter = BLOCK_SHARDS[tp]
    cfg = MuseGlimmerTextConfig(
        num_attention_heads=heads, num_key_value_heads=kv_heads, intermediate_size=inter
    )
    parts = ("attn", "mlp")

    def layer_on(device, dtype):
        torch.manual_seed(3)
        layer = MuseGlimmerTextDecoderLayer(cfg, 0).eval().to(torch.float16)
        attn, mlp = layer.self_attn, layer.mlp
        layer._spyre_fp8 = {
            "attn": fp8._fuse(
                [attn.q_proj, attn.k_proj, attn.v_proj, attn.gate_proj], device, dtype
            ),
            "o": fp8._fuse([attn.o_proj], device, dtype),
            "mlp": fp8._fuse([mlp.gate_proj, mlp.up_proj], device, dtype),
            "down": fp8._fuse([mlp.down_proj], device, dtype),
        }
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj"):
            setattr(attn, name, nn.Identity())
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, name, nn.Identity())
        return layer.to(device=device, dtype=dtype)

    hd, cache = cfg.head_dim, 512
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, tokens, cfg.hidden_size, generator=g).half()
    angle = torch.randn(1, tokens, hd // 2, generator=g)
    c, s = angle.cos(), angle.sin()
    freqs = torch.stack([torch.stack([c, -s], 2), torch.stack([s, c], 2)], 2).half()
    allowed = torch.ones(tokens, cache, dtype=torch.bool).tril(0)
    allowed[:, tokens:] = False
    mask = torch.zeros(1, 1, tokens, cache, dtype=torch.float16).masked_fill(
        ~allowed, torch.finfo(torch.float16).min / 2
    )

    def run(device, dtype):
        layer = layer_on(device, dtype)
        block = fp8._make_block(layer, parts, heads, kv_heads, hd, use_rope, None)
        if device == "cpu":
            block = block._torchdynamo_orig_callable
        k = allocate_kv_cache_tensor(1, kv_heads, cache, hd, dtype, device)
        v = allocate_kv_cache_tensor(1, kv_heads, cache, hd, dtype, device)
        h = x.to(device, dtype)
        if device != "cpu":
            h = to_default_device_layout(h)
        with torch.no_grad():
            out, _, _ = block(
                h,
                freqs.to(device, dtype),
                mask.to(device, dtype),
                k,
                v,
                make_cache_index(0, tokens, device),
            )
        return out.to("cpu").float()

    expected = run("cpu", torch.float32)
    actual = run(DEVICE, torch.float16)
    assert torch.isfinite(actual).all()
    # Compare the layer's own contribution, not the residual it carries. The
    # CPU emulation of the same W8A8 codes run with fp16 activations differs
    # from its fp32 run by 0.037 (1 row) / 0.045 (16 rows) at TP 1, 0.036 /
    # 0.045 at the TP 2 shard and 0.037 / 0.044 at the TP 4 shard: four chained
    # re-quantized groups. Spyre measured 0.044 / 0.053 at TP 1 (fp16
    # accumulation in scaled_mm on top). 0.08 is ~1.8x that reference; a
    # misdescribed layout reads ~1.0.
    rel = _rel(actual - x.float(), expected - x.float())
    print(f"fp8 block tp={tp} rows={tokens} rope={use_rope}: rel {rel:.4f}")
    assert rel < 0.08, rel
