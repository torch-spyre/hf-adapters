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

"""CPU coverage for the opt-in Muse Glimmer W8A8 fp8 linears (``_muse_fp8``).

The CPU emulation multiplies the exact e4m3 codes, so these tests pin the
quantization, the fused-group bookkeeping, the replaced (not duplicated)
weights, and the end-to-end accuracy against the 16-bit adapter.
"""

import pytest
import torch
import torch.nn as nn

from hf_adapters import _muse_fp8, _muse_tree, hf_common, hf_muse_glimmer
from hf_adapters._muse_tree import MuseTreeVerifier
from tests._muse_helpers import VOCAB, prepared_pair, stock_forward


def test_parts_parsing():
    assert _muse_fp8.fp8_parts("") == ()
    assert _muse_fp8.fp8_parts("attn, mlp,lm_head") == ("attn", "mlp", "lm_head")
    with pytest.raises(ValueError, match="unknown"):
        _muse_fp8.fp8_parts("attn,experts")


def test_real_widths_are_stick_aligned_and_the_head_pad_fits_the_ear_limit():
    for n, k in [(8704, 6656), (6656, 4096), (39936, 6656), (6656, 19968)]:
        _muse_fp8._check_aligned("muse", n, k)
    with pytest.raises(ValueError, match="multiples of 128"):
        _muse_fp8._check_aligned("bad", 202048, 6656)
    # 202112 = 1579 sticks of 128 and 1579 is prime: one core's span would be
    # 1.35 GB. The next smooth count is 1580 = 2^2 * 5 * 79.
    assert _muse_fp8.fp8_vocab(202048, 6656) == 202240
    assert hf_common._largest_prime_factor(202240 // 128) * 6656 * 128 <= (
        hf_common._EAR_LIMIT_BYTES
    )


def test_quantized_weight_round_trips_and_the_emulation_is_accurate():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 512, generator=g) * 0.02
    codes, scale = _muse_fp8.quantize_weight(w)
    assert codes.dtype == torch.float8_e4m3fn and scale.dtype == torch.float16
    assert float(codes.float().abs().max()) <= _muse_fp8.W_MAX
    x = torch.randn(4, 3, 512, generator=g)
    assert _muse_fp8.emulated_relative_error(w, x) < 0.06


def test_single_token_input_matches_the_two_row_form():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 384, generator=g) * 0.02
    codes, scale = _muse_fp8.quantize_weight(w)
    wq = codes.t().contiguous()
    x = torch.randn(1, 1, 384, generator=g)
    one = _muse_fp8.fp8_matmul(x, wq, scale.float())
    two = _muse_fp8.fp8_matmul(torch.cat([x, -x], 1), wq, scale.float())
    assert one.shape == (1, 1, 256)
    torch.testing.assert_close(one, two[:, :1])
    q, k = _muse_fp8.fp8_matmul(x, wq, scale.float(), (128, 128))
    torch.testing.assert_close(torch.cat([q, k], -1), one)


PARTS = ["attn", "mlp", "lm_head", "attn,mlp,lm_head"]


def _prefill_logits(model, prompt):
    verifier = MuseTreeVerifier(model, 8, prefill_chunk=64, capacity_multiple=512)
    return verifier.prefill(prompt, max_new_tokens=16)[0, -1, :VOCAB]


def _prompt(seed=2):
    return torch.randint(
        2, VOCAB - 2, (1, 60), generator=torch.Generator().manual_seed(seed)
    )


@pytest.mark.parametrize("parts", PARTS)
def test_with_lossless_codes_the_fp8_graphs_equal_the_16_bit_model(monkeypatch, parts):
    """fp32 "codes" make W8A8 exact, so the fused groups, splits, scales and
    the replaced head must reproduce the 16-bit adapter (and stock HF)."""
    monkeypatch.setattr(_muse_fp8, "FP8", torch.float32)
    reference, adapted = prepared_pair(monkeypatch, _muse_tree, _muse_fp8, parts=parts)
    want = stock_forward(reference, _prompt()).logits[0, -1]
    torch.testing.assert_close(
        _prefill_logits(adapted, _prompt()), want, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("parts", PARTS)
def test_fp8_conversion_replaces_the_weights_and_stays_finite(monkeypatch, parts):
    reference, adapted = prepared_pair(monkeypatch, _muse_tree, _muse_fp8, parts=parts)
    assert adapted._spyre_fp8_parts == tuple(parts.split(","))
    layers = hf_common.get_backbone(adapted).layers
    if "attn" in parts:
        # Replaced, not duplicated: no 16-bit attention linears remain.
        assert all(isinstance(layer.self_attn.q_proj, nn.Identity) for layer in layers)
        qkvg, scale = layers[0]._spyre_fp8["attn"]
        assert qkvg.dtype == torch.float8_e4m3fn
        assert qkvg.shape == (256, 4 * 128 + 2 * 128) and scale.shape == (768,)
    if "mlp" in parts:
        assert all(isinstance(layer.mlp.down_proj, nn.Identity) for layer in layers)
    if "lm_head" in parts:
        assert isinstance(adapted.lm_head, _muse_fp8.Fp8Linear)
        assert adapted.lm_head.weight.shape[1] % 128 == 0
    got = _prefill_logits(adapted, _prompt())
    assert got.shape == (VOCAB,) and torch.isfinite(got).all()
    # Plain generate runs the same fp8 blocks (band-masked SDPA on every
    # request: no anchored sliding-window cache with fp8 layers).
    out = hf_muse_glimmer.generate(adapted, _prompt(), max_new_tokens=2)
    assert out.shape == (1, 62)
    if parts != "lm_head":
        assert adapted._spyre_swa_mode is None
        blocks = adapted._spyre_compiled_blocks
        assert blocks is adapted._spyre_muse_full_blocks
        assert blocks is adapted._spyre_muse_swa_blocks


@pytest.mark.parametrize("prompt_len", [40, 120])
def test_lossless_fp8_generate_equals_stock_greedy(monkeypatch, prompt_len):
    """Plain generate under fp8 (lossless codes) still matches stock greedy,
    within and past the sliding window."""
    monkeypatch.setattr(_muse_fp8, "FP8", torch.float32)
    reference, adapted = prepared_pair(
        monkeypatch, _muse_tree, _muse_fp8, parts="attn,mlp,lm_head"
    )
    prompt = torch.randint(
        2, VOCAB - 2, (1, prompt_len), generator=torch.Generator().manual_seed(5)
    )
    with torch.no_grad():
        want = reference.generate(
            prompt, max_new_tokens=16, min_new_tokens=16, do_sample=False
        )
    got = hf_muse_glimmer.generate(adapted, prompt, max_new_tokens=16)
    assert got.tolist() == want.tolist()


def test_lm_head_alone_stays_close(monkeypatch):
    """The head is the last op (no compounding through random layers), so its
    fp8 error is the single-matmul error: the logits stay close."""
    reference, adapted = prepared_pair(
        monkeypatch, _muse_tree, _muse_fp8, parts="lm_head"
    )
    want = stock_forward(reference, _prompt()).logits[0, -1]
    got = _prefill_logits(adapted, _prompt())
    assert torch.nn.functional.cosine_similarity(got, want, dim=0) > 0.995


def test_unset_knob_leaves_the_model_untouched(monkeypatch):
    reference, adapted = prepared_pair(monkeypatch, _muse_tree)
    assert not hasattr(adapted, "_spyre_fp8_parts")
    assert isinstance(adapted.lm_head, nn.Linear)
    layers = hf_common.get_backbone(adapted).layers
    assert all(not hasattr(layer, "_spyre_fp8") for layer in layers)
