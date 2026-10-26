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

"""Pass/fail rules of the Spyre token-compare harness, checked on CPU (#494)."""

import pytest
import torch

from tests._token_compare_helpers import (
    MAX_LOGIT_DIFF,
    MAX_MEAN_DIFF,
    MIN_COSINE,
    TOP1_TIE_MARGIN,
    compare_step,
)

_VOCAB = 1000


def _logits(**scores: float) -> torch.Tensor:
    """Four-token vocab; ``t<i>=score`` sets token i, the rest stay at -10."""
    out = torch.full((4,), -10.0)
    for name, score in scores.items():
        out[int(name[1:])] = score
    return out


def _randn(seed: int, scale: float) -> torch.Tensor:
    """Wide random logits with token 0 as a clear top-1."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.randn(_VOCAB, generator=gen) * scale
    out[0] = 20.0
    return out


def test_exact_top1_match_passes():
    row = compare_step(_logits(t0=5.0, t1=1.0), _logits(t0=5.1, t1=1.0))
    assert row["top1_match"] and row["top1_ok"]
    assert row["hf_margin"] == 0.0


def test_flip_between_near_tied_tokens_is_tolerated():
    # Gemma 4 12B-it on Spyre (#494): HF ranks Paris 0.125 above The, Spyre
    # ranks The 0.25 above Paris. Both sides call it a tie -> numeric noise.
    row = compare_step(_logits(t0=5.0, t1=4.875), _logits(t0=4.75, t1=5.0))
    assert not row["top1_match"]
    assert row["near_tie"] and row["top1_ok"] and row["step_ok"]
    assert row["hf_margin"] == pytest.approx(0.125)
    assert row["spyre_margin"] == pytest.approx(0.25)


def test_flip_at_one_logit_gap_is_not_a_tie():
    # A 1.0-logit gap is a ~2.7x odds ratio: a real preference, not a tie.
    row = compare_step(_logits(t0=1.0, t1=0.0), _logits(t0=0.0, t1=1.0))
    assert TOP1_TIE_MARGIN < 1.0
    assert not row["near_tie"] and not row["step_ok"]


def test_flip_with_large_hf_margin_fails():
    row = compare_step(_logits(t0=5.0, t1=1.0), _logits(t0=1.0, t1=5.0))
    assert not row["near_tie"] and not row["top1_ok"]


def test_flip_device_confidently_wrong_fails():
    # HF sees a tie, but Spyre is confidently wrong -> a real defect, not noise.
    row = compare_step(_logits(t0=5.0, t1=4.9), _logits(t0=0.0, t1=8.0))
    assert row["hf_margin"] <= TOP1_TIE_MARGIN
    assert row["spyre_margin"] > TOP1_TIE_MARGIN
    assert not row["top1_ok"]


def test_margin_at_tolerance_is_tolerated():
    row = compare_step(
        _logits(t0=TOP1_TIE_MARGIN, t1=0.0), _logits(t0=0.0, t1=TOP1_TIE_MARGIN)
    )
    assert row["top1_ok"]


def test_healthy_bf16_noise_passes():
    hf = _randn(seed=0, scale=3.0)
    noise = torch.randn(_VOCAB, generator=torch.Generator().manual_seed(1)) * 0.1
    row = compare_step(hf, hf + noise)
    assert row["logits_ok"] and row["step_ok"]
    assert row["cosine"] > 0.999


def test_low_cosine_fails_even_when_top1_matches():
    # Small per-logit diffs, but the two distributions are unrelated.
    row = compare_step(_randn(seed=0, scale=0.3), _randn(seed=1, scale=0.3))
    assert row["top1_match"]
    assert row["max_diff"] <= MAX_LOGIT_DIFF and row["mean_diff"] <= MAX_MEAN_DIFF
    assert row["cosine"] < MIN_COSINE
    assert not row["logits_ok"] and not row["step_ok"]


def test_large_max_diff_fails_even_when_top1_matches():
    hf = _randn(seed=0, scale=3.0)
    spyre = hf.clone()
    spyre[5] += MAX_LOGIT_DIFF + 1.0
    row = compare_step(hf, spyre)
    assert row["top1_match"] and row["cosine"] >= MIN_COSINE
    assert row["max_diff"] > MAX_LOGIT_DIFF
    assert not row["logits_ok"] and not row["step_ok"]


def test_large_mean_diff_fails_even_when_top1_matches():
    # Every logit shifted by the same amount: cosine and max stay in bounds.
    hf = torch.full((_VOCAB,), 10.0)
    hf[0] = 11.0
    row = compare_step(hf, hf + MAX_MEAN_DIFF + 0.2)
    assert row["top1_match"] and row["cosine"] >= MIN_COSINE
    assert row["max_diff"] <= MAX_LOGIT_DIFF
    assert row["mean_diff"] > MAX_MEAN_DIFF
    assert not row["logits_ok"] and not row["step_ok"]


def test_diffs_compare_shared_vocab_prefix_only():
    # Spyre logits carry stick padding past the real vocab; it must be ignored.
    hf = _logits(t0=5.0)
    spyre = torch.cat([_logits(t0=5.0), torch.full((4,), 99.0)])
    row = compare_step(hf, spyre)
    assert row["max_diff"] == 0.0 and row["top1_ok"]


def test_nan_logits_are_reported_and_fail():
    spyre = _logits(t0=5.0)
    spyre[2] = float("nan")
    row = compare_step(_logits(t0=5.0), spyre)
    assert row["spyre_nan"] and not row["hf_nan"]
    assert not row["top1_ok"] and not row["step_ok"]
