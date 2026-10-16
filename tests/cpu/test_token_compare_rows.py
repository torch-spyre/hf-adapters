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

from tests._token_compare_helpers import TOP1_TIE_MARGIN, compare_step


def _logits(**scores: float) -> torch.Tensor:
    """Four-token vocab; ``t<i>=score`` sets token i, the rest stay at -10."""
    out = torch.full((4,), -10.0)
    for name, score in scores.items():
        out[int(name[1:])] = score
    return out


def test_exact_top1_match_passes():
    row = compare_step(_logits(t0=5.0, t1=1.0), _logits(t0=5.1, t1=1.0))
    assert row["top1_match"] and row["top1_ok"]
    assert row["hf_margin"] == 0.0


def test_flip_between_near_tied_tokens_is_tolerated():
    # HF: Paris(0) 5.0 vs The(1) 4.6. Spyre flips to The by a hair. Both sides
    # agree the two candidates are within the tolerance -> numeric noise.
    row = compare_step(_logits(t0=5.0, t1=4.6), _logits(t0=4.7, t1=4.8))
    assert not row["top1_match"]
    assert row["near_tie"] and row["top1_ok"]
    assert row["hf_margin"] == pytest.approx(0.4)


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
    assert not row["top1_ok"]
