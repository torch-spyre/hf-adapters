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

"""Per-step pass/fail rule for the HF-vs-Spyre token-compare harness.

A top-1 mismatch is only a defect when the two sides genuinely disagree. When
HF's own top-2 candidates sit within bf16 noise of each other (Gemma 4 12B-it on
"The capital of France is": 'Paris' vs 'The', #494), Spyre picking the other
one is rounding, not a bug. Such a step is a *near tie*: both HF and Spyre must
score the two candidates within ``TOP1_TIE_MARGIN`` of each other, so a
confidently wrong Spyre (large ``spyre_margin``) still fails.
"""

from typing import Any

import torch

# Logit gap below which a top-1 flip counts as numeric noise. Measured bf16
# per-logit error on Spyre is ~0.1 (healthy dense models) to ~0.7 (Gemma 4 12B,
# 48 sandwich-norm layers, softcapped at 30), and a flip needs error on both
# candidates, so 1.0 covers real noise while a broken layer (diffs >10) cannot
# hide under it.
TOP1_TIE_MARGIN = 1.0


def compare_step(hf_logits: torch.Tensor, spyre_logits: torch.Tensor) -> dict[str, Any]:
    """Compare one step's logits; returns diffs, margins and the verdict."""
    vocab = min(hf_logits.shape[0], spyre_logits.shape[0])
    h = hf_logits[:vocab].float()
    a = spyre_logits[:vocab].float()
    hf_nan = bool(h.isnan().any())
    spyre_nan = bool(a.isnan().any())

    diff = (h - a).abs()
    h_top1 = int(h.argmax())
    a_top1 = int(a.argmax())
    match = h_top1 == a_top1
    # How far each side ranks the other side's pick below its own (0 on match).
    hf_margin = float(h[h_top1] - h[a_top1])
    spyre_margin = float(a[a_top1] - a[h_top1])
    near_tie = (
        not match and hf_margin <= TOP1_TIE_MARGIN and spyre_margin <= TOP1_TIE_MARGIN
    )
    return {
        "top1_match": match,
        "near_tie": near_tie,
        "top1_ok": (match or near_tie) and not (hf_nan or spyre_nan),
        "hf_margin": hf_margin,
        "spyre_margin": spyre_margin,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "hf_nan": hf_nan,
        "spyre_nan": spyre_nan,
    }
