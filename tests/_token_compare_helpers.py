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

Decode is teacher-forced, so every step scores the same context on both sides
and the full logit vectors can be held to absolute bounds: a step passes only
when cosine, max diff and mean diff are all within the calibrated limits below.
Top-1 must also agree, with one exception: when HF's own top-2 candidates sit
within bf16 noise of each other (Gemma 4 12B-it on "The capital of France is":
'Paris' vs 'The', #494), Spyre picking the other one is rounding, not a bug.
Such a step is a *near tie*: both HF and Spyre must score the two candidates
within ``TOP1_TIE_MARGIN`` of each other, so a confidently wrong Spyre (large
``spyre_margin``) still fails.
"""

from typing import Any

import torch

# Limits calibrated on Spyre (torch-spyre 601e155f) across all 26 models of the
# default causal collection, teacher-forced, prefill + 4 decode steps (#687).
#
# Logit gap below which a top-1 flip counts as numeric noise. The only flip
# observed was Gemma 4 12B-it prefill: HF margin 0.125, Spyre margin 0.25 (two
# bf16 ulps at |logit| ~20-30). 0.5 is a ~1.65x odds ratio, so any real
# preference between the two candidates still fails.
TOP1_TIE_MARGIN = 0.5
# Worst blocking model per metric, over runs with and without
# CO_OPTIMIZING_LX_PLANNING: cosine 0.9749 and max diff 3.07 (pythia-410m),
# mean diff 0.72 (Phi-3.5-mini). A drifting model sits past these, e.g.
# OLMoE-1B-7B decode with the LX co-optimizer on: cosine 0.91, mean diff 1.8
# -- while top-1 still matched (the same model is at cosine 0.9999 with it off).
MIN_COSINE = 0.95
MAX_LOGIT_DIFF = 5.0
MAX_MEAN_DIFF = 1.0


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
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    cosine = float(torch.nn.functional.cosine_similarity(h, a, dim=0))
    no_nan = not (hf_nan or spyre_nan)
    top1_ok = (match or near_tie) and no_nan
    # NaN compares False, so a NaN step never passes these bounds.
    logits_ok = (
        no_nan
        and cosine >= MIN_COSINE
        and max_diff <= MAX_LOGIT_DIFF
        and mean_diff <= MAX_MEAN_DIFF
    )
    return {
        "top1_match": match,
        "near_tie": near_tie,
        "top1_ok": top1_ok,
        "logits_ok": logits_ok,
        "step_ok": top1_ok and logits_ok,
        "hf_margin": hf_margin,
        "spyre_margin": spyre_margin,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine": cosine,
        "hf_nan": hf_nan,
        "spyre_nan": spyre_nan,
    }
