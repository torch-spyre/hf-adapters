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
Those three are measured on mean-centred logits, because softmax ignores a
common offset: a uniform shift leaves every token probability unchanged and must
not read as drift, and a large shared offset must not make unrelated vectors
look aligned. The uncentred values are kept as ``raw_*`` for reporting only.
Top-1 must also agree exactly. Teacher forcing stops one flip from corrupting
later steps, but the flip itself still fails: Spyre's prefill pick has been seen
to change between runs (#494), and that stays visible until it is explained.
``hf_margin`` and ``spyre_margin`` report how close a flip was; they never
excuse one. These checks compare Spyre with HF, not Spyre with itself, so they
say nothing about run-to-run repeatability.
"""

from typing import Any

import torch

# Limits checked on centred logits, teacher-forced, prefill + 4 decode steps,
# across the 27 models of the default causal collection (#687). Worst blocking
# values: cosine 0.980, max diff 3.51, mean diff 0.45 in CI (pythia-410m,
# torch-spyre 9cffba78); cosine 0.9745, max diff 4.96 on a dev pod (gemma-4-26B
# decode-4, torch-spyre 64a818de). A fresh compile can move these numbers
# (torch-spyre#5561), so one run is not a ceiling.
MIN_COSINE = 0.95
MAX_LOGIT_DIFF = 5.0
MAX_MEAN_DIFF = 1.0


def _distance(h: torch.Tensor, a: torch.Tensor) -> tuple[float, float, float]:
    """Return (max abs diff, mean abs diff, cosine) between two logit vectors."""
    diff = (h - a).abs()
    cosine = torch.nn.functional.cosine_similarity(h, a, dim=0)
    return float(diff.max()), float(diff.mean()), float(cosine)


def validate_forced_tokens(forced_tokens: list[int] | None, num_decode: int) -> None:
    """Reject a forced list too short to cover prefill and every decode step."""
    if forced_tokens is None:
        return
    needed = num_decode + 1
    if len(forced_tokens) < needed:
        raise ValueError(
            f"forced_tokens has {len(forced_tokens)} entries; "
            f"need at least {needed} for {num_decode} decode steps"
        )


def next_fed_token(picked: int, step: int, forced_tokens: list[int] | None) -> int:
    """Token fed after ``step``: the reference token if forcing, else ``picked``."""
    return picked if forced_tokens is None else forced_tokens[step]


def compare_step(hf_logits: torch.Tensor, spyre_logits: torch.Tensor) -> dict[str, Any]:
    """Compare one step's logits; returns diffs, margins and the verdict."""
    vocab = min(hf_logits.shape[0], spyre_logits.shape[0])
    h = hf_logits[:vocab].float()
    a = spyre_logits[:vocab].float()
    hf_nan = bool(h.isnan().any())
    spyre_nan = bool(a.isnan().any())

    h_top1 = int(h.argmax())
    a_top1 = int(a.argmax())
    match = h_top1 == a_top1
    # How far each side ranks the other side's pick below its own (0 on match).
    hf_margin = float(h[h_top1] - h[a_top1])
    spyre_margin = float(a[a_top1] - a[h_top1])
    max_diff, mean_diff, cosine = _distance(h - h.mean(), a - a.mean())
    raw_max_diff, raw_mean_diff, raw_cosine = _distance(h, a)
    no_nan = not (hf_nan or spyre_nan)
    top1_ok = match and no_nan
    # NaN compares False, so a NaN step never passes these bounds.
    logits_ok = (
        no_nan
        and cosine >= MIN_COSINE
        and max_diff <= MAX_LOGIT_DIFF
        and mean_diff <= MAX_MEAN_DIFF
    )
    return {
        "top1_match": match,
        "top1_ok": top1_ok,
        "logits_ok": logits_ok,
        "step_ok": top1_ok and logits_ok,
        "hf_margin": hf_margin,
        "spyre_margin": spyre_margin,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine": cosine,
        "raw_max_diff": raw_max_diff,
        "raw_mean_diff": raw_mean_diff,
        "raw_cosine": raw_cosine,
        "hf_nan": hf_nan,
        "spyre_nan": spyre_nan,
    }
