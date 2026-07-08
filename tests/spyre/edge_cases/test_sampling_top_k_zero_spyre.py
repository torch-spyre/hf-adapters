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

"""Spyre edge case: ``sampling_top_k_zero`` (top_k=0 -> deterministic)."""

from __future__ import annotations

import time

import pytest
import torch
from _generate_edge_case_helpers import SAMPLING_MAX_NEW, SAMPLING_TARGETS, make_prompts
from edge_cases._shared import _setup, _teardown
from model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_sampling_top_k_zero_spyre(model_path: str) -> None:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
        kwargs = dict(do_sample=True, temperature=1.0, top_k=0)
        t0 = time.time()
        torch.manual_seed(2024)
        out1 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        torch.manual_seed(2024)
        out2 = model.generate(
            tokenizer, sampling_prompts, max_new_tokens=SAMPLING_MAX_NEW, **kwargs
        )
        elapsed = time.time() - t0
        ok = out1 == out2 and all(s for s in out1)
        detail = "" if ok else f"out1={out1!r} out2={out2!r}"
        print(f"  sampling_top_k_zero: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, None)
