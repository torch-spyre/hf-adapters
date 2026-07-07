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

"""Spyre edge case: ``zero_new_tokens`` (max_new_tokens=0 returns empty strings)."""

from __future__ import annotations

import time

import pytest
from _generate_edge_case_helpers import make_prompts
from edge_cases._shared import _setup, _teardown
from model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_zero_new_tokens_spyre(model_path: str) -> None:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        prompts = make_prompts(tokenizer, [5, 12])
        t0 = time.time()
        out = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
        elapsed = time.time() - t0
        ok = len(out) == len(prompts) and all(s == "" for s in out)
        detail = "" if ok else f"got={out!r}"
        print(f"  zero_new_tokens: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, None)
