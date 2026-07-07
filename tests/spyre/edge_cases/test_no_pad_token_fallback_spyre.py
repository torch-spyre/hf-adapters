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

"""Spyre edge case: ``no_pad_token_fallback`` (pad_token=None hits eos fallback)."""

from __future__ import annotations

import time

import pytest
from _generate_edge_case_helpers import (
    NoPadTokenizer,
    hf_reference_outputs,
    make_prompts,
)
from edge_cases._shared import _setup, _teardown
from model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_no_pad_token_fallback_spyre(model_path: str) -> None:
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        no_pad_prompts = make_prompts(tokenizer, [5, 12])
        no_pad_max_new = 16
        no_pad_refs = hf_reference_outputs(
            ref_model, tokenizer, no_pad_prompts, no_pad_max_new
        )
        wrapped = NoPadTokenizer(tokenizer)
        t0 = time.time()
        out = model.generate(
            wrapped, no_pad_prompts, max_new_tokens=no_pad_max_new, do_sample=False
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_pad_refs, out))
        detail = "" if ok else f"hf={no_pad_refs!r} spyre={out!r}"
        print(f"  no_pad_token_fallback: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, ref_model)
