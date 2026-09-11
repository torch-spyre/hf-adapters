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

"""Spyre edge case: ``sampling_determinism`` (same seed -> same output)."""

from __future__ import annotations

import time

import pytest
import torch

from tests._generate_edge_case_helpers import (
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
    make_prompts,
)
from tests.conftest import encode_generation_inputs
from tests.model_registry import CAUSAL_PATHS
from tests.spyre.edge_cases._shared import _setup, _teardown

pytestmark = pytest.mark.model_harness("causal")


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_sampling_determinism_spyre(model_path: str) -> None:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
        encoded = encode_generation_inputs(tokenizer, sampling_prompts)
        t0 = time.time()
        torch.manual_seed(1234)
        a1 = model.generate(
            **encoded,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        torch.manual_seed(1234)
        a2 = model.generate(
            **encoded,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        torch.manual_seed(9999)
        b = model.generate(
            **encoded,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        elapsed = time.time() - t0
        ok = torch.equal(a1, a2) and not torch.equal(a1, b)
        detail = "" if ok else f"a1={a1!r} a2={a2!r} b={b!r}"
        print(f"  sampling_determinism: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, None)


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_sampling_top_k_zero_spyre(model_path: str) -> None:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
        encoded = encode_generation_inputs(tokenizer, sampling_prompts)
        kwargs = dict(do_sample=True, temperature=1.0, top_k=0)
        t0 = time.time()
        torch.manual_seed(2024)
        out1 = model.generate(
            **encoded,
            max_new_tokens=SAMPLING_MAX_NEW,
            **kwargs,
        )
        torch.manual_seed(2024)
        out2 = model.generate(
            **encoded,
            max_new_tokens=SAMPLING_MAX_NEW,
            **kwargs,
        )
        elapsed = time.time() - t0
        ok = torch.equal(out1, out2) and out1.shape[1] > encoded["input_ids"].shape[1]
        detail = "" if ok else f"out1={out1!r} out2={out2!r}"
        print(f"  sampling_top_k_zero: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, None)
