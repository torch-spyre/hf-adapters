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

"""Spyre edge case: ``no_eos_runs_full_budget`` (eos_token_id=None crosses block)."""

from __future__ import annotations

import time

import pytest
import torch
from _generate_edge_case_helpers import make_prompts
from edge_cases._shared import _setup, _teardown
from model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_no_eos_runs_full_budget_spyre(model_path: str) -> None:
    ok, detail = _run_no_eos(model_path)
    assert ok, detail


def _run_no_eos(model_path: str) -> tuple[bool, str]:
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        no_eos_prompts = make_prompts(tokenizer, [5, 12])
        no_eos_max_new = 64 + 7
        no_eos_refs = []
        for prompt in no_eos_prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = ref_model.generate(
                    **encoded,
                    max_new_tokens=no_eos_max_new,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                )
            new_ids = out[0][encoded["input_ids"].shape[1] :]
            no_eos_refs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        t0 = time.time()
        out = model.generate(
            tokenizer,
            no_eos_prompts,
            max_new_tokens=no_eos_max_new,
            do_sample=False,
            eos_token_id=None,
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(no_eos_refs, out))
        detail = "" if ok else f"hf={no_eos_refs!r} spyre={out!r}"
        print(f"  no_eos_runs_full_budget: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok, detail
    finally:
        _teardown(model, ref_model)
