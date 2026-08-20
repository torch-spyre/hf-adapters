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

"""Spyre edge cases for EOS-token handling in generation."""

import time

import pytest
from _generate_edge_case_helpers import (
    hf_reference_outputs,
    make_prompt_with_eos_inside,
    make_prompts,
)
from _shared import _setup, _teardown, run_eos_case
from model_registry import CAUSAL_PATHS

pytestmark = pytest.mark.model_harness("causal")


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_first_of_second_block_spyre(model_path: str) -> None:
    ok, detail = run_eos_case(model_path, "eos_first_of_second_block")
    assert ok, detail


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_first_token_spyre(model_path: str) -> None:
    ok, detail = run_eos_case(model_path, "eos_first_token")
    assert ok, detail


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_inside_prompt_spyre(model_path: str) -> None:
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        if tokenizer.eos_token_id is None:
            pytest.skip("tokenizer has no eos_token_id")
        eos_in_prompt = make_prompt_with_eos_inside(
            tokenizer, tokenizer.eos_token_id, target_tokens=12
        )
        eos_in_prompt_max_new = 64 + 8
        eos_in_prompt_refs = hf_reference_outputs(
            ref_model, tokenizer, [eos_in_prompt], eos_in_prompt_max_new
        )
        t0 = time.time()
        out = model.generate(
            tokenizer,
            [eos_in_prompt],
            max_new_tokens=eos_in_prompt_max_new,
            do_sample=False,
        )
        elapsed = time.time() - t0
        ok = eos_in_prompt_refs[0].strip() == out[0].strip()
        detail = "" if ok else f"hf={eos_in_prompt_refs!r} spyre={out!r}"
        print(f"  eos_inside_prompt: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, ref_model)


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_mid_block_spyre(model_path: str) -> None:
    ok, detail = run_eos_case(model_path, "eos_mid_block")
    assert ok, detail


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_on_last_step_spyre(model_path: str) -> None:
    ok, detail = run_eos_case(model_path, "eos_on_last_step")
    assert ok, detail


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_no_eos_runs_full_budget_spyre(model_path: str) -> None:
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    import torch

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
        assert ok, detail
    finally:
        _teardown(model, ref_model)
