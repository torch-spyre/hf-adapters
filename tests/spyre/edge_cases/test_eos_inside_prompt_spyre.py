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

"""Spyre edge case: ``eos_inside_prompt`` (EOS id inside prompt does not stop gen)."""

from __future__ import annotations

import time

import pytest

from tests._generate_edge_case_helpers import (
    hf_reference_outputs,
    make_prompt_with_eos_inside,
)
from tests.conftest import encode_generation_inputs
from tests.model_registry import CAUSAL_PATHS
from tests.spyre.edge_cases._shared import _setup, _teardown


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_eos_inside_prompt_spyre(model_path: str) -> None:
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        if tokenizer.eos_token_id is None:
            import pytest

            pytest.skip("tokenizer has no eos_token_id")
        eos_in_prompt = make_prompt_with_eos_inside(
            tokenizer, tokenizer.eos_token_id, target_tokens=12
        )
        eos_in_prompt_max_new = 64 + 8
        eos_in_prompt_refs = hf_reference_outputs(
            ref_model, tokenizer, [eos_in_prompt], eos_in_prompt_max_new
        )
        encoded = encode_generation_inputs(tokenizer, [eos_in_prompt])
        t0 = time.time()
        out = model.generate(
            **encoded,
            max_new_tokens=eos_in_prompt_max_new,
            do_sample=False,
        )
        spyre_output = tokenizer.decode(
            out[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        elapsed = time.time() - t0
        ok = eos_in_prompt_refs[0].strip() == spyre_output.strip()
        detail = "" if ok else f"hf={eos_in_prompt_refs!r} spyre={out!r}"
        print(f"  eos_inside_prompt: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        assert ok, detail
    finally:
        _teardown(model, ref_model)
