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

"""
Shared scaffolding for the Spyre edge-case test files.

Each per-case file under this directory imports ``run_*`` from here. The
runners load the Spyre model once per call (i.e. once per pytest item), run a
single edge case, and return ``(ok: bool, detail: str)``. Tests assert ``ok``.

Cases that need an HF reference forward capture it on CPU **before** the
``prepare_for_spyre`` move (the RMSNorm patch is global), mirroring the
ordering discipline from the previous one-process driver.
"""

from __future__ import annotations

import gc
import time

from _generate_edge_case_helpers import (
    CASES,
    EOS_CASES,
    forced_eos_expected,
    greedy_token_ids,
    hf_reference_outputs,
    make_prompts,
)
from torch.nn import Module
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
)

from hf_adapters import AutoSpyreModelForCausalLM
from tests.conftest import load_ref_model, resolve_adapter_module_for_test


def _load_spyre_model(model_path: str) -> Module:
    print(f"  Loading {model_path} on Spyre ...")
    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path)
    print(f"  Spyre load+prepare: {time.time() - t0:.1f}s")
    return model


def _setup(
    model_path: str,
    need_ref: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter = resolve_adapter_module_for_test(model_path)

    ref_model = load_ref_model(model_path, adapter_mod=adapter) if need_ref else None
    spyre_model = _load_spyre_model(model_path)
    return model_path, tokenizer, ref_model, spyre_model


def _teardown(
    spyre_model: AutoSpyreModelForCausalLM,
    ref_model: PreTrainedModel | None,
) -> None:
    del spyre_model
    if ref_model is not None:
        del ref_model
    gc.collect()


def run_greedy_case(model_path: str, case_id: str) -> tuple[bool, str]:
    """Greedy-generate case: HF reference == Spyre output, per row."""
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        targets, max_new = CASES[case_id]
        prompts = make_prompts(tokenizer, targets)
        hf_outputs = hf_reference_outputs(ref_model, tokenizer, prompts, max_new)
        t0 = time.time()
        spyre_outputs = model.generate(
            tokenizer, prompts, max_new_tokens=max_new, do_sample=False
        )
        elapsed = time.time() - t0
        ok = all(hf.strip() == sp.strip() for hf, sp in zip(hf_outputs, spyre_outputs))
        detail = "" if ok else f"hf={hf_outputs!r} spyre={spyre_outputs!r}"
        print(f"  {case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok, detail
    finally:
        _teardown(model, ref_model)


def run_eos_case(model_path: str, case_id: str) -> tuple[bool, str]:
    """Forced-EOS case: shared eos_token_id stops each row at its requested offset."""
    info, tokenizer, ref_model, model = _setup(model_path, need_ref=True)
    try:
        eos_offsets, max_new = EOS_CASES[case_id]
        batch_size = len(eos_offsets)
        prompts = make_prompts(tokenizer, [5] * batch_size)
        per_prompt_ids = [
            greedy_token_ids(ref_model, tokenizer, p, max_new) for p in prompts
        ]
        from _generate_edge_case_helpers import pick_forced_eos_id

        eos_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
        if eos_id is None:
            import pytest

            pytest.skip("no clean shared eos token at requested offsets")
        expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)
        t0 = time.time()
        out = model.generate(
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
        )
        elapsed = time.time() - t0
        ok = all(e.strip() == g.strip() for e, g in zip(expected, out))
        detail = "" if ok else f"expected={expected!r} got={out!r}"
        print(f"  forced_eos:{case_id}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok, detail
    finally:
        _teardown(model, ref_model)
