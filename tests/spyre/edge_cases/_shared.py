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

import torch
from _generate_edge_case_helpers import (
    CASES,
    EOS_CASES,
    SAMPLING_KWARGS,
    SAMPLING_MAX_NEW,
    SAMPLING_TARGETS,
    NoPadTokenizer,
    forced_eos_expected,
    greedy_token_ids,
    hf_reference_outputs,
    make_prompt_with_eos_inside,
    make_prompts,
)
from torch.nn import Module
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
)

from hf_adapters import AutoSpyreModelForCausalLM
from hf_adapters.auto_spyre_model import resolve_adapter_module
from tests.conftest import load_ref_model


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
    adapter = resolve_adapter_module(model_path)

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


def run_zero_new_tokens(model_path: str) -> tuple[bool, str]:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        prompts = make_prompts(tokenizer, [5, 12])
        t0 = time.time()
        out = model.generate(tokenizer, prompts, max_new_tokens=0, do_sample=False)
        elapsed = time.time() - t0
        ok = len(out) == len(prompts) and all(s == "" for s in out)
        detail = "" if ok else f"got={out!r}"
        print(f"  zero_new_tokens: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok, detail
    finally:
        _teardown(model, None)


def run_sampling_determinism(model_path: str) -> tuple[bool, str]:
    info, tokenizer, _, model = _setup(model_path, need_ref=False)
    try:
        sampling_prompts = make_prompts(tokenizer, SAMPLING_TARGETS)
        t0 = time.time()
        torch.manual_seed(1234)
        a1 = model.generate(
            tokenizer,
            sampling_prompts,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        torch.manual_seed(1234)
        a2 = model.generate(
            tokenizer,
            sampling_prompts,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        torch.manual_seed(9999)
        b = model.generate(
            tokenizer,
            sampling_prompts,
            max_new_tokens=SAMPLING_MAX_NEW,
            **SAMPLING_KWARGS,
        )
        elapsed = time.time() - t0
        ok = a1 == a2 and a1 != b
        detail = "" if ok else f"a1={a1!r} a2={a2!r} b={b!r}"
        print(f"  sampling_determinism: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok, detail
    finally:
        _teardown(model, None)


def run_no_eos(model_path: str) -> tuple[bool, str]:
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


def run_no_pad(model_path: str) -> tuple[bool, str]:
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
        return ok, detail
    finally:
        _teardown(model, ref_model)


def run_top_k_zero(model_path: str) -> tuple[bool, str]:
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
        return ok, detail
    finally:
        _teardown(model, None)


def run_eos_inside_prompt(model_path: str) -> tuple[bool, str]:
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
        return ok, detail
    finally:
        _teardown(model, ref_model)
