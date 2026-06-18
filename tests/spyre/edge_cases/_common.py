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

"""Shared helpers for the per-case Spyre edge-case tests.

Process isolation between cases comes from the CI matrix: each (model, case)
combination is its own runner cell, which means a fresh Python process per
case and VFIO DMA mappings released between cases. No multiprocessing.Pool
needed — the matrix IS the boundary.

The HF reference forward MUST run BEFORE ``AutoSpyreModelForCausalLM.from_pretrained``
because Spyre prepare patches RMSNorm globally. Each helper here loads the HF
reference (when needed), captures whatever it needs, frees the model, and only
then loads the Spyre model.
"""

from __future__ import annotations

import gc
import time

import torch
from model_registry import CAUSAL_LM_MODELS


def load_spyre_model(info):
    """Load + prepare the adapter on Spyre. Caller is responsible for ``del``."""
    from hf_adapters import AutoSpyreModelForCausalLM

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(info["path"])
    print(f"  Spyre load+prepare: {time.time() - t0:.1f}s")
    return model


def load_hf_reference(info):
    """Load the stock HF causal-LM on CPU. Caller is responsible for ``del``."""
    from transformers import AutoModelForCausalLM

    ref_dtype = torch.float32 if info.get("dtype") == "float32" else torch.float16
    ref = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=ref_dtype, device_map="cpu"
    )
    ref.eval()
    ref.requires_grad_(False)
    return ref


def free(*objs):
    """Release model references and run gc — matches the script's per-case discipline."""
    for obj in objs:
        del obj
    gc.collect()


def model_info(model_key):
    """Return the registry entry for ``model_key``, raising on misses."""
    if model_key not in CAUSAL_LM_MODELS:
        raise KeyError(
            f"unknown model_key {model_key!r}; "
            f"options: {list(CAUSAL_LM_MODELS.keys())}"
        )
    return CAUSAL_LM_MODELS[model_key]


def run_greedy_case(model_key, targets, max_new_tokens):
    """Run a greedy case: HF reference (CPU) BEFORE Spyre load, then assert match.

    ``targets`` is the list of approximate prompt token lengths
    (``len(targets) == batch_size``). ``max_new_tokens`` drives the regime.
    """
    from _generate_edge_case_helpers import hf_reference_outputs, make_prompts
    from transformers import AutoTokenizer

    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    prompts = make_prompts(tokenizer, targets)

    # Capture HF reference BEFORE Spyre prepare (RMSNorm patch is global).
    ref = load_hf_reference(info)
    try:
        hf_outputs = hf_reference_outputs(ref, tokenizer, prompts, max_new_tokens)
    finally:
        free(ref)

    model = load_spyre_model(info)
    try:
        spyre_outputs = model.generate(
            tokenizer, prompts, max_new_tokens=max_new_tokens, do_sample=False
        )
    finally:
        free(model)

    assert len(spyre_outputs) == len(hf_outputs), (
        f"{model_key}: adapter returned {len(spyre_outputs)} outputs, "
        f"expected {len(hf_outputs)}"
    )
    for i, (hf_out, sp_out) in enumerate(zip(hf_outputs, spyre_outputs)):
        assert hf_out.strip() == sp_out.strip(), (
            f"{model_key} prompt[{i}] (target={targets[i]}, "
            f"max_new={max_new_tokens}):\n"
            f"  HF:    {hf_out!r}\n"
            f"  Spyre: {sp_out!r}"
        )


def run_forced_eos_case(model_key, eos_offsets, max_new_tokens):
    """Run a forced-EOS case: pick a stop token, run, assert truncation matches."""
    from _generate_edge_case_helpers import (
        forced_eos_expected,
        greedy_token_ids,
        make_prompts,
        pick_forced_eos_id,
    )
    from transformers import AutoTokenizer

    info = model_info(model_key)
    print(f"\n  {info['name']}: {info['path']}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    batch_size = len(eos_offsets)
    prompts = make_prompts(tokenizer, [5] * batch_size)

    ref = load_hf_reference(info)
    try:
        per_prompt_ids = [
            greedy_token_ids(ref, tokenizer, p, max_new_tokens) for p in prompts
        ]
    finally:
        free(ref)

    eos_id = pick_forced_eos_id(per_prompt_ids, eos_offsets)
    if eos_id is None:
        import pytest

        pytest.skip(
            f"{model_key}: no shared token at offsets {eos_offsets} that is "
            "absent from earlier positions; cannot force a clean batched EOS"
        )

    expected = forced_eos_expected(per_prompt_ids, eos_offsets, tokenizer)

    model = load_spyre_model(info)
    try:
        spyre_outputs = model.generate(
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
        )
    finally:
        free(model)

    assert (
        len(spyre_outputs) == batch_size
    ), f"{model_key}: got {len(spyre_outputs)} outputs, expected {batch_size}"
    for b, (exp, got) in enumerate(zip(expected, spyre_outputs)):
        assert exp.strip() == got.strip(), (
            f"{model_key} row[{b}] (eos_offset={eos_offsets[b]}, "
            f"forced_eos_id={eos_id}):\n"
            f"  expected: {exp!r}\n"
            f"  got:      {got!r}"
        )
