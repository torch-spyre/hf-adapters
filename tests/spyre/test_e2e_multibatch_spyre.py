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

"""E2E ``batch_size > 1`` generate() on Spyre — batching-invariance regression.

This exercises the part of the KV-cache path that a ``batch_size == 1`` run
never touches: the indirect scatter's *batch* dimension. The scatter writes into
a layout-pinned cache whose device layout places the batch dim between the
head-dim sticks and the ``elems_per_stick`` tail; with two left-padded prompts of
different lengths, each sequence carries its own ``prompt_offset`` and its writes
must land in its own batch row. A batch/row mix-up in the pinned layout would
corrupt one sequence's cache while leaving element counts correct — invisible at
batch 1 and invisible on CPU (which has no device layout). This class of bug is
exactly what regressed on Qwen3-0.6B (device saturated the ``-inf`` attention
mask fill; the most-left-padded row decoded garbage).

**Oracle: Spyre-batched vs Spyre-single, NOT Spyre vs HF-CPU.**
The precise question here is "does batching a prompt alongside others (adding
left-pad and a shared cache) change its output versus running it alone?" — i.e.
``spyre_batched[i] == spyre_single(prompt_i)``. Comparing against HF-CPU instead
would fold in an *orthogonal* concern — benign Spyre-vs-HF greedy divergence,
where fp16-on-device flips the argmax on a knife's-edge continuation. That was
observed here: granite-3.3-2b (an instruct model on a bare completion prompt)
decodes ``'Paris.\\n\\nStep 1'`` on Spyre vs ``'Paris.\\n\\nThe capital of'`` on
HF-CPU — identically at batch=1 with zero padding, so it has nothing to do with
batching. Spyre-vs-HF fidelity is already covered by
``test_e2e_token_compare_spyre`` and the ``edge_cases`` suite; this test isolates
batching. The self-consistency oracle is also model-agnostic and still catches
the original corruption (single row = clean, batched row = garbled → mismatch).
HF-CPU output is printed as an informational column but is not asserted on.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_multibatch_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_multibatch_spyre.py -k Qwen3-0.6B
    pytest -s -vvv tests/spyre/test_e2e_multibatch_spyre.py \
        --model-path ibm-granite/granite-3.3-2b-instruct
"""

import gc
import sys
from typing import Any

import pytest
from transformers import AutoTokenizer

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import move_model_to_spyre
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.cpu._generate_helpers import (
    MAX_NEW_TOKENS,
    PROMPTS,
    hf_reference_outputs,
)
from tests.model_registry import (
    CAUSAL_PATHS,
    NON_BLOCKING_CAUSAL_MODELS,
    xfail_non_blocking,
)

pytestmark = pytest.mark.model_harness("causal")


def _print_table(model_path: str, rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per prompt in the batch.

    The asserted column is ``Batch OK`` (batched == single). ``HF (CPU)`` is
    shown for context only.
    """
    print(f"\n## E2E Multi-batch Generate (batching-invariance) — {model_path}\n")
    print("| # | Prompt | Spyre batched | Spyre single | Batch OK | HF (CPU, info) |")
    print("|---|--------|---------------|--------------|----------|----------------|")
    for r in rows:
        ok = "OK" if r["match"] else "FAIL"
        print(
            f"| {r['i']} | {r['prompt']!r} | {r['batched']!r} "
            f"| {r['single']!r} | {ok} | {r['hf']!r} |"
        )


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(CAUSAL_PATHS, table=NON_BLOCKING_CAUSAL_MODELS)
)
def test_e2e_multibatch_spyre(model_path: str) -> None:
    # ``generate`` is looked up off the live (possibly DEVICE-patched) module,
    # matching tests/cpu/test_generate_cpu.py.
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    adapter_mod = resolve_adapter_module_for_test(model_path)

    assert len(PROMPTS) > 1, "multi-batch test needs batch_size > 1"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}  (batch_size={len(PROMPTS)})")
    print(f"{'=' * 70}")

    # HF reference (per-prompt) — informational only. Run BEFORE prepare_for_spyre
    # patches RMSNorm globally. Loaded on CPU via load_ref_model, then discarded.
    model = load_ref_model(model_path, adapter_mod)
    print("  Running HF reference on CPU (per-prompt, informational) ...")
    hf_outputs = hf_reference_outputs(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)
    del model
    gc.collect()

    # One Spyre model, reused for the batched run and each single run (generate
    # allocates a fresh KV cache per call, so runs do not contaminate each other).
    spyre_dtype = torch_dtype_for_model_path(model_path)
    model = load_ref_model(model_path, adapter_mod)
    move_model_to_spyre(model=model, module=adapter_mod, dtype=spyre_dtype)

    # Bind the model as a default arg so the closure holds a local, not a free
    # variable: ``model`` is ``del``-ed below, and ruff resolves closure free
    # vars against the scope's final state (unbound) and flags F821.
    def spyre_generate(prompts: list[str], _model=model) -> list[str]:
        return hf_common_mod.generate(
            adapter_mod._run_forward,
            _model,
            tokenizer,
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Path under test: batched (left-padded, shared cache).
    print("  Running adapter batched generate() on Spyre ...")
    batched = spyre_generate(list(PROMPTS))
    # Oracle: each prompt alone (batch=1, no left-pad, private cache).
    print("  Running adapter single-prompt generate() on Spyre (oracle) ...")
    singles = [spyre_generate([p])[0] for p in PROMPTS]
    del model
    gc.collect()

    rows = [
        {
            "i": i,
            "prompt": prompt,
            "hf": hf_out.strip(),
            "batched": b_out.strip(),
            "single": s_out.strip(),
            "match": b_out.strip() == s_out.strip(),
        }
        for i, (prompt, hf_out, b_out, s_out) in enumerate(
            zip(PROMPTS, hf_outputs, batched, singles)
        )
    ]
    _print_table(model_path, rows)

    mismatches = [
        {
            "i": r["i"],
            "prompt": r["prompt"],
            "batched": r["batched"],
            "single": r["single"],
        }
        for r in rows
        if not r["match"]
    ]
    assert not mismatches, (
        "batched output diverged from single-prompt output (batch/row "
        f"corruption): {mismatches}"
    )
