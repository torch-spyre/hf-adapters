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
E2E token comparison for Granite's EXPERIMENTAL hierarchical whole-forward compile.

Same shape as ``test_e2e_token_compare_spyre.py`` -- HF stock forward on CPU vs
adapter forward on Spyre, comparing logits and greedy tokens at each step -- but
with ``hier_compile=True``, which replaces per-layer ``torch.compile`` with a
single compiled whole-forward whose decoder layers are ``nested_compile_region``
blocks (traced once, reused across all N layers).

That path is opt-in, so no other Spyre suite exercises it: with the flag off
(every existing Granite test) ``prepare_for_spyre`` builds per-layer compiled
blocks and attaches no ``_spyre_run_forward``. This file is its only on-hardware
coverage, and it reuses token-compare's helpers so the two report identically.

Comparing against stock HF (rather than against the default compile path) sets
the higher bar: it catches numeric drift that both Spyre paths could share, and
it is the on-hardware regression guard for the closure-capture bug in
``hf_common._shared_region_block`` -- if the shared region froze layer 0's
weights, every later layer would compute with the wrong weights and top-1 would
diverge from HF. The CPU unit test can only show that with toy ``nn.Linear``
layers.

Covers Granite 3.3 2B and 8B, the two checkpoints ``hf_granite.py`` serves.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_hier_compile_spyre.py
    pytest -s -vvv tests/spyre/test_hier_compile_spyre.py -k 2b
"""

from typing import Any

import pytest

from hf_adapters.auto_spyre_model import _resolve_run_forward_fn, dtype_for_model_path
from hf_adapters.hf_common import encode_prompts, move_model_to_spyre
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.model_registry import CAUSAL_LM_MODELS
from tests.spyre.test_e2e_token_compare_spyre import (
    _compare_results,
    _print_table,
    adapter_greedy_steps,
    hf_greedy_steps,
)

# Required by tests/conftest.py::pytest_generate_tests, which raises
# UsageError("unknown model harness: None") on any test taking ``model_path``
# without this marker as soon as --model-path is passed (the Makefile's `tests`
# target passes it for every tier).
pytestmark = pytest.mark.model_harness("causal")

# Granite 3.3 2B and 8B -- the only models whose prepare_for_spyre understands
# hier_compile. Resolved from the registry by key so a path change there cannot
# leave this test silently running against a stale path.
HIER_COMPILE_PATHS: list[str] = [
    CAUSAL_LM_MODELS[key]["path"] for key in ("granite2b", "granite8b")
]


def _run_hier_model_test(model_path: str, num_decode: int = 4) -> list[dict[str, Any]]:
    """HF-vs-Spyre comparison for one model under hier_compile. Returns rows.

    Mirrors ``test_e2e_token_compare_spyre._run_model_test``; the differences are
    the ``hier_compile=True`` on ``move_model_to_spyre`` and picking the forward
    callable via ``_resolve_run_forward_fn`` instead of hardcoding
    ``adapter._run_forward``.
    """
    from transformers import AutoTokenizer

    adapter = resolve_adapter_module_for_test(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path} (hier_compile=True)")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(model_path=model_path, adapter_mod=adapter)

    prompt = "The capital of France is"
    encoded = encode_prompts(tokenizer, prompt)
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    # HF reference runs on CPU BEFORE prepare_for_spyre, which mutates the
    # decoder layers in place (and, for some families, patches RMSNorm globally).
    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    spyre_dtype = dtype_for_model_path(model_path, target_device="spyre")
    move_model_to_spyre(
        model=model, module=adapter, dtype=spyre_dtype, hier_compile=True
    )

    # Fail loudly if the opt-in did not engage: without this, a flag that stopped
    # being threaded through would silently fall back to the per-layer path and
    # this test would duplicate token-compare instead of covering anything new.
    assert hasattr(model, "_spyre_run_forward"), (
        "hier_compile=True did not attach _spyre_run_forward -- the compiled "
        "whole-forward path is not the one under test"
    )

    # The compiled whole-forward consumes selected_freqs, not position_ids, so it
    # needs the same shim generate() uses rather than the raw adapter._run_forward.
    run_forward_fn = _resolve_run_forward_fn(model, adapter._run_forward)

    print("  Running adapter on Spyre (compiled whole-forward) ...")
    adapter_results = adapter_greedy_steps(
        run_forward_fn,
        model,
        input_ids,
        num_decode=num_decode,
    )

    return _compare_results(hf_results, adapter_results, tokenizer, model_path)


@pytest.mark.parametrize("model_path", HIER_COMPILE_PATHS)
def test_hier_compile_token_compare_spyre(model_path: str) -> None:
    rows = _run_hier_model_test(model_path)
    _print_table(rows)
    mismatches = [r for r in rows if not r["top1_match"]]
    n_match = sum(1 for r in rows if r["top1_match"])
    print(f"\nTop-1 agreement: {n_match}/{len(rows)} steps")
    assert not mismatches, mismatches
