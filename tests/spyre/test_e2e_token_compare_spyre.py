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
E2E token-level comparison: HF generation (CPU) vs adapter generation (Spyre).

For each model, runs prefill + 4 greedy decode steps on both CPU (stock HF)
and Spyre (adapter), comparing logits and greedy tokens at each step.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k qwen3
"""

from typing import Any, Callable

import pytest
import torch
from transformers import PreTrainedModel

from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import (
    encode_prompts,
    generate,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.model_registry import (
    CAUSAL_PATHS,
    NON_BLOCKING_CAUSAL_MODELS,
    REMOTE_CODE_PATHS,
    xfail_non_blocking,
)

pytestmark = pytest.mark.model_harness("causal")


def hf_greedy_steps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Use stock generation so both sides apply the same token-selection rules."""
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),  # One unpadded prompt.
            max_new_tokens=num_decode + 1,
            do_sample=False,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
    return _generation_results(output, input_ids.shape[1], num_decode + 1)


def adapter_greedy_steps(
    run_forward_fn: Callable,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    num_decode: int = 4,
) -> list[dict[str, Any]]:
    """Use the production generation loop once, with fresh request caches."""
    with torch.no_grad():
        output = generate(
            run_forward_fn,
            model,
            input_ids,
            max_new_tokens=num_decode + 1,
            do_sample=False,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
    return _generation_results(output, input_ids.shape[1], num_decode + 1)


def _generation_results(output, prompt_length, steps):
    assert len(output.logits) == steps
    assert all(torch.isfinite(logits).all().item() for logits in output.logits)
    return [
        {
            "step": step,
            "logits": logits[0].float(),
            "token": output.sequences[0, prompt_length + step].item(),
        }
        for step, logits in enumerate(output.logits)
    ]


def _compare_results(
    hf_results: list[dict[str, Any]],
    adapter_results: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
) -> list[dict[str, Any]]:
    """Compare HF vs adapter results, return comparison rows."""
    rows = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h_logits = hf_r["logits"]
        a_logits = ad_r["logits"]

        min_vocab = min(h_logits.shape[0], a_logits.shape[0])
        h = h_logits[:min_vocab]
        a = a_logits[:min_vocab]

        diff = (h - a).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        h_top1 = h.argmax().item()
        a_top1 = a.argmax().item()
        match = h_top1 == a_top1

        # Token-selection rules can change the raw-logit winner. Check the
        # returned token too, so equal raw logits cannot hide different choices.
        assert hf_r["token"] == ad_r["token"], (
            f"{model_name} step {step}: HF token {hf_r['token']} "
            f"!= Spyre token {ad_r['token']}"
        )

        step_label = "prefill" if step == 0 else f"decode-{step}"
        h_str = tokenizer.decode([hf_r["token"]])
        a_str = tokenizer.decode([ad_r["token"]])
        rows.append(
            {
                "model": model_name,
                "step": step_label,
                "hf_token": hf_r["token"],
                "hf_str": h_str,
                "spyre_token": ad_r["token"],
                "spyre_str": a_str,
                "top1_match": match,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "hf_nan": h_logits.isnan().any().item(),
                "spyre_nan": a_logits.isnan().any().item(),
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per step."""
    print("\n## E2E Token Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(
        "| Model | Step | HF Token | Spyre Token | Match "
        "| Max Diff | Mean Diff | HF NaN | Spyre NaN |"
    )
    print(
        "|-------|------|----------|-------------|-------"
        "|----------|-----------|--------|-----------|"
    )
    for r in rows:
        match = "OK" if r["top1_match"] else "FAIL"
        hf_col = f"{r['hf_token']:>5} {r['hf_str']!r}"
        sp_col = f"{r['spyre_token']:>5} {r['spyre_str']!r}"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {r['model']} | {r['step']} | {hf_col} | {sp_col} "
            f"| {match} | {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {hn} | {sn} |"
        )


def _run_model_test(
    model_path: str, num_decode: int = 4, trust_remote_code: bool | None = None
) -> list[dict[str, Any]]:
    """Full comparison for one model. Returns the list of comparison rows."""
    from transformers import AutoTokenizer

    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    adapter = resolve_adapter_module_for_test(
        model_path, trust_remote_code=trust_remote_code
    )

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model = load_ref_model(
        model_path=model_path, adapter_mod=adapter, trust_remote_code=trust_remote_code
    )

    prompt = "The capital of France is"
    # Tokenize following the model's canonical scheme (chat template for
    # instruct models, plain post-processing for base models). The same IDs feed
    # the HF reference and Spyre adapter, keeping the comparison symmetric.
    encoded = encode_prompts(tokenizer, prompt)
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    # Use bf16/fp16 dtype, requested by the registry or based on the model config.
    # (Spyre does not support float32, so float32 entries will use fp16.)
    spyre_dtype = dtype_for_model_path(
        model_path,
        target_device="spyre",
        trust_remote_code=trust_remote_code,
    )
    move_model_to_spyre(model=model, module=adapter, dtype=spyre_dtype)
    print("  Running adapter on Spyre ...")
    adapter_results = adapter_greedy_steps(
        getattr(adapter, "_run_prefill_next_logits", adapter._run_forward),
        model,
        input_ids,
        num_decode=num_decode,
    )

    return _compare_results(hf_results, adapter_results, tokenizer, model_path)


def token_compare_spyre(
    model_path: str,
    trust_remote_code: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _run_model_test(model_path, trust_remote_code=trust_remote_code)
    mismatches = [r for r in rows if not r["top1_match"]]
    return mismatches, rows


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(CAUSAL_PATHS, table=NON_BLOCKING_CAUSAL_MODELS)
)
def test_e2e_token_compare_spyre(
    model_path: str, trust_remote_code: bool | None
) -> None:
    mismatches, rows = token_compare_spyre(
        model_path, trust_remote_code=trust_remote_code
    )
    _print_table(rows)
    n_match = sum(1 for r in rows if r["top1_match"])
    print(f"\nTop-1 agreement: {n_match}/{len(rows)} steps")
    assert not mismatches, mismatches
