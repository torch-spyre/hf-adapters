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
Multi-batch generate() test: verify that ``hf_common.generate()`` produces
correct per-sequence outputs when called with batch_size > 1.

For each registered model, ``test_multibatch[<key>]`` runs the same prompts
through stock HF ``generate(do_sample=False)`` per-prompt, then through the
adapter's batched ``generate()``, and asserts the decoded text matches.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest.
"""

import gc
import sys

import pytest
from transformers import AutoTokenizer

from tests.conftest import (
    encode_generation_inputs,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu._generate_helpers import (
    MAX_NEW_TOKENS,
    PROMPTS,
    hf_reference_outputs,
)
from tests.cpu.conftest import _set_rope_dtype, _unwrap_compiled_blocks
from tests.model_registry import (
    CAUSAL_PATHS,
    NON_BLOCKING_CAUSAL_MODELS,
    REMOTE_CODE_PATHS,
    xfail_non_blocking,
)

pytestmark = pytest.mark.model_harness("causal")


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(CAUSAL_PATHS, table=NON_BLOCKING_CAUSAL_MODELS)
)
def test_multibatch(model_path: str, trust_remote_code: bool | None) -> None:
    from hf_adapters.auto_spyre_model import dtype_for_model_path

    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    adapter_mod = resolve_adapter_module_for_test(
        model_path, trust_remote_code=trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # HF reference (per-prompt, BEFORE patching for cleanliness)
    model = load_ref_model(model_path, adapter_mod, trust_remote_code=trust_remote_code)
    hf_outputs = hf_reference_outputs(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)
    del model
    gc.collect()

    # Adapter batched generate
    encoded = encode_generation_inputs(tokenizer, PROMPTS)
    model = load_ref_model(model_path, adapter_mod, trust_remote_code=trust_remote_code)
    adapter_mod.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)
    dtype = dtype_for_model_path(
        model_path,
        target_device="cpu",
        trust_remote_code=trust_remote_code,
    )
    _set_rope_dtype(model, dtype)
    sequences = hf_common_mod.generate(
        getattr(adapter_mod, "_run_prefill_next_logits", adapter_mod._run_forward),
        model,
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    adapter_outputs = tokenizer.batch_decode(
        sequences[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    del model
    gc.collect()

    for i, (prompt, hf_out, adapter_out) in enumerate(
        zip(PROMPTS, hf_outputs, adapter_outputs)
    ):
        assert (
            hf_out.strip() == adapter_out.strip()
        ), f"prompt[{i}] {prompt!r}: HF {hf_out!r} != adapter {adapter_out!r}"


@pytest.mark.parametrize("chunk_size", [None, 64])
@pytest.mark.parametrize("custom_prefill", [False, True])
@pytest.mark.parametrize("adapter_name", ["hf_gemma4", "hf_gemma4_moe"])
def test_generation_row_optimizations(
    monkeypatch, chunk_size, custom_prefill, adapter_name
):
    """Check the adapter binding, one-row head, and custom callback precedence."""
    from types import SimpleNamespace

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from hf_adapters import auto_spyre_model, hf_gemma4, hf_gemma4_moe

    adapter = {"hf_gemma4": hf_gemma4, "hf_gemma4_moe": hf_gemma4_moe}[adapter_name]

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=11,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    ).eval()
    model.config.final_logit_softcapping = 3.0
    # Include a padded vocabulary entry: generation must still crop it.
    model.lm_head = torch.nn.Linear(4, 12, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.arange(48).reshape(12, 4) % 5)
    head_rows, cache_writes = [], []

    def backbone(model, ids, positions, mask, keys, values, cache_index):
        hidden = (ids[..., None] + torch.arange(4)).float() % 7
        for cache in keys + values:
            cache.index_copy_(2, cache_index, hidden[:, None])
        cache_writes.append(cache_index.tolist())
        return hidden

    monkeypatch.setattr(hf_gemma4, "_run_backbone_forward", backbone)
    model.lm_head.register_forward_pre_hook(
        lambda module, args: head_rows.append(args[0].shape[1])
    )

    def prefill(
        *,
        model,
        input_ids,
        position_ids,
        attention_mask,
        key_caches,
        value_caches,
        cache_index,
    ):
        return hf_gemma4._run_forward(
            model,
            input_ids,
            position_ids,
            attention_mask,
            key_caches,
            value_caches,
            cache_index,
        )

    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    inputs = torch.arange(65).reshape(1, 65) % 11
    outputs, writes = [], []
    for module in (SimpleNamespace(_run_forward=adapter._run_forward), adapter):
        monkeypatch.setattr(
            auto_spyre_model, "resolve_adapter_module", lambda *a, **kw: module
        )
        auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained("test-model")
        head_rows.clear()
        cache_writes.clear()
        outputs.append(
            model.generate(
                inputs,
                max_new_tokens=3,
                do_sample=False,
                eos_token_id=None,
                prefill_chunk_size=chunk_size,
                prefill_fn=prefill if custom_prefill else None,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
            )
        )
        writes.append(list(cache_writes))
        if module is adapter and not custom_prefill:
            assert all(rows == 1 for rows in head_rows)
        else:
            assert head_rows[0] > 1
        assert head_rows[-2:] == [1, 1]  # Decode is unchanged.
    assert writes[0] == writes[1]
    for output in outputs[1:]:
        assert torch.equal(outputs[0].sequences, output.sequences)
        for field in ("logits", "scores"):
            for control, treatment in zip(
                getattr(outputs[0], field), getattr(output, field)
            ):
                assert control.shape == treatment.shape == (1, 11)
                assert torch.equal(control, treatment)


def test_prefill_logit_copy_has_only_the_selected_row():
    import torch

    from hf_adapters.hf_common import _prefill_next_logits

    full = torch.arange(2 * 64 * 12).reshape(2, 64, 12).float()
    selected = _prefill_next_logits(full)
    assert torch.equal(selected, full[:, -1, :])
    assert (
        selected.untyped_storage().nbytes()
        == selected.numel() * selected.element_size()
    )


def test_token_compare_uses_one_generation_with_model_rules():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.spyre.test_e2e_token_compare_spyre import (
        _compare_results,
        adapter_greedy_steps,
    )

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=11,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    ).eval()
    model.generation_config.suppress_tokens = [2]
    model.generation_config.forced_bos_token_id = 3
    calls = []

    def forward(model, ids, positions, mask, keys, values, cache_index):
        calls.append(ids.shape[1])
        logits = torch.zeros(*ids.shape, 11)
        logits[..., 2] = 5
        return logits

    results = adapter_greedy_steps(forward, model, torch.tensor([[1]]), num_decode=2)
    assert len(calls) == 3
    assert calls[1:] == [1, 1]
    assert [result["token"] for result in results] == [3, 0, 0]
    assert all(result["logits"].argmax().item() == 2 for result in results)
    assert model.generation_config.suppress_tokens == [2]
    wrong_token = [{**results[0], "token": 4}, *results[1:]]
    with pytest.raises(AssertionError, match="HF token 3 != Spyre token 4"):
        _compare_results(results, wrong_token, None, "test-model")
