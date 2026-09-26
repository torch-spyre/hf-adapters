# Copyright 2026 The Torch-Spyre Authors.
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

"""Granite requests trailing logits only through its generation entry point."""

import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM

from hf_adapters import hf_granite
from hf_adapters.hf_common import generate, set_rope_dtype


def test_granite_generate_preserves_options_and_requests_last_position(monkeypatch):
    model, input_ids, attention_mask = object(), object(), object()
    calls = {}

    def forward(*args, **kwargs):
        calls["forward"] = (args, kwargs)
        return "logits"

    def generate(run_forward, actual_model, actual_ids, **kwargs):
        calls["generate"] = (actual_model, actual_ids, kwargs)
        return run_forward(actual_model, actual_ids)

    monkeypatch.setattr(hf_granite, "_run_forward", forward)
    monkeypatch.setattr(hf_granite, "generate_common", generate)
    result = hf_granite.generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        prefill_chunk_size=512,
    )
    assert result == "logits"
    assert calls["forward"] == ((model, input_ids), {"logits_to_keep": 1})
    assert calls["generate"] == (
        model,
        input_ids,
        {
            "attention_mask": attention_mask,
            "max_new_tokens": 3,
            "prefill_chunk_size": 512,
        },
    )


@pytest.mark.parametrize("prompt_length", [3, 65])
def test_granite_last_position_generation_matches_full_head(monkeypatch, prompt_length):
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    config = GraniteConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = GraniteForCausalLM(config).eval()
    hf_granite.prepare_for_spyre(model)
    set_rope_dtype(model, torch.float32)
    input_ids = torch.arange(1, prompt_length + 1).repeat(2, 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :2] = 0
    input_ids[1, :2] = 0
    options = dict(
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        prefill_chunk_size=64,
        return_dict_in_generate=True,
        output_logits=True,
    )
    projected_lengths = []
    model.lm_head.register_forward_pre_hook(
        lambda _module, args: projected_lengths.append(args[0].shape[1])
    )
    with torch.inference_mode():
        full = generate(hf_granite._run_forward, model, input_ids, **options)
        assert max(projected_lengths) == 64
        projected_lengths.clear()
        actual = hf_granite.generate(model, input_ids, **options)
    torch.testing.assert_close(actual.sequences, full.sequences)
    assert len(actual.logits) == len(full.logits) == 3
    for result, reference in zip(actual.logits, full.logits):
        torch.testing.assert_close(result, reference)
    assert projected_lengths and set(projected_lengths) == {1}
