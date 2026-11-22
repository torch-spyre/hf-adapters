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

"""Check the shared OLMo setup against stock OLMo 2 generation on CPU."""

import copy
from typing import TypeVar

import pytest
import torch
from transformers import AutoModelForCausalLM, Olmo2Config, PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

from hf_adapters import hf_common, hf_olmo2

_T = TypeVar("_T")


def _skip_compile(fn: _T, **kwargs: object) -> _T:
    return fn


@pytest.mark.parametrize("chunk_size", [None, 64])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_generate_matches_hf(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int | None,
    batch_size: int,
) -> None:
    torch.manual_seed(23)
    monkeypatch.setattr(torch, "compile", _skip_compile)
    config: Olmo2Config = Olmo2Config(
        vocab_size=131,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=None,
        attn_implementation="eager",
    )
    reference: PreTrainedModel = AutoModelForCausalLM.from_config(config).eval()
    # With chunk_size=64, a 129-token prompt spans three prefill chunks.
    ids: torch.Tensor = torch.randint(1, config.vocab_size, (batch_size, 129))
    mask: torch.Tensor = torch.ones_like(ids)
    if batch_size > 1:
        ids[0, :-7] = 0
        mask[0, :-7] = 0

    expected: GenerateDecoderOnlyOutput = reference.generate(
        ids,
        attention_mask=mask,
        max_new_tokens=4,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )

    adapted: PreTrainedModel = copy.deepcopy(reference)
    hf_olmo2.prepare_for_spyre(adapted)
    hf_common.set_rope_dtype(adapted, torch.float32)
    adapted._spyre_prefill_chunk_size = chunk_size
    actual: GenerateDecoderOnlyOutput = hf_common.generate(
        hf_olmo2._run_forward,
        adapted,
        ids,
        attention_mask=mask,
        prefill_backbone_fn=hf_olmo2._run_backbone_forward,
        max_new_tokens=4,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
    )

    torch.testing.assert_close(actual.sequences, expected.sequences)
    assert actual.logits is not None and expected.logits is not None
    assert len(actual.logits) == len(expected.logits) == 4
    for got, want in zip(actual.logits, expected.logits, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
