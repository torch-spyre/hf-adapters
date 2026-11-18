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

"""CPU accuracy test for the OLMo 3 adapter on a tiny random checkpoint.

The registry-driven ``test_adapter_cpu_accuracy`` downloads the real 7B
checkpoint, whose ``sliding_window`` (4096) is far longer than any test prompt,
so the sliding band never masks anything there. This test builds a tiny OLMo 3
whose window is shorter than the prompt, so the sliding layers genuinely see a
narrower mask than the full layers, and compares the adapter's greedy
generation against stock HF on the same weights — including left-padded
batches and chunked prefill, which route the band through
``_ChunkedPrefillMaskBuilder``.
"""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, Olmo3Config, PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

from hf_adapters import hf_common, hf_olmo3

VOCAB_SIZE: int = 131
SLIDING_WINDOW: int = 16
PROMPT_LEN: int = 41
MAX_NEW_TOKENS: int = 6


def _tiny_config() -> Olmo3Config:
    # head_dim=128 keeps head_dim/2 at one Spyre stick, as on the real 7B.
    return Olmo3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        sliding_window=SLIDING_WINDOW,
        # Mirrors the real 7B: default RoPE on sliding layers, YaRN (with its
        # attention_scaling != 1) on full layers. A single shared rotary would
        # get one of the two layer types wrong.
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "original_max_position_embeddings": 256,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
        },
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        pad_token_id=0,
        eos_token_id=None,
        attn_implementation="eager",
    )


@pytest.fixture
def models(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PreTrainedModel, PreTrainedModel]:
    """Return ``(stock_hf_model, prepared_adapter_model)`` sharing weights."""
    torch.manual_seed(23)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    reference: PreTrainedModel = AutoModelForCausalLM.from_config(_tiny_config()).eval()
    # Prepare a copy so the reference keeps stock HF modules and forward.
    adapted: PreTrainedModel = copy.deepcopy(reference)
    hf_olmo3.prepare_for_spyre(adapted)
    hf_common.set_rope_dtype(adapted, torch.float32)
    return reference, adapted


def _prompts(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random prompts, left-padded to a mixed length when ``batch_size > 1``."""
    ids: torch.Tensor = torch.randint(1, VOCAB_SIZE, (batch_size, PROMPT_LEN))
    mask: torch.Tensor = torch.ones_like(ids)
    if batch_size > 1:
        pad: int = PROMPT_LEN - 27  # Row 0 keeps 27 real tokens (> window).
        ids[0, :pad] = 0
        mask[0, :pad] = 0
    return ids, mask


@pytest.mark.parametrize("chunk_size", [None, 64])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_generate_matches_hf(
    models: tuple[PreTrainedModel, PreTrainedModel],
    chunk_size: int | None,
    batch_size: int,
) -> None:
    reference, adapted = models
    adapted._spyre_prefill_chunk_size = chunk_size
    ids, mask = _prompts(batch_size)

    expected: GenerateDecoderOnlyOutput = reference.generate(
        ids,
        attention_mask=mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    actual: GenerateDecoderOnlyOutput = hf_common.generate(
        hf_olmo3._run_forward,
        adapted,
        ids,
        attention_mask=mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
    )

    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_sliding_band_is_exercised(
    models: tuple[PreTrainedModel, PreTrainedModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against a vacuous pass: dropping the band must change the logits."""
    reference, adapted = models
    ids, mask = _prompts(1)
    kwargs: dict[str, object] = dict(
        attention_mask=mask,
        max_new_tokens=1,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
    )
    banded: GenerateDecoderOnlyOutput = hf_common.generate(
        hf_olmo3._run_forward, adapted, ids, **kwargs
    )
    # Drop only the band: sliding layers keep their own RoPE, so any logit
    # change is attributable to the window mask alone.
    monkeypatch.setattr(
        hf_olmo3, "add_causal_sliding_window_band", lambda mask, *a, **k: mask
    )
    unbanded: GenerateDecoderOnlyOutput = hf_common.generate(
        hf_olmo3._run_forward, adapted, ids, **kwargs
    )

    assert not torch.allclose(banded.logits[0], unbanded.logits[0], atol=1e-3)


def test_rejects_unknown_layer_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    config: Olmo3Config = _tiny_config()
    model: PreTrainedModel = AutoModelForCausalLM.from_config(config).eval()
    model.config.layer_types = ["linear_attention"] * config.num_hidden_layers

    with pytest.raises(ValueError, match="does not support layer types"):
        hf_olmo3.prepare_for_spyre(model)
