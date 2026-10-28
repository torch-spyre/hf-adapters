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

"""Stock-HF parity for Muse Glimmer and the shared Gemma 3 decoder block."""

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
    PreTrainedModel,
)

from hf_adapters import hf_gemma3, hf_muse_glimmer
from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM
from hf_adapters.hf_common import get_backbone, set_rope_dtype
from tests.cpu.test_adapter_cpu_accuracy import adapter_greedy_steps, hf_greedy_steps


def _tiny_model(family: str) -> PreTrainedModel:
    """Keep GQA, sandwich norms, and mixed attention at Spyre-aligned widths."""
    if family == "gemma3":
        return (
            Gemma3ForCausalLM(
                Gemma3TextConfig(
                    vocab_size=128,
                    hidden_size=384,
                    intermediate_size=512,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=128,
                    layer_types=["sliding_attention", "full_attention"],
                    sliding_window=64,
                    max_position_embeddings=512,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            )
            .eval()
            .requires_grad_(False)
        )
    return (
        MuseGlimmerForConditionalGeneration(
            MuseGlimmerConfig(
                text_config={
                    "vocab_size": 128,
                    "hidden_size": 384,
                    "intermediate_size": 512,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 128,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "layer_rope_theta": [500000.0, 0.0],
                    "sliding_window": 64,
                    "max_position_embeddings": 512,
                    "pad_token_id": 0,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "output_multiplier": 0.5,
                    "final_logit_softcapping": 1.2,
                },
                vision_config={
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                },
                out_hidden_size=512,
                projector_hidden_size=256,
                image_token_id=126,
                video_token_id=127,
            )
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("family", ["muse_glimmer", "gemma3"])
@pytest.mark.parametrize("swa_mode", [None, "anchored"])
def test_prefill_and_decode_match_stock_hf(family: str, swa_mode: str | None) -> None:
    """Cross the local window and compact-cache shift, comparing every logit."""
    torch.random.default_generator.manual_seed(17)
    model = _tiny_model(family)
    input_ids = torch.randint(3, 125, (1, 73))
    reference: list[dict[str, Any]] = hf_greedy_steps(model, input_ids, num_decode=70)

    model._spyre_swa_mode = swa_mode
    adapter = hf_muse_glimmer if family == "muse_glimmer" else hf_gemma3
    # This checks adapter arithmetic against HF; hardware tests compile it.
    with torch.compiler.set_stance("force_eager"):
        adapter.prepare_for_spyre(model)
        set_rope_dtype(model, torch.float32)
        actual: list[dict[str, Any]] = adapter_greedy_steps(
            adapter._run_forward, model, input_ids, 70
        )
    for expected, result in zip(reference, actual, strict=True):
        torch.testing.assert_close(
            result["logits"], expected["logits"], atol=1e-5, rtol=1e-4
        )
        assert result["token"] == expected["token"]


def test_auto_loader_matches_stock_generation(tmp_path: Path) -> None:
    """Exercise real dispatch/loading and unequal padded prompts from a checkpoint."""
    torch.random.default_generator.manual_seed(29)
    model = _tiny_model("muse_glimmer")
    input_ids = torch.tensor([[1, 4, 5, 6, 7], [0, 0, 1, 8, 9]])
    attention_mask = (input_ids != 0).long()
    with torch.no_grad():
        expected: torch.Tensor = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
        )
    model.save_pretrained(tmp_path)

    with torch.compiler.set_stance("force_eager"):
        adapted: PreTrainedModel = AutoSpyreModelForCausalLM.from_pretrained(
            str(tmp_path)
        )
        actual: torch.Tensor = adapted.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
        )
    assert not hasattr(adapted.model, "vision_tower")
    assert get_backbone(adapted).embed_tokens.embed_norm is not None
    torch.testing.assert_close(actual, expected)
