# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""TP loader and compiled adapter parity; run with torchrun -m pytest."""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    GraniteConfig,
    GraniteForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM

if int(os.environ.get("WORLD_SIZE", "1")) < 2:
    pytest.skip("requires at least two torchrun ranks", allow_module_level=True)


@pytest.mark.parametrize(
    "family", ["qwen3", "granite", "gemma3", "phi3", "qwen3_5", "qwen3_5_moe"]
)
def test_tp_checkpoint_and_compiled_generation(family: str) -> None:
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,spyre:spyreccl")
    rank = dist.get_rank()
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(522)
    config_cls, model_cls = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "granite": (GraniteConfig, GraniteForCausalLM),
        "gemma3": (Gemma3TextConfig, Gemma3ForCausalLM),
        "phi3": (Phi3Config, Phi3ForCausalLM),
        "qwen3_5": (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
        "qwen3_5_moe": (Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM),
    }[family]
    extra: dict[str, Any] = {}
    if family in ("phi3", "qwen3_5", "qwen3_5_moe"):
        extra = {
            "original_max_position_embeddings": 256,
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.75,
            },
        }
    if family == "qwen3_5_moe":
        extra.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
    config = config_cls(
        vocab_size=256,
        hidden_size=512 if family == "phi3" else 256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1 if family == "gemma3" else 4,
        head_dim=128,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
        **extra,
    )
    reference = model_cls(config).to(torch.float16).eval()
    ids = torch.tensor([[1, 3, 5, 7], [1, 4, 6, 8]])
    kwargs: dict[str, Any] = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    with torch.no_grad():
        expected = reference.generate(ids, **kwargs)
    paths: list[str | None] = [
        tempfile.mkdtemp(prefix=f"hf-tp-{family}-") if rank == 0 else None
    ]
    dist.broadcast_object_list(paths, src=0)
    assert paths[0] is not None
    checkpoint = Path(paths[0])
    if rank == 0:
        reference.save_pretrained(checkpoint)
    dist.barrier()
    model = AutoSpyreModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float16, tp_plan="auto"
    )
    assert not any(isinstance(param, DTensor) for param in model.parameters())
    with torch.no_grad():
        actual = model.generate(ids, **kwargs)
    torch.testing.assert_close(actual.sequences.cpu(), expected.sequences)
    for got, want in zip(actual.logits, expected.logits, strict=True):
        torch.testing.assert_close(
            got.cpu().float(), want.float(), atol=0.02, rtol=0.02
        )
        assert torch.isfinite(got.cpu()).all()
