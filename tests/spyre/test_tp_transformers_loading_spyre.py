# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""TP loader and compiled adapter parity; run with torchrun -m pytest."""

import os
import tempfile
from typing import Any

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
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


class _TPNumericalMismatchError(AssertionError):
    """Only numerical parity failures qualify for the tracked MoE xfails."""


@pytest.mark.parametrize(
    "family",
    [
        "qwen3",
        "qwen3_tied_embedding",
        "qwen3_tied_head",
        "granite",
        "gemma3",
        "phi3",
        "phi3_padded",
        "qwen3_5",
        "qwen3_5_moe",
        "gemma4",
        "gemma4_replicated",
        pytest.param(
            "gemma4_moe",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_TPNumericalMismatchError,
                reason="Gemma MoE decode divergence: https://github.com/torch-spyre/hf-adapters/issues/773",
            ),
        ),
        "qwen3_5_hybrid",
        pytest.param(
            "qwen3_5_moe_hybrid",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_TPNumericalMismatchError,
                reason="Rank-dependent MoE routing: https://github.com/torch-spyre/hf-adapters/issues/772",
            ),
        ),
    ],
)
def test_tp_checkpoint_and_compiled_generation(family: str) -> None:
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(522)
    omitted_alias: str | None = {
        "qwen3_tied_embedding": "lm_head.weight",
        "qwen3_tied_head": "model.embed_tokens.weight",
    }.get(family)
    base_family = (
        "qwen3"
        if omitted_alias is not None
        else family.removesuffix("_hybrid")
        .removesuffix("_replicated")
        .removesuffix("_padded")
    )
    config_cls, model_cls = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "granite": (GraniteConfig, GraniteForCausalLM),
        "gemma3": (Gemma3TextConfig, Gemma3ForCausalLM),
        "gemma4": (Gemma4TextConfig, Gemma4ForCausalLM),
        "gemma4_moe": (Gemma4TextConfig, Gemma4ForCausalLM),
        "phi3": (Phi3Config, Phi3ForCausalLM),
        "qwen3_5": (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
        "qwen3_5_moe": (Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM),
    }[base_family]
    extra: dict[str, Any] = {}
    if base_family in ("phi3", "qwen3_5", "qwen3_5_moe"):
        extra = {
            "original_max_position_embeddings": 256,
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.75,
            },
        }
    if family.endswith("_hybrid"):
        extra.update(
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
    if base_family == "qwen3_5_moe":
        extra.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
    if base_family in ("gemma4", "gemma4_moe"):
        extra.update(
            global_head_dim=128,
            vocab_size_per_layer_input=256,
            hidden_size_per_layer_input=0 if base_family == "gemma4_moe" else 64,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=128,
            enable_moe_block=base_family == "gemma4_moe",
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=256,
        )
    config = config_cls(
        vocab_size=256,
        hidden_size=(
            384 if family == "phi3_padded" else 512 if base_family == "phi3" else 256
        ),
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=(
            1 if family == "gemma3" else 2 if family.startswith("gemma4") else 4
        ),
        head_dim=96 if family == "phi3_padded" else 128,
        max_position_embeddings=256,
        tie_word_embeddings=omitted_alias is not None,
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

    def save_checkpoint(path: str) -> None:
        if omitted_alias is None:
            reference.save_pretrained(path)
        else:
            # Preserve the chosen alias direction instead of letting HF's save
            # helper choose which copy of shared storage to keep.
            reference.config.save_pretrained(path)
            state = dict(reference.state_dict())
            state.pop(omitted_alias)
            save_file(state, os.path.join(path, "model.safetensors"))

    # Start with independent seeded checkpoints so the public loader can create
    # the first group. Once it exists, HF saves only on rank zero: share that
    # checkpoint without destroying/reinitializing torchrun's rendezvous group.
    with tempfile.TemporaryDirectory(prefix=f"hf-tp-{family}-") as local_checkpoint:
        checkpoint = local_checkpoint
        if dist.is_initialized():
            paths = [local_checkpoint]
            dist.broadcast_object_list(paths, src=0)
            checkpoint = paths[0]
            if dist.get_rank() == 0:
                save_checkpoint(checkpoint)
            dist.barrier()
        else:
            save_checkpoint(checkpoint)
        model = AutoSpyreModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.float16,
            tp_plan={} if family == "gemma4_replicated" else "auto",
        )
        dist.barrier()  # All ranks must finish reading before rank zero cleans up.
    assert dist.is_initialized()
    assert not any(isinstance(param, DTensor) for param in model.parameters())
    with torch.no_grad():
        actual = model.generate(ids, **kwargs)
    for got in actual.logits:
        assert torch.isfinite(got.cpu()).all()
    try:
        torch.testing.assert_close(actual.sequences.cpu(), expected.sequences)
        for step, (got, want) in enumerate(
            zip(actual.logits, expected.logits, strict=True)
        ):
            torch.testing.assert_close(
                got.cpu().float(),
                want.float(),
                atol=0.02,
                rtol=0.02,
                msg=f"{family}: logits differ at generation step {step}",
            )
    except AssertionError as error:
        raise _TPNumericalMismatchError(str(error)) from error
