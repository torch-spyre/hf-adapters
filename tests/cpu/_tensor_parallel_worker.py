# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Real Gloo TP loading/forward checks, launched by test_tensor_parallel_loading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import (
    AutoModelForCausalLM,
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
from transformers.distributed import DistributedConfig
from transformers.distributed.tensor_parallel import apply_tensor_parallelism

# Install the same CPU adapter environment as the parent pytest process.
import tests.conftest  # noqa: F401, E402
from hf_adapters import (
    auto_spyre_model,
    hf_common,
    hf_gemma3,
    hf_gemma4,
    hf_gemma4_moe,
    hf_granite,
    hf_phi3,
    hf_qwen3,
    hf_qwen3_5,
    hf_qwen3_5_moe,
)
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_CPU_PACKED_COLWISE,
    SPYRE_CPU_ROWWISE,
    SpyreShardOperation,
    prepare_spyre_tp_plan,
    spyre_tp_loading,
)


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


_device_route = hf_qwen3_5_moe._route_decode_on_cpu


def _cpu_route(
    x: torch.Tensor, mlp: nn.Module, top_k: int, stick_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, indices = _device_route(x, mlp, top_k, stick_size)
    # Spyre's keep_by_index accepts float32-transported indices; CPU's tensor
    # indexing requires integers. Preserve their exact values in this emulator.
    return weights, indices.to(torch.int64)


def check_styles(root: Path, world_size: int, rank: int) -> None:
    mesh = dist.init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = nn.Module()
    with torch.device("meta"):
        model.col = nn.Linear(8, 11)
        model.row = nn.Linear(8, 11)
        model.grouped = nn.Linear(8, 8)
        model.embed = nn.Embedding(11, 8, padding_idx=10)
        model.hidden_embed = nn.Embedding(11, 11)
        model.experts = nn.Module()
        model.experts.gate_up_proj = nn.Parameter(torch.empty(2, 16, 8))
        model.experts.down_proj = nn.Parameter(torch.empty(2, 8, 8))
        model.scalar = nn.Linear(8, 8, bias=False)
        model.scalar.bound = nn.Parameter(torch.empty(()))
    full: dict[str, torch.Tensor] = {
        name: torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)
        / 100
        for name, param in model.named_parameters()
    }
    plan = prepare_spyre_tp_plan(
        model,
        {
            "col": "colwise_gather_output",
            "row": "rowwise_split_input",
            "embed": "embedding_rowwise",
            "hidden_embed": "embedding_colwise",
            "experts.gate_up_proj": "packed_colwise",
            "experts.down_proj": "rowwise",
            "scalar": "colwise",
        },
        cpu_staged_modules=("experts.gate_up_proj", "experts.down_proj"),
        grouped_colwise_modules={"grouped": 2},
    )
    with spyre_tp_loading():
        apply_tensor_parallelism(model, mesh, plan)
    for name, placeholder in list(model.named_parameters()):
        assert isinstance(placeholder, DTensor), name
        op = SpyreShardOperation(placeholder)
        shard = op.shard_tensor(full[name], device="cpu", dtype=torch.float32)
        assert shard is not None
        if name.startswith("grouped"):
            expected = full[name].chunk(2, dim=0)[rank // (world_size // 2)]
        elif name == "experts.gate_up_proj":
            expected = torch.cat(
                [
                    part.chunk(world_size, dim=1)[rank]
                    for part in full[name].chunk(2, dim=1)
                ],
                dim=1,
            )
            # The converter also accepts one checkpoint tensor per expert.
            for expert in range(2):
                per_expert = op.shard_tensor(
                    full[name][expert], tensor_idx=expert, device="cpu"
                )
                torch.testing.assert_close(per_expert, expected[expert])
                halves = [
                    op.shard_tensor(part, tensor_idx=expert, device="cpu")
                    for part in full[name][expert].chunk(2, dim=0)
                ]
                torch.testing.assert_close(torch.cat(halves, dim=0), expected[expert])
        elif name in ("row.weight", "experts.down_proj", "hidden_embed.weight"):
            expected = full[name].chunk(world_size, dim=-1)[rank]
        elif name in ("col.weight", "col.bias", "embed.weight", "scalar.weight"):
            expected = full[name].chunk(world_size, dim=0)[rank]
        else:
            expected = full[name]
        torch.testing.assert_close(shard, expected)
        checkpoint = root / f"slice-{rank}.safetensors"
        save_file({"weight": full[name]}, checkpoint)
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            torch.testing.assert_close(
                op.shard_tensor(handle.get_slice("weight"), device="cpu"), expected
            )
        if name == "col.weight":
            # Clipped-linear converters reuse a matrix operation for scalar state.
            torch.testing.assert_close(
                op.shard_tensor(torch.tensor(3.0), device="cpu"), torch.tensor(3.0)
            )
        parent, _, param = name.rpartition(".")
        model.get_submodule(parent)._parameters[param] = nn.Parameter(
            shard, requires_grad=False
        )

    inputs = torch.randn(2, 3, 8)
    torch.testing.assert_close(
        model.col(inputs),
        torch.nn.functional.linear(inputs, full["col.weight"], full["col.bias"]),
    )
    torch.testing.assert_close(
        model.row(inputs),
        torch.nn.functional.linear(inputs, full["row.weight"], full["row.bias"]),
    )
    ids = torch.tensor([[0, 4, 8], [1, 5, 10]])
    torch.testing.assert_close(
        model.embed(ids), torch.nn.functional.embedding(ids, full["embed.weight"])
    )
    torch.testing.assert_close(
        model.hidden_embed(ids),
        torch.nn.functional.embedding(ids, full["hidden_embed.weight"]),
    )
    assert plan["experts.gate_up_proj"] == SPYRE_CPU_PACKED_COLWISE
    assert plan["experts.down_proj"] == SPYRE_CPU_ROWWISE


def check_models(root: Path, world_size: int, rank: int) -> None:
    for family, config_cls, model_cls, adapter, kv_heads in (
        ("qwen3", Qwen3Config, Qwen3ForCausalLM, hf_qwen3, 4),
        ("granite", GraniteConfig, GraniteForCausalLM, hf_granite, 4),
        ("gemma3", Gemma3TextConfig, Gemma3ForCausalLM, hf_gemma3, 1),
        ("gemma4", Gemma4TextConfig, Gemma4ForCausalLM, hf_gemma4, 2),
        ("gemma4_moe", Gemma4TextConfig, Gemma4ForCausalLM, hf_gemma4_moe, 2),
        ("phi3", Phi3Config, Phi3ForCausalLM, hf_phi3, 4),
        ("qwen3_5", Qwen3_5TextConfig, Qwen3_5ForCausalLM, hf_qwen3_5, 4),
        ("qwen3_5_moe", Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, hf_qwen3_5_moe, 4),
    ):
        config = config_cls(
            vocab_size=131,
            hidden_size=512 if family == "phi3" else 256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
            head_dim=128,
            max_position_embeddings=256,
            tie_word_embeddings=False,
            pad_token_id=0,
            eos_token_id=None,
            bos_token_id=1,
            **(
                {
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 256,
                    "shared_expert_intermediate_size": 256,
                    "layer_types": [
                        "full_attention" if family == "phi3" else "linear_attention",
                        "full_attention",
                    ],
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.75,
                    },
                }
                if family in ("phi3", "qwen3_5", "qwen3_5_moe")
                else (
                    {
                        "global_head_dim": 128,
                        "vocab_size_per_layer_input": 131,
                        "hidden_size_per_layer_input": (
                            0 if family == "gemma4_moe" else 64
                        ),
                        "layer_types": ["sliding_attention", "full_attention"],
                        "sliding_window": 128,
                        "enable_moe_block": family == "gemma4_moe",
                        "num_experts": 4,
                        "top_k_experts": 2,
                        "moe_intermediate_size": 256,
                    }
                    if family in ("gemma4", "gemma4_moe")
                    else {}
                )
            ),
        )
        reference = model_cls(config).eval()
        checkpoint = root / family
        if rank == 0:
            reference.save_pretrained(checkpoint)
        dist.barrier()
        ids = torch.tensor([[1, 3, 5, 7], [1, 4, 6, 8]])
        expected = reference(ids, use_cache=False).logits
        loaded = hf_common.load_model_common(
            str(checkpoint),
            adapter,
            torch.float32,
            AutoModelForCausalLM,
            tp_plan="auto",
        )
        assert not any(isinstance(param, DTensor) for param in loaded.parameters())
        torch.testing.assert_close(
            loaded(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5
        )
        if family == "qwen3_5":
            linear_attn: nn.Module = loaded.model.layers[0].linear_attn
            assert linear_attn.in_proj_a.weight.shape[0] == 4
            assert linear_attn.in_proj_b.weight.shape[0] == 4
            assert linear_attn.in_proj_qkv.weight.shape[0] == 1024 // world_size
        del loaded
        if family == "gemma4_moe":
            # Preparation uses Spyre-only packed-expert DMA/operators. The
            # compiled hardware suite covers generation; CPU checks the real
            # checkpoint converter, local placement, and stock TP forward.
            continue
        spyre = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
            str(checkpoint), dtype=torch.float32, tp_plan="auto"
        )
        expected_generation = reference.generate(
            ids,
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        actual = spyre.generate(
            ids,
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        torch.testing.assert_close(actual.sequences, expected_generation.sequences)
        for got, want in zip(actual.logits, expected_generation.logits, strict=True):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)
        del spyre

    # Tied vocabulary weights must follow the embedding's local shard and be
    # gathered exactly once by the adapter's prepared LM head.
    tied_config = Qwen3Config(
        vocab_size=131,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        tie_word_embeddings=True,
        eos_token_id=None,
        pad_token_id=0,
    )
    tied_reference = Qwen3ForCausalLM(tied_config).eval()
    checkpoint = root / "tied"
    if rank == 0:
        tied_reference.save_pretrained(checkpoint)
    dist.barrier()
    tied_model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=torch.float32,
        tp_plan={"model.embed_tokens": "embedding_rowwise"},
    )
    expected_tokens = tied_reference.generate(ids, max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(
        tied_model.generate(ids, max_new_tokens=3, do_sample=False), expected_tokens
    )

    # Explicit overrides must win over a wildcard, and unspecified rules in an
    # explicit plan must not be silently merged back from the model defaults.
    probe = Qwen3ForCausalLM(
        config=Qwen3Config(
            vocab_size=131,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
        )
    )
    plan = prepare_spyre_tp_plan(
        probe,
        {"model.layers.*.self_attn.k_proj": "colwise"},
        grouped_colwise_modules={"model.layers.0.self_attn.k_proj": 2},
    )
    checkpoint = root / "exact"
    if rank == 0:
        probe.save_pretrained(checkpoint)
    dist.barrier()
    with spyre_tp_loading():
        loaded = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            distributed_config=DistributedConfig(tp_size=world_size, tp_plan=plan),
        )
    assert loaded.model.layers[0].self_attn.k_proj.weight.shape[0] == 256
    assert loaded.model.layers[1].self_attn.k_proj.weight.shape[0] == 512 // world_size
    assert loaded.model.layers[0].self_attn.q_proj.weight.shape[0] == 512
    assert loaded.tp_plan["model.layers.*.self_attn.k_proj"] == SPYRE_COLWISE


def check_incomplete_checkpoints(root: Path, rank: int) -> None:
    reference = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
        )
    ).eval()
    module_name = "model.layers.0.self_attn.q_proj"
    key = f"{module_name}.weight"
    for missing in (True, False):
        checkpoint = root / ("missing" if missing else "mismatched")
        if rank == 0:
            state = dict(reference.state_dict())
            if missing:
                state.pop(key)
            else:
                state[key] = state[key][:-1]
            reference.save_pretrained(checkpoint, state_dict=state)
        dist.barrier()
        for staged in (False, True):
            plan = prepare_spyre_tp_plan(
                reference,
                {module_name: "colwise"},
                cpu_replicated_modules={module_name} if staged else (),
            )
            with (
                spyre_tp_loading(),
                pytest.raises(
                    ValueError, match="Spyre TP requires complete checkpoint weights"
                ) as caught,
            ):
                AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    ignore_mismatched_sizes=True,
                    distributed_config=DistributedConfig(
                        tp_size=dist.get_world_size(), tp_plan=plan
                    ),
                )
            assert key in str(caught.value)


def main() -> None:
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(522)
    torch.compile = _eager
    dist.init_process_group("gloo")
    root = Path(sys.argv[1])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        with (
            patch("torch._C._get_accelerator", return_value=torch.device("cpu")),
            patch.object(hf_qwen3_5_moe, "_route_decode_on_cpu", _cpu_route),
            torch.no_grad(),
        ):
            check_styles(root, world_size, rank)
            check_models(root, world_size, rank)
            check_incomplete_checkpoints(root, rank)
        print(
            f"TP{world_size} rank {rank}: all loading, placement and numerical checks passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
