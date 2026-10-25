# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process worker for real Transformers checkpoint/TP integration tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from hf_adapters import hf_common, hf_gemma4
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_COLWISE_GATHER_OUTPUT,
    SPYRE_GROUPED_COLWISE_PREFIX,
    _register_grouped_colwise_style,
    spyre_compiled_all_gather_last_dim,
)


def run(checkpoints: Path) -> None:
    hf_common.DEVICE = "cpu"
    torch.set_num_threads(1)
    adapter = ModuleType("tp_test_adapter")
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    inputs = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])

    for name in ("llama", "olmoe", "gemma4", "qwen3_moe"):
        checkpoint = checkpoints / name
        reference = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float32
        )
        reference.eval().requires_grad_(False)
        with torch.no_grad():
            expected = reference(inputs).logits
        # The first load must initialize the process group itself. Subsequent
        # loads must reuse it, not attempt a second initialization.
        model = hf_common.load_model_common(
            checkpoint,
            hf_gemma4 if name == "gemma4" else adapter,
            dtype=torch.float32,
            auto_model_cls=AutoModelForCausalLM,
            tp_plan="auto",
        )
        assert all(
            not isinstance(p, DTensor) and p.device.type == "cpu"
            for p in model.parameters()
        )
        projection = model.model.layers[0].self_attn.q_proj
        source = reference.model.layers[0].self_attn.q_proj
        torch.testing.assert_close(
            projection.weight, source.weight.chunk(world_size, dim=0)[rank]
        )
        if name != "olmoe":
            assert projection.out_features == source.out_features // world_size
            assert projection._hf_tp_plan == SPYRE_COLWISE
        else:
            assert projection.out_features == source.out_features
            assert projection._hf_tp_plan == SPYRE_COLWISE_GATHER_OUTPUT
        assert projection._hf_device_mesh.size() == world_size
        with torch.no_grad():
            torch.testing.assert_close(
                model(inputs).logits, expected, rtol=1e-5, atol=1e-6
            )
        if name == "llama":
            assert model.lm_head.weight is model.model.embed_tokens.weight
            # The rowwise bias must survive repeated calls and fullgraph capture.
            down = model.model.layers[0].mlp.down_proj
            local_input = torch.randn(2, 3, down.in_features)
            bias = down.bias
            compiled = torch.compile(down, backend="eager", fullgraph=True)
            with torch.no_grad():
                torch.testing.assert_close(compiled(local_input), down(local_input))
            assert down.bias is bias
        elif name in ("olmoe", "qwen3_moe"):
            expert = model.model.layers[0].mlp.experts
            full = reference.model.layers[0].mlp.experts.gate_up_proj
            gate, up = full.chunk(2, dim=1)
            expected_packed = torch.cat(
                (
                    gate.chunk(world_size, dim=1)[rank],
                    up.chunk(world_size, dim=1)[rank],
                ),
                dim=1,
            )
            torch.testing.assert_close(expert.gate_up_proj, expected_packed)
        try:
            model.save_pretrained(checkpoints / "must_not_be_written")
        except NotImplementedError:
            pass
        else:
            raise AssertionError("local TP checkpoint saving was not rejected")

    checkpoint = checkpoints / "llama"
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    reference.eval().requires_grad_(False)
    plan = dict(reference.tp_plan)
    plan.update({"model.embed_tokens": "embedding_rowwise", "lm_head": "colwise"})
    model = hf_common.load_model_common(
        checkpoint,
        adapter,
        dtype=torch.float32,
        auto_model_cls=AutoModelForCausalLM,
        tp_plan=plan,
    )
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with torch.no_grad():
        logits = model(inputs).logits
        sizes = tuple(
            part.shape[0] for part in reference.lm_head.weight.chunk(world_size, dim=0)
        )
        padded = torch.nn.functional.pad(logits, (0, max(sizes) - logits.shape[-1]))
        reconstructed = spyre_compiled_all_gather_last_dim(
            padded,
            world_size,
            model.model.embed_tokens._hf_device_mesh.get_group().group_name,
            shard_sizes=sizes,
        )
        torch.testing.assert_close(
            reconstructed, reference(inputs).logits, rtol=1e-5, atol=1e-6
        )

    # An explicit plan is complete, not an overlay on the HF default plan.
    # Exact layer overrides must pass validation and beat wildcard rules.
    _register_grouped_colwise_style(2)
    plan = {
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.0.self_attn.k_proj": f"{SPYRE_GROUPED_COLWISE_PREFIX}_2",
    }
    model = hf_common.load_model_common(
        checkpoint,
        adapter,
        dtype=torch.float32,
        auto_model_cls=AutoModelForCausalLM,
        tp_plan=plan,
    )
    grouped_rank = rank // (world_size // 2)
    torch.testing.assert_close(
        model.model.layers[0].self_attn.k_proj.weight,
        reference.model.layers[0].self_attn.k_proj.weight.chunk(2, dim=0)[grouped_rank],
    )
    torch.testing.assert_close(
        model.model.layers[1].self_attn.k_proj.weight,
        reference.model.layers[1].self_attn.k_proj.weight.chunk(world_size, dim=0)[
            rank
        ],
    )
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_proj.weight,
        reference.model.layers[0].self_attn.q_proj.weight,
    )
    dist.barrier()
    dist.destroy_process_group()
    print(f"TP integration passed on rank {rank}/{world_size}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    run(parser.parse_args().checkpoints)
