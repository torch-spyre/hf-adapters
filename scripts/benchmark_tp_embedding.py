# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure vocab-parallel host staging versus pre-staged compiled execution.

Run with SPYRE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2
scripts/benchmark_tp_embedding.py. Each measured call copies its result to CPU
to synchronize device work. Compilation and warmup are excluded; the delta
includes host masking, input transfers, and Python dispatch, not just PCIe DMA.
"""

import argparse
import json
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401

from hf_adapters.spyre_tensor_parallel import (
    SpyreEmbeddingRowwiseParallel,
    _copy_embedding,
)


def milliseconds(call: Callable[[], torch.Tensor], iterations: int) -> float:
    for _ in range(5):
        call().cpu()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        call().cpu()
    return 1000 * (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    dist.init_process_group("cpu:gloo,spyre:spyreccl")
    try:
        rank = dist.get_rank()
        size = dist.get_world_size()
        device = torch.device("spyre", rank)
        mesh = dist.init_device_mesh("spyre", (size,), mesh_dim_names=("tp",))
        local_vocab, hidden = 512, 2816
        vocab = local_vocab * size
        start = rank * local_vocab
        layer = torch.nn.Embedding(vocab, hidden, device="meta", dtype=torch.float16)
        SpyreEmbeddingRowwiseParallel().install_forward(layer, mesh)
        weight = (
            torch.arange(start, start + local_vocab)
            .half()[:, None]
            .expand(-1, hidden)
            .contiguous()
        )
        layer.weight = torch.nn.Parameter(
            _copy_embedding(weight, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        for batch, length in ((1, 1), (2, 1), (1, 64), (2, 64)):
            ids = torch.arange(batch * length).reshape(batch, length) % vocab
            device_ids = ids.to(device)
            valid = (ids >= start) & (ids < start + local_vocab)
            local_ids = torch.where(valid, ids - start, 0).to(device)
            device_mask = valid.half().unsqueeze(-1).to(device)

            def full() -> torch.Tensor:
                return layer(device_ids)

            def staged() -> torch.Tensor:
                return layer.forward(local_ids, device_mask)

            with torch.no_grad():
                torch.testing.assert_close(full().cpu(), staged().cpu(), atol=0, rtol=0)
                full_ms = milliseconds(full, args.iterations)
                staged_ms = milliseconds(staged, args.iterations)
            result = {
                "full_ms": full_ms,
                "staged_ms": staged_ms,
                "host_path_delta_ms": full_ms - staged_ms,
            }
            ranks: list[dict[str, float] | None] = [None] * size
            dist.all_gather_object(ranks, result)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch,
                            "sequence": length,
                            "hidden": hidden,
                            "tp": size,
                            "iterations": args.iterations,
                            "ranks": ranks,
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
