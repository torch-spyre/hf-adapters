# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compiled rowwise TP must reduce products and add the bias exactly once."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_spyre  # noqa: F401

from hf_adapters.spyre_tensor_parallel import (
    SpyreRowwiseParallel,
    _copy_default,
    _copy_linear,
)

if int(os.environ.get("WORLD_SIZE", "1")) != 2:
    pytest.skip("requires a two-rank torchrun", allow_module_level=True)


def test_compiled_rowwise_bias_prefill_and_decode() -> None:
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,spyre:spyreccl")
    rank = dist.get_rank()
    device = torch.device("spyre", rank)
    mesh = dist.init_device_mesh("spyre", (2,), mesh_dim_names=("tp",))
    generator = torch.Generator().manual_seed(771)
    weight = torch.randn(128, 256, generator=generator).half() * 0.02
    bias = torch.linspace(0.25, 0.75, 128).half()
    layer = torch.nn.Linear(256, 128, device="meta", dtype=torch.float16)
    SpyreRowwiseParallel().install_forward(layer, mesh)
    layer.weight = torch.nn.Parameter(
        _copy_linear(
            weight[:, rank * 128 : (rank + 1) * 128], device=device, dtype=torch.float16
        ),
        requires_grad=False,
    )
    layer.bias = torch.nn.Parameter(
        _copy_default(bias, device=device, dtype=torch.float16), requires_grad=False
    )
    saved_bias = layer.bias
    compiled = torch.compile(layer, dynamic=False, fullgraph=True)
    with torch.no_grad():
        for sequence_length in (64, 1):
            inputs = torch.randn(2, sequence_length, 256, generator=generator).half()
            expected = F.linear(inputs, weight, bias)
            local = inputs[..., rank * 128 : (rank + 1) * 128].contiguous().to(device)
            for _ in range(3):
                actual = compiled(local).cpu()
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
                assert layer.bias is saved_bias
