# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-/two-card coverage for compiled Spyre TP embedding and linear paths.

Run with::

    SPYRE_DEVICES=0,1 torchrun --nproc-per-node=2 -m pytest -q -s \
        tests/spyre/test_tp_embedding_collective_spyre.py
"""

import os
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401
from torch.distributed.device_mesh import DeviceMesh
from torch_spyre.model_utils import _dma_to_spyre_indirect_access
from transformers.distributed import DistributedConfig

from hf_adapters.spyre_tensor_parallel import (
    SpyreColwiseParallel,
    SpyreEmbeddingRowwiseParallel,
    SpyreRowwiseParallel,
    _initialize_tp_mesh,
    finalize_spyre_tp,
)

if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) not in (1, 2):
    pytest.skip("requires a one- or two-rank torchrun", allow_module_level=True)


def _runtime_mesh() -> DeviceMesh:
    _, meshes = _initialize_tp_mesh(
        DistributedConfig(tp_size=int(os.environ["WORLD_SIZE"])), "spyre"
    )
    return meshes.get_mesh("tp")


def test_mask_on_host_then_compile_embedding_and_reduce_on_device() -> None:
    mesh = _runtime_mesh()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("spyre", rank)
    local_vocab = 1024 // world_size
    hidden_size = 2816  # Gemma 4 26B-A4B text hidden size.
    vocab_start = rank * local_vocab

    # Every embedding coordinate equals its global vocabulary row.  That makes
    # incorrect masking, local-ID translation, or physical-position reduction
    # visible without relying on approximate random values.
    rows = torch.arange(
        vocab_start,
        vocab_start + local_vocab,
        dtype=torch.float16,
    )
    weight_cpu = rows.unsqueeze(1).expand(local_vocab, hidden_size).contiguous()
    weight = _dma_to_spyre_indirect_access(weight_cpu, device=device)

    global_ids = torch.tensor([[0, 511, 512, 1000]], dtype=torch.int32).repeat(1, 16)
    # HF installs the style on the full-sized meta module before replacing its
    # weight with a local shard. Reproduce that order, including the global
    # vocabulary size used to calculate ownership on rank 1.
    embedding = torch.nn.Embedding(local_vocab * world_size, hidden_size, device="meta")
    style = SpyreEmbeddingRowwiseParallel()
    style.install_forward(embedding, mesh)
    embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
    reduced = embedding(global_ids.to(device))
    expected = global_ids.to(torch.float16).unsqueeze(-1).expand_as(reduced.cpu())
    torch.testing.assert_close(reduced.cpu(), expected, rtol=0, atol=0)

    dist.barrier()
    dist.destroy_process_group()


def test_local_linear_styles_compile_with_layouts_and_one_bias() -> None:
    mesh = _runtime_mesh()
    torch.manual_seed(123)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.SiLU(), torch.nn.Linear(256, 128)
        )
        .to(dtype=torch.float16)
        .eval()
        .requires_grad_(False)
    )
    reference = deepcopy(model)
    inputs = torch.randn(64, 128, dtype=torch.float16)
    expected = reference(inputs)

    for index, style in ((0, SpyreColwiseParallel()), (2, SpyreRowwiseParallel())):
        for name, _ in list(model[index].named_parameters()):
            style.shard_param(model[index], name, mesh)
        style.install_forward(model[index], mesh)
    finalize_spyre_tp(model)
    model.to(f"spyre:{mesh.get_local_rank()}")
    bias = model[2].bias
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    device_inputs = inputs.to(f"spyre:{mesh.get_local_rank()}")
    with torch.no_grad():
        for _ in range(2):
            actual = compiled(device_inputs).cpu()
            torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    assert model[2].bias is bias
    dist.barrier()
    dist.destroy_process_group()
