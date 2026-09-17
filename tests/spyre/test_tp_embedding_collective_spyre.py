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

"""Two-card coverage for the Spyre vocab-parallel embedding data path.

Run with::

    SPYRE_DEVICES=0,1 torchrun --nproc-per-node=2 -m pytest -q -s \
        tests/spyre/test_tp_embedding_collective_spyre.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401
from torch_spyre.model_utils import _dma_to_spyre_indirect_access

from hf_adapters.spyre_tensor_parallel import SpyreEmbeddingRowwiseParallel

if int(os.environ.get("WORLD_SIZE", "1")) != 2:
    pytest.skip("requires a two-rank torchrun", allow_module_level=True)


class _WorldMesh:
    @staticmethod
    def size():
        return dist.get_world_size()

    @staticmethod
    def get_group():
        return dist.group.WORLD

    @staticmethod
    def get_local_rank():
        return dist.get_rank()


def test_mask_on_host_then_compile_embedding_and_reduce_on_device():
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,spyre:spyreccl")

    rank = dist.get_rank()
    device = torch.device("spyre", rank)
    local_vocab = 512
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
    embedding = torch.nn.Embedding(local_vocab, hidden_size)
    embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
    style = SpyreEmbeddingRowwiseParallel()
    style.prepare_module_tp(embedding, _WorldMesh())
    reduced = embedding(global_ids.to(device))
    expected = global_ids.to(torch.float16).unsqueeze(-1).expand_as(reduced.cpu())
    torch.testing.assert_close(reduced.cpu(), expected, rtol=0, atol=0)

    dist.barrier()
    dist.destroy_process_group()
