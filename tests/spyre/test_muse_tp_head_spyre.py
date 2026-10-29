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

"""Muse Glimmer TP 2 / TP 4 LM head on Spyre: every vocabulary shard correct.

Regression: an eager device slice ``logits[..., :vocab]`` of the all-gathered
logits returned the first rank's columns correctly and garbage after them
(relative error 1.0 on the second shard), so TP 2 verifies picked wrong tokens
whenever the argmax fell past the first shard of the vocabulary. The adapter
now returns the padded gathered logits and crops on the host.

Usage::

    SPYRE_DEVICES=2,3 python -m torch.distributed.run --standalone \\
        --nproc-per-node=2 -m pytest -s tests/spyre/test_muse_tp_head_spyre.py
    SPYRE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \\
        --nproc-per-node=4 -m pytest -s tests/spyre/test_muse_tp_head_spyre.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_spyre  # noqa: F401

pytestmark = pytest.mark.requires_spyre

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
if WORLD not in (2, 4):
    pytest.skip("requires a two- or four-rank torchrun", allow_module_level=True)


class _Cfg:
    vocab_size = 202048
    output_multiplier = 0.19611613513818404
    final_logit_softcapping = 20.0


class _Model:
    config = _Cfg()


@pytest.mark.parametrize("rows", [1, 16])
def test_gathered_head_is_right_on_every_shard(rows):
    from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped

    from hf_adapters import _muse_tp

    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,spyre:spyreccl")
    rank = dist.get_rank()
    device = torch.device("spyre", rank)
    vocab, hidden = 202048, 6656
    weight = (
        torch.randn(vocab, hidden, generator=torch.Generator().manual_seed(0)) * 0.02
    ).half()
    model = _Model()
    model.lm_head = nn.Linear(hidden, vocab, bias=False)
    model.lm_head.weight = nn.Parameter(weight.clone(), requires_grad=False)
    ctx = _muse_tp.TPContext(rank, WORLD, dist.group.WORLD.group_name)
    model._spyre_tp = ctx
    _muse_tp._shard_lm_head(model, ctx)
    local = model.lm_head.weight.detach()
    model.lm_head.weight = nn.Parameter(
        _dma_to_spyre_dim_order_swapped(
            local, target_dtype=torch.float16, device=device
        ),
        requires_grad=False,
    )
    x = torch.randn(
        1, rows, hidden, generator=torch.Generator().manual_seed(rows)
    ).half()
    padded = model._spyre_lm_head_forward(x.to(device)).to("cpu").float()
    per = local.shape[0]
    assert padded.shape[-1] == per * WORLD and per % 128 == 0, padded.shape
    got = padded[..., :vocab]
    ref = (
        torch.tanh((x.float() @ weight.float().t()) * _Cfg.output_multiplier / 20) * 20
    )
    for lo in range(0, vocab, per):
        hi = min(lo + per, vocab)
        rel = (got[..., lo:hi] - ref[..., lo:hi]).norm() / ref[..., lo:hi].norm()
        assert rel < 0.02, (lo, hi, float(rel))
    dist.barrier()
