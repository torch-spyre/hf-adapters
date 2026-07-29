# Copyright 2025 The Torch-Spyre Authors.
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

"""
Spyre test for the DSpark speculative-decoding DRAFT adapters (qwen3/gemma4/
granite). Unlike the causal-LM adapters, a DSpark drafter is a block proposer:
its public forward is ``_run_draft_block`` (target context features + a noise
block -> block hidden states), not ``generate``. This test loads each drafter via
its adapter, runs ``prepare_for_spyre`` + ``_run_draft_block`` on the Spyre card
against a synthetic context, and asserts the block hidden states are finite with
the expected shape.

The drafter checkpoints come from the model registry (``DSPARK_PATHS``), and the
library resolves each path to its adapter (``resolve_adapter_module``) — no env
vars needed. Skipped unless DeepSpec (the drafter modeling classes) is importable.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_dspark_draft_spyre.py
    pytest -s -vvv tests/spyre/test_dspark_draft_spyre.py -k qwen3
"""

import pytest
import torch
from model_registry import DSPARK_PATHS

pytest.importorskip("deepspec", reason="DSpark drafter modeling requires DeepSpec")


@pytest.mark.parametrize("ckpt", DSPARK_PATHS)
def test_dspark_draft_block(ckpt):
    """prepare_for_spyre + _run_draft_block produce finite block hidden states."""
    import torch_spyre  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    from hf_adapters import hf_common
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    torch.spyre.set_device(0)
    dev = torch.device("spyre:0")
    hf_common.DEVICE = dev

    # The library resolves the checkpoint to its adapter by architecture
    # (``*DSparkModel``); confirm it lands on a DSpark draft adapter module.
    resolved = resolve_adapter_module(ckpt)
    assert resolved.__name__.rsplit(".", 1)[-1].startswith(
        "hf_dspark_"
    ), f"{ckpt} resolved to {resolved.__name__}, expected a DSpark draft adapter"

    arch = (AutoConfig.from_pretrained(ckpt).architectures or [""])[0]
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float16, attn_implementation="sdpa"
    ).eval()
    model.requires_grad_(False)

    resolved.prepare_for_spyre(model)
    hf_common._move_to_spyre_with_layout(model, torch.float16)

    block = int(model.block_size)
    hid = model.config.hidden_size
    feat = len(model.target_layer_ids) * hid
    ctx_pad = resolved.CTX_PAD
    kv_pad = ((ctx_pad + block + 31) // 32) * 32

    ctx = torch.randn(1, ctx_pad, feat, dtype=torch.float16).to(dev)
    ids = torch.full((1, block), int(model.mask_token_id), dtype=torch.long)
    ids[:, 0] = 5
    pos = torch.arange(kv_pad).unsqueeze(0)
    freqs = model._spyre_rope(ctx.new_zeros(1, kv_pad, hid), pos)

    block_hidden = resolved._run_draft_block(model, ids, ctx, freqs, ctx_valid_len=12)
    hc = block_hidden.detach().to("cpu").float()

    assert hc.shape == (
        1,
        block,
        hid,
    ), f"unexpected block-hidden shape {tuple(hc.shape)}"
    assert torch.isfinite(hc).all(), f"non-finite block hidden states for {arch}"
