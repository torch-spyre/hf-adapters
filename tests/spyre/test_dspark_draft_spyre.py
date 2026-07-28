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

Skipped unless (a) DeepSpec (the drafter modeling classes) is importable and
(b) a checkpoint path is provided per family via env:

    DSPARK_DRAFT_QWEN3=<path> DSPARK_DRAFT_GRANITE=<path> DSPARK_DRAFT_GEMMA4=<path>

Usage (on Spyre pod)::

    DSPARK_DRAFT_QWEN3=deepseek-ai/dspark_qwen3_4b_block7 \
      pytest -s -vvv tests/spyre/test_dspark_draft_spyre.py -k qwen3
"""

import os

import pytest
import torch

pytest.importorskip("deepspec", reason="DSpark drafter modeling requires DeepSpec")

_FAMILIES = {
    "qwen3": ("DSPARK_DRAFT_QWEN3", "hf_adapters.hf_dspark_qwen3"),
    "granite": ("DSPARK_DRAFT_GRANITE", "hf_adapters.hf_dspark_granite"),
    "gemma4": ("DSPARK_DRAFT_GEMMA4", "hf_adapters.hf_dspark_gemma4"),
}


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_dspark_draft_block(family):
    """prepare_for_spyre + _run_draft_block produce finite block hidden states."""
    import importlib

    env_var, module_name = _FAMILIES[family]
    ckpt = os.environ.get(env_var)
    if not ckpt:
        pytest.skip(f"set {env_var}=<checkpoint> to run the {family} draft test")

    import torch_spyre  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    from hf_adapters import hf_common
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    torch.spyre.set_device(0)
    dev = torch.device("spyre:0")
    hf_common.DEVICE = dev

    # The adapter is selected by architecture (``*DSparkModel``); confirm the
    # architecture-name dispatch routes this checkpoint to the expected module.
    resolved = resolve_adapter_module(ckpt)
    assert resolved is importlib.import_module(
        module_name
    ), f"{ckpt} resolved to {resolved.__name__}, expected {module_name}"

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
