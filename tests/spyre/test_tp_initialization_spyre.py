# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Public TP loader bootstrap; run this file in its own two-rank torchrun."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401
from transformers import Qwen3Config, Qwen3ForCausalLM

from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM

if int(os.environ.get("WORLD_SIZE", "1")) != 2:
    pytest.skip("requires a two-rank torchrun", allow_module_level=True)


def test_public_tp_loader_initializes_process_group() -> None:
    assert not dist.is_initialized(), "Run this file in its own torchrun invocation"
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(771)
    reference = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=256,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=128,
                max_position_embeddings=256,
                tie_word_embeddings=False,
                eos_token_id=None,
                pad_token_id=0,
                bos_token_id=1,
            )
        )
        .to(torch.float16)
        .eval()
    )
    ids = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        expected = reference.generate(
            ids,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

    try:
        # Each rank saves its seeded checkpoint before a process group exists.
        with tempfile.TemporaryDirectory(prefix="hf-tp-initialize-") as checkpoint:
            reference.save_pretrained(checkpoint)
            model = AutoSpyreModelForCausalLM.from_pretrained(
                checkpoint, dtype=torch.float16, tp_plan="auto"
            )
        assert dist.is_initialized()
        assert dist.get_world_size() == 2
        assert model.model.layers[0].self_attn.q_proj.weight.shape[0] == 256

        # CPU synchronization uses Gloo; compiled TP generation uses SpyreCCL.
        count = torch.ones(1)
        dist.all_reduce(count)
        torch.testing.assert_close(count, torch.tensor([2.0]))
        with torch.no_grad():
            actual = model.generate(
                ids,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
        assert torch.isfinite(actual.logits[0].cpu()).all()
        torch.testing.assert_close(actual.sequences.cpu(), expected.sequences)
        torch.testing.assert_close(
            actual.logits[0].cpu().float(),
            expected.logits[0].float(),
            atol=0.02,
            rtol=0.02,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
