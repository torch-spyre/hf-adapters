# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generation must project only the last prefill row without losing KV state."""

import importlib
import inspect

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    GPT2Config,
    GraniteConfig,
    LlamaConfig,
    OPTConfig,
)

from hf_adapters import auto_spyre_model, hf_common
from tests.model_registry import CAUSAL_LM_MODELS


@pytest.fixture(params=["granite", "llama", "gpt2", "opt", "gemma2"])
def prepared_model(monkeypatch, request):
    torch.manual_seed(17)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    configs = dict(
        granite=GraniteConfig(
            vocab_size=129,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=2048,
            logits_scaling=2.0,
            pad_token_id=0,
            eos_token_id=None,
        ),
        llama=LlamaConfig(
            vocab_size=129,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        gpt2=GPT2Config(
            vocab_size=129,
            n_embd=128,
            n_head=2,
            n_layer=2,
            n_positions=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        opt=OPTConfig(
            vocab_size=129,
            hidden_size=128,
            ffn_dim=256,
            word_embed_proj_dim=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        gemma2=Gemma2Config(
            vocab_size=129,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=2048,
            sliding_window=128,
            pad_token_id=0,
            eos_token_id=None,
        ),
    )
    adapter = importlib.import_module(f"hf_adapters.hf_{request.param}")
    model = AutoModelForCausalLM.from_config(configs[request.param]).eval()
    adapter.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    monkeypatch.setattr(
        auto_spyre_model, "resolve_adapter_module", lambda *args, **kwargs: adapter
    )
    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained("tiny-decoder")
    return model, adapter


@pytest.mark.parametrize("chunk_size", [None, 64, 512])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_prefill_head_matches_full_logits(prepared_model, chunk_size, batch_size):
    model, adapter = prepared_model
    model._spyre_prefill_chunk_size = chunk_size
    ids = torch.randint(1, 129, (batch_size, 139))
    mask = torch.ones_like(ids)
    if batch_size > 1:
        mask[0, 71:] = 0  # Mixed-length, right-padded prompts.
        ids[0, 71:] = 0
    kwargs = dict(
        attention_mask=mask,
        max_new_tokens=4,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
    )
    expected = hf_common.generate(adapter._run_forward, model, ids, **kwargs)

    head_rows = []
    backbone_shapes = []
    original_head = model._spyre_lm_head_forward
    original_block = model._spyre_compiled_blocks[0]

    def record_head(hidden):
        head_rows.append(tuple(hidden.shape))
        return original_head(hidden)

    def record_block(hidden, freqs, attn_mask, *args):
        backbone_shapes.append((hidden.shape[1], attn_mask.shape[-1]))
        return original_block(hidden, freqs, attn_mask, *args)

    model._spyre_lm_head_forward = record_head
    model._spyre_compiled_blocks[0] = record_block
    actual = model.generate(ids, **kwargs)

    torch.testing.assert_close(actual.sequences, expected.sequences)
    for got, want in zip(actual.logits, expected.logits):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        assert got.shape == (batch_size, 129)  # No vocabulary padding exposed.
    for got, want in zip(actual.scores, expected.scores):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    assert head_rows == [(batch_size, 1, model.config.hidden_size)] * 4
    prefill_shapes = [shape for shape in backbone_shapes if shape[0] > 1]
    assert len(prefill_shapes) == (3 if chunk_size == 64 else 1)
    assert len(set(prefill_shapes)) == 1  # No per-prefix attention specialization.


@torch.no_grad()
def test_custom_prefill_hook_takes_precedence(prepared_model):
    model, adapter = prepared_model
    ids = torch.tensor([[11, 12, 13]])
    calls = []

    def unused_backbone(*args, **kwargs):
        pytest.fail("custom prefill must not call the text backbone hook")

    def prefill(**kwargs):
        calls.append("prefill")
        return adapter._run_forward(
            kwargs["model"],
            kwargs["input_ids"],
            kwargs["position_ids"],
            kwargs["attention_mask"],
            kwargs["key_caches"],
            kwargs["value_caches"],
            kwargs["cache_index"],
        )

    expected = hf_common.generate(
        adapter._run_forward, model, ids, max_new_tokens=2, do_sample=False
    )
    actual = hf_common.generate(
        adapter._run_forward,
        model,
        ids,
        max_new_tokens=2,
        do_sample=False,
        prefill_backbone_fn=unused_backbone,
        prefill_fn=prefill,
    )
    assert calls == ["prefill"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "adapter_name",
    sorted(
        {
            info["adapter"].removesuffix(".py")
            for info in CAUSAL_LM_MODELS.values()
            if info.get("kind") != "dspark_draft"
        }
    ),
)
def test_registered_decoders_use_prefill_backbone(monkeypatch, adapter_name):
    """Cover auto-class dispatch for every registry adapter, including MoE aliases."""
    adapter = importlib.import_module(f"hf_adapters.{adapter_name}")
    model = type("PreparedModel", (), {})()
    monkeypatch.setattr(
        auto_spyre_model, "resolve_adapter_module", lambda *args, **kwargs: adapter
    )
    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    if adapter_name == "hf_diffusion_gemma":
        # Denoising consumes every canvas row; its custom loop must remain active.
        expected = object()
        monkeypatch.setattr(adapter, "generate", lambda *args, **kwargs: expected)
    else:
        backbone = adapter._run_backbone_forward
        inspect.signature(backbone).bind(*([None] * 7))
        expected = object()

        def generate(forward, actual_model, input_ids, **kwargs):
            assert actual_model is model
            assert forward is adapter._run_forward
            assert kwargs["prefill_backbone_fn"] is backbone
            return expected

        monkeypatch.setattr(hf_common, "generate", generate)
    loaded = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        "registry-model"
    )
    assert loaded.generate(torch.tensor([[1]])) is expected
