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

"""Embedding backbones share model math while the runtime separates requests."""

import copy
import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    BertConfig,
    DistilBertConfig,
    ModernBertConfig,
    MPNetConfig,
    Qwen3Config,
    RobertaConfig,
    XLMRobertaConfig,
)

from hf_adapters.execution import prepare_adapter_execution


def make_backbone(family, head_dim=64, **kwargs):
    torch.manual_seed(42)
    cls = {
        "bert": BertConfig,
        "distilbert": DistilBertConfig,
        "mpnet": MPNetConfig,
        "modernbert": ModernBertConfig,
        "roberta": RobertaConfig,
        "xlm-roberta": XLMRobertaConfig,
        "qwen3": Qwen3Config,
    }[family]
    config = cls(
        hidden_size=2 * head_dim,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=head_dim,
        vocab_size=101,
        max_position_embeddings=kwargs.pop("max_position_embeddings", 66),
        pad_token_id=0,
        attn_implementation="eager" if family == "mpnet" else "sdpa",
        **({"num_key_value_heads": 1} if family == "qwen3" else {}),
        **({"hidden_dim": 256} if family == "distilbert" else {}),
        **(
            {
                "local_attention": 4,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "cls_token_id": 1,
                "sep_token_id": 2,
            }
            if family == "modernbert"
            else {}
        ),
        **kwargs,
    )
    return AutoModel.from_config(config).eval().requires_grad_(False)


class RequestAttention(nn.Module):
    def __init__(self, spec, lengths, extent):
        super().__init__()
        self.spec = spec
        self.lengths = lengths
        self.extent = extent

    def forward(self, q, k, v):
        outputs = []
        start = 0
        for length in self.lengths:
            end = start + length
            bias = self.spec.position_bias
            mask = bias()[:, :, :length, :length] if bias is not None else None
            output = F.scaled_dot_product_attention(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                attn_mask=mask,
                scale=self.spec.scale,
                is_causal=self.spec.is_causal,
                enable_gqa=True,
            )
            width = self.extent or length
            outputs.append(F.pad(output, (0, 0, 0, width - length)))
            start += width
        return torch.cat(outputs, dim=2)


@pytest.mark.parametrize(
    "family,head_dim,is_causal",
    [
        ("bert", 32, False),
        ("bert", 64, False),
        ("distilbert", 32, False),
        ("distilbert", 64, False),
        ("mpnet", 64, False),
        ("modernbert", 32, False),
        ("modernbert", 64, False),
        ("roberta", 32, False),
        ("xlm-roberta", 64, False),
        ("qwen3", 128, True),
        ("qwen3", 128, False),
    ],
)
@pytest.mark.parametrize("extent", [None, 16], ids=["packed", "rectangle"])
def test_embedding_execution_matches_separate_hf_requests(
    monkeypatch, family, head_dim, is_causal, extent
):
    stock = make_backbone(family, head_dim, is_causal=is_causal)
    has_token_types = family in ("bert", "roberta", "xlm-roberta")
    external = copy.deepcopy(stock)
    lengths = [5, 9]
    ids = [torch.arange(1, n + 1).unsqueeze(0) for n in lengths]
    token_types = [torch.arange(n).remainder(2).unsqueeze(0) for n in lengths]
    offset = 1 if family in ("roberta", "xlm-roberta", "mpnet") else 0
    positions = [torch.arange(n).unsqueeze(0) + offset for n in lengths]
    with torch.inference_mode():
        expected = [
            stock(
                x,
                position_ids=p,
                **({"token_type_ids": t} if has_token_types else {}),
                **(
                    {"attention_mask": torch.zeros(1, 1, x.shape[1], x.shape[1])}
                    if family == "qwen3" and not is_causal
                    else {}
                ),
            ).last_hidden_state
            for x, p, t in zip(ids, positions, token_types)
        ]

    compile_calls = []
    original_compile = torch.compile

    def compile_cpu(fn, **kwargs):
        compile_calls.append(kwargs)
        return original_compile(fn, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_cpu)
    prepared = prepare_adapter_execution(external, max_model_len=64, task="embed")
    assert not compile_calls
    assert len(prepared.runtime.attentions) == 2
    for index, attention in enumerate(prepared.runtime.attentions):
        spec = attention.spec
        assert spec.layer_idx == index
        assert spec.uses_cache is False
        assert spec.is_causal is is_causal
        assert spec.head_size == max(head_dim, 128 if family == "modernbert" else 64)
        assert spec.scale == pytest.approx(head_dim**-0.5)
        attention.bind(RequestAttention(spec, lengths, extent))
    prepared.runtime.compile_body(external)
    assert compile_calls == [{"dynamic": False, "fullgraph": True}] * (
        3 if family in ("qwen3", "modernbert") else 2
    )

    def pack(tensors):
        return torch.cat(
            [F.pad(t, (0, (extent or n) - n)) for t, n in zip(tensors, lengths)], dim=1
        )

    with torch.inference_mode():
        actual = prepared.forward(
            pack(ids),
            pack(positions),
            token_type_ids=pack(token_types) if has_token_types else None,
        )
    start = 0
    for n, reference in zip(lengths, expected):
        torch.testing.assert_close(
            actual[:, start : start + n], reference, atol=2e-6, rtol=2e-5
        )
        start += extent or n


@pytest.mark.parametrize(
    "family", ["bert", "distilbert", "roberta", "xlm-roberta", "mpnet", "modernbert"]
)
def test_encoder_standalone_and_external_default_segments_agree(monkeypatch, family):
    monkeypatch.setattr("hf_adapters.hf_common.DEVICE", "cpu")
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    stock = make_backbone(family, 64 if family == "mpnet" else 32)
    standalone, external = copy.deepcopy(stock), copy.deepcopy(stock)
    ids = torch.tensor([[1, 2, 3, 4]])
    offset = 1 if family in ("roberta", "xlm-roberta", "mpnet") else 0
    positions = torch.arange(4).unsqueeze(0) + offset
    with torch.inference_mode():
        expected = stock(ids).last_hidden_state
    module = importlib.import_module(
        f"hf_adapters.hf_{family}"
        if family in ("bert", "distilbert", "mpnet", "modernbert")
        else "hf_adapters.hf_xlm_roberta"
    )
    if family == "mpnet":
        monkeypatch.setattr(module, "DEVICE", "cpu")
    module.prepare_for_spyre(standalone)
    prepared = prepare_adapter_execution(external, max_model_len=64, task="embed")
    for attention in prepared.runtime.attentions:
        attention.bind(RequestAttention(attention.spec, [4], None))
    with torch.inference_mode():
        dense = module._run_backbone_forward(
            standalone, ids, torch.zeros(1, 1, 4, 4), positions, torch.zeros_like(ids)
        )
        actual = prepared.forward(ids, positions)
    torch.testing.assert_close(dense, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("setting", ["is_causal", "use_bidirectional_attention"])
def test_decoder_embedder_can_declare_bidirectional_attention(setting):
    model = make_backbone("qwen3", 128, **{setting: setting != "is_causal"})
    prepared = prepare_adapter_execution(model, max_model_len=64, task="embed")
    assert all(
        not a.spec.is_causal and not a.spec.uses_cache
        for a in prepared.runtime.attentions
    )


@pytest.mark.parametrize(
    "kwargs,max_length,error",
    [
        ({"is_decoder": True}, 64, "bidirectional"),
        ({"position_embedding_type": "relative_key"}, 64, "absolute"),
        ({}, 67, "position table"),
    ],
)
def test_encoder_rejects_unsupported_semantics_before_mutation(
    kwargs, max_length, error
):
    model = make_backbone("bert", 32, **kwargs)
    query = model.encoder.layer[0].attention.self.query
    with pytest.raises(ValueError, match=error):
        prepare_adapter_execution(model, max_model_len=max_length, task="embed")
    assert model.encoder.layer[0].attention.self.query is query
    assert not hasattr(model, "_spyre_compiled_blocks")


def test_encoder_requires_embedding_task():
    with pytest.raises(ValueError, match="requires task='embed'"):
        prepare_adapter_execution(make_backbone("bert"), max_model_len=64)


@pytest.mark.parametrize("family", ["roberta", "mpnet"])
def test_context_accounts_for_position_offset(family):
    with pytest.raises(ValueError, match="position table"):
        prepare_adapter_execution(make_backbone(family), max_model_len=66, task="embed")


@pytest.mark.parametrize("family", ["distilbert", "mpnet", "modernbert"])
def test_encoder_without_segment_embeddings_rejects_token_types(family):
    prepared = prepare_adapter_execution(
        make_backbone(family), max_model_len=64, task="embed"
    )
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="token_type_ids"):
        prepared.forward(
            ids, torch.tensor([[0, 1]]), token_type_ids=torch.zeros_like(ids)
        )


def test_mpnet_bias_follows_model_placement_and_is_not_checkpoint_state():
    model = make_backbone("mpnet", max_position_embeddings=130)
    prepared = prepare_adapter_execution(model, max_model_len=65, task="embed")
    bias = prepared.runtime.attentions[0].spec.position_bias
    assert all(a.spec.position_bias is bias for a in prepared.runtime.attentions)
    assert bias().shape == (1, 2, 128, 128)
    assert not any("_spyre_position_bias" in name for name in model.state_dict())
    model.to(dtype=torch.float64)
    assert bias().dtype == torch.float64


def test_modernbert_rope_and_local_bias_follow_model_placement():
    model = make_backbone("modernbert", max_position_embeddings=130)
    prepared = prepare_adapter_execution(model, max_model_len=65, task="embed")
    global_spec, local_spec = [a.spec for a in prepared.runtime.attentions]
    assert global_spec.position_bias is None
    bias = local_spec.position_bias()
    assert bias.shape == (1, 1, 128, 128)
    # local_attention=4 is an inclusive radius of two, not a four-token radius.
    assert bias[0, 0, 0, 2] == 0 and bias[0, 0, 2, 0] == 0
    assert bias[0, 0, 0, 3] < -1000 and bias[0, 0, 3, 0] < -1000
    assert isinstance(model._spyre_rope, nn.ModuleDict)
    assert set(model._spyre_rope) == {"full_attention", "sliding_attention"}
    assert all(
        rope._frozen and rope._cached_len == 65 for rope in model._spyre_rope.values()
    )
    assert not torch.equal(
        model._spyre_rope["full_attention"]._freq_cache,
        model._spyre_rope["sliding_attention"]._freq_cache,
    )
    assert not any(
        "_freq_cache" in key or "_spyre_position_bias" in key
        for key in model.state_dict()
    )
    model.to(dtype=torch.float64)
    assert local_spec.position_bias().dtype == torch.float64
    assert all(
        rope._freq_cache.dtype == torch.float64 for rope in model._spyre_rope.values()
    )


@pytest.mark.parametrize("failure", ["context", "rope", "window", "layer_type"])
def test_modernbert_rejects_unsupported_config_before_mutation(failure):
    model = make_backbone("modernbert")
    max_model_len = 64
    error = ""
    if failure == "context":
        max_model_len, error = 67, "context limit"
    elif failure == "rope":
        model.config.rope_parameters["sliding_attention"]["rope_type"] = "dynamic"
        error = "frozen dynamic"
    elif failure == "window":
        model.config.local_attention = 0
        error = "positive local attention window"
    else:
        model.config.layer_types[1] = "linear_attention"
        error = "full attention"
    with pytest.raises(ValueError, match=error):
        prepare_adapter_execution(model, max_model_len=max_model_len, task="embed")
    assert not hasattr(model, "_spyre_compiled_blocks")
    assert not hasattr(model.layers[0].attn, "q_proj")
