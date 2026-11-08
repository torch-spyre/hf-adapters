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

"""Existing adapter bodies with dense and external attention execution."""

import copy
import importlib
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPTNeoConfig,
    GPTNeoXConfig,
    GraniteConfig,
    LlamaConfig,
    MinistralConfig,
    MistralConfig,
    Olmo2Config,
    OlmoConfig,
    OPTConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    SmolLM3Config,
)

from hf_adapters.execution import DenseAttention, prepare_adapter_execution
from hf_adapters.hf_common import get_backbone


def make_model(family, head_dim, partial, **config_kwargs):
    torch.manual_seed(29)
    config_class = {
        "gpt2": GPT2Config,
        "gpt_neo": GPTNeoConfig,
        "gpt_neox": GPTNeoXConfig,
        "granite": GraniteConfig,
        "llama": LlamaConfig,
        "mistral": MistralConfig,
        "ministral": MinistralConfig,
        "olmo": OlmoConfig,
        "olmo2": Olmo2Config,
        "opt": OPTConfig,
        "qwen2": Qwen2Config,
        "qwen3": Qwen3Config,
        "phi3": Phi3Config,
        "smollm3": SmolLM3Config,
    }[family]
    if family == "smollm3":
        config_kwargs.setdefault("no_rope_layers", [1, 0])
    if family == "gpt_neo":
        config_kwargs.setdefault("attention_types", [[["global", "local"], 1]])
    if family == "gpt2":
        config_kwargs.setdefault("n_inner", 256)
    if family == "opt":
        config_kwargs.setdefault("ffn_dim", 256)
        config_kwargs.setdefault("word_embed_proj_dim", 2 * head_dim)
    cfg = config_class(
        hidden_size=2 * head_dim,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=head_dim,
        vocab_size=129,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=64,
        original_max_position_embeddings=64,
        partial_rotary_factor=partial,
        attn_implementation="eager" if family == "gpt_neo" else "sdpa",
        **config_kwargs,
        **(
            {}
            if family in ("gpt2", "gpt_neo", "gpt_neox", "opt")
            else {"num_key_value_heads": 1}
        ),
    )
    if family == "granite":
        cfg.embedding_multiplier = 1.7
        cfg.residual_multiplier = 0.7
        cfg.attention_multiplier = 0.11
        cfg.logits_scaling = 2.5
    model = AutoModelForCausalLM.from_config(cfg).eval().requires_grad_(False)
    if family in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.uniform_(0.6, 1.4)
                if module.bias is not None:
                    module.bias.normal_(std=0.1)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.normal_(std=0.1)
    for layer in getattr(get_backbone(model), "layers", []):
        if family in ("qwen3", "olmo2"):
            layer.self_attn.q_norm.weight.uniform_(0.6, 1.4)
            layer.self_attn.k_norm.weight.uniform_(0.6, 1.4)
        if family == "olmo2":
            layer.post_attention_layernorm.weight.uniform_(0.6, 1.4)
            layer.post_feedforward_layernorm.weight.uniform_(0.6, 1.4)
        if family == "qwen2":
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                projection.bias.normal_(std=0.1)
    return model


class ReferenceAttention(nn.Module):
    def __init__(self, spec, batch, length):
        super().__init__()
        shape = (batch, spec.num_kv_heads, length, spec.head_size)
        self.register_buffer("key", torch.zeros(shape))
        self.register_buffer("value", torch.zeros(shape))
        self.index = None
        self.dense = DenseAttention(spec.scale)

    def forward(self, q, k, v):
        mask = torch.where(
            torch.arange(self.key.shape[2])[None, :] <= self.index[:, None],
            0.0,
            -torch.inf,
        )[None, None]
        output, key, value = self.dense(q, k, v, mask, self.key, self.value, self.index)
        assert key is self.key and value is self.value
        return output


@pytest.mark.parametrize(
    "family,head_dim,partial,config_kwargs",
    [
        ("granite", 64, 1.0, {}),
        ("granite", 128, 1.0, {}),
        ("llama", 64, 1.0, {}),
        ("llama", 128, 1.0, {}),
        ("mistral", 64, 1.0, {}),
        ("mistral", 128, 1.0, {}),
        ("ministral", 64, 1.0, {}),
        ("ministral", 128, 1.0, {}),
        ("ministral", 128, 1.0, {"layer_types": ["full_attention"] * 2}),
        (
            "ministral",
            64,
            1.0,
            {
                "sliding_window": 64,
                "layer_types": ["full_attention", "sliding_attention"],
            },
        ),
        ("gpt_neo", 32, 1.0, {}),
        ("gpt_neo", 64, 1.0, {}),
        ("gpt_neo", 128, 1.0, {}),
        ("gpt_neo", 64, 1.0, {"window_size": 64}),
        ("gpt_neo", 32, 1.0, {"attention_types": [[["global"], 2]], "window_size": 1}),
        ("gpt_neo", 64, 1.0, {"activation_function": "relu"}),
        ("qwen2", 64, 1.0, {}),
        ("qwen2", 128, 1.0, {}),
        ("qwen3", 128, 1.0, {}),
        ("phi3", 96, 1.0, {}),
        ("phi3", 128, 0.75, {}),
        ("phi3", 96, 0.5, {}),
        ("olmo", 64, 1.0, {}),
        ("olmo", 128, 1.0, {}),
        ("olmo", 64, 1.0, {"clip_qkv": 0.1}),
        ("olmo", 128, 1.0, {"clip_qkv": 0.1}),
        ("olmo2", 128, 1.0, {}),
        ("olmo2", 256, 1.0, {}),
        ("smollm3", 128, 1.0, {}),
        ("smollm3", 256, 1.0, {}),
        ("gpt_neox", 64, 0.25, {}),
        ("gpt_neox", 96, 0.5, {}),
        ("gpt_neox", 128, 0.25, {}),
        ("gpt_neox", 128, 1.0, {}),
        ("gpt_neox", 64, 0.25, {"use_parallel_residual": False}),
        ("gpt_neox", 64, 0.25, {"attention_bias": False}),
        ("gpt2", 64, 1.0, {}),
        ("gpt2", 128, 1.0, {}),
        ("gpt2", 64, 1.0, {"scale_attn_weights": False}),
        ("gpt2", 64, 1.0, {"scale_attn_by_inverse_layer_idx": True}),
        ("gpt2", 64, 1.0, {"activation_function": "relu"}),
        ("opt", 32, 1.0, {}),
        ("opt", 64, 1.0, {}),
        ("opt", 128, 1.0, {}),
        ("opt", 64, 1.0, {"do_layer_norm_before": False}),
    ],
)
def test_existing_body_matches_stock_and_standalone(
    monkeypatch, family, head_dim, partial, config_kwargs
):
    from hf_adapters import hf_common

    monkeypatch.setattr(hf_common, "DEVICE", "cpu")
    compile_calls = []

    def record_compile(function, **kwargs):
        compile_calls.append(function)
        return function

    monkeypatch.setattr(torch, "compile", record_compile)
    stock = make_model(family, head_dim, partial, **config_kwargs)
    dense = copy.deepcopy(stock)
    external = copy.deepcopy(stock)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14]])
    positions = torch.arange(7).unsqueeze(0) + torch.tensor([[7], [19]])
    with torch.inference_mode():
        expected = stock(ids, position_ids=positions, use_cache=False).logits
        adapter = importlib.import_module(f"hf_adapters.hf_{family}")
        adapter.prepare_for_spyre(dense)
        if family in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
            backbone = get_backbone(dense)
            norm_owner = backbone.decoder if family == "opt" else backbone
            norm_name = "ln_f" if family in ("gpt2", "gpt_neo") else "final_layer_norm"
            final_norm = getattr(norm_owner, norm_name)
            assert all(function is not final_norm for function in compile_calls)
        if hasattr(dense, "_spyre_rope"):
            dense._spyre_rope.set_dtype(torch.float32)
        prepared = prepare_adapter_execution(external, max_model_len=64)
        assert prepared.driver is adapter._run_backbone_forward
        assert not hasattr(external, "_spyre_lm_head_forward")
        assert len(prepared.runtime.attentions) == 2
        kernels = []
        for executor in prepared.runtime.attentions:
            kernel = ReferenceAttention(executor.spec, 2, 7)
            executor.bind(kernel)
            kernels.append(kernel)
        shape = kernels[0].key.shape
        keys = [torch.zeros(shape) for _ in kernels]
        values = [torch.zeros(shape) for _ in kernels]
        dense_outputs, external_outputs = [], []
        for start, end in ((0, 4), (4, 5), (5, 7)):
            index = torch.arange(start, end)
            for kernel in kernels:
                kernel.index = index
            hidden = prepared.forward(ids[:, start:end], positions[:, start:end])
            logits = external.get_output_embeddings()(hidden)
            if prepared.runtime.logits_scale is not None:
                logits = logits * prepared.runtime.logits_scale
            external_outputs.append(logits)
            mask = torch.where(
                torch.arange(7)[None, :] <= index[:, None], 0.0, -torch.inf
            )[None, None]
            dense_outputs.append(
                adapter._run_forward(
                    dense,
                    ids[:, start:end],
                    positions[:, start:end],
                    mask,
                    keys,
                    values,
                    index,
                )[..., :129]
            )
        actual = torch.cat(external_outputs, dim=1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            stock(ids, position_ids=positions, use_cache=False).logits, expected
        )
        torch.testing.assert_close(
            torch.cat(dense_outputs, dim=1), expected, atol=2e-6, rtol=2e-5
        )
        for index, kernel in enumerate(kernels):
            torch.testing.assert_close(kernel.key, keys[index])
            torch.testing.assert_close(kernel.value, values[index])
        if family == "phi3":
            assert external._spyre_q_projs[0].out_features == 256
            assert external._spyre_gate_projs[0].out_features == 256
            assert prepared.runtime.blocks[0].function.__qualname__.startswith(
                "_make_compiled_block."
            )
        elif family not in ("gpt2", "gpt_neo", "opt"):
            assert get_backbone(external).layers


@pytest.mark.parametrize(
    "family,head_dim",
    [
        ("llama", 64),
        ("llama", 128),
        ("mistral", 64),
        ("mistral", 128),
        ("ministral", 64),
        ("ministral", 128),
        ("gpt_neo", 32),
        ("gpt_neo", 64),
        ("olmo", 64),
        ("olmo", 128),
        ("olmo2", 128),
        ("olmo2", 256),
        ("smollm3", 128),
        ("smollm3", 256),
        ("gpt_neox", 64),
        ("gpt_neox", 128),
        ("gpt2", 64),
        ("gpt2", 128),
        ("opt", 32),
        ("opt", 64),
    ],
)
def test_packed_backbone_preserves_tied_embeddings_and_compiles_after_binding(
    monkeypatch, family, head_dim
):
    model = make_model(
        family,
        head_dim,
        0.25 if family == "gpt_neox" else 1.0,
        tie_word_embeddings=True,
    )
    backbone = get_backbone(model)
    config = backbone.config
    original_config = config.to_dict()
    embedding = backbone.get_input_embeddings().weight
    original_embedding = embedding.detach().clone()
    ids = torch.tensor([[1, 3, 5, 2, 4, 6, 8]])
    positions = torch.tensor([[5, 6, 7, 19, 20, 21, 22]])
    with torch.inference_mode():
        expected = torch.cat(
            [
                backbone(
                    ids[:, start:end],
                    position_ids=positions[:, start:end],
                    use_cache=False,
                ).last_hidden_state
                for start, end in ((0, 3), (3, 7))
            ],
            dim=1,
        )

    compile_calls = []
    torch_compile = torch.compile

    def compile_on_cpu(function, **kwargs):
        compile_calls.append((function, kwargs))
        return torch_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_on_cpu)
    prepared = prepare_adapter_execution(backbone, max_model_len=64)
    assert not compile_calls
    assert backbone.config is config
    assert config.to_dict() == original_config
    assert backbone.get_input_embeddings().weight is embedding
    assert model.get_output_embeddings().weight is embedding
    torch.testing.assert_close(embedding, original_embedding, rtol=0, atol=0)
    assert not hasattr(backbone, "_spyre_lm_head_forward")
    assert prepared.runtime.logits_scale is None
    with pytest.raises(RuntimeError, match="Bind attention before compiling"):
        prepared.runtime.compile_body(backbone)

    class PackedAttention(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            requests = torch.tensor([0, 0, 0, 1, 1, 1, 1])
            mask = (requests[:, None] == requests[None, :]).tril()
            self.register_buffer("mask", mask)

        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=self.mask, scale=self.scale, enable_gqa=True
            )

    for layer_idx, executor in enumerate(prepared.runtime.attentions):
        spec = executor.spec
        assert spec.layer_idx == layer_idx
        assert (spec.num_heads, spec.num_kv_heads, spec.head_size) == (
            2,
            2 if family in ("gpt2", "gpt_neo", "gpt_neox", "opt") else 1,
            max(64 if family in ("gpt2", "gpt_neo", "opt") else 128, head_dim),
        )
        assert spec.scale == pytest.approx(
            1.0 if family in ("gpt_neo", "opt") else head_dim**-0.5
        )
        executor.bind(PackedAttention(spec.scale))
    prepared.runtime.compile_body(backbone)
    eager_final_norm = family in ("gpt2", "gpt_neo", "gpt_neox", "opt")
    expected_compile_count = 2 if eager_final_norm else 3
    assert [kwargs for _, kwargs in compile_calls] == [
        {"dynamic": False, "fullgraph": True}
    ] * expected_compile_count
    if eager_final_norm:
        norm_owner = backbone.decoder if family == "opt" else backbone
        final_norm = getattr(
            norm_owner, "ln_f" if family in ("gpt2", "gpt_neo") else "final_layer_norm"
        )
        assert all(function is not final_norm for function, _ in compile_calls)
    with torch.inference_mode():
        actual = prepared.forward(ids, positions)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_scaled_llama_rope_is_rejected_before_mutation():
    model = make_model("llama", 64, 1.0)
    model.config.rope_parameters["rope_type"] = "llama3"
    original_attention = model.model.layers[0].self_attn
    with pytest.raises(ValueError, match="does not support frozen llama3 RoPE"):
        prepare_adapter_execution(model, max_model_len=64)
    assert model.model.layers[0].self_attn is original_attention
    assert not hasattr(model, "_spyre_rope")


def test_external_attention_requires_binding_and_rejects_dense_state():
    model = make_model("qwen2", 64, 1.0)
    prepared = prepare_adapter_execution(model, max_model_len=64)
    executor = prepared.runtime.attentions[0]
    q = torch.zeros(1, 2, 1, 128)
    k = v = q[:, :1]
    with pytest.raises(RuntimeError, match="Bind external"):
        executor(q, k, v, None, None, None, None)
    executor.bind(lambda q, k, v: q)
    with pytest.raises(ValueError, match="owns masks"):
        executor(q, k, v, None, k, v, torch.tensor([0]))
    with pytest.raises(RuntimeError, match="already been bound"):
        executor.bind(lambda q, k, v: q)
    with pytest.raises(ValueError, match="already been prepared"):
        prepare_adapter_execution(model, max_model_len=64)


def test_longrope_boundary_is_rejected_before_mutation():
    model = make_model("phi3", 128, 0.75)
    model.config.rope_parameters["rope_type"] = "longrope"
    with pytest.raises(ValueError, match="short-context limit"):
        prepare_adapter_execution(model, max_model_len=65)
    assert not hasattr(model, "_spyre_rope")


@pytest.mark.parametrize(
    "family,partial", [("phi3", 0.75), ("mistral", 1.0), ("ministral", 1.0)]
)
@pytest.mark.parametrize("window", [64, 262144])
def test_window_covering_context_is_accepted(family, partial, window):
    model = make_model(family, 128, partial)
    model.config.sliding_window = window
    prepared = prepare_adapter_execution(model, max_model_len=64)
    assert len(prepared.runtime.attentions) == 2
    assert model.config.sliding_window == window


@pytest.mark.parametrize(
    "family,partial", [("phi3", 0.75), ("mistral", 1.0), ("ministral", 1.0)]
)
def test_active_sliding_window_is_rejected_before_mutation(family, partial):
    model = make_model(family, 128, partial)
    model.config.sliding_window = 63
    with pytest.raises(ValueError, match="full attention within max_model_len"):
        prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(model, "_spyre_rope")


@pytest.mark.parametrize(
    "family",
    [
        "granite",
        "llama",
        "mistral",
        "ministral",
        "gpt_neo",
        "olmo",
        "olmo2",
        "smollm3",
        "gpt2",
        "gpt_neox",
        "opt",
    ],
)
def test_adapter_preparation_does_not_import_serving_libraries_tp_or_registry(family):
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import importlib.abc
                import sys

                class NoServingImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split('.')[0] in {'spyre_inference', 'vllm', 'sglang', 'sglang_spyre_backend'}:
                            raise AssertionError(f'Unexpected serving import: {fullname}')

                sys.meta_path.insert(0, NoServingImports())
                from hf_adapters.execution import prepare_adapter_execution
                from transformers import AutoConfig, AutoModel

                model = AutoModel.from_config(AutoConfig.for_model(sys.argv[1],
                    hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                    vocab_size=129, max_position_embeddings=64,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2,
                    **({"attention_types": [[["global", "local"], 1]]}
                       if sys.argv[1] == "gpt_neo" else {}),
                )).eval()
                prepared = prepare_adapter_execution(model, max_model_len=64)
                assert len(prepared.runtime.attentions) == 2
                assert 'hf_adapters.auto_spyre_model' not in sys.modules
                assert 'hf_adapters.spyre_tensor_parallel' not in sys.modules
                assert not any(
                    name.split('.')[0] in {'spyre_inference', 'vllm', 'sglang', 'sglang_spyre_backend'}
                    for name in sys.modules
                )
                """),
            family,
        ],
        check=True,
    )


def test_frozen_rope_uses_registered_device_cache_without_host_sync(monkeypatch):
    model = make_model("granite", 64, 1.0)
    prepare_adapter_execution(model, max_model_len=32)
    rope = model._spyre_rope
    assert rope._freq_cache.shape == (32, 2, 2, 64)
    assert "_freq_cache" in dict(rope.named_buffers())
    assert "_freq_cache" not in rope.state_dict()
    rope.to(dtype=torch.float64)
    assert rope._freq_cache.dtype == torch.float64

    def no_item(*args):
        raise AssertionError("Frozen RoPE must not synchronize positions to the host")

    monkeypatch.setattr(torch.Tensor, "item", no_item)
    positions = torch.tensor([[0, 3], [4, 31]])
    torch.testing.assert_close(rope(None, positions), rope._freq_cache[positions])


@pytest.mark.parametrize("family", ["gpt2", "gpt_neo", "opt"])
def test_learned_positions_need_no_rope_and_cannot_be_prepared_twice(family):
    model = make_model(family, 64, 1.0)
    prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(model, "_spyre_rope")
    with pytest.raises(ValueError, match="already been prepared"):
        prepare_adapter_execution(model, max_model_len=64)


@pytest.mark.parametrize("option", ["add_cross_attention", "reorder_and_upcast_attn"])
@pytest.mark.parametrize("execution", ["standalone", "external"])
def test_unsupported_gpt2_attention_rejected_before_mutation(option, execution):
    from hf_adapters import hf_gpt2

    model = make_model("gpt2", 64, 1.0, **{option: True})
    attention = model.transformer.h[0].attn
    with pytest.raises(ValueError, match="does not support"):
        if execution == "standalone":
            hf_gpt2.prepare_for_spyre(model)
        else:
            prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(attention, "q_proj")


def test_projected_opt_requires_serving_output_head_support(monkeypatch):
    from hf_adapters import hf_common, hf_opt

    monkeypatch.setattr(hf_common, "DEVICE", "cpu")
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    model = make_model("opt", 64, 1.0, word_embed_proj_dim=64)
    ids = torch.tensor([[1, 3, 5, 7]])
    positions = torch.arange(4).unsqueeze(0)
    with torch.inference_mode():
        expected = model(ids, position_ids=positions, use_cache=False).logits
    with pytest.raises(ValueError, match="word_embed_proj_dim"):
        prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(model, "_spyre_compiled_blocks")

    hf_opt.prepare_for_spyre(model)
    keys = [torch.zeros(1, 2, 4, 64) for _ in range(2)]
    values = [torch.zeros_like(key) for key in keys]
    mask = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    with torch.inference_mode():
        actual = hf_opt._run_forward(
            model, ids, positions, mask, keys, values, torch.arange(4)
        )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backbone_only", [False, True])
@pytest.mark.parametrize("window", [1, 63])
def test_gpt_neo_active_local_window_rejected_before_mutation(backbone_only, window):
    model = make_model("gpt_neo", 32, 1.0, window_size=window)
    if backbone_only:
        model = get_backbone(model)
    backbone = get_backbone(model)
    attention = backbone.h[0].attn.attention
    projection = attention.q_proj
    activation_forward = type(backbone.h[0].mlp.act).forward
    original_config = model.config.to_dict()
    with pytest.raises(ValueError, match="window_size >= max_model_len"):
        prepare_adapter_execution(model, max_model_len=64)
    assert attention.q_proj is projection
    assert type(backbone.h[0].mlp.act).forward is activation_forward
    assert model.config.to_dict() == original_config
    assert not hasattr(model, "_spyre_compiled_blocks")
    assert not hasattr(model, "_spyre_lm_head_forward")


def test_gpt_neo_learned_position_limit_rejected_before_mutation():
    model = make_model("gpt_neo", 32, 1.0)
    attention = model.transformer.h[0].attn.attention
    projection = attention.q_proj
    with pytest.raises(ValueError, match="learned position table"):
        prepare_adapter_execution(model, max_model_len=65)
    assert attention.q_proj is projection
    assert not hasattr(model, "_spyre_compiled_blocks")


@pytest.mark.parametrize("window", [None, 0])
def test_sliding_layers_without_a_covering_window_rejected(window):
    model = make_model("ministral", 64, 1.0)
    model.config.sliding_window = window
    with pytest.raises(ValueError, match="full attention within max_model_len"):
        prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(model, "_spyre_rope")


def test_non_attention_layer_rejected_even_with_covering_window():
    model = make_model("ministral", 64, 1.0)
    model.config.layer_types[0] = "linear_attention"
    with pytest.raises(ValueError, match="full attention within max_model_len"):
        prepare_adapter_execution(model, max_model_len=64)
    assert not hasattr(model, "_spyre_rope")
