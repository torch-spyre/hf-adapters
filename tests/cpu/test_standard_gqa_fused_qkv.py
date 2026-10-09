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

"""The standard GQA block fuses Q/K/V when ``SPYRE_FUSE_QKV=1``.

Q, K and V read the same normalized input, so ``StandardGQAAttention`` stacks
them into one ``qkv_proj`` and splits its output. These tests check that the
fused block computes what the three separate projections computed (prefill
and decode, with and without bias), that KV-cache sizing still sees the
rank-local head counts, which layer sets ``fuse_linears`` declines, that the
fused projection is a registered model parameter on every adapter path, that
fusion drops only hooks with no forward effect (Transformers' plain colwise TP
hooks) and keeps projections with any other hook separate, and that head
padding followed by fusion computes what padding alone computed.
"""

import copy
import importlib
import logging
import types

import pytest
import torch
from torch import nn
from transformers import (
    GraniteConfig,
    GraniteForCausalLM,
    GraniteSWAConfig,
    GraniteSWAForCausalLM,
)
from transformers.integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    add_tensor_parallel_hooks_to_module,
)
from transformers.models.granite.modeling_granite import (
    GraniteDecoderLayer,
    GraniteRotaryEmbedding,
)

from hf_adapters import hf_common, hf_granite
from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    StandardGQABlock,
    _local_query_head_count,
    _materialize_decode_mask_heads,
    allocate_kv_caches,
    build_decode_mask,
    build_prefill_mask,
    fuse_linears,
    kv_cache_shapes,
    make_cache_index,
    move_model_to_spyre,
    prepare_standard_gqa_blocks,
    untie_embedding_and_lm_head,
)
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_COLWISE_GATHER_OUTPUT,
    register_spyre_tp_styles,
)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 256, 4, 2, 64


@pytest.fixture(autouse=True)
def _enable_fusion(monkeypatch):
    """Exercise the opt-in path throughout this fusion-specific suite."""
    monkeypatch.setenv("SPYRE_FUSE_QKV", "1")


@pytest.mark.parametrize("flag", [None, "0"])
def test_qkv_fusion_is_opt_in(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("SPYRE_FUSE_QKV")
    else:
        monkeypatch.setenv("SPYRE_FUSE_QKV", flag)

    def unexpected_fusion(_projections):
        pytest.fail("disabled fusion must not construct a fused projection")

    monkeypatch.setattr(hf_common, "fuse_linears", unexpected_fusion)
    _, layer = _tiny_granite_layer()
    block = StandardGQABlock(layer, is_res_mul=True)
    assert block.self_attn.qkv_proj is None
    for name in ("q_proj", "k_proj", "v_proj"):
        assert f"self_attn.{name}.weight" in block.state_dict()
    assert "self_attn.qkv_proj.weight" not in block.state_dict()


def _tiny_granite_layer(attention_bias=False):
    torch.manual_seed(0)
    config = GraniteConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        attention_bias=attention_bias,
        attention_multiplier=HEAD_DIM**-0.5,
        residual_multiplier=0.22,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
    )
    layer = GraniteDecoderLayer(config, layer_idx=0).eval().requires_grad_(False)
    return config, layer


def _prefill_then_decode(block, config, prompt_len=48, chunk=64, cache_len=128):
    """One left-padded prefill chunk, then one decode step, as ``generate`` runs them."""
    offset = chunk - prompt_len
    rope = PrecomputedRotaryEmbedding(GraniteRotaryEmbedding(config=config))
    rope.set_dtype(torch.float32)
    key_cache = torch.zeros(1, KV_HEADS, cache_len, HEAD_DIM)
    value_cache = torch.zeros_like(key_cache)
    generator = torch.Generator().manual_seed(1)

    h = torch.randn(1, chunk, HIDDEN, generator=generator)
    positions = torch.clamp(torch.arange(chunk) - offset, min=0).view(1, chunk)
    mask = build_prefill_mask(1, chunk, cache_len, offset, dtype=torch.float32)
    prefill, key_cache, value_cache = block(
        h,
        rope(h, positions),
        mask,
        key_cache,
        value_cache,
        make_cache_index(0, chunk),
    )

    h = torch.randn(1, 1, HIDDEN, generator=generator)
    mask = _materialize_decode_mask_heads(
        build_decode_mask(1, cache_len, chunk, offset, dtype=torch.float32), HEADS
    )
    decode, key_cache, value_cache = block(
        h,
        rope(h, torch.tensor([[prompt_len]])),
        mask,
        key_cache,
        value_cache,
        make_cache_index(chunk, 1),
    )
    return prefill, decode, key_cache, value_cache


@pytest.mark.parametrize("attention_bias", [False, True])
def test_fused_qkv_block_matches_separate_projections(monkeypatch, attention_bias):
    config, layer = _tiny_granite_layer(attention_bias)
    attn = layer.self_attn
    fused = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)

    qkv = fused.self_attn.qkv_proj
    assert fused.self_attn.qkv_sizes == (
        HEADS * HEAD_DIM,
        KV_HEADS * HEAD_DIM,
        KV_HEADS * HEAD_DIM,
    )
    assert not hasattr(fused.self_attn, "q_proj")
    projections = (attn.q_proj, attn.k_proj, attn.v_proj)
    assert torch.equal(qkv.weight, torch.cat([p.weight for p in projections]))
    if attention_bias:
        assert torch.equal(qkv.bias, torch.cat([p.bias for p in projections]))
    else:
        assert qkv.bias is None

    monkeypatch.setattr(hf_common, "fuse_linears", lambda linears: None)
    separate = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)
    assert separate.self_attn.qkv_proj is None

    with torch.no_grad():
        got = _prefill_then_decode(fused, config)
        want = _prefill_then_decode(separate, config)
    for name, g, w in zip(("prefill", "decode", "key_cache", "value_cache"), got, want):
        torch.testing.assert_close(g, w, msg=lambda m, name=name: f"{name}: {m}")


def test_fused_qkv_keeps_rank_local_head_counts(monkeypatch):
    """KV-cache sizing still sees this rank's heads once Q/K/V are fused."""
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)
    config, layer = _tiny_granite_layer()
    # Colwise TP2 shards: this rank owns half of the query and KV heads.
    for name, heads in (
        ("q_proj", HEADS // 2),
        ("k_proj", KV_HEADS // 2),
        ("v_proj", KV_HEADS // 2),
    ):
        full = getattr(layer.self_attn, name)
        shard = nn.Linear(HIDDEN, heads * HEAD_DIM, bias=False)
        shard.weight = nn.Parameter(
            full.weight[: heads * HEAD_DIM].clone(), requires_grad=False
        )
        setattr(layer.self_attn, name, shard)
    model = types.SimpleNamespace(config=config, layers=nn.ModuleList([layer]))

    prepare_standard_gqa_blocks(model.layers, True)

    assert model.layers[0].self_attn.qkv_proj is not None
    assert kv_cache_shapes(model) == [(KV_HEADS // 2, HEAD_DIM, HEAD_DIM)]
    assert _local_query_head_count(model, HEAD_DIM) == HEADS // 2


def test_fuse_linears_declines_layers_it_cannot_split_into_views():
    def linear(in_features, out_features, bias=True, device=None):
        return nn.Linear(in_features, out_features, bias=bias, device=device)

    fused = fuse_linears([linear(128, 64), linear(128, 96)])
    assert (fused.in_features, fused.out_features) == (128, 160)
    # Not a plain Linear, different inputs, mixed bias, mixed dtype, mixed device.
    assert fuse_linears([nn.Identity(), linear(128, 64)]) is None
    assert fuse_linears([linear(128, 64), linear(64, 64)]) is None
    assert fuse_linears([linear(128, 64), linear(128, 64, bias=False)]) is None
    assert fuse_linears([linear(128, 64), linear(128, 64).half()]) is None
    assert fuse_linears([linear(128, 64), linear(128, 64, device="meta")]) is None
    # Same weight dtype, mixed bias dtype.
    wide_bias = linear(128, 64)
    wide_bias.bias = nn.Parameter(wide_bias.bias.detach().double())
    assert fuse_linears([linear(128, 64), wide_bias]) is None
    # A split point inside a stick would make the parts copies, not views.
    assert fuse_linears([linear(128, 96), linear(128, 64)]) is None


def _tiny_causal_lm(adapter_name):
    torch.manual_seed(0)
    sizes = dict(
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=HEADS // 2,  # head_dim 128: no head padding
        num_key_value_heads=KV_HEADS // 2,
        vocab_size=256,
    )
    if adapter_name == "hf_granite_swa":
        config = GraniteSWAConfig(
            **sizes,
            sliding_window=64,
            layer_types=["full_attention", "sliding_attention"],
        )
        return GraniteSWAForCausalLM(config).eval().requires_grad_(False)
    config = GraniteConfig(**sizes)
    return GraniteForCausalLM(config).eval().requires_grad_(False)


@pytest.mark.parametrize("adapter_name", ["hf_granite", "hf_granite_swa"])
def test_fused_projection_is_a_registered_model_parameter(monkeypatch, adapter_name):
    """The model's move to Spyre places only parameters registered in the model.

    Granite builds its blocks with ``prepare_standard_gqa_blocks`` and Granite
    4.1 SWA with ``make_standard_gqa_block``. Either way every parameter a
    compiled standard block uses, including the fused projection it owns, must
    be reachable from the model, and the separate Q weights it replaced must
    not stay behind as a second copy.
    """
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)
    adapter = importlib.import_module(f"hf_adapters.{adapter_name}")
    model = _tiny_causal_lm(adapter_name)
    layers = model.model.layers
    separate_q = [layer.self_attn.q_proj.weight for layer in layers]

    # As move_model_to_spyre runs it, before the move itself.
    untie_embedding_and_lm_head(model)
    adapter.prepare_for_spyre(model)

    registered = {id(p) for p in model.parameters()}
    standard = [
        (i, block)
        for i, block in enumerate(model._spyre_compiled_blocks)
        if isinstance(block, StandardGQABlock)
    ]
    assert standard
    for i, block in standard:
        assert layers[i] is block
        assert block.self_attn.qkv_proj is not None
        assert {id(p) for p in block.parameters()} <= registered
        assert id(separate_q[i]) not in registered


class _Mesh:
    """Stands in for a DeviceMesh; installing these TP hooks needs only its size."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _add_tp_hooks(projections, plan, mesh):
    """Install the hooks Transformers' TP loader puts on modules planned ``plan``."""
    register_spyre_tp_styles()
    for projection in projections:
        add_tensor_parallel_hooks_to_module(
            types.SimpleNamespace(config=None), projection, plan, "proj", mesh
        )


# Test ids must not contain "spyre": tests/conftest.py skips those off-device.
@pytest.mark.parametrize(
    "plan", ["colwise", SPYRE_COLWISE], ids=["transformers", "hf-adapters"]
)
def test_fused_qkv_drops_only_colwise_hooks_without_forward_effect(monkeypatch, plan):
    """Plain colwise TP hooks change nothing going forward, so fusion may drop them.

    ``tp_plan="auto"`` puts them on every standard GQA Q/K/V projection
    (hf-adapters loads ``colwise`` entries as ``spyre_colwise``). The reference
    keeps the separate projections, which still run the hooks.
    """
    config, layer = _tiny_granite_layer(attention_bias=True)
    attn = layer.self_attn
    _add_tp_hooks((attn.q_proj, attn.k_proj, attn.v_proj), plan, _Mesh(2))

    fused = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)
    assert fused.self_attn.qkv_proj is not None

    monkeypatch.setattr(hf_common, "fuse_linears", lambda linears: None)
    separate = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)
    for projection in (separate.self_attn.q_proj, separate.self_attn.v_proj):
        assert len(projection._forward_pre_hooks) == len(projection._forward_hooks) == 1

    with torch.no_grad():
        got = _prefill_then_decode(fused, config)
        want = _prefill_then_decode(separate, config)
    for name, g, w in zip(("prefill", "decode", "key_cache", "value_cache"), got, want):
        torch.testing.assert_close(g, w, msg=lambda m, name=name: f"{name}: {m}")


def _double_output(module, args, output):
    return output * 2


def test_fused_qkv_keeps_projections_with_other_hooks_separate(monkeypatch, caplog):
    """A hook the fused layer would drop keeps Q/K/V separate, where it still runs."""
    config, layer = _tiny_granite_layer()
    unhooked = copy.deepcopy(layer)
    layer.self_attn.v_proj.register_forward_hook(_double_output)

    with caplog.at_level(logging.DEBUG, logger=hf_common.__name__):
        block = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)
    assert block.self_attn.qkv_proj is None
    assert "keeps 3 layers separate: layer 2 has a forward hook" in caplog.text

    monkeypatch.setattr(hf_common, "fuse_linears", lambda linears: None)
    reference = StandardGQABlock(copy.deepcopy(layer), is_res_mul=True)
    without_hook = StandardGQABlock(unhooked, is_res_mul=True)
    with torch.no_grad():
        got = _prefill_then_decode(block, config)
        want = _prefill_then_decode(reference, config)
        *_, key_cache, value_cache = _prefill_then_decode(without_hook, config)
    for name, g, w in zip(("prefill", "decode", "key_cache", "value_cache"), got, want):
        torch.testing.assert_close(g, w, msg=lambda m, name=name: f"{name}: {m}")
    # The hook really ran: it doubled V and left K alone.
    torch.testing.assert_close(got[2], key_cache)
    torch.testing.assert_close(got[3], 2 * value_cache)


class _DoublingInputColwise(ColwiseParallel):
    """A colwise TP style whose input hook doubles the input going forward."""

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        return inputs[0] * 2


class _DoublingOutputColwise(ColwiseParallel):
    """A colwise TP style whose output hook doubles the output going forward."""

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        return outputs * 2


def test_fuse_linears_declines_hooks_it_cannot_drop(monkeypatch):
    def linears(count=3):
        return [nn.Linear(128, 64) for _ in range(count)]

    plain = linears()
    _add_tp_hooks(plain, SPYRE_COLWISE, _Mesh(2))
    assert fuse_linears(plain) is not None
    # colwise_gather_output all-gathers every output across the ranks.
    gathered = linears()
    _add_tp_hooks(gathered, SPYRE_COLWISE_GATHER_OUTPUT, _Mesh(2))
    assert fuse_linears(gathered) is None
    # A colwise subclass that overrides one hook method. Transformers wraps it
    # in the same lambdas as plain colwise; only the method inside shows that
    # this hook changes the forward pass.
    for style in (_DoublingInputColwise(), _DoublingOutputColwise()):
        monkeypatch.setitem(ALL_PARALLEL_STYLES, "custom_colwise", style)
        custom = linears()
        _add_tp_hooks(custom, "custom_colwise", _Mesh(2))
        assert fuse_linears(custom) is None, type(style).__name__
    # The layers must carry the same hooks: here only the first is colwise.
    mixed = linears()
    _add_tp_hooks(mixed[:1], SPYRE_COLWISE, _Mesh(2))
    assert fuse_linears(mixed) is None
    # Any other hook: a pre-hook, a kwargs hook, an always-call hook, a
    # backward hook, or a replaced forward (the way accelerate hooks a module).
    for install in (
        lambda p: p.register_forward_pre_hook(lambda module, args: None),
        lambda p: p.register_forward_hook(
            lambda module, args, kwargs, output: None, with_kwargs=True
        ),
        lambda p: p.register_forward_hook(
            lambda module, args, output: None, always_call=True
        ),
        lambda p: p.register_full_backward_hook(lambda module, gin, gout: None),
        lambda p: setattr(p, "forward", p.forward),
    ):
        hooked = linears(2)
        install(hooked[1])
        assert fuse_linears(hooked) is None


def _tiny_padded_granite():
    """Granite with head_dim 64 and Q/K/V bias: preparation pads heads to 128."""
    torch.manual_seed(0)
    config = GraniteConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        attention_bias=True,
        vocab_size=256,
        attention_multiplier=HEAD_DIM**-0.5,
        residual_multiplier=0.22,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
    )
    return GraniteForCausalLM(config).eval().requires_grad_(False)


def _model_prefill_then_decode(adapter, model, prompt_len=48, chunk=64, cache_len=128):
    """``_prefill_then_decode`` for a whole prepared model, through its adapter."""
    offset = chunk - prompt_len
    key_caches, value_caches = allocate_kv_caches(model, 1, cache_len, torch.float32)
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, model.config.vocab_size, (1, chunk + 1), generator=generator)

    positions = torch.clamp(torch.arange(chunk) - offset, min=0).view(1, chunk)
    mask = build_prefill_mask(1, chunk, cache_len, offset, dtype=torch.float32)
    prefill = adapter._run_forward(
        model,
        ids[:, :chunk],
        positions,
        mask,
        key_caches,
        value_caches,
        make_cache_index(0, chunk),
    )

    mask = _materialize_decode_mask_heads(
        build_decode_mask(1, cache_len, chunk, offset, dtype=torch.float32),
        model._spyre_decode_mask_num_heads,
    )
    decode = adapter._run_forward(
        model,
        ids[:, chunk:],
        torch.tensor([[prompt_len]]),
        mask,
        key_caches,
        value_caches,
        make_cache_index(chunk, 1),
    )
    return prefill, decode, key_caches, value_caches


def test_fusion_after_head_padding_matches_padding_alone(monkeypatch):
    """Heads are padded first, then the padded Q/K/V are fused.

    That is the real order for head_dim 64 (Granite 2B, TinyLlama, Granite
    Vision, Granite 4.0 Micro): ``prepare_rope_and_heads`` interleave-pads Q
    and K for RoPE and end-pads V, and ``prepare_standard_gqa_blocks`` then
    stacks the padded projections and their biases.
    """
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)
    source = _tiny_padded_granite()
    fused = copy.deepcopy(source)
    move_model_to_spyre(fused, hf_granite, torch.float32)
    with monkeypatch.context() as patch:
        patch.setattr(hf_common, "fuse_linears", lambda linears: None)
        separate = copy.deepcopy(source)
        move_model_to_spyre(separate, hf_granite, torch.float32)

    padded = 2 * HEAD_DIM
    widths = (HEADS * padded, KV_HEADS * padded, KV_HEADS * padded)
    for fused_layer, separate_layer in zip(fused.model.layers, separate.model.layers):
        attn = separate_layer.self_attn
        projections = (attn.q_proj, attn.k_proj, attn.v_proj)
        assert tuple(p.out_features for p in projections) == widths
        assert fused_layer.self_attn.qkv_sizes == widths
        qkv = fused_layer.self_attn.qkv_proj
        assert torch.equal(qkv.weight, torch.cat([p.weight for p in projections]))
        assert torch.equal(qkv.bias, torch.cat([p.bias for p in projections]))
    assert kv_cache_shapes(fused) == [(KV_HEADS, padded, padded)] * 2
    assert kv_cache_shapes(separate) == kv_cache_shapes(fused)

    with torch.no_grad():
        got = _model_prefill_then_decode(hf_granite, fused)
        want = _model_prefill_then_decode(hf_granite, separate)
    names = ("prefill logits", "decode logits", "key caches", "value caches")
    for name, g, w in zip(names, got, want):
        torch.testing.assert_close(g, w, msg=lambda m, name=name: f"{name}: {m}")
