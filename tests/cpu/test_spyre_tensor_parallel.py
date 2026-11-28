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

from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from hf_adapters import hf_gemma4, hf_gemma4_mm, hf_gemma4_moe
from hf_adapters.hf_common import (
    _resolve_tp_plan,
    padded_vocab_size,
    prepare_lm_head_for_spyre,
    run_lm_head,
    untie_embedding_and_lm_head,
)
from hf_adapters.hf_gemma4 import spyre_tp_grouped_colwise_modules
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_CPU_PACKED_COLWISE,
    SPYRE_CPU_REPLICATED,
    SPYRE_CPU_ROWWISE,
    SPYRE_EMBEDDING_COLWISE,
    SPYRE_EMBEDDING_ROWWISE,
    SPYRE_GROUPED_COLWISE_PREFIX,
    SPYRE_REPLICATED_EMBEDDING,
    SPYRE_REPLICATED_LINEAR,
    SPYRE_ROWWISE,
    SpyreEmbeddingColwiseParallel,
    SpyreEmbeddingRowwiseParallel,
    SpyreRowwiseParallel,
    SpyreRowwiseSplitInputParallel,
    _matches_plan,
    _reassemble_all_gather_last_dim,
    prepare_spyre_tp_plan,
    spyre_tp_loading,
)


@pytest.mark.parametrize(
    ("device", "world_size", "tp_plan", "initialized", "should_initialize"),
    [
        ("spyre:1", 2, "auto", False, True),
        ("spyre:1", 2, {}, False, True),
        ("spyre:1", 2, "auto", True, False),
        ("spyre:1", 2, {}, True, False),
        ("spyre:0", 1, "auto", False, False),
        ("spyre:0", 2, None, False, False),
        ("cpu", 2, "auto", False, False),
        ("cpu", 2, "auto", True, False),
    ],
)
def test_public_loader_initializes_spyre_tp_before_loading(
    device: str,
    world_size: int,
    tp_plan: str | dict[str, str] | None,
    initialized: bool,
    should_initialize: bool,
    tp_plan_model: PreTrainedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hf_adapters import _model_loading, hf_common

    monkeypatch.setattr(hf_common, "DEVICE", device)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(_model_loading, "model_subfolder", Mock(return_value=""))
    monkeypatch.setattr(
        hf_common,
        "_resolve_tp_plan",
        Mock(return_value=({"lm_head": SPYRE_REPLICATED_LINEAR}, tp_plan_model.config)),
    )
    monkeypatch.setattr(
        torch.distributed, "is_initialized", Mock(return_value=initialized)
    )
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    loader = Mock()
    loader.from_pretrained.return_value = tp_plan_model
    calls = Mock()
    calls.attach_mock(initialize, "initialize")
    calls.attach_mock(loader.from_pretrained, "load")

    assert (
        hf_common.load_model_common(
            "checkpoint", None, auto_model_cls=loader, tp_plan=tp_plan
        )
        is tp_plan_model
    )
    if should_initialize:
        initialize.assert_called_once_with(backend="cpu:gloo,spyre:spyreccl")
        assert [entry[0] for entry in calls.mock_calls] == ["initialize", "load"]
    else:
        initialize.assert_not_called()
        loader.from_pretrained.assert_called_once()


def test_custom_loader_keeps_ownership_of_distributed_initialization(
    tp_plan_model: PreTrainedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hf_adapters import hf_common

    monkeypatch.setattr(hf_common, "DEVICE", "spyre:0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", Mock(return_value=False))
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    adapter = ModuleType("custom_adapter")

    def load_hf_model(
        model_path: str, dtype: torch.dtype, tp_plan: str | dict[str, str] | None = None
    ) -> PreTrainedModel:
        assert tp_plan == "auto"
        return tp_plan_model

    adapter.load_hf_model = load_hf_model
    assert (
        hf_common.load_model_common("checkpoint", adapter, tp_plan="auto")
        is tp_plan_model
    )
    initialize.assert_not_called()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 16)
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.q_proj = nn.Linear(16, 32, bias=False)
            layer.o_proj = nn.Linear(32, 16, bias=False)
            layer.router = nn.Linear(16, 4, bias=False)
            layer.experts = nn.Module()
            layer.experts.gate_up_proj = nn.Parameter(torch.empty(2, 32, 16))
            layer.experts.down_proj = nn.Parameter(torch.empty(2, 16, 16))
        self.lm_head = nn.Linear(16, 64, bias=False)


class _FakeMesh:
    _group = SimpleNamespace(group_name="test_group")

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    @staticmethod
    def get_group() -> SimpleNamespace:
        return _FakeMesh._group


def test_reassemble_dim0_all_gather_on_last_dimension():
    rank_outputs = [
        torch.arange(24).reshape(2, 3, 4),
        torch.arange(24, 48).reshape(2, 3, 4),
    ]
    gathered = torch.cat(rank_outputs, dim=0)

    result = _reassemble_all_gather_last_dim(
        gathered, rank_outputs[0].shape, len(rank_outputs)
    )

    assert torch.equal(result, torch.cat(rank_outputs, dim=-1))


def test_reassemble_dim0_all_gather_removes_per_rank_padding():
    rank_outputs = [
        torch.tensor([[[10, 11, 12, 99]]]),
        torch.tensor([[[20, 21, 98, 99]]]),
    ]
    gathered = torch.cat(rank_outputs, dim=0)

    result = _reassemble_all_gather_last_dim(
        gathered,
        rank_outputs[0].shape,
        len(rank_outputs),
        shard_sizes=(3, 2),
    )

    assert torch.equal(result, torch.tensor([[[10, 11, 12, 20, 21]]]))


def test_plan_matching_uses_transformers_numeric_wildcards() -> None:
    plan = {"model.layers.*.self_attn.q_proj": "colwise"}

    assert _matches_plan("model.layers.3.self_attn.q_proj.weight", plan)
    assert not _matches_plan("model.layers.3.self_attn.k_proj.weight", plan)
    assert not _matches_plan("model.layers.3.self_attn.q_proj.weight", {})


@pytest.mark.parametrize(
    ("name", "key", "matched"),
    [
        ("model.layers.1.blocks.2.mlp.weight", "model.layers.*.mlp", False),
        ("model.layers.decoder.mlp.weight", "model.layers.*.mlp", False),
        ("model.layers.layer1.mlp.weight", "model.layers.*.mlp", False),
        ("model.layers.1.mlp.weight", "model.layers.*.mlp", True),
        ("model.layers.1.mlp.weight", "model.layers.*.mlp.weight", True),
        (
            "model.layers.1.blocks.2.mlp.weight",
            "model.layers.*.blocks.*.mlp",
            True,
        ),
        ("model.layers.1.mlp.weight", "model.layers.*.m*", False),
        ("weight", "weight", True),
    ],
)
def test_plan_matching_agrees_with_hf_numeric_wildcards(
    name: str, key: str, matched: bool
) -> None:
    from transformers.distributed import mixin

    plan = {key: SPYRE_COLWISE}
    expected = SPYRE_COLWISE if matched else None
    assert mixin._get_parameter_plan(name, plan) == expected
    assert _matches_plan(name, MappingProxyType(plan)) is matched
    with spyre_tp_loading():
        assert mixin._get_parameter_plan(name, plan) == expected


@pytest.mark.parametrize(
    "name",
    [
        "vision.model.layers.3.self_attn.q_proj.weight",
        "modelXlayers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.q_projection.weight",
        "model.layers.3.self_attn.q_proj.extra.weight",
    ],
)
def test_plan_matching_rejects_partial_names(name: str) -> None:
    assert not _matches_plan(name, {"model.layers.*.self_attn.q_proj": SPYRE_COLWISE})


@pytest.mark.parametrize("split_input", [False, True])
def test_biased_rowwise_forward_compiles_once(
    monkeypatch: pytest.MonkeyPatch, split_input: bool
) -> None:
    from torch._dynamo.testing import CompileCounter

    from hf_adapters import spyre_tensor_parallel

    def reduce_twice(value: torch.Tensor, group_name: str) -> torch.Tensor:
        return value * 2

    monkeypatch.setattr(
        spyre_tensor_parallel, "spyre_compiled_all_reduce", reduce_twice
    )
    layer = nn.Linear(8, 4)
    bias = layer.bias
    style = SpyreRowwiseSplitInputParallel() if split_input else SpyreRowwiseParallel()
    style.install_forward(layer, _FakeMesh(size=2, rank=1))
    layer.weight = nn.Parameter(torch.randn(4, 4))
    counter = CompileCounter()
    compiled = torch.compile(layer, backend=counter, fullgraph=True)
    for _ in range(3):
        inputs = torch.randn(2, 8 if split_input else 4)
        local = inputs[:, 4:] if split_input else inputs
        expected = torch.nn.functional.linear(local, layer.weight) * 2 + bias
        torch.testing.assert_close(compiled(inputs), expected)
        assert layer.bias is bias
    assert counter.frame_count == 1


@pytest.mark.parametrize("padding_idx", [None, 0, 4])
def test_vocab_parallel_embedding_preserves_custom_forward_and_uneven_range(
    monkeypatch: pytest.MonkeyPatch,
    padding_idx: int | None,
) -> None:
    class ScaledEmbedding(nn.Embedding):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return super().forward(input_ids) * 3

    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        "hf_adapters.spyre_tensor_parallel.spyre_compiled_all_reduce",
        lambda outputs, _group_name: outputs,
    )

    embedding = ScaledEmbedding(5, 2, padding_idx=padding_idx)
    style = SpyreEmbeddingRowwiseParallel()
    style.install_forward(embedding, _FakeMesh(size=2, rank=1))
    # Simulate Transformers replacing the full meta parameter with rank 1's
    # uneven local shard after TP hooks have been installed.
    embedding.weight = nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False
    )

    output = embedding(torch.tensor([0, 2, 3, 4]))

    assert torch.equal(
        output,
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 6.0], [9.0, 12.0]]),
    )


def test_resolve_tp_plan_forwards_trust_remote_code(monkeypatch):
    seen = {}

    def from_pretrained(model_path, *, subfolder, trust_remote_code):
        seen["args"] = (model_path, subfolder, trust_remote_code)
        return SimpleNamespace()

    class AutoModel:
        @staticmethod
        def from_config(_config):
            model = _ToyModel()
            model.tp_plan = {}
            return model

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        from_pretrained,
    )

    _resolve_tp_plan(
        "custom/model",
        AutoModel,
        {},
        adapter_module=SimpleNamespace(),
        trust_remote_code=True,
    )

    assert seen["args"] == ("custom/model", "", True)


def test_shared_tp_lm_head_uses_pre_padding_shard_shape(monkeypatch):
    mesh = _FakeMesh(size=2, rank=1)
    backbone = SimpleNamespace(
        embed_tokens=SimpleNamespace(_hf_device_mesh=mesh),
    )
    model = nn.Module()
    model.lm_head = nn.Linear(2, 64, bias=False)
    cfg = SimpleNamespace(vocab_size=129, final_logit_softcapping=None)
    model.config = cfg
    model.get_input_embeddings = lambda: backbone.embed_tokens
    model._spyre_lm_head_was_tied = True
    gathered = {}

    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)

    def fake_all_gather(logits, group_size, group_name, *, shard_sizes):
        gathered["args"] = (
            logits.shape[-1],
            group_size,
            group_name,
            shard_sizes,
        )
        return logits.new_zeros(*logits.shape[:-1], sum(shard_sizes))

    monkeypatch.setattr(
        "hf_adapters.spyre_tensor_parallel.spyre_compiled_all_gather_last_dim",
        fake_all_gather,
    )

    prepare_lm_head_for_spyre(model, logits_processor=lambda logits: logits + 1)
    output = run_lm_head(model, torch.ones(1, 1, 2))

    assert model._spyre_lm_head_tp_mesh is mesh
    assert model.lm_head.weight.shape[0] == 128
    assert output.shape[-1] == cfg.vocab_size
    assert torch.equal(output, torch.ones_like(output))
    assert gathered["args"] == (128, 2, "test_group", (65, 64))


def test_untie_records_tied_lm_head_for_later_tp_detection():
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(64, 2)
            self.lm_head = nn.Linear(2, 64, bias=False)
            self.lm_head.weight = self.embedding.weight
            self.config = SimpleNamespace(tie_word_embeddings=True)

        def get_input_embeddings(self):
            return self.embedding

    model = TiedModel()
    untie_embedding_and_lm_head(model)

    assert model._spyre_lm_head_was_tied is True
    assert model.lm_head.weight.data_ptr() != model.embedding.weight.data_ptr()
    assert model.config.tie_word_embeddings is False


def test_shared_replicated_lm_head_is_not_mistaken_for_tp_after_padding(monkeypatch):
    mesh = _FakeMesh(size=2, rank=0)
    backbone = SimpleNamespace(
        embed_tokens=SimpleNamespace(_hf_device_mesh=mesh),
    )
    model = nn.Module()
    model.lm_head = nn.Linear(2, 65, bias=False)
    cfg = SimpleNamespace(vocab_size=65, final_logit_softcapping=None)
    model.config = cfg
    model.get_input_embeddings = lambda: backbone.embed_tokens
    model._spyre_lm_head_was_tied = True
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)

    prepare_lm_head_for_spyre(model)

    assert model.lm_head.weight.shape[0] == 128
    assert model._spyre_lm_head_tp_mesh is None
    assert run_lm_head(model, torch.ones(1, 1, 2)).shape[-1] == 128


def test_gemma4_multimodal_logits_use_shared_lm_head(monkeypatch):
    hidden_states = torch.tensor([[[1.0, 2.0]]])
    monkeypatch.setattr(
        hf_gemma4_mm.hf_gemma4,
        "_run_blocks_over_embeds",
        lambda *_args, **_kwargs: hidden_states,
    )

    model = SimpleNamespace(
        config=SimpleNamespace(
            final_logit_softcapping=None,
            enable_moe_block=False,
        ),
        _spyre_has_ple=False,
        _spyre_lm_head_forward=lambda value: value + 1,
        lm_head=lambda _value: (_ for _ in ()).throw(
            AssertionError("eager LM head should not run")
        ),
    )

    output = hf_gemma4_mm._logits_from_embeds(
        model,
        hidden_states,
        position_ids=None,
        attn_mask=None,
        key_caches=None,
        value_caches=None,
        cache_index=None,
    )

    assert torch.equal(output, hidden_states + 1)


def test_prepare_spyre_tp_plan_translates_and_covers_placement():
    model = _ToyModel()
    plan = {
        "model.embed_tokens": "embedding_rowwise",
        "model.layers.*.q_proj": "colwise",
        "model.layers.*.o_proj": "rowwise",
        "model.layers.*.experts.gate_up_proj": "packed_colwise",
        "model.layers.*.experts.down_proj": "rowwise",
    }
    staged = {
        "model.layers.*.experts.gate_up_proj",
        "model.layers.*.experts.down_proj",
    }

    result = prepare_spyre_tp_plan(
        model,
        plan,
        cpu_staged_modules=staged,
        replicated_linear_modules={"lm_head"},
        replicated_embedding_modules={"model.embed_tokens"},
    )

    assert result["model.embed_tokens"] == SPYRE_REPLICATED_EMBEDDING
    assert result["model.layers.*.q_proj"] == SPYRE_COLWISE
    assert result["model.layers.*.o_proj"] == SPYRE_ROWWISE
    assert result["model.layers.*.experts.gate_up_proj"] == SPYRE_CPU_PACKED_COLWISE
    assert result["model.layers.*.experts.down_proj"] == SPYRE_CPU_ROWWISE
    assert result["model.layers.*.router"] == SPYRE_REPLICATED_LINEAR
    assert result["lm_head"] == SPYRE_REPLICATED_LINEAR


@pytest.mark.parametrize(
    ("module_cls", "sharded_style", "fallback_style"),
    [
        (nn.Linear, SPYRE_COLWISE, SPYRE_REPLICATED_LINEAR),
        (nn.Embedding, SPYRE_EMBEDDING_ROWWISE, SPYRE_REPLICATED_EMBEDDING),
    ],
)
def test_prepare_spyre_tp_plan_covers_nested_weight_modules(
    module_cls: type[nn.Linear] | type[nn.Embedding],
    sharded_style: str,
    fallback_style: str,
) -> None:
    from transformers.distributed import mixin

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    layer: nn.Module = model.model.layers[1]
    layer.mlp = module_cls(4, 4)
    layer.blocks = nn.ModuleList([nn.Module(), nn.Module(), nn.Module()])
    layer.blocks[2].mlp = module_cls(4, 4)

    # The rule legitimately covers the direct module, but not the nested one.
    plan = prepare_spyre_tp_plan(model, {"model.layers.*.mlp": sharded_style})
    assert plan["model.layers.*.mlp"] == sharded_style
    assert plan["model.layers.*.blocks.*.mlp"] == fallback_style
    with spyre_tp_loading():
        expanded: dict[str, str] = mixin.DistributedMixin._expand_parallel_plan(
            model, plan
        )
    for suffix in ("", ".weight"):
        assert expanded["model.layers.1.mlp" + suffix] == sharded_style
        assert expanded["model.layers.1.blocks.2.mlp" + suffix] == fallback_style


def test_embedding_styles_reconstruct_with_their_registered_dimension() -> None:
    prepare_spyre_tp_plan(_ToyModel(), {})

    row_style = ALL_PARALLEL_STYLES[SPYRE_EMBEDDING_ROWWISE]
    col_style = ALL_PARALLEL_STYLES[SPYRE_EMBEDDING_COLWISE]
    reconstructed_row = row_style.__class__()
    reconstructed_col = col_style.__class__()

    assert isinstance(reconstructed_row, SpyreEmbeddingRowwiseParallel)
    assert reconstructed_row.shard_dim == 0
    assert isinstance(reconstructed_col, SpyreEmbeddingColwiseParallel)
    assert reconstructed_col.shard_dim == 1


def test_explicit_embedding_and_lm_head_styles_are_not_forced_replicated():
    result = prepare_spyre_tp_plan(
        _ToyModel(),
        {
            "model.embed_tokens": "embedding_rowwise",
            "lm_head": "colwise",
        },
    )

    assert result["model.embed_tokens"] == SPYRE_EMBEDDING_ROWWISE
    assert result["lm_head"] == SPYRE_COLWISE


def test_gemma4_tp4_groups_kv_only_when_a_shard_would_split_a_head():
    assert (
        hf_gemma4_moe.spyre_tp_grouped_colwise_modules
        is spyre_tp_grouped_colwise_modules
    )

    class Attention(nn.Module):
        def __init__(self, kv_heads):
            super().__init__()
            head_dim = 256
            self.q_proj = nn.Linear(16, 8 * head_dim, bias=False)
            self.k_proj = nn.Linear(16, kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(16, kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(8 * head_dim, 16, bias=False)

    class GemmaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            self.model.language_model.layers[0].self_attn = Attention(kv_heads=2)
            self.model.language_model.layers[1].self_attn = Attention(kv_heads=4)
            layer_cfg = SimpleNamespace(head_dim=256)
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(per_layer_config=[layer_cfg, layer_cfg])
            )

    model = GemmaModel()
    plan = {
        "model.language_model.layers.*.self_attn.q_proj": "colwise",
        "model.language_model.layers.*.self_attn.k_proj": "colwise",
        "model.language_model.layers.*.self_attn.v_proj": "colwise",
        "model.language_model.layers.*.self_attn.o_proj": "rowwise",
    }
    grouped = spyre_tp_grouped_colwise_modules(model, tp_size=4)
    result = prepare_spyre_tp_plan(model, plan, grouped_colwise_modules=grouped)

    prefix = "model.language_model.layers"
    grouped_style = f"{SPYRE_GROUPED_COLWISE_PREFIX}_2"
    assert result[f"{prefix}.0.self_attn.k_proj"] == grouped_style
    assert result[f"{prefix}.0.self_attn.v_proj"] == grouped_style
    assert list(result).index(f"{prefix}.0.self_attn.k_proj") < list(result).index(
        f"{prefix}.*.self_attn.k_proj"
    )
    assert result[f"{prefix}.*.self_attn.q_proj"] == SPYRE_COLWISE
    assert result[f"{prefix}.*.self_attn.o_proj"] == SPYRE_ROWWISE
    assert f"{prefix}.1.self_attn.k_proj" not in result
    assert f"{prefix}.1.self_attn.v_proj" not in result


def _run_gemma4_finish_block(tp_group_name, has_ple):
    spec = SimpleNamespace(
        tp_group_name=tp_group_name,
        has_ple=has_ple,
        post_attention_norm_eps=1e-6,
        pre_feedforward_norm_eps=1e-6,
        post_feedforward_norm_eps=1e-6,
        post_ple_norm_eps=1e-6,
    )
    eye = torch.eye(2)
    ones = torch.ones(1, 1, 2)
    hf_gemma4._finish_block(
        spec,
        residual=ones,
        attn_out=ones,
        o_weight=eye,
        o_bias=torch.tensor([3.0, 4.0]),
        post_attn_norm_weight=torch.ones(2),
        pre_ffn_norm_weight=torch.ones(2),
        post_ffn_norm_weight=torch.ones(2),
        gate_weight=eye,
        up_weight=eye,
        down_weight=eye,
        ple_gate_weight=eye if has_ple else None,
        ple_projection_weight=eye if has_ple else None,
        post_ple_norm_weight=torch.ones(2) if has_ple else None,
        layer_scalar=torch.tensor(1.0),
        per_layer_input=ones if has_ple else None,
        query_row_mask=None,
    )


def test_gemma4_finish_block_reduces_only_rowwise_projections(monkeypatch):
    events = []
    reduced_inputs = []

    def fake_all_reduce(tensor, group_name):
        events.append(("reduce", group_name))
        reduced_inputs.append(tensor.clone())
        return tensor

    def fake_rms_norm(tensor, _weight, _eps):
        events.append(("norm", None))
        return tensor

    monkeypatch.setattr(hf_gemma4, "spyre_compiled_all_reduce", fake_all_reduce)
    monkeypatch.setattr(hf_gemma4, "_gemma4_rms_norm", fake_rms_norm)

    # PLE linears are replicated, so a TP block still has exactly the two
    # rowwise reductions, each before its corresponding sandwich norm.
    _run_gemma4_finish_block("test_group", has_ple=True)
    assert events == [
        ("reduce", "test_group"),
        ("norm", None),
        ("norm", None),
        ("reduce", "test_group"),
        ("norm", None),
        ("norm", None),
    ]
    # o_proj's replicated bias must not be part of the partial sum.
    assert torch.equal(reduced_inputs[0], torch.ones(1, 1, 2))

    events.clear()
    reduced_inputs.clear()
    _run_gemma4_finish_block(None, has_ple=False)
    assert all(event != "reduce" for event, _ in events)


def test_gemma4_vlm_forwards_text_policies_and_replicates_vision_encoder():
    class ClippableLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16, bias=False)

    model = nn.Module()
    model.model = nn.Module()
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.encoder = nn.Module()
    model.model.vision_tower.encoder.layers = nn.ModuleList([nn.Module()])
    layer = model.model.vision_tower.encoder.layers[0]
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = ClippableLinear()
    layer.self_attn.q_norm = nn.LayerNorm(16)

    cpu_replicated = hf_gemma4_mm.spyre_tp_cpu_replicated_modules(model, tp_size=2)
    result = prepare_spyre_tp_plan(
        model,
        {
            "model.vision_tower.encoder.layers.*.self_attn.q_proj": "colwise",
            "model.vision_tower.encoder.layers.*.self_attn.q_norm": (
                "replicated_with_grad_allreduce"
            ),
        },
        cpu_replicated_modules=cpu_replicated,
    )

    prefix = "model.vision_tower.encoder.layers.0.self_attn"
    assert result[f"{prefix}.q_proj"] == SPYRE_CPU_REPLICATED
    assert result[f"{prefix}.q_proj.linear"] == SPYRE_CPU_REPLICATED
    assert result[f"{prefix}.q_norm"] == SPYRE_CPU_REPLICATED
    assert (
        hf_gemma4_mm.SPYRE_TP_CPU_STAGED_MODULES
        is hf_gemma4_moe.SPYRE_TP_CPU_STAGED_MODULES
    )
    assert (
        hf_gemma4_mm.spyre_tp_grouped_colwise_modules
        is spyre_tp_grouped_colwise_modules
    )


def test_spyre_tp_plan_lookup_precedence() -> None:
    from transformers.distributed import mixin

    parameter = "model.layers.5.self_attn.k_proj.weight"
    rules = {
        parameter: SPYRE_CPU_REPLICATED,
        "model.layers.5.self_attn.k_proj": SPYRE_REPLICATED_LINEAR,
        "model.layers.*.self_attn.k_proj.weight": SPYRE_COLWISE,
        "model.layers.*.self_attn.k_proj": SPYRE_ROWWISE,
    }

    assert mixin._get_parameter_plan(parameter, rules) == SPYRE_COLWISE
    plan = dict(rules)
    with spyre_tp_loading():
        for key, style in rules.items():
            assert mixin._get_parameter_plan(parameter, plan) == style
            assert _matches_plan(parameter, MappingProxyType(plan))
            del plan[key]
        assert mixin._get_parameter_plan(parameter, plan) is None
        assert not _matches_plan(parameter, plan)
    assert mixin._get_parameter_plan(parameter, rules) == SPYRE_COLWISE


@pytest.mark.parametrize(
    "parent", ["model.layers.5.self_attn", "model.layers.*.self_attn"]
)
def test_spyre_tp_module_lookup_does_not_inherit_parent_rule(parent: str) -> None:
    from transformers.distributed import mixin

    module = "model.layers.5.self_attn.k_proj"
    plan = {parent: SPYRE_COLWISE}
    with spyre_tp_loading():
        assert mixin._get_parameter_plan(module, plan, is_weight=False) is None
        assert mixin._get_parameter_plan(module, plan, is_weight=True) == SPYRE_COLWISE


def test_tied_equality_compares_on_cpu_inside_tp_loading() -> None:
    from transformers import modeling_utils

    class _Probe:
        def __init__(self, cpu_copy: str | None = None) -> None:
            self.cpu_copy = cpu_copy

        def detach(self) -> "_Probe":
            return self

        def cpu(self) -> "_Probe":
            return _Probe(cpu_copy="cpu")

    calls: list[tuple[str | None, str | None]] = []

    def record(source: _Probe, target: _Probe) -> bool:
        calls.append((source.cpu_copy, target.cpu_copy))
        return True

    original = modeling_utils._tied_params_are_equal
    modeling_utils._tied_params_are_equal = record
    try:
        with spyre_tp_loading():
            assert modeling_utils._tied_params_are_equal(_Probe(), _Probe())
    finally:
        modeling_utils._tied_params_are_equal = original
    assert calls == [("cpu", "cpu")]


def test_single_stick_vocabulary_shard_needs_no_prime_factor() -> None:
    assert padded_vocab_size(33, 256, 4) == 64


@pytest.mark.parametrize("fail", [False, True])
def test_tp_loader_hooks_restore(fail: bool) -> None:
    from contextlib import nullcontext

    from transformers import PreTrainedModel, core_model_loading, modeling_utils
    from transformers.distributed import mixin

    originals = (
        core_model_loading.DtensorShardOperation,
        core_model_loading._dtensor_from_local_like,
        core_model_loading.get_device,
        modeling_utils._tied_params_are_equal,
        mixin._get_parameter_plan,
        mixin.DistributedMixin._resolve_parallel_plans,
        mixin.DistributedMixin._validate_parallel_plan,
        PreTrainedModel._move_missing_keys_from_meta_to_device,
    )
    with (
        pytest.raises(RuntimeError, match="checkpoint failure")
        if fail
        else nullcontext()
    ):
        with spyre_tp_loading():
            if fail:
                raise RuntimeError("checkpoint failure")
    assert originals == (
        core_model_loading.DtensorShardOperation,
        core_model_loading._dtensor_from_local_like,
        core_model_loading.get_device,
        modeling_utils._tied_params_are_equal,
        mixin._get_parameter_plan,
        mixin.DistributedMixin._resolve_parallel_plans,
        mixin.DistributedMixin._validate_parallel_plan,
        PreTrainedModel._move_missing_keys_from_meta_to_device,
    )


@pytest.mark.parametrize(
    ("loaded_name", "materialized"),
    [
        ("source", True),
        ("first", True),
        ("second", True),
        (None, False),
        ("second", False),
    ],
)
def test_missing_tied_group_requires_a_materialized_member(
    loaded_name: str | None, materialized: bool
) -> None:
    from transformers import PretrainedConfig, PreTrainedModel

    model = PreTrainedModel(PretrainedConfig())
    names = ("source", "first", "second")
    for name in names:
        layer = nn.Linear(4, 4, bias=False, device="meta")
        setattr(layer.weight, "_spyre_copy_kind", "linear")
        setattr(model, name, layer)
    model.all_tied_weights_keys = {
        "first.weight": "source.weight",
        "second.weight": "source.weight",
    }
    missing = {f"{name}.weight" for name in names if name != loaded_name}
    # HF marks mapping targets initialized even when they have not been loaded.
    for key in model.all_tied_weights_keys:
        setattr(model.get_parameter(key), "_is_hf_initialized", True)
    if materialized:
        assert loaded_name is not None
        model.get_submodule(loaded_name).weight = nn.Parameter(
            torch.arange(16.0).reshape(4, 4)
        )

    with spyre_tp_loading():
        if not materialized:
            with pytest.raises(
                ValueError, match="Spyre TP requires complete checkpoint weights"
            ):
                model._move_missing_keys_from_meta_to_device(
                    missing, {"": "cpu"}, None, None
                )
            return
        before = missing.copy()
        model._move_missing_keys_from_meta_to_device(missing, {"": "cpu"}, None, None)
        assert missing == before
        for key in missing:
            parameter = model.get_parameter(key)
            assert parameter.is_meta
            assert parameter._is_hf_initialized

    model.tie_weights(missing_keys=missing, recompute_mapping=False)
    assert not missing
    assert loaded_name is not None
    loaded = model.get_parameter(f"{loaded_name}.weight")
    assert all(model.get_parameter(f"{name}.weight") is loaded for name in names)


@pytest.fixture
def tp_plan_model() -> PreTrainedModel:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.device("meta"):
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=64,
            )
        )
    # HF defaults may name optional branches absent from a particular config.
    model._tp_plan["model.layers.*.optional_proj"] = "colwise"
    return model


@pytest.mark.parametrize(
    ("key", "accepted"),
    [
        ("model.layers.0.self_attn.q_proj", True),
        ("model.layers.0.self_attn.q_proj.weight", True),
        ("model.layers.*.self_attn.q_proj", True),
        ("model.layers.*.self_attn.q_proj.weight", True),
        ("model.layers.*.optional_proj", True),
        ("model.layers.*.undeclared_proj", False),
        ("model.layers.9.self_attn.q_proj", False),
        ("vision.model.layers.0.self_attn.q_proj", False),
        ("model.layers.*.self_attn.q_*", False),
    ],
)
def test_tp_plan_validation_accepts_only_resolvable_or_declared_keys(
    key: str,
    accepted: bool,
    tp_plan_model: PreTrainedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.distributed import DistributedConfig, mixin

    model = tp_plan_model
    warning = Mock()
    monkeypatch.setattr(mixin.logger, "warning", warning)
    plan = prepare_spyre_tp_plan(model, {key: "colwise"})
    config = DistributedConfig(tp_size=2, tp_plan=plan)

    with spyre_tp_loading():
        if not accepted:
            with pytest.raises(ValueError, match="TP plan keys match no model"):
                model._resolve_parallel_plans(config)
            return
        resolved, ep_plan, fsdp_plan = model._resolve_parallel_plans(config)
        warning.assert_not_called()
        assert not ep_plan
        assert not fsdp_plan
        assert all("*" not in name for name in resolved)
        parameter = key.replace("*", "0")
        if not parameter.endswith(".weight"):
            parameter += ".weight"
        if "optional_proj" in key:
            assert parameter not in resolved
        else:
            assert resolved[parameter] == SPYRE_COLWISE
            module = parameter.removesuffix(".weight")
            if not key.endswith(".weight"):
                assert resolved[module] == SPYRE_COLWISE
                assert list(resolved).index(parameter) < list(resolved).index(module)


@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_tp_plan_validation_preserves_exact_override(
    suffix: str, tp_plan_model: PreTrainedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.distributed import DistributedConfig, mixin

    warning = Mock()
    monkeypatch.setattr(mixin.logger, "warning", warning)
    module = "model.layers.0.self_attn.q_proj"
    plan = prepare_spyre_tp_plan(
        tp_plan_model,
        {
            "model.layers.*.self_attn.q_proj": "colwise",
            module + suffix: SPYRE_REPLICATED_LINEAR,
        },
    )
    with spyre_tp_loading():
        resolved, _, _ = tp_plan_model._resolve_parallel_plans(
            DistributedConfig(tp_size=2, tp_plan=plan)
        )
    warning.assert_not_called()
    assert resolved[module + ".weight"] == SPYRE_REPLICATED_LINEAR
    assert resolved["model.layers.1.self_attn.q_proj.weight"] == SPYRE_COLWISE
    assert tp_plan_model.tp_plan[module + suffix] == SPYRE_REPLICATED_LINEAR


@pytest.mark.parametrize(
    "key", ["model.layers.0.self_attn.q_proj", "model.layers.*.optional_proj"]
)
def test_tp_plan_validation_rejects_unsupported_styles(
    key: str, tp_plan_model: PreTrainedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.distributed import DistributedConfig, mixin

    warning = Mock()
    monkeypatch.setattr(mixin.logger, "warning", warning)
    plan = prepare_spyre_tp_plan(tp_plan_model, {key: "spyre_unknown_style"})
    with spyre_tp_loading():
        with pytest.raises(ValueError, match="Unsupported `tp_plan` styles"):
            tp_plan_model._resolve_parallel_plans(
                DistributedConfig(tp_size=2, tp_plan=plan)
            )
        # A failed resolution must also discard its accepted-name exemption.
        tp_plan_model._validate_parallel_plan(
            {key: SPYRE_COLWISE}, ALL_PARALLEL_STYLES, "tp_plan"
        )
        warning.assert_called_once()
        assert key in warning.call_args.args[0]


@pytest.mark.parametrize(
    ("plan_name", "style"),
    [("ep_plan", "grouped_gemm"), ("fsdp_plan", "free_full_weight")],
)
def test_tp_plan_validation_keeps_ep_and_fsdp_diagnostics(
    plan_name: str,
    style: str,
    tp_plan_model: PreTrainedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.distributed import DistributedConfig, mixin

    warning = Mock()
    monkeypatch.setattr(mixin.logger, "warning", warning)
    optional = "model.layers.*.optional_proj"
    setattr(tp_plan_model, f"_{plan_name}", {optional: style})
    plan = prepare_spyre_tp_plan(tp_plan_model, {optional: "colwise"})
    with spyre_tp_loading():
        tp_plan_model._resolve_parallel_plans(
            DistributedConfig(
                tp_size=2,
                tp_plan=plan,
                ep_size=2 if plan_name == "ep_plan" else 1,
                fsdp_size=2 if plan_name == "fsdp_plan" else 1,
            )
        )
    warning.assert_called_once()
    message: str = warning.call_args.args[0]
    assert f"`{plan_name}` rules" in message and optional in message


def test_tp_plan_validation_keeps_unrelated_diagnostics(
    tp_plan_model: PreTrainedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.distributed import DistributedConfig, mixin

    warning = Mock()
    monkeypatch.setattr(mixin.logger, "warning", warning)
    exact = "model.layers.0.self_attn.q_proj"
    optional = "model.layers.*.optional_proj"
    plan = prepare_spyre_tp_plan(tp_plan_model, {exact: "colwise", optional: "colwise"})
    original_validate = mixin.DistributedMixin._validate_parallel_plan

    with spyre_tp_loading():
        tp_plan_model._resolve_parallel_plans(
            DistributedConfig(tp_size=2, tp_plan=plan)
        )
        warning.assert_not_called()
        # Once resolution ends, even this model gets the upstream diagnostics.
        tp_plan_model._validate_parallel_plan(plan, ALL_PARALLEL_STYLES, "tp_plan")
        warning.assert_called_once()
        message: str = warning.call_args.args[0]
        assert exact in message and optional in message

        warning.reset_mock()
        tp_plan_model._resolve_parallel_plans(
            DistributedConfig(tp_size=2, tp_plan={exact: "colwise"})
        )
        warning.assert_called_once()
        assert exact in warning.call_args.args[0]

    assert mixin.DistributedMixin._validate_parallel_plan is original_validate


def test_cpu_staging_survives_converter_final_device_lookup() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM, core_model_loading
    from transformers.distributed import DistributedConfig

    with torch.device("meta"):
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=64,
            )
        )
    staged = "model.layers.0.self_attn.q_proj"
    plan = prepare_spyre_tp_plan(
        model,
        {"model.layers.*.self_attn.q_proj": "colwise"},
        cpu_replicated_modules={staged},
    )
    device_map = {"": "cuda:7"}
    with spyre_tp_loading():
        model._resolve_parallel_plans(DistributedConfig(tp_size=2, tp_plan=plan))
        assert core_model_loading.get_device(device_map, f"{staged}.weight") == "cpu"
        assert core_model_loading.get_device(
            device_map, f"{staged}.weight", valid_torch_device=True
        ) == torch.device("cpu")
        assert (
            core_model_loading.get_device(
                device_map, "model.layers.1.self_attn.q_proj.weight"
            )
            == "cuda:7"
        )
    assert core_model_loading.get_device(device_map, f"{staged}.weight") == "cuda:7"
