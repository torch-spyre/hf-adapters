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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
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
    _matches_plan,
    _reassemble_all_gather_last_dim,
    prepare_spyre_tp_plan,
    spyre_tp_loading,
)


def test_spyre_loader_initializes_cpu_and_device_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.distributed import utils

    original = utils._ensure_torch_distributed
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", Mock(return_value=False))
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)

    with spyre_tp_loading():
        utils._ensure_torch_distributed("spyre")

    initialize.assert_called_once()
    assert initialize.call_args.kwargs["backend"] == "cpu:gloo,spyre:spyreccl"
    assert initialize.call_args.kwargs.get("device_id") is None
    assert utils._ensure_torch_distributed is original


def test_spyre_loader_keeps_existing_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.distributed import utils

    monkeypatch.setattr(torch.distributed, "is_initialized", Mock(return_value=True))
    initialize = Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    with spyre_tp_loading():
        utils._ensure_torch_distributed("spyre")
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
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    @staticmethod
    def get_group():
        return SimpleNamespace(group_name="test_group")


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


def test_plan_matching_uses_transformers_glob_semantics():
    plan = {"model.layers.*.self_attn.q_proj": "colwise"}

    assert _matches_plan("model.layers.3.self_attn.q_proj.weight", plan)
    assert not _matches_plan("model.layers.3.self_attn.k_proj.weight", plan)
    assert not _matches_plan("model.layers.3.self_attn.q_proj.weight", {})


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

    def from_pretrained(model_path, *, trust_remote_code):
        seen["args"] = (model_path, trust_remote_code)
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

    assert seen["args"] == ("custom/model", True)


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


def test_exact_tp_plan_entry_precedes_transformers_wildcard_lookup() -> None:
    from transformers.distributed import tensor_parallel

    parameter = "model.layers.5.self_attn.k_proj.weight"
    plan = {
        "model.layers.*.self_attn.k_proj": SPYRE_COLWISE,
        "model.layers.5.self_attn.k_proj": SPYRE_REPLICATED_LINEAR,
    }

    assert tensor_parallel._get_parameter_plan(parameter, plan) == SPYRE_COLWISE
    with spyre_tp_loading():
        assert (
            tensor_parallel._get_parameter_plan(parameter, plan)
            == SPYRE_REPLICATED_LINEAR
        )
    assert tensor_parallel._get_parameter_plan(parameter, plan) == SPYRE_COLWISE


def test_single_stick_vocabulary_shard_needs_no_prime_factor() -> None:
    assert padded_vocab_size(33, 256, 4) == 64


def test_tp_loader_hooks_restore_after_failure() -> None:
    from transformers import core_model_loading
    from transformers.distributed import mixin, tensor_parallel, utils

    originals = (
        core_model_loading.DtensorShardOperation,
        core_model_loading._dtensor_from_local_like,
        core_model_loading.get_device,
        tensor_parallel._get_parameter_plan,
        mixin.resolve_parallel_plans,
        utils._ensure_torch_distributed,
    )
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        with spyre_tp_loading():
            raise RuntimeError("checkpoint failure")
    assert originals == (
        core_model_loading.DtensorShardOperation,
        core_model_loading._dtensor_from_local_like,
        core_model_loading.get_device,
        tensor_parallel._get_parameter_plan,
        mixin.resolve_parallel_plans,
        utils._ensure_torch_distributed,
    )


def test_cpu_staging_survives_converter_final_device_lookup() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM, core_model_loading
    from transformers.distributed import DistributedConfig, mixin

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
        mixin.resolve_parallel_plans(model, DistributedConfig(tp_size=2, tp_plan=plan))
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
