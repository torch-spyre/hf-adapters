# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real shard-on-read coverage: do not substitute direct legacy hook calls."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    OlmoeConfig,
    OlmoeForCausalLM,
    PretrainedConfig,
    Qwen3MoeConfig,
)
from transformers.distributed import DistributedConfig, mixin

from hf_adapters.hf_common import _prefer_exact_tp_plan_entries, load_model_common
from hf_adapters.spyre_tensor_parallel import prepare_spyre_tp_plan, spyre_tp_loading


def _gemma4_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=False,
    )


def _qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


@pytest.mark.parametrize(
    ("model_config", "absent_rule"),
    (
        (_gemma4_config(), "model.layers.*.experts.gate_up_proj"),
        (_qwen3_moe_config(), "model.layers.*.mlp.gate_proj"),
    ),
    ids=("dense_gemma4", "all_expert_qwen3_moe"),
)
def test_default_tp_rules_allow_absent_optional_modules(
    model_config: PretrainedConfig, absent_rule: str
) -> None:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
    assert absent_rule in model.tp_plan
    # A partial explicit plan containing a known optional rule must also remain
    # complete: accepting the rule must not merge back the HF default plan.
    plan = prepare_spyre_tp_plan(model, {absent_rule: model.tp_plan[absent_rule]})
    config = DistributedConfig(tp_size=2, tp_plan=plan)
    with spyre_tp_loading(config, target_device="cpu"):
        resolved, expert_plan = mixin.resolve_parallel_plans(model, config)
    assert resolved == plan
    assert model.tp_plan == plan
    assert expert_plan == {}


@pytest.mark.parametrize(
    "rule",
    ("model.layers.*.self_attn.typo_proj", "model.layers.99.self_attn.q_proj"),
)
def test_unknown_unmatched_tp_rules_are_rejected(rule: str) -> None:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(_qwen3_moe_config())
    plan = prepare_spyre_tp_plan(model, {rule: "colwise"})
    config = DistributedConfig(tp_size=2, tp_plan=plan)
    with spyre_tp_loading(config, target_device="cpu"):
        with pytest.raises(ValueError, match="matches no module or parameter"):
            mixin.resolve_parallel_plans(model, config)


@pytest.mark.parametrize("style", ("embedding_colwise", "spyre_embedding_colwise"))
def test_colwise_embedding_is_rejected_before_checkpoint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, style: str
) -> None:
    _qwen3_moe_config().save_pretrained(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")

    def unexpected_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unsupported TP plan reached checkpoint loading")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="does not support 'embedding_colwise'"):
        load_model_common(
            tmp_path,
            ModuleType("adapter"),
            auto_model_cls=AutoModelForCausalLM,
            tp_plan={"model.embed_tokens": style},
        )


def test_loading_patches_are_restored_after_failure() -> None:
    from transformers import core_model_loading, modeling_utils
    from transformers.distributed import DistributedConfig, mixin, tensor_parallel

    original_initialize = mixin.initialize_distributed_mesh
    original_resolve = mixin.resolve_parallel_plans
    original_shard = core_model_loading.DtensorShardOperation
    original_lookup = tensor_parallel._get_parameter_plan
    original_model_lookup = modeling_utils._get_parameter_plan
    config = DistributedConfig(tp_size=2, tp_plan={})
    with pytest.raises(RuntimeError, match="load failed"):
        with (
            spyre_tp_loading(config, target_device="cpu"),
            _prefer_exact_tp_plan_entries(),
        ):
            raise RuntimeError("load failed")
    assert mixin.initialize_distributed_mesh is original_initialize
    assert mixin.resolve_parallel_plans is original_resolve
    assert core_model_loading.DtensorShardOperation is original_shard
    assert tensor_parallel._get_parameter_plan is original_lookup
    assert modeling_utils._get_parameter_plan is original_model_lookup


def test_tp_size_one_uses_the_ordinary_cpu_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    options: dict[str, Any] = {}

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> torch.nn.Module:
            options.update(kwargs)
            return torch.nn.Linear(4, 4)

    model = load_model_common(
        "unused", ModuleType("adapter"), auto_model_cls=AutoModel, tp_plan="auto"
    )
    assert options["device_map"] == "cpu"
    assert "distributed_config" not in options
    assert not model.training
    assert not any(param.requires_grad for param in model.parameters())
    assert not hasattr(model, "_spyre_tp_parameters")


def test_invalid_tp_string_is_rejected_at_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="tp_plan"):
        load_model_common("unused", ModuleType("adapter"), tp_plan="invalid")


@pytest.mark.parametrize("world_size", [2, 4])
def test_hf_checkpoint_loading_across_ranks(tmp_path: Path, world_size: int) -> None:
    torch.manual_seed(42)
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=True,
            attention_bias=True,
            mlp_bias=True,
        )
    )
    llama.save_pretrained(tmp_path / "llama")
    olmoe = OlmoeForCausalLM(
        OlmoeConfig(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=2,
            num_experts_per_tok=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    olmoe.save_pretrained(tmp_path / "olmoe")
    for name, model_config in (
        ("gemma4", _gemma4_config()),
        ("qwen3_moe", _qwen3_moe_config()),
    ):
        AutoModelForCausalLM.from_config(model_config).save_pretrained(tmp_path / name)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "-m",
            "tests.cpu.tp_loading_worker",
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("TP integration passed on rank") == world_size
    assert not (tmp_path / "must_not_be_written").exists()
