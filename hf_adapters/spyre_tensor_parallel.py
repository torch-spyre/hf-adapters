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

"""Transformers 5.19 TP loading with ordinary rank-local Spyre execution.

HF owns checkpoint conversion and shard-on-read through CPU DTensor placeholders.
After loading, finalize_spyre_tp removes those wrappers before adapter preparation
and the usual module-aware Spyre DMA. No full checkpoint is loaded on every rank.
Runtime styles use local tensors and functional collectives, including when an
adapter compiles a whole block. The narrowly scoped loading patches are restored
even if from_pretrained fails.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from threading import RLock
from types import MethodType
from typing import Any, cast
from unittest.mock import patch
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
    _StridedShard,
)
from transformers.core_model_loading import build_glob_alternation
from transformers.distributed import DistributedConfig
from transformers.distributed.sharding_utils import DtensorShardOperation
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    TensorParallelLayer,
    replace_layer_number_by_wildcard,
)
from transformers.distributed.utils import TransformersDeviceMesh

from hf_adapters.hf_common import spyre_compiled_all_reduce

SPYRE_COLWISE = "spyre_colwise"
SPYRE_COLWISE_GATHER_OUTPUT = "spyre_colwise_gather_output"
SPYRE_ROWWISE = "spyre_rowwise"
SPYRE_ROWWISE_SPLIT_INPUT = "spyre_rowwise_split_input"
SPYRE_EMBEDDING_ROWWISE = "spyre_embedding_rowwise"
SPYRE_EMBEDDING_COLWISE = "spyre_embedding_colwise"
SPYRE_REPLICATED = "spyre_replicated"
SPYRE_REPLICATED_LINEAR = "spyre_replicated_linear"
SPYRE_REPLICATED_EMBEDDING = "spyre_replicated_embedding"
SPYRE_GROUPED_COLWISE_PREFIX = "spyre_grouped_colwise"
SPYRE_CPU_PACKED_COLWISE = "spyre_cpu_packed_colwise"
SPYRE_CPU_ROWWISE = "spyre_cpu_rowwise"
SPYRE_CPU_REPLICATED = "spyre_cpu_replicated"
SPYRE_MOE_EXPERTS = "spyre_moe_experts"

_LOADING_LOCK = RLock()
_CPU_MESHES: WeakKeyDictionary[DeviceMesh, dict[int, DeviceMesh]] = WeakKeyDictionary()


def _copy_default(
    shard: torch.Tensor, *, device: str | torch.device, dtype: torch.dtype | None
) -> torch.Tensor:
    from torch_spyre.model_utils import _dma_to_spyre_default

    return cast(
        torch.Tensor, _dma_to_spyre_default(shard, target_dtype=dtype, device=device)
    )


def _copy_linear(
    shard: torch.Tensor, *, device: str | torch.device, dtype: torch.dtype | None
) -> torch.Tensor:
    from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped

    if shard.ndim == 2:
        return cast(
            torch.Tensor,
            _dma_to_spyre_dim_order_swapped(shard, target_dtype=dtype, device=device),
        )
    return _copy_default(shard, device=device, dtype=dtype)


def _loading_mesh(mesh: DeviceMesh, num_shards: int | None = None) -> DeviceMesh:
    """CPU placeholders share runtime ranks, but never move shards to Spyre."""
    num_shards = mesh.size() if num_shards is None else num_shards
    if mesh.size() % num_shards:
        raise ValueError(
            f"TP size {mesh.size()} is not divisible by {num_shards} KV shards"
        )
    meshes = _CPU_MESHES.setdefault(mesh, {})
    if num_shards not in meshes:
        if num_shards == mesh.size():
            meshes[num_shards] = DeviceMesh.from_group(
                mesh.get_group(), "cpu", mesh_dim_names=("tp",)
            )
        else:
            # Contiguous rank groups share one KV shard: TP=4, KV=2 gives
            # shard ownership [0, 0, 1, 1], not [0, 1, 0, 1].
            meshes[num_shards] = DeviceMesh(
                "cpu",
                mesh.mesh.reshape(num_shards, mesh.size() // num_shards),
                mesh_dim_names=("kv_shard", "kv_replica"),
            )
    return meshes[num_shards]


class _SpyreParallel(TensorParallelLayer):
    """Inference-only styles; the mesh and parameters belong to each module."""

    name = SPYRE_REPLICATED

    def placement(self, module: nn.Module, param: str) -> Placement:
        return Replicate()

    def validate_param(
        self,
        module: nn.Module,
        param: str,
        mesh: DeviceMesh,
        parameter_name: str | None = None,
    ) -> None:
        weight = module._parameters.get(param)
        if weight is None:
            return
        placement = self.placement(module, param)
        if isinstance(placement, Shard):
            shards = getattr(self, "num_shards", mesh.size())
            last_size, _ = Shard.local_shard_size_and_offset(
                weight.shape[placement.dim], shards, shards - 1
            )
            if last_size == 0:
                raise ValueError(
                    f"{parameter_name or param}: Spyre TP does not support empty shards"
                )

    def shard_param(self, module: nn.Module, param: str, mesh: DeviceMesh) -> None:
        weight = module._parameters.get(param)
        if weight is None:
            return
        placement = self.placement(module, param)
        if isinstance(placement, Replicate):
            return
        cpu_mesh = _loading_mesh(mesh)
        module._parameters[param] = nn.Parameter(
            distribute_tensor(weight, cpu_mesh, [placement], src_data_rank=None),
            requires_grad=False,
        )

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        # Keep the existing adapter consumers working without depending on
        # private metadata that Transformers no longer supplies.
        setattr(module, "_hf_tp_plan", self.name)
        setattr(module, "_hf_device_mesh", mesh)
        return module


class SpyreColwiseParallel(_SpyreParallel):
    name = SPYRE_COLWISE

    def placement(self, module: nn.Module, param: str) -> Placement:
        weight = module._parameters[param]
        assert weight is not None
        if weight.ndim == 0:
            return Replicate()
        dim = 1 if isinstance(module, nn.Embedding) else max(0, weight.ndim - 2)
        return Shard(dim)


class SpyreColwiseGatherOutputParallel(SpyreColwiseParallel):
    name = SPYRE_COLWISE_GATHER_OUTPUT

    def validate_param(
        self,
        module: nn.Module,
        param: str,
        mesh: DeviceMesh,
        parameter_name: str | None = None,
    ) -> None:
        super().validate_param(module, param, mesh, parameter_name)
        weight = module._parameters.get(param)
        if weight is not None and weight.ndim > 0:
            placement = self.placement(module, param)
            assert isinstance(placement, Shard)
            if weight.shape[placement.dim] % mesh.size():
                raise ValueError(
                    f"{parameter_name or param}: gathered output must divide TP size"
                )

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        super().install_forward(module, mesh)
        original_forward = module.forward
        group_size, group_name = mesh.size(), mesh.get_group().group_name

        def gathered_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            output = original_forward(*args, **kwargs)
            return spyre_compiled_all_gather_last_dim(output, group_size, group_name)

        module.forward = gathered_forward
        return module


class SpyreRowwiseParallel(_SpyreParallel):
    name = SPYRE_ROWWISE
    split_input = False

    def placement(self, module: nn.Module, param: str) -> Placement:
        weight = module._parameters[param]
        assert weight is not None
        return Shard(-1) if weight.ndim > 1 else Replicate()

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        super().install_forward(module, mesh)
        original_forward = module.forward
        group_name = mesh.get_group().group_name
        rank, size = mesh.get_local_rank(), mesh.size()
        split_input = self.split_input

        def rowwise_forward(
            inputs: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            if split_input:
                local_size, offset = Shard.local_shard_size_and_offset(
                    inputs.shape[-1], size, rank
                )
                inputs = inputs.narrow(-1, offset, local_size)
            # Preserve custom forwards, but add the replicated bias only once.
            # Restore it even on failure, and keep .bias available to adapters
            # that bypass this wrapper and call F.linear directly.
            bias = module._parameters.get("bias")
            if bias is not None:
                module._parameters["bias"] = None
            try:
                output = original_forward(inputs, *args, **kwargs)
            finally:
                if bias is not None:
                    module._parameters["bias"] = bias
            output = spyre_compiled_all_reduce(output, group_name)
            return output if bias is None else output + bias

        module.forward = rowwise_forward
        return module


class SpyreRowwiseSplitInputParallel(SpyreRowwiseParallel):
    name = SPYRE_ROWWISE_SPLIT_INPUT
    split_input = True


class SpyreEmbeddingRowwiseParallel(_SpyreParallel):
    """Vocab sharding with host ID masking and a compiled lookup/reduction."""

    name = SPYRE_EMBEDDING_ROWWISE

    def placement(self, module: nn.Module, param: str) -> Placement:
        return Shard(0)

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        super().install_forward(module, mesh)
        group_name = mesh.get_group().group_name
        weight = module.get_parameter("weight")
        local_size, vocab_start = Shard.local_shard_size_and_offset(
            weight.shape[0], mesh.size(), mesh.get_local_rank()
        )
        if local_size == 0:
            raise ValueError("Spyre vocab TP does not support empty embedding shards")
        vocab_end = vocab_start + local_size
        original_forward = module.forward

        def prepare_inputs(
            mod: nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            weight = cast(torch.Tensor, mod.weight)
            assert local_size == weight.shape[0], "incorrect local vocabulary size"
            input_tensor = inputs[0]
            host_input = input_tensor.cpu()
            invalid = (host_input < vocab_start) | (host_input >= vocab_end)
            masked_input = host_input.clone() - vocab_start
            masked_input[invalid] = 0
            valid = (~invalid).to(weight.dtype).unsqueeze(-1).to(input_tensor.device)
            return masked_input.to(input_tensor.device), valid

        def embedding_forward(
            masked_input: torch.Tensor, valid: torch.Tensor
        ) -> torch.Tensor:
            return spyre_compiled_all_reduce(
                original_forward(masked_input) * valid, group_name
            )

        module.forward = torch.compile(embedding_forward, dynamic=False, fullgraph=True)
        module.register_forward_pre_hook(prepare_inputs)
        return module


class SpyreReplicatedInferenceParallel(_SpyreParallel):
    """Replicate without training-only gradient hooks."""


class SpyreReplicatedLinearParallel(_SpyreParallel):
    name = SPYRE_REPLICATED_LINEAR


class SpyreReplicatedEmbeddingParallel(_SpyreParallel):
    name = SPYRE_REPLICATED_EMBEDDING


class SpyreGroupedColwiseParallel(SpyreColwiseParallel):
    num_shards = 1

    def shard_param(self, module: nn.Module, param: str, mesh: DeviceMesh) -> None:
        weight = module._parameters.get(param)
        if weight is None:
            return
        placement = self.placement(module, param)
        if isinstance(placement, Replicate):
            return
        cpu_mesh = _loading_mesh(mesh, self.num_shards)
        placements = [placement] if cpu_mesh.ndim == 1 else [placement, Replicate()]
        module._parameters[param] = nn.Parameter(
            distribute_tensor(weight, cpu_mesh, placements, src_data_rank=None),
            requires_grad=False,
        )


class SpyreCpuPackedColwiseParallel(SpyreColwiseParallel):
    name = SPYRE_CPU_PACKED_COLWISE

    def placement(self, module: nn.Module, param: str) -> Placement:
        weight = module._parameters[param]
        assert weight is not None
        if weight.ndim < 2:
            return super().placement(module, param)
        return _StridedShard(weight.ndim - 2, split_factor=2)


class SpyreCpuRowwiseParallel(SpyreRowwiseParallel):
    name = SPYRE_CPU_ROWWISE


class SpyreCpuReplicatedParallel(_SpyreParallel):
    name = SPYRE_CPU_REPLICATED


class SpyreMoeExpertsParallel(_SpyreParallel):
    name = SPYRE_MOE_EXPERTS

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        super().install_forward(module, mesh)
        sharded = any(
            isinstance(param, DTensor)
            and any(not p.is_replicate() for p in param.placements)
            for param in module.parameters()
        )
        if sharded:
            original_forward = module.forward
            group_name = mesh.get_group().group_name

            def experts_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                return spyre_compiled_all_reduce(
                    original_forward(*args, **kwargs), group_name
                )

            module.forward = experts_forward
        return module


_STYLE_INSTANCES: dict[str, _SpyreParallel] = {
    style.name: style
    for style in (
        SpyreColwiseParallel(),
        SpyreColwiseGatherOutputParallel(),
        SpyreRowwiseParallel(),
        SpyreRowwiseSplitInputParallel(),
        SpyreEmbeddingRowwiseParallel(),
        SpyreReplicatedInferenceParallel(),
        SpyreReplicatedLinearParallel(),
        SpyreReplicatedEmbeddingParallel(),
        SpyreCpuPackedColwiseParallel(),
        SpyreCpuRowwiseParallel(),
        SpyreCpuReplicatedParallel(),
        SpyreMoeExpertsParallel(),
    )
}
_STYLE_REWRITES = {
    "colwise": SPYRE_COLWISE,
    "colwise_gather_output": SPYRE_COLWISE_GATHER_OUTPUT,
    "colwise_rep": SPYRE_COLWISE_GATHER_OUTPUT,
    "rowwise": SPYRE_ROWWISE,
    "rowwise_split_input": SPYRE_ROWWISE_SPLIT_INPUT,
    "rowwise_rep": SPYRE_ROWWISE_SPLIT_INPUT,
    "embedding_rowwise": SPYRE_EMBEDDING_ROWWISE,
    "replicated_with_grad_allreduce": SPYRE_REPLICATED,
    "packed_colwise": SPYRE_CPU_PACKED_COLWISE,
    "moe_tp_experts": SPYRE_MOE_EXPERTS,
}
_CPU_STYLE_REWRITES = {
    "packed_colwise": SPYRE_CPU_PACKED_COLWISE,
    "rowwise": SPYRE_CPU_ROWWISE,
}
_GROUPED_COLWISE_STYLES: dict[str, SpyreGroupedColwiseParallel] = {}


def _register_grouped_colwise_style(num_shards: int) -> str:
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    name = f"{SPYRE_GROUPED_COLWISE_PREFIX}_{num_shards}"
    if name not in _GROUPED_COLWISE_STYLES:
        style = SpyreGroupedColwiseParallel()
        style.name, style.num_shards = name, num_shards
        _GROUPED_COLWISE_STYLES[name] = style
        # HF's generic registry annotates Callable, but its TP registry stores styles.
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]
    return name


def register_spyre_tp_styles() -> None:
    for name, style in _STYLE_INSTANCES.items():
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]


def _matches_plan(parameter_name: str, plan: Mapping[str, str]) -> bool:
    if not plan:
        return False
    alternation, _, _ = build_glob_alternation(list(plan))
    return alternation.search(parameter_name) is not None


# Embedding styles that keep every rank's lookup full-width: vocabulary
# sharding all-reduces complete rows, and the replicated styles do not shard.
_EMBEDDING_SAFE_STYLES = frozenset(
    {
        SPYRE_EMBEDDING_ROWWISE,
        SPYRE_REPLICATED,
        SPYRE_REPLICATED_LINEAR,
        SPYRE_REPLICATED_EMBEDDING,
        SPYRE_CPU_REPLICATED,
    }
)


def _embedding_weight_style(module_name: str, plan: Mapping[str, str]) -> str | None:
    """Resolve an embedding weight's style the way loading looks it up.

    Exact parameter and module entries win (``_prefer_exact_tp_plan_entries``),
    then HF's layer-index-generic names.
    """
    generic_name = replace_layer_number_by_wildcard(module_name)
    for key in (
        f"{module_name}.weight",
        module_name,
        f"{generic_name}.weight",
        generic_name,
    ):
        if key in plan:
            return plan[key]
    return None


def _reject_hidden_sharded_embeddings(
    named_modules: Mapping[str, nn.Module], plan: Mapping[str, str]
) -> None:
    """Reject plans that would shard an embedding's hidden dimension.

    A Linear style such as ``colwise`` (``Shard(1)`` for an embedding) or
    ``rowwise`` (``Shard(-1)``) on an embedding is ``embedding_colwise`` under
    another name: each rank would return a partial-width lookup.
    """
    for name, module in named_modules.items():
        if not isinstance(module, nn.Embedding):
            continue
        style = _embedding_weight_style(name, plan)
        if style is not None and style not in _EMBEDDING_SAFE_STYLES:
            raise ValueError(
                f"Spyre TP cannot apply {style!r} to embedding {name!r}: it would "
                "shard the hidden dimension, which requires a verified full-width "
                "output gather. Use replicated embeddings or 'embedding_rowwise' "
                "instead."
            )


def prepare_spyre_tp_plan(
    model: nn.Module,
    plan: Mapping[str, str],
    *,
    cpu_staged_modules: Iterable[str] = (),
    cpu_replicated_modules: Iterable[str] = (),
    replicated_linear_modules: Iterable[str] = (),
    replicated_embedding_modules: Iterable[str] = (),
    grouped_colwise_modules: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Translate the complete plan, preserving caller and adapter overrides."""
    if {"embedding_colwise", SPYRE_EMBEDDING_COLWISE}.intersection(plan.values()):
        raise ValueError(
            "Spyre TP does not support 'embedding_colwise' (or "
            "'spyre_embedding_colwise'): hidden-dimension sharding requires a "
            "verified full-width output gather. Use replicated embeddings or "
            "'embedding_rowwise' instead."
        )
    register_spyre_tp_styles()
    staged = set(cpu_staged_modules)
    cpu_replicated_modules = set(cpu_replicated_modules)
    replicated_linear_modules = set(replicated_linear_modules)
    replicated_embedding_modules = set(replicated_embedding_modules)
    grouped_colwise_modules = dict(grouped_colwise_modules or {})
    named_modules = dict(model.named_modules())
    rewritten: dict[str, str] = {}
    for module_name, style in plan.items():
        if module_name in staged:
            try:
                rewritten[module_name] = _CPU_STYLE_REWRITES[style]
            except KeyError as error:
                raise ValueError(
                    f"CPU-staged TP module {module_name!r} uses unsupported style {style!r}"
                ) from error
        else:
            rewritten[module_name] = _STYLE_REWRITES.get(style, style)

    for modules, style in (
        (replicated_linear_modules, SPYRE_REPLICATED_LINEAR),
        (replicated_embedding_modules, SPYRE_REPLICATED_EMBEDDING),
        (cpu_replicated_modules, SPYRE_CPU_REPLICATED),
    ):
        for module_name in modules:
            if module_name in named_modules:
                rewritten[module_name] = style
    for module_name, num_shards in grouped_colwise_modules.items():
        if module_name in named_modules:
            rewritten[module_name] = _register_grouped_colwise_style(num_shards)

    for name, module in named_modules.items():
        if not name or getattr(module, "weight", None) is None:
            continue
        if _matches_plan(f"{name}.weight", rewritten):
            continue
        generic_name = replace_layer_number_by_wildcard(name)
        if isinstance(module, nn.Embedding):
            rewritten.setdefault(generic_name, SPYRE_REPLICATED_EMBEDDING)
        elif isinstance(module, nn.Linear):
            rewritten.setdefault(generic_name, SPYRE_REPLICATED_LINEAR)

    # Checked after the adapter overrides, which may replicate an embedding the
    # caller planned; any remaining non-vocabulary style is hidden sharding.
    _reject_hidden_sharded_embeddings(named_modules, rewritten)

    unsupported = (
        set(rewritten.values())
        - _STYLE_INSTANCES.keys()
        - _GROUPED_COLWISE_STYLES.keys()
    )
    if unsupported:
        raise ValueError(f"Unsupported Spyre TP styles: {sorted(unsupported)}")
    exact_overrides = {
        name: rewritten.pop(name)
        for name in (
            *replicated_linear_modules,
            *replicated_embedding_modules,
            *cpu_replicated_modules,
            *grouped_colwise_modules,
        )
        if name in rewritten
    }
    return {**exact_overrides, **rewritten}


class _ScalarSafeShardOperation(DtensorShardOperation):
    """Converter scalars belong to a matrix but have no shardable dimension."""

    def shard_tensor(
        self,
        source: Any,
        tensor_idx: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        shape = source.shape if isinstance(source, torch.Tensor) else source.get_shape()
        if len(shape) == 0:
            return cast(torch.Tensor, source[...].to(device=device, dtype=dtype))
        return super().shard_tensor(
            source, tensor_idx=tensor_idx, device=device, dtype=dtype
        )


def _initialize_tp_mesh(
    config: DistributedConfig, target_device: str
) -> tuple[torch.device, TransformersDeviceMesh]:
    if config.ep_size != 1 or config.fsdp_size != 1 or config.pp_size != 1:
        raise ValueError("Spyre loading supports tensor parallelism only")
    device_type = torch.device(target_device).type
    if device_type == "spyre":
        import torch_spyre  # noqa: F401

        getattr(torch, "spyre").set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,spyre:spyreccl" if device_type == "spyre" else "gloo",
            timeout=timedelta(hours=2),
        )
    if dist.get_world_size() != config.tp_size:
        raise ValueError("TP size must match the initialized process-group size")
    if device_type == "spyre":
        backends = dist.get_backend_config()
        if "cpu:gloo" not in backends or "spyre:spyreccl" not in backends:
            raise ValueError(
                "Spyre TP requires a process group initialized with "
                "backend='cpu:gloo,spyre:spyreccl'; the existing group is "
                f"{backends!r}. It will not be reinitialized automatically."
            )
    dense = dist.init_device_mesh(
        device_type, (1, 1, config.tp_size), mesh_dim_names=("pp", "fsdp", "tp")
    )
    expert = dist.init_device_mesh(
        device_type, (1, config.tp_size, 1), mesh_dim_names=("pp", "efsdp", "ep")
    )
    return torch.device("cpu"), TransformersDeviceMesh(dense, expert)


@contextmanager
def spyre_tp_loading(
    config: DistributedConfig, *, target_device: str
) -> Iterator[None]:
    """Adapt only this load to CPU materialization and full-plan semantics."""
    from transformers import core_model_loading
    from transformers.distributed import mixin

    original_initialize = mixin.initialize_distributed_mesh
    original_resolve = mixin.resolve_parallel_plans

    def initialize(current: DistributedConfig) -> Any:
        if current is not config:
            return original_initialize(current)
        return _initialize_tp_mesh(current, target_device)

    def resolve(model: nn.Module, current: DistributedConfig) -> Any:
        if current is not config:
            return original_resolve(model, current)
        plan = dict(cast(dict[str, str], config.tp_plan))
        # Unlike HF 5.19's merge, our resolved plan is authoritative. HF's
        # original plan may name optional modules absent from this config (e.g.
        # Gemma 4 experts). Accept those known rules without merging the defaults
        # back in, but still reject unknown rules that match no actual name.
        default_plan: dict[str, str] = getattr(model, "tp_plan", None) or {}
        names = {name for name, _ in model.named_modules()}
        names.update(name for name, _ in model.named_parameters())
        for rule in plan:
            if rule in names or rule in default_plan:
                continue
            pattern, _, _ = build_glob_alternation([rule])
            if not any(pattern.search(name) for name in names):
                raise ValueError(f"TP rule {rule!r} matches no module or parameter")
        setattr(model, "_tp_plan", plan)
        return plan, {}

    with _LOADING_LOCK, ExitStack() as stack:
        stack.enter_context(
            patch.object(mixin, "initialize_distributed_mesh", initialize)
        )
        stack.enter_context(patch.object(mixin, "resolve_parallel_plans", resolve))
        stack.enter_context(
            patch.object(
                core_model_loading, "DtensorShardOperation", _ScalarSafeShardOperation
            )
        )
        yield


@dataclass(frozen=True)
class _ParameterShard:
    shape: torch.Size
    placements: tuple[Placement, ...]
    mesh: DeviceMesh


def _reject_local_tp_save(model: nn.Module, *args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(
        "save_pretrained is not supported for prepared Spyre TP models: parameters "
        "are local shards and adapters may pad or repack them. Save the original "
        "unsharded HF model instead."
    )


def finalize_spyre_tp(model: nn.Module) -> None:
    """Remove DTensors before adapter preparation, preserving aliases and shapes."""
    aliases: dict[int, nn.Parameter] = {}
    shard_specs: dict[str, _ParameterShard] = {}
    for name, module in model.named_modules():
        for param_name, param in list(
            module.named_parameters(recurse=False, remove_duplicate=False)
        ):
            if not isinstance(param, DTensor):
                continue
            full_name = f"{name}.{param_name}" if name else param_name
            shard_specs[full_name] = _ParameterShard(
                param.shape, tuple(param.placements), param.device_mesh
            )
            if id(param) not in aliases:
                aliases[id(param)] = nn.Parameter(
                    param.to_local().detach(), requires_grad=False
                )
            module._parameters[param_name] = aliases[id(param)]
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            if hasattr(module, "in_features"):
                module.in_features = weight.shape[1]
            if (
                hasattr(module, "out_features")
                and getattr(module, "_hf_tp_plan", None) != SPYRE_COLWISE_GATHER_OUTPUT
            ):
                module.out_features = weight.shape[0]
            if isinstance(module, nn.Embedding):
                module.num_embeddings, module.embedding_dim = weight.shape
    setattr(model, "_spyre_tp_parameters", shard_specs)
    # HF 5.19 only gathers DTensors on save. After this deliberate local-tensor
    # boundary it would silently write incomplete checkpoints.
    setattr(model, "save_pretrained", MethodType(_reject_local_tp_save, model))


def _reassemble_all_gather_last_dim(
    gathered: torch.Tensor,
    input_shape: Sequence[int],
    group_size: int,
    shard_sizes: Sequence[int] | None = None,
) -> torch.Tensor:
    rank_major = gathered.reshape(group_size, *input_shape)
    if shard_sizes is not None:
        assert len(shard_sizes) == group_size
        assert all(size <= input_shape[-1] for size in shard_sizes)
        return torch.cat(
            [rank_major[rank, ..., :size] for rank, size in enumerate(shard_sizes)],
            dim=-1,
        )
    input_ndim = len(input_shape)
    rank_next_to_last = rank_major.permute(*range(1, input_ndim), 0, input_ndim)
    return rank_next_to_last.reshape(*input_shape[:-1], input_shape[-1] * group_size)


def spyre_compiled_all_gather_last_dim(
    tensor: torch.Tensor,
    group_size: int,
    group_name: str,
    shard_sizes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Gather equal padded shards, then restore last-axis order and trim padding."""
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(
        tensor, group_size, group_name
    )
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)
    return _reassemble_all_gather_last_dim(
        gathered, tensor.shape, group_size, shard_sizes=shard_sizes
    )
