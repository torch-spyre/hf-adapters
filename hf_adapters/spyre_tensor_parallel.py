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

"""Spyre placement policies for Hugging Face tensor-parallel loading.

Transformers' TP loader owns checkpoint slicing and module hooks.  These
styles keep those responsibilities in Transformers while replacing the final
CPU-to-device copy with torch-spyre's layout-aware DMA helpers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from datetime import timedelta
from functools import lru_cache
from types import MethodType
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, _StridedShard
from transformers import PreTrainedModel
from transformers.distributed.sharding_utils import DtensorShardOperation
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    EmbeddingParallel,
    PackedColwiseParallel,
    RowwiseParallel,
    TensorParallelLayer,
    replace_layer_number_by_wildcard,
)

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

_CopyKind = Literal["default", "linear", "embedding", "cpu"]


def _copy_default(
    shard: torch.Tensor,
    *,
    device: torch.device | str | int | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if device is None or torch.device(device).type != "spyre":
        return shard.to(device=device, dtype=dtype)
    from torch_spyre.model_utils import _dma_to_spyre_default

    result: torch.Tensor = _dma_to_spyre_default(
        shard, target_dtype=dtype, device=device
    )
    return result


def _copy_linear(
    shard: torch.Tensor,
    *,
    device: torch.device | str | int | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if device is None or torch.device(device).type != "spyre":
        return shard.to(device=device, dtype=dtype)
    from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped

    if shard.ndim == 2:
        result: torch.Tensor = _dma_to_spyre_dim_order_swapped(
            shard, target_dtype=dtype, device=device
        )
        return result
    return _copy_default(shard, device=device, dtype=dtype)


def _copy_embedding(
    shard: torch.Tensor,
    *,
    device: torch.device | str | int | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if device is None or torch.device(device).type != "spyre":
        return shard.to(device=device, dtype=dtype)
    from torch_spyre.model_utils import _dma_to_spyre_indirect_access

    if shard.ndim == 2:
        moved: torch.Tensor | None = _dma_to_spyre_indirect_access(
            shard, target_dtype=dtype, device=device
        )
        if moved is not None:
            return moved
    return _copy_default(shard, device=device, dtype=dtype)


class SpyreShardOperation(DtensorShardOperation):
    """Use HF's shard-on-read algorithm, then select the Spyre DMA layout."""

    def __init__(self, param: DTensor) -> None:
        self.copy_kind: _CopyKind = getattr(param, "_spyre_copy_kind", "default")
        self.param_name: str = getattr(param, "_spyre_param_name", "<unnamed>")
        self.param_shape = param.shape
        # Upstream's constructor indexes axis 0 even for replicated scalars.
        if param.ndim != 0:
            super().__init__(param)

    def shard_tensor(
        self,
        source: Any,
        tensor_idx: int | None = None,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        # Weight converters can feed scalar clipping bounds through the same
        # operation as their matrix. Replicate that state on every rank.
        shape = source.shape if isinstance(source, torch.Tensor) else source.get_shape()
        # Without a converter, the disk tensor must match the global parameter.
        # Checking only the local shard can hide a truncated tensor on all but
        # the final rank. Converters always supply tensor_idx and may legitimately
        # feed smaller pieces (e.g. separate gate/up or per-expert matrices).
        if len(shape) != 0 and tensor_idx is None and tuple(shape) != self.param_shape:
            raise ValueError(
                "Spyre TP requires complete checkpoint weights; "
                f"{self.param_name}: checkpoint shape {tuple(shape)} "
                f"!= planned shape {self.param_shape}"
            )
        shard: torch.Tensor | None
        if len(shape) == 0:
            shard = source[...].to(device="cpu", dtype=dtype)
        elif any(isinstance(p, _StridedShard) for p in self.placements):
            # HF 5.22's per-expert path drops the packed split factor. A
            # converter may also supply gate/up separately instead of a fused
            # matrix. Describe the source's actual layout before shard-on-read.
            dropped_axis = int(
                tensor_idx is not None and len(shape) == self.param_ndim - 1
            )
            placements: list[Placement] = []
            for placement in self.placements:
                if isinstance(placement, (Shard, _StridedShard)):
                    param_dim = placement.dim % self.param_ndim
                    if param_dim < dropped_axis:
                        raise ValueError(
                            "Packed Spyre TP does not shard the expert axis"
                        )
                    dim = param_dim - dropped_axis
                    if (
                        isinstance(placement, _StridedShard)
                        and shape[dim] == self.param_shape[param_dim]
                    ):
                        placements.append(
                            _StridedShard(dim, split_factor=placement.split_factor)
                        )
                    else:
                        placements.append(Shard(dim))
                else:
                    placements.append(placement)
            source_meta = distribute_tensor(
                torch.empty(tuple(shape), device="meta"),
                self.device_mesh,
                placements,
                src_data_rank=None,
            )
            shard = DtensorShardOperation(source_meta).shard_tensor(
                source, device="cpu", dtype=dtype
            )
        else:
            shard = super().shard_tensor(source, tensor_idx, device="cpu", dtype=dtype)
        if shard is None or self.copy_kind == "cpu":
            return shard
        if (
            shard.numel() == 0
            and device is not None
            and torch.device(device).type == "spyre"
        ):
            raise ValueError(
                "TP produced an empty shard, which Spyre cannot load; reduce the TP size"
            )
        copy = {
            "linear": _copy_linear,
            "embedding": _copy_embedding,
            "default": _copy_default,
        }[self.copy_kind]
        return copy(shard, device=device, dtype=dtype)


class _SpyreLocalParallel(TensorParallelLayer):
    """DTensor placeholders for loading; ordinary local tensors for inference."""

    copy_kind: _CopyKind = "default"
    style_name = SPYRE_REPLICATED
    _replicated = False

    def shard_param(self, module: nn.Module, param: str, mesh: DeviceMesh) -> None:
        value = module._parameters.get(param)
        if value is None:
            return
        if value.ndim == 0 or type(self)._replicated:
            module._parameters[param] = nn.Parameter(
                distribute_tensor(value, mesh, [Replicate()], src_data_rank=None),
                requires_grad=value.requires_grad,
            )
        else:
            super().shard_param(module, param, mesh)
        placeholder = module._parameters[param]
        if not isinstance(placeholder, DTensor):
            raise TypeError(f"{self.style_name} did not create a DTensor for {param}")
        setattr(placeholder, "_spyre_copy_kind", self.copy_kind)
        setattr(module, "_hf_device_mesh", mesh)
        setattr(module, "_hf_tp_plan", self.style_name)

    def context_around_forward(self, module: nn.Module, mesh: DeviceMesh) -> Any:
        return nullcontext()

    def transform_inputs_pre_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    def transform_output_post_forward(
        self, module: nn.Module, output: torch.Tensor, mesh: DeviceMesh
    ) -> torch.Tensor:
        return output

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        setattr(module, "_hf_device_mesh", mesh)
        setattr(module, "_hf_tp_plan", self.style_name)
        result: nn.Module = super().install_forward(module, mesh)
        return result


def _module_size(module: nn.Module, name: str) -> int:
    value = getattr(module, name)
    if not isinstance(value, int):
        raise TypeError(f"{type(module).__name__}.{name} must be an integer")
    return value


class SpyreColwiseParallel(_SpyreLocalParallel, ColwiseParallel):
    copy_kind: _CopyKind = "linear"
    style_name = SPYRE_COLWISE

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        result = super().install_forward(module, mesh)
        setattr(
            module,
            "out_features",
            Shard.local_shard_size_and_offset(
                _module_size(module, "out_features"), mesh.size(), mesh.get_local_rank()
            )[0],
        )
        return result


class SpyreColwiseGatherOutputParallel(SpyreColwiseParallel):
    style_name = SPYRE_COLWISE_GATHER_OUTPUT

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        setattr(
            module, "_spyre_global_out_features", _module_size(module, "out_features")
        )
        result: nn.Module = super().install_forward(module, mesh)
        return result

    def transform_output_post_forward(
        self, module: nn.Module, output: torch.Tensor, mesh: DeviceMesh
    ) -> torch.Tensor:
        return _gather_uneven_last_dim(
            output, _module_size(module, "_spyre_global_out_features"), mesh
        )


class SpyreRowwiseParallel(_SpyreLocalParallel, RowwiseParallel):
    copy_kind: _CopyKind = "linear"
    style_name = SPYRE_ROWWISE

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        standard_linear = (
            isinstance(module, nn.Linear)
            and getattr(module.forward, "__func__", None) is nn.Linear.forward
        )
        if module._parameters.get("bias") is not None and not standard_linear:
            raise ValueError("Biased Spyre rowwise TP requires a standard nn.Linear")
        result = super().install_forward(module, mesh)
        setattr(
            module,
            "in_features",
            Shard.local_shard_size_and_offset(
                _module_size(module, "in_features"), mesh.size(), mesh.get_local_rank()
            )[0],
        )
        if standard_linear and isinstance(module, nn.Linear):

            def linear_forward(input: torch.Tensor) -> torch.Tensor:
                args, _ = self.transform_inputs_pre_forward(module, (input,), {}, mesh)
                # Form the local partial without bias. Do not mutate module state
                # inside Dynamo; add the replicated bias once, after reduction.
                output = F.linear(args[0], module.weight)
                return self.transform_output_post_forward(module, output, mesh)

            setattr(module, "forward", linear_forward)
        return result

    def transform_output_post_forward(
        self, module: nn.Module, output: torch.Tensor, mesh: DeviceMesh
    ) -> torch.Tensor:
        output = spyre_compiled_all_reduce(output, mesh.get_group().group_name)
        bias = module._parameters.get("bias")
        return output if bias is None else output + bias


class SpyreRowwiseSplitInputParallel(SpyreRowwiseParallel):
    style_name = SPYRE_ROWWISE_SPLIT_INPUT

    def transform_inputs_pre_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        value: torch.Tensor = args[0]
        size, offset = Shard.local_shard_size_and_offset(
            value.shape[-1], mesh.size(), mesh.get_local_rank()
        )
        return (value[..., offset : offset + size], *args[1:]), kwargs


class SpyreEmbeddingRowwiseParallel(_SpyreLocalParallel, EmbeddingParallel):
    copy_kind: _CopyKind = "embedding"
    style_name = SPYRE_EMBEDDING_ROWWISE

    def __init__(self) -> None:
        super().__init__(shard_dim=0)

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        setattr(module, "_hf_device_mesh", mesh)
        setattr(module, "_hf_tp_plan", self.style_name)
        if not isinstance(module, nn.Embedding):
            raise TypeError("Vocab-parallel placement requires an Embedding")
        group_name = mesh.get_group().group_name
        size, start = Shard.local_shard_size_and_offset(
            module.weight.shape[0], mesh.size(), mesh.get_local_rank()
        )
        end = start + size
        if module.padding_idx is not None:
            module.padding_idx = (
                module.padding_idx - start
                if start <= module.padding_idx < end
                else None
            )
        original_forward = module.forward

        def prepare_inputs(
            mod: nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert isinstance(mod, nn.Embedding)
            assert (
                size == mod.weight.shape[0]
            ), "embedding vocabulary shard has the wrong size"
            input_tensor = inputs[0]
            # Spyre integer masking is host-staged: each call transfers IDs to
            # CPU, then local IDs and the validity mask back to the input device.
            # This also applies to one-token decode; benchmark the complete
            # embedding call (including these copies), not just its compiled body.
            host = input_tensor.cpu()
            invalid = (host < start) | (host >= end)
            masked = host - start
            masked[invalid] = 0
            valid = (
                (~invalid).to(mod.weight.dtype).unsqueeze(-1).to(input_tensor.device)
            )
            return masked.to(input_tensor.device), valid

        def embedding_forward(
            masked: torch.Tensor, valid: torch.Tensor
        ) -> torch.Tensor:
            return spyre_compiled_all_reduce(
                original_forward(masked) * valid, group_name
            )

        module.num_embeddings = size
        setattr(
            module,
            "forward",
            torch.compile(embedding_forward, dynamic=False, fullgraph=True),
        )
        module.register_forward_pre_hook(prepare_inputs)
        return module


class SpyreEmbeddingColwiseParallel(_SpyreLocalParallel, EmbeddingParallel):
    copy_kind: _CopyKind = "embedding"
    style_name = SPYRE_EMBEDDING_COLWISE

    def __init__(self) -> None:
        super().__init__(shard_dim=1)

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        setattr(
            module, "_spyre_global_embedding_dim", _module_size(module, "embedding_dim")
        )
        setattr(
            module,
            "embedding_dim",
            Shard.local_shard_size_and_offset(
                _module_size(module, "embedding_dim"),
                mesh.size(),
                mesh.get_local_rank(),
            )[0],
        )
        result: nn.Module = super().install_forward(module, mesh)
        return result

    def transform_output_post_forward(
        self, module: nn.Module, output: torch.Tensor, mesh: DeviceMesh
    ) -> torch.Tensor:
        return _gather_uneven_last_dim(
            output, _module_size(module, "_spyre_global_embedding_dim"), mesh
        )


class SpyreReplicatedInferenceParallel(_SpyreLocalParallel):
    _replicated = True


class SpyreReplicatedLinearParallel(SpyreReplicatedInferenceParallel):
    copy_kind: _CopyKind = "linear"
    style_name = SPYRE_REPLICATED_LINEAR


class SpyreReplicatedEmbeddingParallel(SpyreReplicatedInferenceParallel):
    copy_kind: _CopyKind = "embedding"
    style_name = SPYRE_REPLICATED_EMBEDDING


def _grouped_mesh(mesh: DeviceMesh, num_shards: int) -> DeviceMesh:
    if mesh.size() % num_shards:
        raise ValueError(
            f"TP size {mesh.size()} is not divisible by {num_shards} grouped-colwise shards"
        )
    # Keep children with their owning mesh, not in a process-global cache that
    # would retain every mesh/process group after the caller releases it.
    grouped: dict[int, DeviceMesh] | None = getattr(mesh, "_spyre_grouped_meshes", None)
    if grouped is None:
        grouped = {}
        setattr(mesh, "_spyre_grouped_meshes", grouped)
    if num_shards not in grouped:
        grouped[num_shards] = mesh._unflatten(
            0,
            (num_shards, mesh.size() // num_shards),
            (f"spyre_shard_{num_shards}", f"spyre_replica_{num_shards}"),
        )
    return grouped[num_shards]


class SpyreGroupedColwiseParallel(_SpyreLocalParallel):
    copy_kind: _CopyKind = "linear"
    num_shards = 1

    def shard_param(self, module: nn.Module, param: str, mesh: DeviceMesh) -> None:
        value = module._parameters.get(param)
        if value is None:
            return
        grouped_mesh = _grouped_mesh(mesh, self.num_shards)
        placement = Replicate() if value.ndim == 0 else Shard(max(0, value.ndim - 2))
        placeholder = nn.Parameter(
            distribute_tensor(
                value, grouped_mesh, [placement, Replicate()], src_data_rank=None
            ),
            requires_grad=value.requires_grad,
        )
        setattr(placeholder, "_spyre_copy_kind", self.copy_kind)
        module._parameters[param] = placeholder
        setattr(module, "_hf_device_mesh", mesh)
        setattr(module, "_hf_tp_plan", self.style_name)

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        if hasattr(module, "out_features"):
            setattr(
                module,
                "out_features",
                Shard.local_shard_size_and_offset(
                    _module_size(module, "out_features"),
                    self.num_shards,
                    mesh.get_local_rank() // (mesh.size() // self.num_shards),
                )[0],
            )
        result: nn.Module = super().install_forward(module, mesh)
        return result


class SpyreCpuPackedColwiseParallel(_SpyreLocalParallel, PackedColwiseParallel):
    copy_kind: _CopyKind = "cpu"
    style_name = SPYRE_CPU_PACKED_COLWISE


class SpyreCpuRowwiseParallel(_SpyreLocalParallel, RowwiseParallel):
    copy_kind: _CopyKind = "cpu"
    style_name = SPYRE_CPU_ROWWISE


class SpyreCpuReplicatedParallel(SpyreReplicatedInferenceParallel):
    copy_kind: _CopyKind = "cpu"
    style_name = SPYRE_CPU_REPLICATED


class SpyreMoeExpertsParallel(_SpyreLocalParallel):
    """Reduce expert partials even after loading replaces DTensor placeholders."""

    style_name = SPYRE_MOE_EXPERTS

    def shard_param(self, module: nn.Module, param: str, mesh: DeviceMesh) -> None:
        # The per-parameter packed/rowwise rules own expert placement.
        return None

    def install_forward(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        sharded = any(
            isinstance(param, DTensor)
            and any(not placement.is_replicate() for placement in param.placements)
            for param in module.parameters()
        )
        setattr(module, "_spyre_reduce_experts", sharded)
        return super().install_forward(module, mesh)

    def transform_output_post_forward(
        self, module: nn.Module, output: torch.Tensor, mesh: DeviceMesh
    ) -> torch.Tensor:
        if getattr(module, "_spyre_reduce_experts", False):
            return spyre_compiled_all_reduce(output, mesh.get_group().group_name)
        return output


def _gather_uneven_last_dim(
    tensor: torch.Tensor, full_size: int, mesh: DeviceMesh
) -> torch.Tensor:
    sizes = tuple(
        Shard.local_shard_size_and_offset(full_size, mesh.size(), rank)[0]
        for rank in range(mesh.size())
    )
    padded = torch.nn.functional.pad(tensor, (0, max(sizes) - tensor.shape[-1]))
    return spyre_compiled_all_gather_last_dim(
        padded, mesh.size(), mesh.get_group().group_name, sizes
    )


def _reject_local_tp_save(model: nn.Module, *args: Any, **kwargs: Any) -> None:
    """HF 5.22 cannot reconstruct global weights from our ordinary local shards."""
    raise NotImplementedError(
        "save_pretrained() is not supported for Spyre TP models with local weights. "
        "Load the original checkpoint without Spyre TP to save a complete model."
    )


@contextmanager
def spyre_tp_loading() -> Iterator[None]:
    """Bridge HF's DTensor loader to Spyre's local-parameter compiled adapters.

    Only placeholders tagged by our styles take this path. HF retains ownership
    of shard-on-read, weight conversion, shape validation and missing-key checks.
    Rewrapping a loaded shard as a DTensor would hide its device layout and make
    adapter padding operate on the global shape, so retain the local tensor.

    These patches affect the entire process. Callers must serialize HF model
    loads (including non-Spyre loads) while this context is active. The context
    covers HF's own checkpoint worker threads, but is not safe for concurrent
    independent from_pretrained calls.
    """
    from transformers import core_model_loading
    from transformers.distributed import mixin, tensor_parallel, utils

    original_shard = core_model_loading.DtensorShardOperation
    original_wrap = core_model_loading._dtensor_from_local_like
    original_resolve = mixin.resolve_parallel_plans
    original_lookup = tensor_parallel._get_parameter_plan
    original_device = core_model_loading.get_device
    original_initialize = utils._ensure_torch_distributed
    # HF annotates missing_keys as list[str] but passes/operates on a set.
    original_move_missing: Callable[..., None] = (
        PreTrainedModel._move_missing_keys_from_meta_to_device
    )
    staged_parameters: set[str] = set()
    cleanup = ExitStack()

    def preserve_placement_tags(
        model: nn.Module,
        state: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
    ) -> None:
        # HF reads loading placeholders through state_dict(), which detaches
        # DTensors and loses Python attributes. Restore placement on that view
        # so checkpoint slicing still selects our layout-aware shard operation.
        for name, parameter in model.named_parameters():
            copy_kind: _CopyKind | None = getattr(parameter, "_spyre_copy_kind", None)
            if copy_kind is not None and prefix + name in state:
                setattr(state[prefix + name], "_spyre_copy_kind", copy_kind)
                setattr(state[prefix + name], "_spyre_param_name", prefix + name)

    def initialize_distributed(device_type: str | None = None) -> None:
        if device_type is None:
            device_type = torch._C._get_accelerator().type
        if device_type != "spyre":
            original_initialize(device_type)
        elif not torch.distributed.is_initialized():
            # HF's backend map omits Spyre. Do not pass device_id here: that
            # makes DeviceMesh use split_group, unsupported by SpyreCCL.
            # HF binds LOCAL_RANK when it constructs the mesh afterward.
            torch.distributed.init_process_group(
                backend="cpu:gloo,spyre:spyreccl", timeout=timedelta(hours=2)
            )

    def parameter_device(
        device_map: dict[str, Any] | None,
        param_name: str,
        valid_torch_device: bool = False,
    ) -> torch.device | str | int:
        # force_cpu converters move their result to get_device() after the
        # shard operation. Keep staged experts on CPU through that final step.
        if param_name in staged_parameters:
            return torch.device("cpu") if valid_torch_device else "cpu"
        result: torch.device | str | int = original_device(
            device_map, param_name, valid_torch_device=valid_torch_device
        )
        return result

    def shard_operation(param: DTensor) -> DtensorShardOperation:
        return (
            SpyreShardOperation(param)
            if hasattr(param, "_spyre_copy_kind")
            else original_shard(param)
        )

    def local_parameter(local: torch.Tensor, ref: DTensor) -> torch.Tensor:
        return local if hasattr(ref, "_spyre_copy_kind") else original_wrap(local, ref)

    def move_missing(
        model: PreTrainedModel,
        missing_keys: set[str],
        device_map: dict[str, Any] | None,
        device_mesh: Any,
        hf_quantizer: Any,
    ) -> None:
        # modeling_utils has direct imports of get_device/_dtensor_from_local_like.
        # Reject incomplete planned weights before that path allocates generic
        # device tensors or wraps local shards back into DTensors. HF tying is
        # symmetric: any successfully loaded member can supply its tied group.
        tied_neighbors: dict[str, set[str]] = {}
        for target, source in model.all_tied_weights_keys.items():
            tied_neighbors.setdefault(target, set()).add(source)
            tied_neighbors.setdefault(source, set()).add(target)
        remaining = set(tied_neighbors)
        loaded_aliases: set[str] = set()
        embedding_sources: dict[str, torch.Tensor] = {}
        while remaining:
            group: set[str] = set()
            pending = [remaining.pop()]
            while pending:
                name = pending.pop()
                group.add(name)
                neighbors = tied_neighbors[name] & remaining
                remaining.difference_update(neighbors)
                pending.extend(neighbors)
            loaded = [
                name
                for name in group - missing_keys
                if not model.get_parameter_or_buffer(name).is_meta
            ]
            if not loaded:
                continue
            loaded_aliases.update(group)
            for name in group & missing_keys:
                ref = model.get_parameter_or_buffer(name)
                if getattr(ref, "_spyre_copy_kind", None) == "embedding":
                    embedding_sources.update((alias, ref) for alias in loaded)
        unsupported = sorted(
            name
            for name in missing_keys - loaded_aliases
            if hasattr(model.get_parameter_or_buffer(name), "_spyre_copy_kind")
        )
        if unsupported:
            raise ValueError(
                "Spyre TP requires complete checkpoint weights; missing or mismatched "
                f"planned parameters: {', '.join(unsupported)}"
            )
        # Reverse tying can make an embedding inherit a linear DMA layout (or
        # a full vocabulary instead of its local shard). Restore the embedding's
        # placement before HF shares the loaded parameter with the missing alias.
        # Adapter preparation later gives the LM head its own linear layout.
        for name, ref in embedding_sources.items():
            parameter = model.get_parameter_or_buffer(name)
            parent_name, _, param_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            local_shape = (
                ref.to_local().shape if isinstance(ref, DTensor) else ref.shape
            )
            moved: torch.Tensor | None
            if parameter.shape == local_shape:
                if parameter.device.type != "spyre" or isinstance(parent, nn.Embedding):
                    continue
                moved = _copy_embedding(
                    parameter.detach().cpu(),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            elif isinstance(ref, DTensor) and parameter.shape == ref.shape:
                moved = SpyreShardOperation(ref).shard_tensor(
                    parameter.detach().cpu(),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                assert moved is not None
            else:
                raise ValueError(
                    f"Tied parameter {name} has an incompatible embedding shard shape"
                )
            replacement = nn.Parameter(moved, requires_grad=parameter.requires_grad)
            setattr(replacement, "_is_hf_initialized", True)
            setattr(parent, param_name, replacement)
        # Preserve the missing-key set used by HF's later symmetric tie_weights.
        # Until then, leave resolvable aliases on meta and skip their random init.
        for name in missing_keys & loaded_aliases:
            setattr(model.get_parameter_or_buffer(name), "_is_hf_initialized", True)
        original_move_missing(
            model, missing_keys - loaded_aliases, device_map, device_mesh, hf_quantizer
        )

    def exact_first(
        parameter_name: str, plan: dict[str, str], is_weight: bool = True
    ) -> str | None:
        if parameter_name in plan:
            return plan[parameter_name]
        if is_weight and "." in parameter_name:
            parent = parameter_name.rsplit(".", 1)[0]
            if parent in plan:
                return plan[parent]
        return original_lookup(parameter_name, plan, is_weight=is_weight)

    def resolve_plans(
        model: Any, distributed_config: Any
    ) -> tuple[dict[str, str], dict[str, str]]:
        plan = distributed_config.tp_plan
        if isinstance(plan, dict) and any(
            style.startswith("spyre_") for style in plan.values()
        ):
            # Our plan is already complete, and exact per-layer overrides are
            # valid. HF 5.22's override validator only accepts wildcard names.
            names = {name for name, _ in model.named_modules()} | {
                name for name, _ in model.named_parameters()
            }
            # HF plans may include optional branches absent from this config
            # (e.g. Gemma's MoE rules on a dense checkpoint). Preserve those
            # declared defaults while validating additional caller overrides.
            known_names = (
                set(model.tp_plan)
                | names
                | {replace_layer_number_by_wildcard(name) for name in names}
            )
            # HF lookup supports exact names and numeric-component wildcards,
            # not arbitrary globs. A regex scan both costs keys × names and
            # accepts patterns that HF would never apply to a parameter.
            unknown = [key for key in plan if key not in known_names]
            if unknown:
                raise ValueError(
                    f"TP plan keys match no model modules or parameters: {unknown}"
                )
            for name, _ in model.named_parameters():
                style = _STYLE_INSTANCES.get(exact_first(name, plan) or "")
                if style is not None and style.copy_kind == "cpu":
                    staged_parameters.add(name)
            model._tp_plan = dict(plan)
            # HF's save helper only gathers DTensors. Its ordinary-tensor path
            # would silently write rank zero's local shards as full weights.
            # Keep this per-model guard after the temporary loader patches exit,
            # including before adapter preparation or process-group teardown.
            model.save_pretrained = MethodType(_reject_local_tp_save, model)
            handle = model.register_state_dict_post_hook(preserve_placement_tags)
            cleanup.callback(handle.remove)
            return original_resolve(model, replace(distributed_config, tp_plan=None))
        return original_resolve(model, distributed_config)

    replacements: tuple[tuple[object, str, Any], ...] = (
        (utils, "_ensure_torch_distributed", initialize_distributed),
        (core_model_loading, "DtensorShardOperation", shard_operation),
        (core_model_loading, "_dtensor_from_local_like", local_parameter),
        (core_model_loading, "get_device", parameter_device),
        (tensor_parallel, "_get_parameter_plan", exact_first),
        (mixin, "resolve_parallel_plans", resolve_plans),
        (PreTrainedModel, "_move_missing_keys_from_meta_to_device", move_missing),
    )
    with cleanup:
        for target, name, replacement in replacements:
            original = getattr(target, name)
            setattr(target, name, replacement)
            cleanup.callback(setattr, target, name, original)
        yield


_STYLE_INSTANCES = {
    SPYRE_COLWISE: SpyreColwiseParallel(),
    SPYRE_COLWISE_GATHER_OUTPUT: SpyreColwiseGatherOutputParallel(),
    SPYRE_ROWWISE: SpyreRowwiseParallel(),
    SPYRE_ROWWISE_SPLIT_INPUT: SpyreRowwiseSplitInputParallel(),
    SPYRE_EMBEDDING_ROWWISE: SpyreEmbeddingRowwiseParallel(),
    SPYRE_EMBEDDING_COLWISE: SpyreEmbeddingColwiseParallel(),
    SPYRE_REPLICATED: SpyreReplicatedInferenceParallel(),
    SPYRE_REPLICATED_LINEAR: SpyreReplicatedLinearParallel(),
    SPYRE_REPLICATED_EMBEDDING: SpyreReplicatedEmbeddingParallel(),
    SPYRE_CPU_PACKED_COLWISE: SpyreCpuPackedColwiseParallel(),
    SPYRE_CPU_ROWWISE: SpyreCpuRowwiseParallel(),
    SPYRE_CPU_REPLICATED: SpyreCpuReplicatedParallel(),
    SPYRE_MOE_EXPERTS: SpyreMoeExpertsParallel(),
}

_STYLE_REWRITES = {
    "colwise": SPYRE_COLWISE,
    "colwise_gather_output": SPYRE_COLWISE_GATHER_OUTPUT,
    "colwise_rep": SPYRE_COLWISE_GATHER_OUTPUT,
    "rowwise": SPYRE_ROWWISE,
    "rowwise_split_input": SPYRE_ROWWISE_SPLIT_INPUT,
    "rowwise_rep": SPYRE_ROWWISE_SPLIT_INPUT,
    "embedding_rowwise": SPYRE_EMBEDDING_ROWWISE,
    "embedding_colwise": SPYRE_EMBEDDING_COLWISE,
    "replicated_with_grad_allreduce": SPYRE_REPLICATED,
    "replicate": SPYRE_REPLICATED,
    "moe_tp_experts": SPYRE_MOE_EXPERTS,
}

_CPU_STYLE_REWRITES = {
    "packed_colwise": SPYRE_CPU_PACKED_COLWISE,
    "rowwise": SPYRE_CPU_ROWWISE,
}

_GROUPED_COLWISE_STYLES: dict[str, TensorParallelLayer] = {}


def _register_grouped_colwise_style(num_shards: int) -> str:
    """Register a reconstructable grouped-colwise class for ``num_shards``."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    name = f"{SPYRE_GROUPED_COLWISE_PREFIX}_{num_shards}"
    if name not in _GROUPED_COLWISE_STYLES:
        cls = type(
            f"SpyreGroupedColwise{num_shards}Parallel",
            (SpyreGroupedColwiseParallel,),
            {"num_shards": num_shards, "style_name": name},
        )
        style = cls()
        _GROUPED_COLWISE_STYLES[name] = style
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]
    return name


def register_spyre_tp_styles() -> None:
    """Register the Spyre TP styles with Transformers (idempotently)."""
    for name, style in _STYLE_INSTANCES.items():
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]


@lru_cache(maxsize=512)
def _plan_pattern(key: str) -> re.Pattern[str]:
    """Compile a TP glob; dots and other non-wildcard characters are literal."""
    return re.compile(re.escape(key).replace(r"\*", ".*"))


def _matches_plan(parameter_name: str, plan: Mapping[str, str]) -> bool:
    parent = parameter_name.rpartition(".")[0]
    return any(
        _plan_pattern(key).fullmatch(parameter_name)
        or _plan_pattern(key).fullmatch(parent)
        for key in plan
    )


def grouped_kv_projection_modules(
    model: nn.Module,
    attention_layers: Iterable[tuple[nn.Module, int]],
    tp_size: int,
) -> dict[str, int]:
    """Select whole-head K/V sharding for unsharded attention layers.

    Adapters supply ``(attention, head_dim)`` pairs so heterogeneous head
    dimensions remain their responsibility. Ordinary colwise TP is retained
    when KV heads divide evenly among ranks. Otherwise each complete KV head
    is replicated within its corresponding group of Q ranks. Missing K/V
    projections (shared caches or K-equals-V attention) need no placement rule.
    """
    from hf_adapters.hf_common import _projection_head_count

    if tp_size <= 0:
        raise ValueError(f"TP size must be positive, got {tp_size}")
    module_names = {id(module): name for name, module in model.named_modules()}
    grouped: dict[str, int] = {}
    for attention, head_dim in attention_layers:
        q_proj = attention.q_proj
        # A missing/uninspectable projection must not fall back to a global
        # config count: placement needs its actual, pre-sharding geometry.
        num_q_heads = _projection_head_count(q_proj, 0, head_dim)
        if num_q_heads % tp_size != 0:
            raise ValueError(
                f"{module_names[id(q_proj)]} cannot split {num_q_heads} query "
                f"heads across {tp_size} TP ranks without splitting a head"
            )
        for name in ("k_proj", "v_proj"):
            projection = getattr(attention, name, None)
            if projection is None:
                continue
            num_kv_heads = _projection_head_count(projection, 0, head_dim)
            if num_kv_heads % tp_size == 0:
                continue
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"{module_names[id(projection)]} cannot distribute "
                    f"{num_kv_heads} KV heads across {tp_size} TP ranks"
                )
            grouped[module_names[id(projection)]] = num_kv_heads
    return grouped


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
    """Translate an HF TP plan and cover otherwise-unplanned weight modules.

    Unplanned Linear and Embedding parameters would otherwise be copied with a
    generic ``Tensor.to(spyre)`` before ``nn.Module.to`` can see their module
    type.  Explicit replicated placement entries make those copies layout
    aware without changing their tensor-parallel semantics.
    """
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
                    f"CPU-staged TP module {module_name!r} uses unsupported "
                    f"style {style!r}"
                ) from error
        else:
            rewritten[module_name] = _STYLE_REWRITES.get(style, style)

    # Some adapters intentionally turn off sharding for selected modules. Keep
    # explicit placement entries so those weights still take a semantic DMA.
    for module_name in replicated_linear_modules:
        if module_name in named_modules:
            rewritten[module_name] = SPYRE_REPLICATED_LINEAR
    for module_name in replicated_embedding_modules:
        if module_name in named_modules:
            rewritten[module_name] = SPYRE_REPLICATED_EMBEDDING
    for module_name in cpu_replicated_modules:
        if module_name in named_modules:
            rewritten[module_name] = SPYRE_CPU_REPLICATED
    for module_name, num_shards in grouped_colwise_modules.items():
        if module_name in named_modules:
            rewritten[module_name] = _register_grouped_colwise_style(num_shards)

    for name, module in named_modules.items():
        if not name or getattr(module, "weight", None) is None:
            continue
        parameter_name = f"{name}.weight"
        if _matches_plan(parameter_name, rewritten):
            continue
        generic_name = replace_layer_number_by_wildcard(name)
        if isinstance(module, nn.Embedding):
            rewritten.setdefault(generic_name, SPYRE_REPLICATED_EMBEDDING)
        elif isinstance(module, nn.Linear):
            rewritten.setdefault(generic_name, SPYRE_REPLICATED_LINEAR)

    # Keep exact adapter overrides first for inspection and plan consumers.
    # spyre_tp_loading applies exact-first lookup to both sharding and hooks.
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


def spyre_compiled_all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    """All-reduce *tensor* through Spyre's graph-lowered collective path.

    Unlike Transformers' ``all_reduce_forward`` autograd wrapper, the
    functional collective is visible to Dynamo/Inductor.  Torch-Spyre lowers
    it to a cached ``allreduce_plan`` and an in-graph ``allreduce_run``.  The
    lowering plans against the realized device allocation, so a separate
    PyTorch flatten/copy is neither necessary nor desirable here.

    This helper is intended to be called from inside a ``torch.compile``
    region.  Eager module hooks should continue using the layout-aware helpers
    above and below.
    """
    reduced: torch.Tensor = torch.ops._c10d_functional.all_reduce(
        tensor, "sum", group_name
    )
    result: torch.Tensor = torch.ops._c10d_functional.wait_tensor(reduced)
    return result


def _reassemble_all_gather_last_dim(
    gathered: torch.Tensor,
    input_shape: tuple[int, ...],
    group_size: int,
    shard_sizes: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Move rank-major dim-0 gather chunks onto the input's last dimension."""
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
    shard_sizes: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """All-gather last-dimension shards through Spyre's compiled path.

    ``_c10d_functional.all_gather_into_tensor`` concatenates rank inputs along
    dimension 0, while tensor-parallel linear layers shard their output along
    the last dimension.  Reinterpret the gathered rank-major result and move
    the rank dimension next to the local shard before flattening the two into
    the reconstructed output dimension.

    This helper is intended for a ``torch.compile`` region.  Torch-Spyre then
    emits a cached ``allgather_plan`` and an in-graph ``allgather_run`` instead
    of Transformers' eager list-based ``dist.all_gather`` plus ``torch.cat``.
    """
    input_shape = tensor.shape
    gathered: torch.Tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        tensor, group_size, group_name
    )
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)
    return _reassemble_all_gather_last_dim(
        gathered, input_shape, group_size, shard_sizes=shard_sizes
    )
