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

import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
from torch.distributed.tensor.placement_types import Shard
from transformers.core_model_loading import build_glob_alternation
from transformers.integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    EmbeddingParallel,
    PackedColwiseParallel,
    ReplicatedWithGradAllReduce,
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


def _copy_default(shard, *, device, dtype):
    from torch_spyre.model_utils import _dma_to_spyre_default

    return _dma_to_spyre_default(shard, target_dtype=dtype, device=device)


def _copy_linear(shard, *, device, dtype):
    from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped

    if shard.ndim == 2:
        return _dma_to_spyre_dim_order_swapped(shard, target_dtype=dtype, device=device)
    return _copy_default(shard, device=device, dtype=dtype)


def _copy_embedding(shard, *, device, dtype):
    from torch_spyre.model_utils import (
        _dma_to_spyre_default,
        _dma_to_spyre_indirect_access,
    )

    if shard.ndim == 2:
        moved = _dma_to_spyre_indirect_access(shard, target_dtype=dtype, device=device)
        if moved is not None:
            return moved
    return _dma_to_spyre_default(shard, target_dtype=dtype, device=device)


class SpyreColwiseParallel(ColwiseParallel):
    """Colwise checkpoint slicing followed by a Linear-aware Spyre DMA."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )
        return _copy_linear(shard, device=device, dtype=dtype)


class SpyreColwiseGatherOutputParallel(SpyreColwiseParallel):
    """State-preserving variant for HF's ``colwise_gather_output`` style."""

    def __init__(self, **kwargs):
        super().__init__(gather_output=True, **kwargs)


class SpyreRowwiseParallel(RowwiseParallel):
    """Rowwise checkpoint slicing followed by a Linear-aware Spyre DMA."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )
        return _copy_linear(shard, device=device, dtype=dtype)


class SpyreRowwiseSplitInputParallel(SpyreRowwiseParallel):
    """State-preserving variant for HF's ``rowwise_split_input`` style."""

    def __init__(self, **kwargs):
        super().__init__(split_input=True, **kwargs)


class _SpyreEmbeddingParallel(EmbeddingParallel):
    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )
        return _copy_embedding(shard, device=device, dtype=dtype)


class SpyreEmbeddingRowwiseParallel(_SpyreEmbeddingParallel):
    """Vocab-sharded embedding with host-side ID masking for Spyre."""

    def __init__(self, **kwargs):
        super().__init__(embedding_dim_sharding=0, **kwargs)

    def _prepare_input_fn(self, mod, inputs, *, vocab_start, vocab_end):
        input_tensor = inputs[0] if inputs else inputs
        host_input = input_tensor.cpu()
        invalid = (host_input < vocab_start) | (host_input >= vocab_end)

        masked_input = host_input.clone() - vocab_start
        masked_input[invalid] = 0
        # Spyre does not yet support comparisons producing bool from int32, so
        # construct the small ownership mask on the host.  Transfer it once and
        # keep both masking the embedding result and the collective on-device.
        valid = (~invalid).to(mod.weight.dtype).unsqueeze(-1).to(input_tensor.device)
        return masked_input.to(input_tensor.device), valid

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        # Keep the small integer ownership test in the eager CPU pre-hook, but
        # compile everything after its two H2D transfers as one graph.  In
        # particular, this prevents both the embedding lookup and its
        # all-reduce from becoming eager graph boundaries.
        group_name = device_mesh.get_group().group_name
        rank = device_mesh.get_local_rank()
        # TP hooks are installed before checkpoint loading, while the module
        # still exposes its full-sized meta weight. Capture that global size
        # now; by the first forward, ``module.weight`` has been replaced by the
        # rank-local shard.
        global_vocab_size = module.weight.shape[0]
        local_size, vocab_start = Shard.local_shard_size_and_offset(
            global_vocab_size, device_mesh.size(), rank
        )
        vocab_end = vocab_start + local_size
        original_forward = module.forward

        def prepare_inputs(mod, inputs):
            assert local_size == mod.weight.shape[0], (
                "vocabulary shard size does not match the loaded embedding weight: "
                f"expected {local_size}, got {mod.weight.shape[0]}"
            )
            return self._prepare_input_fn(
                mod,
                inputs,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
            )

        def embedding_forward(masked_input, valid):
            outputs = original_forward(masked_input)
            return spyre_compiled_all_reduce(outputs * valid, group_name)

        module.forward = torch.compile(embedding_forward, dynamic=False, fullgraph=True)
        module.register_forward_pre_hook(prepare_inputs)
        return module


class SpyreEmbeddingColwiseParallel(_SpyreEmbeddingParallel):
    """Hidden-dimension-sharded embedding with a fixed dim-1 policy."""

    def __init__(self, **kwargs):
        super().__init__(embedding_dim_sharding=1, **kwargs)


class SpyreReplicatedInferenceParallel(ReplicatedWithGradAllReduce):
    """Replicated inference parameter without HF's training-only hook."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )
        return _copy_default(shard, device=device, dtype=dtype)

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        return None


class SpyreReplicatedLinearParallel(TensorParallelLayer):
    """Replicate a Linear parameter while selecting its device layout at DMA."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = param[...].to(device="cpu", dtype=dtype)
        return _copy_linear(shard, device=device, dtype=dtype)

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        return None


class SpyreReplicatedEmbeddingParallel(TensorParallelLayer):
    """Replicate an Embedding while selecting indirect-access layout at DMA."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shard = param[...].to(device="cpu", dtype=dtype)
        return _copy_embedding(shard, device=device, dtype=dtype)

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        return None


class SpyreGroupedColwiseParallel(TensorParallelLayer):
    """Shard output features across rank groups and replicate within a group."""

    num_shards = 1

    def _shard_rank(self):
        world_size = self.device_mesh.size()
        if world_size % self.num_shards != 0:
            raise ValueError(
                f"TP size {world_size} is not divisible by {self.num_shards} "
                "grouped-colwise shards"
            )
        return self.rank // (world_size // self.num_shards)

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        shape = list(param.shape) if hasattr(param, "shape") else param.get_shape()
        dim = 0 if len(shape) == 1 else len(shape) - 2
        shard_size = math.ceil(shape[dim] / self.num_shards)
        start = self._shard_rank() * shard_size
        end = min(start + shard_size, shape[dim])
        indices = [slice(None)] * len(shape)
        indices[dim] = slice(start, end)
        shard = param[tuple(indices)].to(device="cpu", dtype=dtype)
        return _copy_linear(shard, device=device, dtype=dtype)

    def get_expected_sharded_shape(self, full_shape):
        shape = list(full_shape)
        dim = 0 if len(shape) == 1 else len(shape) - 2
        shape[dim] = math.ceil(shape[dim] / self.num_shards)
        return tuple(shape)

    def update_module_attributes(self, module):
        if hasattr(module, "out_features"):
            module.out_features = self.get_expected_sharded_shape(
                (module.out_features,)
            )[0]

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        return None


class SpyreCpuPackedColwiseParallel(PackedColwiseParallel):
    """Shard a packed parameter but leave it on CPU for adapter preparation."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        return super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )


class SpyreCpuRowwiseParallel(RowwiseParallel):
    """Create a rowwise shard on CPU for a later adapter-specific DMA."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        return super().shard_tensor(
            param, tensor_idx=tensor_idx, device="cpu", dtype=dtype
        )


class SpyreCpuReplicatedParallel(TensorParallelLayer):
    """Keep an unsharded parameter on CPU for adapter-side preparation."""

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        return param[...].to(device="cpu", dtype=dtype)

    def prepare_module_tp(self, module, device_mesh, **kwargs):
        return None


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
}

_STYLE_DIMS = {
    SPYRE_COLWISE: (-2, -1),
    SPYRE_COLWISE_GATHER_OUTPUT: (-2, -1),
    SPYRE_ROWWISE: (-1, None),
    SPYRE_ROWWISE_SPLIT_INPUT: (-1, None),
    SPYRE_EMBEDDING_ROWWISE: (0, None),
    SPYRE_EMBEDDING_COLWISE: (1, None),
    SPYRE_REPLICATED: (None, None),
    SPYRE_REPLICATED_LINEAR: (None, None),
    SPYRE_REPLICATED_EMBEDDING: (None, None),
    SPYRE_CPU_PACKED_COLWISE: (-2, -1),
    SPYRE_CPU_ROWWISE: (-1, None),
    SPYRE_CPU_REPLICATED: (None, None),
}

_STYLE_REWRITES = {
    "colwise": SPYRE_COLWISE,
    "colwise_gather_output": SPYRE_COLWISE_GATHER_OUTPUT,
    "rowwise": SPYRE_ROWWISE,
    "rowwise_split_input": SPYRE_ROWWISE_SPLIT_INPUT,
    "embedding_rowwise": SPYRE_EMBEDDING_ROWWISE,
    "embedding_colwise": SPYRE_EMBEDDING_COLWISE,
    "replicated_with_grad_allreduce": SPYRE_REPLICATED,
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
            {"num_shards": num_shards},
        )
        style = cls()
        _GROUPED_COLWISE_STYLES[name] = style
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]
        ALL_PARALLEL_STYLES.register_plan_to_weight_dim(name, -2)
        ALL_PARALLEL_STYLES.register_plan_to_bias_dim(name, -1)
    return name


def register_spyre_tp_styles() -> None:
    """Register the Spyre TP styles with Transformers (idempotently)."""
    for name, style in _STYLE_INSTANCES.items():
        ALL_PARALLEL_STYLES.register(name, style)  # type: ignore[arg-type]
        weight_dim, bias_dim = _STYLE_DIMS[name]
        ALL_PARALLEL_STYLES.register_plan_to_weight_dim(name, weight_dim)
        ALL_PARALLEL_STYLES.register_plan_to_bias_dim(name, bias_dim)


def _matches_plan(parameter_name: str, plan: Mapping[str, str]) -> bool:
    if not plan:
        return False

    alternation, _, _ = build_glob_alternation(list(plan))
    return alternation.search(parameter_name) is not None


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
    rewritten = {}
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

    # Transformers' current core loader resolves TP globs by the first regex
    # match. Put exact adapter overrides before generic ``layers.*`` entries so
    # the checkpoint materializer uses the same style as the module hook path.
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


def spyre_compiled_all_reduce(tensor, group_name):
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
    reduced = torch.ops._c10d_functional.all_reduce(tensor, "sum", group_name)
    return torch.ops._c10d_functional.wait_tensor(reduced)


def _reassemble_all_gather_last_dim(
    gathered, input_shape, group_size, shard_sizes=None
):
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
    tensor, group_size, group_name, shard_sizes=None
):
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
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(
        tensor, group_size, group_name
    )
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)
    return _reassemble_all_gather_last_dim(
        gathered, input_shape, group_size, shard_sizes=shard_sizes
    )
