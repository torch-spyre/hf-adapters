# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Make private API drift explicit when changing the Transformers/Torch pins."""

import inspect
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import core_model_loading, modeling_utils
from transformers.distributed import mixin, tensor_parallel, utils


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (core_model_loading.DtensorShardOperation, ("param",)),
        (core_model_loading._dtensor_from_local_like, ("local_tensor", "ref")),
        (
            core_model_loading.get_device,
            ("device_map", "param_name", "valid_torch_device"),
        ),
        (tensor_parallel._get_parameter_plan, ("parameter_name", "plan", "is_weight")),
        (mixin.resolve_parallel_plans, ("model", "distributed_config")),
        (utils._ensure_torch_distributed, ("device_type",)),
        (
            modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device,
            ("self", "missing_keys", "device_map", "device_mesh", "hf_quantizer"),
        ),
        (
            DeviceMesh._unflatten,
            ("self", "dim", "mesh_sizes", "mesh_dim_names", "backend_override"),
        ),
        (_StridedShard, ("dim", "split_factor", "sf")),
    ],
)
def test_private_tp_signatures(
    function: Callable[..., Any], parameters: tuple[str, ...]
) -> None:
    assert tuple(inspect.signature(function).parameters) == parameters


def test_missing_key_aliases_and_accelerator_contract() -> None:
    assert modeling_utils.get_device is core_model_loading.get_device
    assert (
        modeling_utils._dtensor_from_local_like
        is core_model_loading._dtensor_from_local_like
    )
    assert isinstance(torch._C._get_accelerator(), torch.device)
