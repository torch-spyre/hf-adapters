# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Make private API drift explicit when changing the Transformers/Torch pins."""

import inspect
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import core_model_loading, modeling_utils
from transformers.distributed import mixin, parallel_utils, tensor_parallel


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (core_model_loading.DtensorShardOperation, ("param",)),
        (core_model_loading._dtensor_from_local_like, ("local_tensor", "ref")),
        (
            core_model_loading.get_device,
            ("device_map", "param_name", "valid_torch_device"),
        ),
        (mixin._get_parameter_plan, ("parameter_name", "plan", "is_weight")),
        (
            mixin.DistributedMixin._resolve_parallel_plans,
            ("self", "distributed_config"),
        ),
        (mixin.DistributedMixin._expand_parallel_plan, ("self", "plan")),
        (
            mixin.DistributedMixin._validate_parallel_plan,
            ("self", "plan", "supported_styles", "plan_name", "names"),
        ),
        (tensor_parallel.apply_tensor_parallelism, ("model", "tp_mesh", "tp_plan")),
        (
            modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device,
            ("self", "missing_keys", "device_map", "device_mesh", "hf_quantizer"),
        ),
        (_StridedShard, ("dim", "split_factor", "sf")),
    ],
)
def test_private_tp_signatures(
    function: Callable[..., Any], parameters: tuple[str, ...]
) -> None:
    assert tuple(inspect.signature(function).parameters) == parameters


def test_missing_key_aliases_and_accelerator_contract() -> None:
    assert mixin._get_parameter_plan is parallel_utils._get_parameter_plan
    assert modeling_utils.get_device is core_model_loading.get_device
    assert (
        modeling_utils._dtensor_from_local_like
        is core_model_loading._dtensor_from_local_like
    )
    assert isinstance(torch._C._get_accelerator(), torch.device)
