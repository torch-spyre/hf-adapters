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

"""Shared inputs and CPU-thread bounds for the multimodal embedding suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch

# The cat caption (index 1) must rank first against the sample image.
TEXT_PROMPTS: list[str] = [
    "Two dogs at the beach",
    "A cat in the snow",
    "A picture of London at night",
]


@contextmanager
def bounded_cpu_threads() -> Iterator[None]:
    previous: int = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def forbid_model_device_moves(model: torch.nn.Module) -> Iterator[None]:
    """Reject even temporary parameter/buffer relocation during encode calls.

    Guard module conversions rather than Tensor.to so activation transfers and
    compiled tensor operations remain unmodified.
    """
    watched: set[torch.nn.Module] = set(model.modules())
    original_apply = torch.nn.Module._apply

    def guarded_apply(
        module: torch.nn.Module,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> torch.nn.Module:
        if module not in watched:
            return original_apply(module, fn, recurse=recurse)

        def check_device(tensor: torch.Tensor) -> torch.Tensor:
            previous: torch.device = tensor.device
            result = fn(tensor)
            assert result.device == previous, (
                f"{type(module).__name__} moved a model tensor "
                f"from {previous} to {result.device} during inference"
            )
            return result

        return original_apply(module, check_device, recurse=recurse)

    with patch.object(torch.nn.Module, "_apply", guarded_apply):
        yield
