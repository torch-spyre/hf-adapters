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

from collections.abc import Iterator
from contextlib import contextmanager

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
