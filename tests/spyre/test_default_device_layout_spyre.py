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

"""to_default_device_layout gives a Spyre tensor the default layout.

Run on a Spyre pod: pytest tests/spyre/test_default_device_layout_spyre.py
"""

import torch

from hf_adapters.hf_common import to_default_device_layout


def _make(x, layout):
    from torch_spyre._C import copy_tensor, spyre_empty_with_layout

    torch.empty(0, dtype=x.dtype, device="spyre")  # start the device runtime
    dev = spyre_empty_with_layout(x.size(), x.stride(), x.dtype, layout)
    copy_tensor(x, dev, non_blocking=False)
    return dev


def test_other_layout_becomes_default_and_keeps_values():
    from torch_spyre._C import SpyreTensorLayout

    x = torch.randn(1, 128, 256, dtype=torch.float16)
    # Tile along a different dim than the default layout does.
    other = SpyreTensorLayout(list(x.shape), list(x.stride()), x.dtype, [1, 0, 2])
    dev = _make(x, other)
    default = SpyreTensorLayout(list(x.shape), x.dtype)
    assert dev.device_tensor_layout() != default

    out = to_default_device_layout(dev)

    assert out.device_tensor_layout() == default
    assert torch.equal(out.cpu(), dev.cpu())


def test_default_layout_keeps_layout_and_values():
    from torch_spyre._C import SpyreTensorLayout

    x = torch.randn(1, 128, 256, dtype=torch.float16)
    default = SpyreTensorLayout(list(x.shape), x.dtype)
    dev = _make(x, default)

    out = to_default_device_layout(dev)

    assert out.device_tensor_layout() == default
    assert torch.equal(out.cpu(), dev.cpu())
