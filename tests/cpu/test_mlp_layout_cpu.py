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

import pytest
import torch

from hf_adapters.hf_common import _run_mlp_2d


class _RankTwoMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shapes = []

    def forward(self, x):
        self.input_shapes.append(tuple(x.shape))
        assert x.ndim == 2
        return x * 2


@pytest.mark.parametrize("shape", [(1, 1, 128), (1, 512, 128), (2, 7, 128)])
def test_run_mlp_2d_flattens_tokens_and_restores_shape(shape):
    mlp = _RankTwoMLP()
    h = torch.randn(shape)

    out = _run_mlp_2d(mlp, h)

    assert mlp.input_shapes == [(shape[0] * shape[1], shape[2])]
    assert out.shape == h.shape
    torch.testing.assert_close(out, h * 2)
