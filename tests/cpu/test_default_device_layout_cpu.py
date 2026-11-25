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

"""CPU tests for embed_text_tokens and to_default_device_layout.

On CPU the layout copy does nothing, so these check the embedding math.
"""

import torch
from torch import nn

from hf_adapters.hf_common import embed_text_tokens, to_default_device_layout


class _Backbone(nn.Module):
    def __init__(self, multiplier=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        if multiplier is not None:
            self.embedding_multiplier = multiplier


class _Model(nn.Module):
    """Stands in for a causal-LM wrapper: the backbone is ``model.model``."""

    def __init__(self, backbone):
        super().__init__()
        self.model = backbone


def test_cpu_tensor_is_returned_as_is():
    x = torch.randn(1, 8, 64, dtype=torch.float16)
    assert to_default_device_layout(x) is x


def test_embed_without_multiplier_is_the_plain_lookup():
    backbone = _Backbone()
    ids = torch.tensor([[1, 5, 7]])
    out = embed_text_tokens(_Model(backbone), ids)
    assert torch.equal(out, backbone.embed_tokens(ids))


def test_embed_applies_the_multiplier():
    backbone = _Backbone(multiplier=12.0)
    ids = torch.tensor([[2, 3]])
    out = embed_text_tokens(_Model(backbone), ids)
    assert torch.equal(out, backbone.embed_tokens(ids) * 12.0)


def test_embed_uses_the_given_backbone():
    """OPT keeps embed_tokens on the decoder, so it passes the decoder in."""
    decoder = _Backbone()
    ids = torch.tensor([[4, 0, 9]])
    out = embed_text_tokens(nn.Module(), ids, backbone=decoder)
    assert torch.equal(out, decoder.embed_tokens(ids))
