# Copyright 2025 The Torch-Spyre Authors.
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

"""Gemma 4 multimodal masks match Transformers' composition order."""

import torch

from hf_adapters.hf_gemma4_mm import _blockwise_band, _build_mm_masks


def test_vision_block_is_bidirectional_only_on_sliding_layers_and_within_window():
    seqlen = 8
    causal = torch.zeros(1, 1, seqlen, seqlen)
    future = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool), diagonal=1)
    causal.masked_fill_(future, -torch.inf)

    # Positions [1, 6] form one vision block. With W=3, query 1 may attend
    # future position 5, while query 5 may not attend old position 1.
    token_types = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    blockwise = _blockwise_band(token_types, seqlen, seqlen, causal.dtype)
    masks = _build_mm_masks(causal, blockwise, sliding_window=3)

    assert torch.isneginf(masks["full_attention"][0, 0, 1, 5])
    assert masks["sliding_attention"][0, 0, 1, 5] == 0
    assert torch.isneginf(masks["sliding_attention"][0, 0, 5, 1])
    assert masks["sliding_attention"][0, 0, 5, 4] == 0
    assert torch.isneginf(masks["sliding_attention"][0, 0, 0, 7])
