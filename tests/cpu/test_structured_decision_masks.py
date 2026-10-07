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

from hf_adapters.hf_laya import _pad_explicit_masks


def test_pad_explicit_masks_blocks_padded_queries_and_keys():
    full = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    sliding = torch.eye(3, dtype=torch.bool)[None, None]

    masks = _pad_explicit_masks(
        {"full_attention": full, "sliding_attention": sliding}, 2
    )

    for mask in masks.values():
        assert mask.shape == (1, 1, 5, 5)
        assert not mask[..., 3:, :].any()
        assert not mask[..., :, 3:].any()
    assert torch.equal(masks["sliding_attention"][..., :3, :3], sliding)


def test_pad_explicit_masks_accepts_only_present_layer_types():
    full = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    padded = _pad_explicit_masks({"full_attention": full}, 0)
    assert list(padded) == ["full_attention"]
    assert torch.equal(padded["full_attention"], full)


def test_pad_explicit_masks_rejects_shape_mismatch():
    masks = {
        name: torch.ones((1, 1, 3, 3), dtype=torch.bool)
        for name in ("full_attention", "sliding_attention")
    }
    with pytest.raises(ValueError, match="sequence size"):
        _pad_explicit_masks(masks, 0, batch_size=1, sequence_length=4)


def test_pad_explicit_masks_rejects_additive_masks():
    masks = {
        name: torch.zeros((1, 1, 3, 3))
        for name in ("full_attention", "sliding_attention")
    }
    with pytest.raises(ValueError, match="must be boolean"):
        _pad_explicit_masks(masks, 0)
