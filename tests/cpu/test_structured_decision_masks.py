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

from hf_adapters.laya_backend import (
    _build_marker_selector,
    _decision_keep_mask,
    _pad_explicit_masks,
)


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


def test_decision_keep_mask_blocks_batch_and_compiler_padding():
    padding = torch.tensor([[False, False, False], [False, True, True]])

    mask = _decision_keep_mask(padding, 5)

    assert mask.shape == (2, 1, 5, 5)
    assert mask[0, 0, :, :3].all()
    assert not mask[0, 0, :, 3:].any()
    assert mask[1, 0, :, 0].all()
    assert not mask[1, 0, :, 1:].any()


def test_decision_keep_mask_rejects_left_padding():
    padding = torch.tensor([[True, False, False]])

    with pytest.raises(ValueError, match="right-padded"):
        _decision_keep_mask(padding, 64)


def test_decision_keep_mask_rejects_invalid_shape_and_dtype():
    with pytest.raises(ValueError, match="boolean"):
        _decision_keep_mask(torch.zeros((1, 3)), 64)
    with pytest.raises(ValueError, match="boolean"):
        _decision_keep_mask(torch.zeros((1, 1, 3), dtype=torch.bool), 64)
    with pytest.raises(ValueError, match="sequence length"):
        _decision_keep_mask(torch.zeros((1, 0), dtype=torch.bool), 0)


def test_build_marker_selector_packs_cls_and_markers():
    positions = torch.tensor([[3, 3, 1, 99], [4, -2, 0, 99]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    selector = _build_marker_selector(positions, mask, 5, torch.bfloat16)

    assert selector.shape == (2, 5, 5)
    assert selector.dtype == torch.bfloat16
    assert torch.equal(selector.sum(-1), torch.ones((2, 5), dtype=torch.bfloat16))
    assert selector[0].argmax(-1).tolist() == [0, 3, 3, 1, 0]
    assert selector[1].argmax(-1).tolist() == [0, 4, 0, 0, 0]


def test_build_marker_selector_rejects_valid_out_of_range_position():
    with pytest.raises(ValueError, match="exceeds"):
        _build_marker_selector(
            torch.tensor([[5]]), torch.tensor([[True]]), 5, torch.bfloat16
        )
