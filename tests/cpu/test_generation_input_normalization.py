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

import pytest
import torch

from hf_adapters.hf_common import (
    build_prefill_mask,
    encode_prompts,
    generation_cache_len,
    normalize_generation_inputs,
)


def _normalize(input_ids, attention_mask=None):
    return normalize_generation_inputs(
        torch.tensor(input_ids, dtype=torch.long),
        (
            None
            if attention_mask is None
            else torch.tensor(attention_mask, dtype=torch.long)
        ),
    )


class _RecordingTokenizer:
    eos_token = "<eos>"
    pad_token = None

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.call = None

    def apply_chat_template(self, conversations, **kwargs):
        self.call = ("chat", conversations, kwargs)
        return {"input_ids": "chat-ids", "attention_mask": "chat-mask"}

    def __call__(self, prompts, **kwargs):
        self.call = ("plain", prompts, kwargs)
        return {"input_ids": "plain-ids", "attention_mask": "plain-mask"}


def test_encode_prompts_uses_chat_template_by_default():
    tokenizer = _RecordingTokenizer(chat_template="template")

    encoded = encode_prompts(tokenizer, "hello")

    assert encoded["input_ids"] == "chat-ids"
    kind, conversations, kwargs = tokenizer.call
    assert kind == "chat"
    assert conversations == [[{"role": "user", "content": "hello"}]]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["padding_side"] == "left"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_encode_prompts_can_force_plain_right_padding():
    tokenizer = _RecordingTokenizer(chat_template="template")

    encoded = encode_prompts(
        tokenizer, ["short", "longer"], chat=False, padding_side="right"
    )

    assert encoded["input_ids"] == "plain-ids"
    kind, prompts, kwargs = tokenizer.call
    assert kind == "plain"
    assert prompts == ["short", "longer"]
    assert kwargs["padding_side"] == "right"


@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        (
            [[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]],
            [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]],
        ),
        (
            [[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        ),
        (
            [[0, 11, 12, 13, 0], [21, 22, 23, 24, 25]],
            [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
        ),
    ],
)
def test_padding_layouts_normalize_identically(input_ids, attention_mask):
    padded_ids, lengths, padded_len, offsets, position_ids = _normalize(
        input_ids, attention_mask
    )

    assert padded_len == 64
    assert lengths.tolist() == [3, 5]
    assert offsets.tolist() == [61, 59]
    assert padded_ids[0, -3:].tolist() == [11, 12, 13]
    assert padded_ids[1, -5:].tolist() == [21, 22, 23, 24, 25]
    assert position_ids[0, -3:].tolist() == [0, 1, 2]
    assert position_ids[1, -5:].tolist() == [0, 1, 2, 3, 4]


def test_right_padding_regression_keeps_the_whole_prompt():
    padded_ids, _, _, offsets, _ = _normalize([[11, 12, 13, 99, 99]], [[1, 1, 1, 0, 0]])

    assert offsets.tolist() == [61]
    assert padded_ids[0, -3:].tolist() == [11, 12, 13]


@pytest.mark.parametrize(
    ("length", "expected_padded_len", "expected_offset"),
    [(1, 64, 63), (63, 64, 1), (64, 64, 0), (65, 128, 63)],
)
def test_block_boundary_lengths(length, expected_padded_len, expected_offset):
    ids = torch.arange(1, length + 1).unsqueeze(0)
    mask = torch.ones_like(ids)

    padded_ids, lengths, padded_len, offsets, position_ids = (
        normalize_generation_inputs(ids, mask)
    )

    assert lengths.tolist() == [length]
    assert padded_len == expected_padded_len
    assert offsets.tolist() == [expected_offset]
    assert padded_ids[0, expected_offset:].tolist() == ids[0].tolist()
    assert position_ids[0, expected_offset:].tolist() == list(range(length))


def test_external_overpadding_does_not_inflate_cache_geometry():
    input_ids = torch.full((1, 4096), 99, dtype=torch.long)
    input_ids[0, :20] = torch.arange(1, 21)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[0, :20] = 1

    _, lengths, padded_len, _, _ = normalize_generation_inputs(
        input_ids, attention_mask
    )

    assert padded_len == 64
    assert generation_cache_len(lengths.max().item(), 1) == 128


def test_missing_mask_treats_every_id_as_content():
    padded_ids, lengths, _, offsets, _ = _normalize([[7, 0, 7]])

    assert lengths.tolist() == [3]
    assert offsets.tolist() == [61]
    assert padded_ids[0, -3:].tolist() == [7, 0, 7]


def test_prefill_mask_matches_normalized_offsets():
    padded_ids, _, padded_len, offsets, _ = _normalize(
        [[11, 12, 13, 99, 99], [21, 22, 23, 24, 25]],
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
    )
    mask = build_prefill_mask(2, padded_len, 128, offsets)

    assert padded_ids[:, -1].tolist() == [13, 25]
    assert (mask[0, 0, -1, : offsets[0]] < 0).all()
    assert (mask[0, 0, -1, offsets[0] : padded_len] == 0).all()
    assert (mask[0, 0, -1, padded_len:] < 0).all()


def test_inputs_are_not_mutated():
    input_ids = torch.tensor([[11, 12, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    original_ids = input_ids.clone()
    original_mask = attention_mask.clone()

    normalize_generation_inputs(input_ids, attention_mask)

    assert torch.equal(input_ids, original_ids)
    assert torch.equal(attention_mask, original_mask)


@pytest.mark.parametrize(
    ("input_ids", "attention_mask", "error", "message"),
    [
        (torch.tensor([1, 2]), None, ValueError, "shape"),
        (
            torch.empty((0, 2), dtype=torch.long),
            None,
            ValueError,
            "at least one sequence",
        ),
        (torch.empty((1, 0), dtype=torch.long), None, ValueError, "at least one token"),
        (torch.tensor([[1.0, 2.0]]), None, TypeError, "integer dtype"),
        (torch.tensor([[1, 2]]), torch.tensor([1, 1]), ValueError, "same shape"),
        (torch.tensor([[1, 2]]), torch.tensor([[1, 2]]), ValueError, "0 or 1"),
        (torch.tensor([[1, 2]]), torch.tensor([[0, 0]]), ValueError, "unmasked token"),
        (
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 0, 1, 0]]),
            ValueError,
            "contiguous span",
        ),
    ],
)
def test_invalid_inputs_raise(input_ids, attention_mask, error, message):
    with pytest.raises(error, match=message):
        normalize_generation_inputs(input_ids, attention_mask)


def test_non_tensor_inputs_raise():
    with pytest.raises(TypeError, match="input_ids"):
        normalize_generation_inputs([[1, 2]])
    with pytest.raises(TypeError, match="attention_mask"):
        normalize_generation_inputs(torch.tensor([[1, 2]]), [[1, 1]])
