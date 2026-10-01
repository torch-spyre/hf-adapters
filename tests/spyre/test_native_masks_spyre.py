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

"""Device coverage for adapter masks that previously ran on the CPU."""

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch_spyre  # noqa: F401
from torch_spyre.ops.fallbacks import FallbackWarning

from hf_adapters.hf_common import build_prefill_mask, fairseq_position_ids
from hf_adapters.hf_gemma4 import _query_row_mask
from hf_adapters.hf_gemma4_mm import _embed_and_scatter, _replace_image_token_ids
from hf_adapters.hf_lfm2 import _padding_mask as lfm2_padding_mask
from hf_adapters.hf_qwen3_5 import _padding_mask as qwen_padding_mask

pytestmark = pytest.mark.filterwarnings(
    "error::torch_spyre.ops.fallbacks.FallbackWarning"
)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("seq_len", (1, 64))
def test_query_row_mask_handles_padding_and_all_masked_batches(dtype, seq_len):
    mask = torch.full((3, 1, seq_len, 128), -16384.0, dtype=dtype)
    mask[0, :, :, 9] = 0
    mask[1, :, seq_len // 2 :, 35] = 0
    hidden_cpu = torch.arange(64, dtype=dtype).repeat(3, seq_len, 1)
    hidden = hidden_cpu.to("spyre")
    expected = torch.zeros(3, seq_len, 1, dtype=dtype)
    expected[0] = 1
    expected[1, seq_len // 2 :] = 1

    actual = _query_row_mask(hidden, mask.to(hidden.device))

    assert actual.device == hidden.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    masked_hidden = torch.compile(lambda h, m: h * m, fullgraph=True, dynamic=False)(
        hidden, actual
    )
    torch.testing.assert_close(
        masked_hidden.cpu(), hidden_cpu * expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("seq_len, start", ((64, 0), (64, 64), (1, 80)))
@pytest.mark.parametrize("batch_size", (1, 3))
def test_causal_query_validity_matches_the_cache_diagonal(
    dtype, seq_len, start, batch_size
):
    offsets = torch.tensor([0, 5, 75])[:batch_size]
    mask = build_prefill_mask(
        batch_size, seq_len, 192, offsets, dtype, query_start=start
    )
    rows = torch.arange(seq_len)
    expected = (mask[:, 0, rows, start + rows] == 0).to(dtype)
    mask = mask.to("spyre")
    cache_index = torch.arange(start, start + seq_len, dtype=torch.int32).to("spyre")

    actual_qwen = qwen_padding_mask(mask, seq_len, cache_index)
    actual_lfm2 = lfm2_padding_mask(mask, seq_len, cache_index)

    torch.testing.assert_close(actual_qwen.cpu(), expected, rtol=0, atol=0)
    if seq_len == 1:
        expected = torch.nn.functional.pad(expected, (0, 63))
    torch.testing.assert_close(actual_lfm2.cpu(), expected, rtol=0, atol=0)


def test_fairseq_positions_handle_internal_padding_and_large_token_ids():
    ids = torch.tensor(
        [[1, 128001, 1, 128002, 128003, 1, 1, 1], [128001, 128002, 1, 1, 1, 1, 1, 1]],
        dtype=torch.int32,
    ).repeat(1, 8)
    expected = torch.ones_like(ids)
    for row in range(ids.shape[0]):
        position = 1
        for col in range(ids.shape[1]):
            if ids[row, col] != 1:
                position += 1
                expected[row, col] = position

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "default",
            message="aten.cumsum.default is falling back to cpu",
            category=FallbackWarning,
        )
        actual = fairseq_position_ids(ids.to("spyre"), padding_idx=1)

    assert actual.device.type == "spyre"
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", ((1, 64), (2, 64), (1, 1)))
def test_image_token_replacement_preserves_neighboring_large_ids(shape):
    token = 262144
    ids = torch.tensor([token, token - 1, token + 1, 0], dtype=torch.int32)
    ids = ids.repeat(32)[: shape[0] * shape[1]].reshape(shape)
    expected = ids.clone()
    expected[ids == token] = 0

    actual, image_mask = _replace_image_token_ids(ids.to("spyre"), token, 0)

    assert actual.device.type == image_mask.device.type == "spyre"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(image_mask.cpu(), (ids == token).float(), rtol=0, atol=0)


@pytest.mark.parametrize("has_ple", (False, True))
def test_image_token_replacement_feeds_the_embedding_and_feature_scatter(has_ple):
    from torch_spyre.model_utils import _dma_to_spyre_indirect_access

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 64, dtype=torch.float16)
    weight = torch.arange(8, dtype=torch.float16)[:, None].expand(8, 64).contiguous()
    model.model.embed_tokens.weight = torch.nn.Parameter(
        _dma_to_spyre_indirect_access(weight, device="spyre"), requires_grad=False
    )
    model.config = SimpleNamespace(image_token_id=262144, pad_token_id=0)
    model._spyre_has_ple = has_ple
    ids = torch.tensor([[1, 262144, 2, 3]], dtype=torch.int32).repeat(1, 16)
    features = torch.full((16, 64), 17.0, dtype=torch.float16)
    expected_ids = ids.clone()
    expected_ids[ids == 262144] = 0
    expected = weight[expected_ids]
    expected[ids == 262144] = features

    actual, ple_context, actual_ids = _embed_and_scatter(
        model, ids.to("spyre"), features
    )

    assert actual_ids.device.type == "spyre"
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if has_ple:
        torch.testing.assert_close(
            ple_context.cpu(), weight[expected_ids], rtol=0, atol=0
        )
    else:
        assert ple_context is None
