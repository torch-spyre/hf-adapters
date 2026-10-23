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

"""Last-token selection must release the full prefill storage before the head."""

from types import SimpleNamespace

import pytest
import torch

from hf_adapters.hf_common import run_lm_head


@pytest.mark.parametrize("batch", [2])
def test_last_prefill_token_has_compact_independent_storage(batch):
    hidden = torch.arange(batch * 7 * 64).reshape(batch, 7, 64)
    model = SimpleNamespace(_spyre_lm_head_forward=lambda value: value)

    selected = run_lm_head(model, hidden, logits_to_keep=1)

    assert torch.equal(selected, hidden[:, -1:, :])
    assert selected.is_contiguous()
    assert selected.storage_offset() == 0
    assert selected.untyped_storage().nbytes() == (
        selected.numel() * selected.element_size()
    )
    selected.zero_()
    assert torch.count_nonzero(hidden[:, -1:, :]) == batch * 64
