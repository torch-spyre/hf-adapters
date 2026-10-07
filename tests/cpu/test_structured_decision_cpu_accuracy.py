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
import torch.nn.functional as F
from _structured_decision_helpers import build_structured_decision_batch

from hf_adapters import hf_laya
from hf_adapters.hf_common import move_model_to_spyre
from tests.conftest import load_ref_model
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import STRUCTURED_DECISION_PATHS

pytestmark = pytest.mark.model_harness("structured_decision")


@pytest.mark.parametrize(
    "model_path", STRUCTURED_DECISION_PATHS, ids=STRUCTURED_DECISION_PATHS
)
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_structured_decision_cpu_accuracy(model_path, explicit_positions):
    batch = build_structured_decision_batch(
        model_path, explicit_positions=explicit_positions
    )
    model = load_ref_model(model_path, adapter_mod=hf_laya)

    with torch.no_grad():
        ref_option_logits, ref_act_logits = model(**batch)

    move_model_to_spyre(model, hf_laya, next(model.parameters()).dtype)
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        option_logits, act_logits = model(**batch)

    valid = batch["marker_mask"]
    assert option_logits.shape == ref_option_logits.shape
    assert act_logits.shape == ref_act_logits.shape
    assert torch.isfinite(option_logits[valid]).all()
    assert torch.isfinite(act_logits).all()
    assert torch.equal(option_logits.argmax(-1), ref_option_logits.argmax(-1))
    assert torch.equal(act_logits.argmax(-1), ref_act_logits.argmax(-1))
    assert (
        F.cosine_similarity(option_logits[valid], ref_option_logits[valid], dim=0)
        > 0.998
    )
    assert F.cosine_similarity(act_logits, ref_act_logits, dim=-1).min() > 0.999
