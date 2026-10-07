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
from model_registry import STRUCTURED_DECISION_PATHS

from hf_adapters import hf_laya
from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import move_model_to_spyre
from tests.conftest import load_ref_model

pytestmark = pytest.mark.model_harness("structured_decision")


@pytest.mark.parametrize(
    "model_path", STRUCTURED_DECISION_PATHS, ids=STRUCTURED_DECISION_PATHS
)
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_e2e_structured_decision_compare_spyre(model_path, explicit_positions):
    batch = build_structured_decision_batch(
        model_path, explicit_positions=explicit_positions
    )
    dtype = dtype_for_model_path(model_path, target_device="spyre")
    model = load_ref_model(model_path, adapter_mod=hf_laya)

    with torch.no_grad():
        ref_option_logits, ref_act_logits = model(**batch)

    move_model_to_spyre(model, hf_laya, dtype)
    for rope in model.encoder._spyre_rope.values():
        assert rope._freq_cache is not None
        assert rope._freq_cache.dtype == dtype
        assert rope._cached_len >= model.encoder.config.max_position_embeddings
    with torch.no_grad():
        option_logits, act_logits = model(**batch)
        repeat_option_logits, repeat_act_logits = model(**batch)

    valid = batch["marker_mask"]
    option_cos = F.cosine_similarity(
        option_logits.masked_fill(~valid, 0),
        ref_option_logits.masked_fill(~valid, 0),
        dim=-1,
    )
    act_cos = F.cosine_similarity(act_logits, ref_act_logits, dim=-1)

    assert next(model.encoder.parameters()).device.type == "spyre"
    for name in ("head", "type_emb", "scorer", "act_head"):
        module = getattr(model, name)
        if module is not None:
            assert next(module.parameters()).device.type == "cpu"
    assert torch.isfinite(option_logits[valid]).all()
    assert torch.isfinite(act_logits).all()
    assert option_cos.min() >= 0.98
    assert act_cos.min() >= 0.99
    assert torch.equal(option_logits.argmax(-1), ref_option_logits.argmax(-1))
    assert torch.equal(act_logits.argmax(-1), ref_act_logits.argmax(-1))
    assert torch.equal(repeat_option_logits, option_logits)
    assert torch.equal(repeat_act_logits, act_logits)
