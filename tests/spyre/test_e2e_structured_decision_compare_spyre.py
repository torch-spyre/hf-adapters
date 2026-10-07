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
from _structured_decision_helpers import (
    STRUCTURED_DECISION_QUESTIONS,
    STRUCTURED_DECISION_STATE,
    build_structured_decision_batch,
)
from model_registry import STRUCTURED_DECISION_PATHS

from hf_adapters import hf_laya
from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import move_model_to_spyre

pytestmark = pytest.mark.model_harness("structured_decision")


@pytest.mark.parametrize(
    "model_path", STRUCTURED_DECISION_PATHS, ids=STRUCTURED_DECISION_PATHS
)
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_e2e_structured_decision_compare_spyre(model_path, explicit_positions):
    batch = build_structured_decision_batch(
        model_path, explicit_positions=explicit_positions
    )
    from laya import load as load_laya

    dtype = dtype_for_model_path(model_path, target_device="spyre")
    model = load_laya(model_path, device="cpu", backend="eager").model.to(dtype=dtype)

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


@pytest.mark.parametrize(
    "model_path", STRUCTURED_DECISION_PATHS, ids=STRUCTURED_DECISION_PATHS
)
def test_laya_agent_predict_spyre(model_path):
    from laya import load as load_laya

    reference = load_laya(model_path, device="cpu", backend="eager")
    expected = reference.predict(
        STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS
    )

    agent = hf_laya.load(model_path)
    actual = agent.predict(STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS)
    repeated = agent.predict(STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS)

    assert agent.device.type == "cpu"
    assert next(agent.model.encoder.parameters()).device.type == "spyre"
    for name in ("head", "type_emb", "scorer", "act_head"):
        module = getattr(agent.model, name)
        if module is not None:
            assert next(module.parameters()).device.type == "cpu"

    assert actual["answers"].keys() == expected["answers"].keys()
    for key in expected["answers"]:
        expected_answer = expected["answers"][key]
        actual_answer = actual["answers"][key]
        assert actual_answer["type"] == expected_answer["type"]
        if expected_answer["type"] == "choice":
            assert actual_answer["choice"] == expected_answer["choice"]
        elif expected_answer["type"] == "score":
            assert actual_answer["score"] == pytest.approx(
                expected_answer["score"], abs=0.05
            )
        else:
            assert (actual_answer["noul"] >= 0.5) == (expected_answer["noul"] >= 0.5)
    assert repeated == actual

    batch = agent.predict_batch(
        [STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_STATE],
        STRUCTURED_DECISION_QUESTIONS,
    )
    for result in batch:
        assert result["usage"] == actual["usage"]
        for key in actual["answers"]:
            expected_answer = actual["answers"][key]
            batch_answer = result["answers"][key]
            assert batch_answer["type"] == expected_answer["type"]
            if expected_answer["type"] == "choice":
                assert batch_answer["choice"] == expected_answer["choice"]
            elif expected_answer["type"] == "score":
                assert batch_answer["score"] == pytest.approx(
                    expected_answer["score"], abs=0.05
                )
            else:
                assert (batch_answer["noul"] >= 0.5) == (expected_answer["noul"] >= 0.5)
