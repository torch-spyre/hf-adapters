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

from hf_adapters import laya_backend
from hf_adapters.auto_spyre_model import dtype_for_model_path
from hf_adapters.hf_common import move_model_to_spyre
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
    from laya import load as load_laya

    dtype = dtype_for_model_path(model_path, target_device="cpu")
    model = load_laya(model_path, device="cpu", backend="eager").model.to(dtype=dtype)

    with torch.no_grad():
        ref_option_logits, ref_act_logits = model(**batch)

    move_model_to_spyre(model, laya_backend, dtype)
    for rope in model.encoder._spyre_rope.values():
        assert rope._freq_cache is not None
        assert rope._freq_cache.dtype == dtype
        assert rope._cached_len >= model.encoder.config.max_position_embeddings
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
        F.cosine_similarity(
            option_logits[valid].float(), ref_option_logits[valid].float(), dim=0
        )
        > 0.998
    )
    assert (
        F.cosine_similarity(act_logits.float(), ref_act_logits.float(), dim=-1).min()
        > 0.999
    )


@pytest.mark.parametrize(
    "model_path", STRUCTURED_DECISION_PATHS, ids=STRUCTURED_DECISION_PATHS
)
def test_laya_agent_predict(model_path):
    from laya import load as load_laya

    reference = load_laya(model_path, device="cpu", backend="eager")
    expected = reference.predict(
        STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS
    )

    agent = laya_backend.load(model_path)
    _unwrap_compiled_blocks(agent.model)
    actual = agent.predict(STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS)

    assert agent.device.type == "cpu"
    assert type(agent) is type(reference)
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
            assert actual_answer["noul"] == pytest.approx(
                expected_answer["noul"], abs=0.01
            )
        for option, probability in expected_answer.get("probabilities", {}).items():
            assert actual_answer["probabilities"][option] == pytest.approx(
                probability, abs=0.01
            )
        assert actual_answer["confidence"] == pytest.approx(
            expected_answer["confidence"], abs=0.01
        )
        assert actual_answer["action"] == expected_answer["action"]

    repeated = agent.predict(STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_QUESTIONS)
    assert repeated == actual

    batch = agent.predict_batch(
        [STRUCTURED_DECISION_STATE, STRUCTURED_DECISION_STATE],
        STRUCTURED_DECISION_QUESTIONS,
    )
    assert batch == [actual, actual]


def test_laya_load_rejects_unsupported_checkpoint():
    with pytest.raises(ValueError, match="Unsupported Laya checkpoint"):
        laya_backend.load("convaiinnovations/unknown")
