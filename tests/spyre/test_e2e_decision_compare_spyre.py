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

"""Spyre comparison for Transformers decision models (``AutoModelForDecision``).

The stock CPU ``decide`` is the reference. The prepared model runs its text
decoder (including the Gated DeltaNet core) and its decision head on Spyre;
``decide``'s bookkeeping (marks, logits processors) stays on CPU.
"""

import pytest
import torch
import torch.nn.functional as F
from _decision_helpers import decide_request
from model_registry import DECISION_PATHS
from transformers import AutoModelForDecision, AutoTokenizer

from hf_adapters import AutoSpyreModelForDecision
from hf_adapters.auto_spyre_model import dtype_for_model_path

pytestmark = pytest.mark.model_harness("decision")

COS_THRESHOLD: float = 0.99
PROB_ATOL: float = 0.05


def _assert_module_device(module, expected):
    devices = {parameter.device.type for parameter in module.parameters()}
    devices.update(buffer.device.type for buffer in module.buffers())
    assert devices == {expected}


@pytest.mark.parametrize("model_path", DECISION_PATHS, ids=DECISION_PATHS)
def test_e2e_decision_compare_spyre(model_path):
    dtype = dtype_for_model_path(model_path, target_device="spyre")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = AutoModelForDecision.from_pretrained(model_path, dtype=dtype)
    ref_scores, ref_answers, _ = decide_request(ref_model, tokenizer)
    del ref_model

    model = AutoSpyreModelForDecision.from_pretrained(model_path)
    scores, answers, _ = decide_request(model, tokenizer)
    repeat_scores, repeat_answers, _ = decide_request(model, tokenizer)

    backbone = model.base_model
    _assert_module_device(backbone.language_model, "spyre")
    _assert_module_device(model.proj, "spyre")
    _assert_module_device(backbone.visual, "cpu")

    assert len(scores) == len(ref_scores)
    for score, ref_score, repeat_score in zip(scores, ref_scores, repeat_scores):
        assert score.device.type == "cpu"
        assert score.shape == ref_score.shape
        assert torch.isfinite(score).all()
        assert torch.equal(repeat_score, score)
        assert score.argmax() == ref_score.argmax()
        assert F.cosine_similarity(score, ref_score, dim=0) >= COS_THRESHOLD
        torch.testing.assert_close(
            score.softmax(-1), ref_score.softmax(-1), atol=PROB_ATOL, rtol=0
        )

    assert answers.keys() == ref_answers.keys()
    for name, ref_answer in ref_answers.items():
        assert answers[name]["type"] == ref_answer["type"]
    assert repeat_answers == answers
