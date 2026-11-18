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

"""CPU accuracy test for Transformers decision models (``AutoModelForDecision``).

Scores one request with stock ``model.decide`` as the reference, then prepares
a fresh copy with ``prepare_for_spyre`` (compiled blocks unwrapped so they run on
CPU) and scores it again through the same ``decide``. The prompts are longer
than one stick, so the Qwen3.5 linear-attention layers step through several
prefill chunks inside the one-shot readout.
"""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from _decision_helpers import decide_request
from transformers import AutoModelForDecision, AutoTokenizer

from hf_adapters.hf_common import BLOCK_SIZE, SpyreUnsupportedModelError
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import DECISION_PATHS

pytestmark = pytest.mark.model_harness("decision")

COS_THRESHOLD: float = 0.999
PROB_ATOL: float = 0.01


@pytest.mark.parametrize("model_path", DECISION_PATHS, ids=DECISION_PATHS)
def test_decision_cpu_accuracy(model_path: str) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = auto_spyre_model.dtype_for_model_path(model_path, target_device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ref_model = AutoModelForDecision.from_pretrained(model_path, dtype=dtype)
    ref_scores, ref_answers, prompt_lengths = decide_request(ref_model, tokenizer)
    del ref_model
    gc.collect()
    assert max(prompt_lengths) > BLOCK_SIZE

    model = AutoModelForDecision.from_pretrained(model_path, dtype=dtype)
    assert auto_spyre_model.prepare_for_spyre(model) is model
    _unwrap_compiled_blocks(model)
    scores, answers, _ = decide_request(model, tokenizer)
    repeat_scores, _, _ = decide_request(model, tokenizer)

    assert len(scores) == len(ref_scores)
    for score, ref_score, repeat_score in zip(scores, ref_scores, repeat_scores):
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


def test_prepare_for_spyre_rejects_non_decision_models() -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    with pytest.raises(SpyreUnsupportedModelError, match="AutoModelForDecision"):
        auto_spyre_model.prepare_for_spyre(torch.nn.Linear(2, 2))
