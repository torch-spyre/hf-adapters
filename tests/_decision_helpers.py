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

"""Shared request and scoring helpers for Transformers decision-model tests."""

from typing import Any

import torch
from _structured_decision_helpers import (
    STRUCTURED_DECISION_QUESTIONS,
    STRUCTURED_DECISION_STATE,
)

# A System One request with one question of each type (choice, score, noul).
DECISION_REQUEST: dict[str, Any] = {
    "state": STRUCTURED_DECISION_STATE,
    "questions": STRUCTURED_DECISION_QUESTIONS,
}


def decide_request(
    model, tokenizer, request: dict[str, Any] = DECISION_REQUEST
) -> tuple[list[torch.Tensor], dict[str, dict], list[int]]:
    """Score ``request`` with ``model.decide``, as ``DecisionPipeline`` does.

    Returns the option scores of every question (in request order), the answers
    keyed by question name, and the token length of each prompt.
    """
    scores: list[torch.Tensor] = []
    answers: dict[str, dict] = {}
    prompt_lengths: list[int] = []
    for inputs, questions in model.decision_config.prepare_inputs(request, tokenizer):
        prompt_lengths.append(int(inputs["input_ids"].shape[1]))
        prompt_scores = model.decide(**inputs, decision_questions=questions)
        scores.extend(prompt_scores)
        answers |= {
            question.name: question.answer(score.softmax(-1).tolist())
            for question, score in zip(questions, prompt_scores)
        }
    return scores, answers, prompt_lengths
