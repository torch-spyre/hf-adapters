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

import os

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

STRUCTURED_DECISION_STATE = (
    "The primary color is blue. The production API is down and customers cannot log in."
)
STRUCTURED_DECISION_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "payments and refunds",
            "technical": "bugs and outages",
            "other": "anything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "Rate the urgency.",
        "criteria": ["low", "medium", "high", "critical"],
    },
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}


def build_structured_decision_batch(model_path, *, explicit_positions=False):
    from laya.common import QTYPES, build_sequence, collate_items, parallel_layout

    model_dir = snapshot_download(model_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    questions = [
        {
            "t": question["type"],
            "ins": question["instructions"],
            "crit": question.get("criteria"),
        }
        for question in STRUCTURED_DECISION_QUESTIONS.values()
    ]
    items = []
    for question in questions:
        ids, markers = build_sequence(
            tokenizer,
            STRUCTURED_DECISION_STATE,
            question,
        )
        item = {"ids": ids, "markers": markers, "qtype": QTYPES[question["t"]]}
        if explicit_positions:
            closing_head_sep = markers[-1] + 1
            while (
                closing_head_sep < len(ids)
                and ids[closing_head_sep] != tokenizer.sep_token_id
            ):
                closing_head_sep += 1
            item["layout"] = parallel_layout(markers, closing_head_sep + 1, len(ids))
        items.append(item)
    batch = collate_items([items], tokenizer.pad_token_id)
    return {
        name: batch[name]
        for name in (
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
            "position_ids",
            "option_ids",
        )
        if name in batch
    }
