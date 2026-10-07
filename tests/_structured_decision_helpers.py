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

from transformers import AutoTokenizer


def build_structured_decision_batch(model_path, *, explicit_positions=False):
    from laya.common import QTYPES, build_sequence, collate_items, parallel_layout

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    questions = [
        {
            "t": "choice",
            "ins": "Choose the primary color named in the state.",
            "crit": {"red": "red", "blue": "blue", "green": "green"},
        },
        {
            "t": "score",
            "ins": "Rate the urgency.",
            "crit": ["low", "medium", "high", "critical"],
        },
        {"t": "noul", "ins": "Does the state request immediate help?"},
    ]
    items = []
    for question in questions:
        ids, markers = build_sequence(
            tokenizer,
            "The primary color is blue. The request needs immediate help.",
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
