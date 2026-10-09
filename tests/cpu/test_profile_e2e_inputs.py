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

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts import profile_e2e_spyre as profiler


@pytest.fixture
def tokenizer():
    # Like Gemma 4, direct tokenization has no special-token post-processor:
    # the chat template is responsible for the BOS and generation prefix.
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "user": 3,
                "model": 4,
                "The": 5,
                "capital": 6,
                "of": 7,
                "France": 8,
                "is": 9,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        chat_template=(
            "{{ bos_token }}user {{ messages[0]['content'] }}{{ eos_token }}"
            "{% if add_generation_prompt %}model{% endif %}"
        ),
    )


@pytest.mark.parametrize("raw_prompt", [False, True])
def test_profile_chat_and_raw_inputs(tokenizer, raw_prompt):
    encoded = profiler.build_inputs(
        tokenizer, "The capital of France is", 2, None, raw_prompt=raw_prompt
    )
    expected = [5, 6, 7, 8, 9] if raw_prompt else [1, 3, 5, 6, 7, 8, 9, 2, 4]
    assert encoded["input_ids"].tolist() == [expected, expected]
    assert encoded["attention_mask"].tolist() == [[1] * len(expected)] * 2


def test_profile_base_model_without_chat_template(tokenizer):
    tokenizer.chat_template = None
    encoded = profiler.build_inputs(tokenizer, "The capital of France is", 1, None)
    assert encoded["input_ids"].tolist() == [[5, 6, 7, 8, 9]]


@pytest.mark.parametrize("length", [4, 16])
def test_profile_synthetic_workload_has_requested_shape(tokenizer, length):
    encoded = profiler.build_inputs(tokenizer, "The capital of France is", 2, length)
    assert encoded["input_ids"].shape == (2, length)
    assert torch.equal(encoded["input_ids"][0], encoded["input_ids"][1])
    assert encoded["attention_mask"].all()


def test_profile_resolves_checkpoint_dtype(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "dtype": "bfloat16", "text_config": {}})
    )
    monkeypatch.setitem(profiler.MODELS, "gemma4_26b_a4b", str(tmp_path))
    captured = {}
    monkeypatch.setattr(
        profiler, "run_profile", lambda **kwargs: captured.update(kwargs)
    )
    profiler.main(["--model", "gemma4_26b_a4b"])
    assert captured["dtype"] == torch.bfloat16
    assert captured["raw_prompt"] is False
