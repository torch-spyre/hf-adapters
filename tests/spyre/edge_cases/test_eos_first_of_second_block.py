# Copyright 2025 The Torch-Spyre Authors.
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

"""Forced EOS at offset BLOCK_SIZE: first token of second block (expansion arm)."""

import pytest
from _common import run_forced_eos_case
from _generate_edge_case_helpers import BLOCK_SIZE
from model_registry import CAUSAL_LM_MODELS


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_eos_first_of_second_block(model_key):
    run_forced_eos_case(
        model_key, eos_offsets=[BLOCK_SIZE], max_new_tokens=BLOCK_SIZE + 16
    )
