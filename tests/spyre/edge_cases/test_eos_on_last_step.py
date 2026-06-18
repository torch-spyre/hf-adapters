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

"""EOS on the very last allowed step: ``finished.all()`` and the
``i == max_new_tokens - 1`` loop end coincide.
"""

import pytest
from _common import run_forced_eos_case
from model_registry import CAUSAL_LM_MODELS


@pytest.mark.parametrize("model_key", list(CAUSAL_LM_MODELS.keys()))
def test_eos_on_last_step(model_key):
    # max_new_tokens = eos_offset + 1 so the EOS token is the last one emitted.
    run_forced_eos_case(model_key, eos_offsets=[7], max_new_tokens=8)
