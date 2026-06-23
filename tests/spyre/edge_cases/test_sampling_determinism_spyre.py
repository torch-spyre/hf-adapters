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

"""Spyre edge case: ``sampling_determinism`` (same seed -> same output)."""

import pytest
from _shared import run_sampling_determinism
from model_registry import CAUSAL_KEYS


@pytest.mark.parametrize("model_key", CAUSAL_KEYS, ids=CAUSAL_KEYS)
@pytest.mark.slow
def test_sampling_determinism_spyre(model_key):
    ok, detail = run_sampling_determinism(model_key)
    assert ok, detail
