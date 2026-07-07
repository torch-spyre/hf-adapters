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

"""Spyre edge case: ``no_eos_runs_full_budget`` (eos_token_id=None crosses block)."""

import pytest
from _shared import run_no_eos
from model_registry import CAUSAL_PATHS


@pytest.mark.parametrize("model_path", CAUSAL_PATHS, ids=CAUSAL_PATHS)
@pytest.mark.slow
def test_no_eos_runs_full_budget_spyre(model_path: str) -> None:
    ok, detail = run_no_eos(model_path)
    assert ok, detail
