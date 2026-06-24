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

"""
Conftest for the Spyre-pod test lane.

Unlike the parent ``tests/conftest.py`` (which CPU-patches ``hf_adapters`` so
the CPU accuracy tests can run on a laptop), this conftest leaves the real
``hf_adapters`` package alone so ``DEVICE="spyre"`` and the actual torch_spyre
kernels are exercised.

The parent conftest detects a Spyre-targeted invocation via ``sys.argv`` and
skips its patch block; we then populate ``model_registry.CAUSAL_KEYS`` /
``EMBED_KEYS`` ourselves with the *full* registries. CI shards one job per
``model_key`` and uses ``-k <key>`` to select it, so collapsing to one model
per adapter (as ``select_representative_models`` does for local pytest runs)
would silently deselect every job whose key wasn't the chosen representative.
"""

import os
import sys

# Make tests/ importable so model_registry and _helpers resolve from this subdir.
_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import model_registry  # noqa: E402

model_registry.CAUSAL_KEYS, model_registry.EMBED_KEYS = (
    model_registry.select_representative_models()
)
