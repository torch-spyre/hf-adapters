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
Conftest for the Spyre generate() edge-case suite.

Each case file in this directory is auto-marked ``slow`` so plain
``pytest tests/spyre/edge_cases`` deselects them by default. CI's daily lane
opts in with ``--run-slow``; PR runs do not.

This conftest also adds the edge-case directory to ``sys.path`` so the case
files can ``from _shared import ...`` without needing the directory to be a
Python package (matching the rest of the tests/ layout — no __init__.py).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
