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
Conftest for the Spyre test suite.

Adds ``tests/spyre/`` to ``sys.path`` so test modules in this directory can do
bare ``from _seq_classification_helpers import ...`` (and any future
``_*_helpers`` files) without the directory needing to be a Python package.
Mirrors the same pattern used by ``tests/spyre/edge_cases/conftest.py``.
"""

import os
import sys

import pytest
from _pytest.nodes import Item

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# Modules whose tests share one loaded checkpoint at a time and therefore must
# run grouped by ``model_path`` (see ``_ModelRunCache`` in the module).
_GROUP_BY_MODEL_PATH_MODULES: frozenset[str] = frozenset(
    {"test_e2e_multimodal_embed_compare_spyre.py"}
)


def _model_path_of(item: Item) -> str | None:
    callspec = getattr(item, "callspec", None)
    return callspec.params.get("model_path") if callspec is not None else None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[Item]) -> None:
    """Group the listed modules' tests by ``model_path``, keeping test order.

    pytest orders parametrized items test by test, which would interleave
    checkpoints and force a reload on every switch. Only those modules' slots
    in ``items`` are reordered; every other item keeps its position.
    """
    slots: list[int] = [
        index
        for index, item in enumerate(items)
        if item.path.name in _GROUP_BY_MODEL_PATH_MODULES
        and _model_path_of(item) is not None
    ]
    grouped: list[Item] = [items[index] for index in slots]
    first_seen: dict[str | None, int] = {}
    for item in grouped:
        first_seen.setdefault(_model_path_of(item), len(first_seen))
    grouped.sort(key=lambda item: first_seen[_model_path_of(item)])
    for index, item in zip(slots, grouped):
        items[index] = item
