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

"""What a model-module test declares as its capability verdict (JUnit `capability.*`).

Loaded with ``-p`` by ``make model-module-tests``: the upstream ``test_modules.py`` wrapper
runs outside this repo's conftest. The verdict's status is the case outcome, so this only
names the capability and says which backend it ran on.
"""

from __future__ import annotations

import functools
import json
import os
import re
import warnings
from pathlib import Path

import pytest
import yaml

_TEST_FILE = re.compile(r"(^|/)test_modules(_custom)?(__oot_wrapper)?\.py$")
_TEST_NAME = re.compile(
    r"^(test_forward|test_eager_vs_compile|test_layout_stride|test_with_cpu|test_\w+?)"
    r"_(\w+?)_spyre_(\w+)$"
)
_MODEL_TAG = re.compile(r"\bmodel__([\w.\-]+)")
_FALLBACKS = pytest.StashKey[list]()


def split_test_name(name: str) -> tuple[str, str, str] | None:
    """(test label, module, dtype) from ``test_<label>_<module>_spyre_<dtype>``."""
    m = _TEST_NAME.match(name.split("[", 1)[0])
    return (m.group(1), m.group(2), f"torch.{m.group(3)}") if m else None


@functools.lru_cache(maxsize=None)
def subject_for(config_path: str) -> str:
    """The config's one ``model__`` tag, else its file stem."""
    try:
        models = set(_MODEL_TAG.findall(Path(config_path).read_text()))
    except OSError:
        models = set()
    return models.pop() if len(models) == 1 else Path(config_path).stem


@functools.lru_cache(maxsize=None)
def module_entries(config_path: str) -> dict[str, dict]:
    """The config's module entries (those with a ``module_path``) by name."""
    try:
        doc = yaml.safe_load(Path(config_path).read_text())
    except (OSError, yaml.YAMLError):
        return {}
    found: dict[str, dict] = {}
    todo = [doc]
    while todo:
        node = todo.pop()
        if isinstance(node, dict):
            if "name" in node and "module_path" in node:
                found[str(node["name"])] = node
            todo.extend(node.values())
        elif isinstance(node, list):
            todo.extend(node)
    return found


def fallback_op(message: str) -> str:
    m = re.match(r"\s*(aten\.\S+)", message)
    if m:
        return m.group(1)
    return re.sub(r"\s*(is )?falling back to cpu\s*$", "", message).strip()


def capability_properties(
    name: str, config_path: str, fallbacks: list[str]
) -> list[tuple[str, str]]:
    """The `capability.*` properties for one module test, or [] for any other test."""
    parts = split_test_name(name)
    if not parts or not config_path:
        return []
    label, module, dtype = parts
    entry = module_entries(config_path).get(module, {})
    # A module is run with plain or device-layout parameters (the adapter path); the
    # _adapter configs repeat base entries under the latter, so it is part of the identity.
    props = [
        ("capability.test_type", "model_modules"),
        ("capability.subject", subject_for(config_path)),
        ("capability.name", module),
        ("capability.sig.test", label),
        ("capability.sig.dtype", dtype),
        (
            "capability.sig.device_layout",
            str(bool(entry.get("apply_device_layout"))).lower(),
        ),
        ("capability.backend", "cpu" if fallbacks else "spyre"),
    ]
    if entry.get("module_path"):
        props.append(("capability.prop.module_path", str(entry["module_path"])))
    if fallbacks:
        props.append(
            ("capability.prop.fallback_ops", json.dumps(list(dict.fromkeys(fallbacks))))
        )
    return props


def _is_module_test(item: pytest.Item) -> bool:
    return bool(_TEST_FILE.search(item.path.as_posix()))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        from torch_spyre.ops.fallbacks import FallbackWarning
    except Exception:
        FallbackWarning = None
    if FallbackWarning is None or not _is_module_test(item):
        yield
        return
    # FallbackWarning is filtered "once" per process, so it is recorded per test ("always")
    # and every caught warning re-issued, leaving the log as it was.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        yield
    ops = []
    for w in caught:
        # v1 module tests: delete once the dashboard reads v2 capabilities
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if issubclass(w.category, FallbackWarning):
            ops.append(fallback_op(str(w.message)))
    item.stash[_FALLBACKS] = ops


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    # The teardown report copies item.user_properties, and junitxml writes it.
    if call.when == "teardown" and _is_module_test(item):
        item.user_properties.extend(
            capability_properties(
                item.name,
                os.environ.get("PYTORCH_TEST_CONFIG", ""),
                item.stash.get(_FALLBACKS, []),
            )
        )
    yield
