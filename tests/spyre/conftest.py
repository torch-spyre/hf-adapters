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

"""Spyre-side scaffolding.

The CPU sibling at ``tests/conftest.py`` reloads ``hf_adapters.hf_common`` with
``DEVICE = "cpu"`` so the adapters can run on a CPU host. This file does the
opposite: it leaves the canonical install untouched so Spyre tests bind to
``DEVICE = "spyre"`` (the source default).

If ``torch_spyre`` is not importable, every test in this directory is skipped
at collection time. That keeps a plain ``pytest tests/`` run on a CPU host from
failing — Spyre tests are meaningful only on the pod.
"""

import gc
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# tests/_helpers.py and tests/model_registry.py live one level up; make them
# importable from this subdir.
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def pytest_collection_modifyitems(config, items):
    """Skip every Spyre test if the runtime is not present, or if the parent
    ``tests/conftest.py`` already CPU-patched ``hf_common`` (mixed pytest run
    like ``pytest tests/``). Spyre tests need ``DEVICE = "spyre"`` and the
    canonical install — running them against the CPU-patched module would
    silently produce wrong results.
    """
    skip_no_spyre = None
    try:
        import torch_spyre  # noqa: F401
    except ImportError:
        skip_no_spyre = pytest.mark.skip(reason="torch_spyre not installed")

    skip_cpu_patched = None
    if "hf_adapters.hf_common" in sys.modules:
        if getattr(sys.modules["hf_adapters.hf_common"], "DEVICE", None) == "cpu":
            skip_cpu_patched = pytest.mark.skip(
                reason=(
                    "hf_common was CPU-patched by tests/conftest.py "
                    "(mixed pytest tests/ run); invoke pytest with tests/spyre/... only"
                )
            )

    if skip_no_spyre is None and skip_cpu_patched is None:
        return
    for item in items:
        if skip_no_spyre is not None:
            item.add_marker(skip_no_spyre)
        if skip_cpu_patched is not None:
            item.add_marker(skip_cpu_patched)


@pytest.fixture(scope="session")
def hf_common_mod():
    from hf_adapters import hf_common

    return hf_common


@pytest.fixture(scope="session")
def auto_spyre_model():
    from hf_adapters import auto_spyre_model as mod

    return mod


@pytest.fixture
def load_adapter():
    """Return ``hf_adapters.<name>`` as a real module (no path tricks)."""
    import importlib

    def _load(filename):
        mod_name = f"hf_adapters.{filename.replace('.py', '')}"
        return importlib.import_module(mod_name)

    return _load


@pytest.fixture
def set_rope_dtype():
    """Propagate dtype to the model's precomputed RoPE freq cache."""

    def _set(model, dtype):
        from hf_adapters.hf_common import set_rope_dtype as _impl

        _impl(model, dtype)

    return _set


@pytest.fixture(autouse=True)
def _gc_after_test():
    yield
    gc.collect()
