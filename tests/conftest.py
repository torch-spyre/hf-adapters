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
Root conftest — CPU-patch scaffolding and global pytest hooks.

Module-level code below runs at conftest import — i.e. before any test module
in tests/ is loaded — so adapter modules bind to a CPU-patched ``hf_common``.
The trick: load ``hf_common.py`` via importlib, set ``DEVICE = "cpu"``, install
it in ``sys.modules`` under the canonical name, then synthesize an
``hf_adapters`` package pointing at the source directory. Subsequent
``import hf_adapters.X`` calls find our patched version first.

The defensive ``assert`` below fails loudly if anything imported ``hf_adapters``
before pytest reached us — which would lock in the un-patched DEVICE and
silently break CPU tests.

Spyre-targeted runs (``pytest tests/spyre/...``) are detected via ``sys.argv``
and skip the CPU-patching block entirely; the Spyre lane imports
``hf_adapters`` normally with the real ``DEVICE="spyre"``, so no separate
``tests/spyre/conftest.py`` is needed.

``model_registry`` populates ``CAUSAL_KEYS`` / ``EMBED_KEYS`` itself at import
time off ``hf_adapters.auto_spyre_model.CONFIG_TO_ADAPTER_MODULE_MAPPING``. In
the CPU lane, the patched ``auto_spyre_model`` must already be in
``sys.modules`` before ``model_registry`` is imported — the block below
arranges that ordering.

CPU-lane test helpers and fixtures live in ``tests/cpu/conftest.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from transformers import AutoModelForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")

# ---------------------------------------------------------------------------
# Shared helpers — available to all test lanes via `from conftest import ...`
# ---------------------------------------------------------------------------


def load_hf_causal_lm(
    info: dict,
    torch_dtype: torch.dtype,
    adapter_mod: types.ModuleType | None = None,
) -> AutoModelForCausalLM:
    """Load the HF causal-LM reference, honoring the per-entry ``load_fn`` flag.

    When ``load_fn`` is set, the adapter module is expected to expose
    ``load_hf_model(path, dtype)`` (used for non-standard loading paths like
    granite-vision).
    """
    if info.get("load_fn"):
        if adapter_mod is None:
            raise RuntimeError("load_fn=True requires adapter_mod")
        return adapter_mod.load_hf_model(info["path"], torch_dtype)
    return AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )


# ---------------------------------------------------------------------------

# Make tests/ importable so model_registry and helpers resolve from any subdir.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Spyre-targeted runs (`pytest tests/spyre/...`) need the unpatched hf_adapters
# with DEVICE="spyre". Detect that here and skip the CPU patching block.
_TARGETS_SPYRE = any(
    "tests/spyre" in a or a.rstrip("/").endswith("tests/spyre") for a in sys.argv
)

if not _TARGETS_SPYRE:
    assert "hf_adapters.hf_common" not in sys.modules, (
        "hf_adapters.hf_common was imported before tests/conftest.py ran; "
        "the DEVICE='cpu' patch will not apply. Check for plugins or other "
        "conftests that import hf_adapters at collection time."
    )

    _common_path = os.path.join(ADAPTERS_DIR, "hf_common.py")
    _common_spec = importlib.util.spec_from_file_location(
        "hf_adapters.hf_common", _common_path
    )
    _common_mod = importlib.util.module_from_spec(_common_spec)
    sys.modules["hf_adapters.hf_common"] = _common_mod
    _common_spec.loader.exec_module(_common_mod)
    _common_mod.DEVICE = "cpu"

    _pkg = types.ModuleType("hf_adapters")
    _pkg.__path__ = [ADAPTERS_DIR]
    sys.modules["hf_adapters"] = _pkg

    # Pre-load auto_spyre_model with the patched hf_common already in sys.modules
    # so model_registry's top-level import reuses the patched modules.
    _auto_path = os.path.join(ADAPTERS_DIR, "auto_spyre_model.py")
    _auto_spec = importlib.util.spec_from_file_location(
        "hf_adapters.auto_spyre_model", _auto_path
    )
    _auto_mod = importlib.util.module_from_spec(_auto_spec)
    sys.modules["hf_adapters.auto_spyre_model"] = _auto_mod
    _auto_spec.loader.exec_module(_auto_mod)
    setattr(_pkg, "auto_spyre_model", _auto_mod)

# Spyre lane: hf_adapters is imported normally (real DEVICE="spyre").
# model_registry populates CAUSAL_KEYS / EMBED_KEYS at import time in both lanes.
import model_registry  # noqa: E402, F401


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_spyre: mark test as requiring the Spyre backend device",
    )


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (deselected by default)",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Skip spyre tests if torch_spyre is not installed / device unavailable; skip slow tests unless --run-slow."""
    try:
        import torch
        import torch_spyre  # noqa: F401 — side effect: registers "spyre" device

        # Verify the device actually registered
        _ = torch.device("spyre")
        spyre_available = True
    except (ImportError, RuntimeError):
        spyre_available = False

    if not spyre_available:
        skip_spyre = pytest.mark.skip(
            reason="torch_spyre not installed or spyre device unavailable"
        )
        for item in items:
            if "spyre" in item.nodeid or item.get_closest_marker("requires_spyre"):
                item.add_marker(skip_spyre)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow test; pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def torch_dtype_for(info: dict) -> torch.dtype:
    """Map a registry entry's ``dtype`` field to a torch dtype.

    Defaults to float16. ``"float32"`` (e.g. Granite 4 1B, where fp16 overflows
    on CPU) and ``"bfloat16"`` (e.g. EmbeddingGemma, which is bf16-native and
    overflows fp16) are recognized explicitly.
    """
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(info.get("dtype"), torch.float16)
