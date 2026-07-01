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
from _pytest.python import Metafunc
from transformers import AutoModelForCausalLM

# NOTE: do NOT import hf_adapters at module top level. The CPU patch block below
# rebuilds ``hf_adapters.hf_common`` with ``DEVICE='cpu'`` and asserts that no
# import has materialized it yet; a top-level import here would always trip that
# assert. ``MODEL_PATH_TO_TORCH_DTYPE`` / ``MODEL_PATH_WITH_LOAD_FN`` are pulled
# in lazily inside the helpers that use them.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")

# ---------------------------------------------------------------------------
# Shared helpers — available to all test lanes via `from conftest import ...`
# ---------------------------------------------------------------------------


def load_hf_vlm(model_path, torch_dtype, adapter_mod=None):
    """Load the HF multimodal (image→text) reference, honoring ``load_fn``.

    Mirrors :func:`load_hf_causal_lm` for VLM adapters: when ``load_fn`` is set,
    the adapter module is expected to expose ``load_hf_model(path, dtype)`` (a
    non-standard loading path); otherwise the stock
    ``AutoModelForImageTextToText`` auto class is used.
    """
    from hf_adapters.auto_spyre_model import MODEL_PATH_WITH_LOAD_FN

    if model_path in MODEL_PATH_WITH_LOAD_FN:
        if adapter_mod is None:
            raise RuntimeError("load_fn=True requires adapter_mod")
        return adapter_mod.load_hf_model(model_path, torch_dtype)
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch_dtype, device_map="cpu"
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

# This module body may execute more than once: pytest first imports it as the
# rootdir conftest (bare name ``conftest``), and a test doing
# ``from tests.conftest import ...`` triggers a second import under the dotted
# package name. The second run must be a no-op for the patch block — the CPU
# patch is already installed — so guard on whether our patched hf_common is
# already present. A bare assert here would misfire on that benign re-import.
_ALREADY_PATCHED = (
    getattr(sys.modules.get("hf_adapters.hf_common"), "DEVICE", None) == "cpu"
)

if not _TARGETS_SPYRE and not _ALREADY_PATCHED:
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

    # model_registry's top-level import will reuse the patched modules.
    import model_registry  # noqa: E402, F401

elif not _ALREADY_PATCHED:
    # Spyre lane: hf_adapters is imported normally (real DEVICE="spyre").
    # model_registry populates CAUSAL_PATHS / EMBED_PATHS at import time.
    pass  # noqa: E402

# When _ALREADY_PATCHED (benign re-import via ``from tests.conftest import ...``)
# both branches are skipped: hf_adapters is patched and the registry is
# already populated from the first execution.


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
    parser.addoption(
        "--model-path",
        action="append",
        default=[],
        help=(
            "Override the ``model_path`` parametrization for every test that "
            "takes it. Repeat the flag to run against multiple models, e.g. "
            "``--model-path foo/bar --model-path baz/qux``. When set, the "
            "registry-derived CAUSAL_PATHS / EMBED_PATHS / VISION_PATHS lists "
            "in the test decorators are ignored."
        ),
    )


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Rewrite ``model_path`` parametrization when ``--model-path`` is given.

    Every spyre / cpu test that runs over the registry declares
    ``@pytest.mark.parametrize("model_path", CAUSAL_PATHS | EMBED_PATHS | ...)``.
    When the user passes ``--model-path`` on the command line, strip the
    decorator's parametrize markers for ``model_path`` and reparametrize with
    the user-supplied list so any HF model path can be exercised — including
    ones that are not in ``tests/model_registry.py``.
    """
    overrides: list[str] = metafunc.config.getoption("--model-path") or []
    if not overrides:
        return
    if "model_path" not in metafunc.fixturenames:
        return

    # Drop the decorator's own parametrize markers for ``model_path`` so pytest
    # doesn't raise "duplicate parametrization" when we call metafunc.parametrize
    # below. Other parameter names on the same @parametrize marker are preserved.
    kept: list = []
    for marker in metafunc.definition.iter_markers("parametrize"):
        argnames = marker.args[0] if marker.args else ""
        names = [n.strip() for n in argnames.replace(",", " ").split()]
        if "model_path" in names:
            continue
        kept.append(marker)
    metafunc.definition.own_markers = [
        m for m in metafunc.definition.own_markers if m.name != "parametrize"
    ] + kept

    metafunc.parametrize("model_path", overrides, ids=overrides)


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


def torch_dtype_for_model_path(model_path: str) -> torch.dtype:
    """Map a registry entry's ``dtype`` field to a torch dtype.

    Defaults to float16. ``"float32"`` (e.g. Granite 4 1B, where fp16 overflows
    on CPU) and ``"bfloat16"`` (e.g. EmbeddingGemma, which is bf16-native and
    overflows fp16) are recognized explicitly.
    """
    from hf_adapters.auto_spyre_model import MODEL_PATH_TO_TORCH_DTYPE

    return MODEL_PATH_TO_TORCH_DTYPE.get(model_path, torch.float16)


def _load_hf_causal_lm(
    model_path: str,
    torch_dtype: torch.dtype,
    adapter_mod: types.ModuleType | None = None,
) -> AutoModelForCausalLM:
    """Load the HF causal-LM reference, honoring the per-entry ``load_fn`` flag.

    When ``load_fn`` is set, the adapter module is expected to expose
    ``load_hf_model(path, dtype)`` (used for non-standard loading paths like
    granite-vision).
    """

    from hf_adapters.auto_spyre_model import MODEL_PATH_WITH_LOAD_FN

    if model_path in MODEL_PATH_WITH_LOAD_FN:
        if adapter_mod is None:
            raise RuntimeError("load_fn=True requires adapter_mod")
        return adapter_mod.load_hf_model(model_path, torch_dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="cpu"
    )


def load_ref_model(
    model_path: str,
    adapter_mod: types.ModuleType | None = None,
) -> AutoModelForCausalLM:
    """Load the HF reference model, using the adapter's custom loader when load_fn=True."""
    dtype = torch_dtype_for_model_path(model_path)
    ref_model = _load_hf_causal_lm(
        model_path=model_path, torch_dtype=dtype, adapter_mod=adapter_mod
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    return ref_model
