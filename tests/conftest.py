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
Shared scaffolding for CPU accuracy tests.

Module-level code below runs at conftest import — i.e. before any test module
in tests/ is loaded — so adapter modules bind to a CPU-patched ``hf_common``.
The trick: load ``hf_common.py`` via importlib, set ``DEVICE = "cpu"``, install
it in ``sys.modules`` under the canonical name, then synthesize an
``hf_adapters`` package pointing at the source directory. Subsequent
``import hf_adapters.X`` calls find our patched version first.

The defensive ``assert`` at the top of this file fails loudly if anything
imported ``hf_adapters`` before pytest reached us — which would lock in the
un-patched DEVICE and silently break CPU tests.
"""

import gc
import importlib.util
import os
import sys
import types

import pytest
from _helpers import (  # noqa: F401  (re-exported for tests via `from conftest import ...`)
    cosine_per_row,
    encode_padded,
    load_hf_causal_lm,
    min_cosine,
    torch_dtype_for,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")


def _running_spyre_only():
    """True iff every positional pytest target lives under tests/spyre/.

    Spyre tests live in a sibling subdir with their own conftest and must run
    against the canonical (DEVICE="spyre") install — not the CPU-patched
    module this file would otherwise load. When the user runs
    ``pytest tests/spyre/...`` directly, skip the CPU-patching block below so
    the Spyre tests see the unpatched ``hf_common``.

    Subtlety: a few pytest flags take a value as the *next* argv entry —
    ``-k qwen3``, ``-m spyre_block``, etc. A naive walk over ``sys.argv``
    would misclassify ``qwen3`` as a positional that doesn't match
    ``tests/spyre`` and then conclude "not spyre-only," which silently
    re-enables the CPU patch on Spyre runs and skips every Spyre test. We
    enumerate the value-taking flags we actually care about and skip their
    value when parsing.

    Mixed invocations (e.g. ``pytest tests/`` or ``pytest tests/spyre/x.py
    tests/test_load_cpu.py``) keep the CPU patch active. The Spyre conftest
    has its own backstop that skips every Spyre test if ``DEVICE == "cpu"``.
    """
    # Pytest options whose value is the NEXT argv entry. Limited to the ones
    # we use in practice plus a few common neighbours; not a full parser.
    VALUE_TAKING_FLAGS = {
        "-k",
        "-m",
        "-p",
        "-o",
        "--tb",
        "--rootdir",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--basetemp",
        "--maxfail",
        "--log-level",
    }

    positionals = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in VALUE_TAKING_FLAGS:
            skip_next = True
            continue
        if a.startswith("-"):
            continue  # boolean flag (-s, -vvv) or `--flag=value` form
        positionals.append(a)

    if not positionals:
        return False
    return all("tests/spyre" in p.replace("\\", "/") for p in positionals)


_SPYRE_ONLY = _running_spyre_only()

if not _SPYRE_ONLY:
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


def _load_adapter(filename):
    """Load an adapter .py file under hf_adapters/ as a real submodule."""
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not _SPYRE_ONLY:
        setattr(_pkg, filename.replace(".py", ""), mod)
    return mod


if not _SPYRE_ONLY:
    # Pre-load every adapter referenced by CONFIG_TO_ADAPTER_MODULE_MAPPING,
    # then auto_spyre_model itself. Doing this here means tests can grab
    # AutoSpyre* off the module without paying the cost on first use.
    _auto_path = os.path.join(ADAPTERS_DIR, "auto_spyre_model.py")
    _auto_spec = importlib.util.spec_from_file_location(
        "hf_adapters.auto_spyre_model", _auto_path
    )
    _auto_mod = importlib.util.module_from_spec(_auto_spec)
    sys.modules["hf_adapters.auto_spyre_model"] = _auto_mod
    _auto_spec.loader.exec_module(_auto_mod)
    setattr(_pkg, "auto_spyre_model", _auto_mod)


def _unwrap_compiled_blocks(model):
    """Replace torch.compile-wrapped blocks with their CPU-runnable originals."""
    if not hasattr(model, "_spyre_compiled_blocks"):
        return
    unwrapped = []
    for cb in model._spyre_compiled_blocks:
        orig = getattr(cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None))
        unwrapped.append(orig if orig is not None else cb)
    model._spyre_compiled_blocks = unwrapped


def _set_rope_dtype(model, dtype):
    """Propagate the chosen dtype to the model's precomputed RoPE freq cache.

    The manual CPU-test paths load via ``AutoModel`` + ``prepare_for_spyre``
    directly, bypassing ``load_model_common`` / ``_move_to_spyre_with_layout``
    (where the production paths make this explicit ``set_dtype`` call). Mirror
    it here so a non-fp16 model (e.g. bf16 EmbeddingGemma) gets a matching freq
    cache instead of the fp16 default — otherwise ``apply_rope_matmul`` promotes
    the query to fp32 and SDPA rejects the mismatched key/value dtype.
    """
    sys.modules["hf_adapters.hf_common"].set_rope_dtype(model, dtype)


@pytest.fixture(scope="session")
def hf_common_mod():
    return sys.modules["hf_adapters.hf_common"]


@pytest.fixture(scope="session")
def auto_spyre_model():
    return sys.modules["hf_adapters.auto_spyre_model"]


@pytest.fixture
def load_adapter():
    return _load_adapter


@pytest.fixture
def unwrap_compiled_blocks():
    return _unwrap_compiled_blocks


@pytest.fixture
def set_rope_dtype():
    return _set_rope_dtype


@pytest.fixture(autouse=True)
def _gc_after_test():
    yield
    gc.collect()
