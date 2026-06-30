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

Shared test helpers (``cosine_per_row``, ``encode_padded``, ``load_hf_causal_lm``,
``min_cosine``, ``torch_dtype_for``) are defined here as plain functions so that
test files can import exactly what they need via ``from conftest import ...``.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Shared test helpers — plain functions (not fixtures) importable via
# `from conftest import ...` in any test file under tests/.
# ---------------------------------------------------------------------------

# REFACTOR_BENJ : why keeping 2 conftests?


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


def encode_padded(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch with right-padding, returning ``(input_ids, attention_mask)``.

    Sets ``pad_token`` to ``eos_token`` if the tokenizer has none — common for
    decoder-only models repurposed as embedders.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="right"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def min_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> float:
    """Minimum cosine similarity between ``a`` and ``b`` along the last dim.

    Args:
        a, b: tensors with matching shape; cosine is computed over ``dim=-1``.
        attention_mask: optional ``[B, L]`` mask. When provided, the cosine is
            taken over real tokens only (``mask == 1``); without it, every
            element of the result is considered (per-row cosine for ``[B, H]``
            inputs, per-token for ``[B, L, H]``).
    """
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
    if attention_mask is not None:
        cos = cos[attention_mask.bool()]
    return cos.min().item()


def cosine_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity for ``[B, H]`` tensors. Returns a 1-D tensor."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")
# REFACTOR_BENJ : why?
# Spyre-targeted runs (`pytest tests/spyre/...`) need the unpatched hf_adapters
# with DEVICE="spyre". Detect that here and skip the CPU patching block — the
# tests/spyre/conftest.py picks up from there with the real module.
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


def _load_adapter(filename: str) -> types.ModuleType:
    """Load an adapter .py file under hf_adapters/ as a real submodule."""
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    setattr(_pkg, filename.replace(".py", ""), mod)
    return mod


# Pre-load every adapter referenced by CONFIG_TO_ADAPTER_MODULE_MAPPING, then
# auto_spyre_model itself. Doing this here means tests can grab AutoSpyre*
# off the module without paying the cost on first use.
if not _TARGETS_SPYRE:
    _auto_path = os.path.join(ADAPTERS_DIR, "auto_spyre_model.py")
    _auto_spec = importlib.util.spec_from_file_location(
        "hf_adapters.auto_spyre_model", _auto_path
    )
    _auto_mod = importlib.util.module_from_spec(_auto_spec)
    sys.modules["hf_adapters.auto_spyre_model"] = _auto_mod
    _auto_spec.loader.exec_module(_auto_mod)
    setattr(_pkg, "auto_spyre_model", _auto_mod)

    # Now that auto_spyre_model is loaded with patched hf_common, populate the model lists
    # Import model_registry here (after patching) and update its CAUSAL_KEYS/EMBED_KEYS
    import model_registry  # noqa: E402

    model_registry.CAUSAL_KEYS, model_registry.EMBED_KEYS = (
        model_registry.select_representative_models(
            _auto_mod.CONFIG_TO_ADAPTER_MODULE_MAPPING
        )
    )


def _unwrap_compiled_blocks(model: types.ModuleType) -> None:
    """Replace torch.compile-wrapped blocks with their CPU-runnable originals.

    Covers every block list an adapter may attach: ``_spyre_compiled_blocks``
    (the common case) plus ``_spyre_text_blocks`` for two-tower VLMs like Granite
    Vision, whose text decoder is compiled separately from the vision tower.
    """
    for attr in ("_spyre_compiled_blocks", "_spyre_text_blocks"):
        blocks = getattr(model, attr, None)
        if blocks is None:
            continue
        unwrapped = []
        for cb in blocks:
            orig = getattr(
                cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None)
            )
            unwrapped.append(orig if orig is not None else cb)
        setattr(model, attr, unwrapped)


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_spyre: mark test as requiring the Spyre backend device",
    )


def _set_rope_dtype(model: types.ModuleType, dtype: torch.dtype) -> None:
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
def hf_common_mod() -> types.ModuleType:
    return sys.modules["hf_adapters.hf_common"]


@pytest.fixture(scope="session")
def auto_spyre_model() -> types.ModuleType:
    return sys.modules["hf_adapters.auto_spyre_model"]


@pytest.fixture
def load_adapter() -> types.ModuleType:
    return _load_adapter


@pytest.fixture
def unwrap_compiled_blocks() -> None:
    return _unwrap_compiled_blocks


@pytest.fixture
def set_rope_dtype() -> None:
    return _set_rope_dtype


@pytest.fixture(autouse=True)
def _gc_after_test() -> None:
    yield
    gc.collect()


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
