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
Conftest for the CPU test lane.

Makes ``tests/`` importable so that ``model_registry``, ``_generate_edge_case_helpers``,
and ``_vision_helpers`` all resolve correctly when pytest is invoked against this
subdirectory.

The parent ``tests/conftest.py`` is picked up automatically by pytest (it is a
parent-directory conftest) and applies the ``DEVICE="cpu"`` patch to ``hf_adapters``
and populates the model-registry keys before any test module is collected here.

Fixtures and helper functions that are exclusive to the CPU test lane live here
rather than in the parent conftest so the parent stays lean and focused on the
patching infrastructure shared with the Spyre lane.
"""

from __future__ import annotations

import gc
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

# Make tests/ importable so bare-module imports in test files resolve:
#   from conftest import ...          -> this file (or tests/conftest.py for shared helpers)
#   from model_registry import ...    -> tests/model_registry.py
#   from _generate_edge_case_helpers  -> tests/_generate_edge_case_helpers.py
#   from _vision_helpers              -> tests/_vision_helpers.py
_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


# ---------------------------------------------------------------------------
# Shared test helpers — plain functions importable via `from conftest import ...`
# ---------------------------------------------------------------------------


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
# Private helpers backing fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
