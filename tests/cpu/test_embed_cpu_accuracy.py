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
CPU accuracy test for the embedding path.

Covers both decoder-backbone embedders (Qwen3-Embedding, etc.)
and encoder-only models (e.g., BERT-family).

For each registered embedder backbone, two parametrized test cases run:

  test_manual_path[<key>]
    Loads the backbone via stock ``AutoModel`` on CPU, captures the
    reference ``last_hidden_state``, then loads a fresh copy and applies
    the adapter directly (``prepare_for_spyre`` + the appropriate prefill
    driver):
    - ``prefill_encoder`` for encoder-only adapters
      (adapter sets ``_is_encoder_only = True``).
    - ``prefill_embed`` for decoder-backbone embedders.
    Asserts per-token cosine similarity stays above the threshold.

  test_auto_loader[<key>]
    Same comparison, but the adapter side goes through
    ``AutoSpyreModel.from_pretrained``. Exercises the auto-resolution
    path that production callers use.

The DEVICE='cpu' patching of ``hf_common`` happens once in
``tests/conftest.py``; this file is plain pytest.
"""

import gc
import sys
import types

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from tests.conftest import (
    get_dtype_for_cpu,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks, encode_padded, min_cosine
from tests.model_registry import EMBED_PATHS

PROMPTS: list[str] = [
    "The capital of France is Paris.",
    "Sentence embeddings are useful.",
]
COS_THRESHOLD: float = 0.999


def _run_prefill(
    adapter_mod: types.ModuleType,
    hf_common_mod: types.ModuleType,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to prefill_encoder or prefill_embed based on adapter type."""
    if getattr(adapter_mod, "_is_encoder_only", False):
        return hf_common_mod.prefill_encoder(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )
    else:
        return hf_common_mod.prefill_embed(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_auto_loader(model_path: str) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    torch_dtype = get_dtype_for_cpu(model_path=model_path)
    adapter_module = resolve_adapter_module_for_test(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids, attention_mask = encode_padded(tokenizer, PROMPTS)

    # HF reference (loaded fresh, before the auto-loader path).
    ref_model = load_ref_model(
        model_path=model_path, adapter_mod=adapter_module, auto_model_cls=AutoModel
    )

    with torch.no_grad():
        ref_hidden = ref_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
    del ref_model
    gc.collect()

    # Auto-loader path
    model = auto_spyre_model.AutoSpyreModel.from_pretrained(
        model_path, dtype=torch_dtype
    )
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        adapter_hidden, _ = _run_prefill(
            adapter_module, hf_common_mod, model, input_ids, attention_mask
        )
    del model
    gc.collect()

    assert (
        adapter_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: adapter {adapter_hidden.shape} vs ref {ref_hidden.shape}"
    min_cos = min_cosine(adapter_hidden, ref_hidden, attention_mask)
    assert (
        min_cos >= COS_THRESHOLD
    ), f"min per-token cosine {min_cos:.6f} < threshold {COS_THRESHOLD}"
