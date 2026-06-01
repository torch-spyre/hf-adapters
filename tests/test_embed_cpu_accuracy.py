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

import pytest
import torch
from conftest import encode_padded, min_cosine, torch_dtype_for
from model_registry import EMBEDDING_MODELS
from transformers import AutoModel, AutoTokenizer

PROMPTS = [
    "The capital of France is Paris.",
    "Sentence embeddings are useful.",
]
COS_THRESHOLD = 0.9999

MODELS = {k: v for k, v in EMBEDDING_MODELS.items() if v.get("adapter") is not None}


def _run_prefill(adapter_mod, hf_common_mod, model, input_ids, attention_mask):
    """Dispatch to prefill_encoder or prefill_embed based on adapter type."""
    if getattr(adapter_mod, "_is_encoder_only", False):
        return hf_common_mod.prefill_encoder(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )
    else:
        return hf_common_mod.prefill_embed(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_manual_path(model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    torch_dtype = torch_dtype_for(info)

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    input_ids, attention_mask = encode_padded(tokenizer, PROMPTS)

    # HF reference
    model = AutoModel.from_pretrained(info["path"], dtype=torch_dtype, device_map="cpu")
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        ref_hidden = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
    del model
    gc.collect()

    # Adapter (fresh load — prepare_for_spyre is destructive on the instance)
    model = AutoModel.from_pretrained(info["path"], dtype=torch_dtype, device_map="cpu")
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)
    with torch.no_grad():
        adapter_hidden, _ = _run_prefill(
            adapter_mod, hf_common_mod, model, input_ids, attention_mask
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


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_auto_loader(
    model_key, auto_spyre_model, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    torch_dtype = torch_dtype_for(info)

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    input_ids, attention_mask = encode_padded(tokenizer, PROMPTS)

    # HF reference (loaded fresh, before the auto-loader path).
    ref_model = AutoModel.from_pretrained(
        info["path"], dtype=torch_dtype, device_map="cpu"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    with torch.no_grad():
        ref_hidden = ref_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
    del ref_model
    gc.collect()

    # Auto-loader path
    model = auto_spyre_model.AutoSpyreModel.from_pretrained(
        info["path"], dtype=torch_dtype
    )
    unwrap_compiled_blocks(model)
    adapter_module = auto_spyre_model._resolve_adapter_module(info["path"])
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
