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

For each registered embedder backbone, two parametrized test cases run:

  test_manual_path[<key>]
    Loads the backbone via stock ``AutoModel`` on CPU, captures the
    reference ``last_hidden_state``, then loads a fresh copy and applies
    the adapter directly (``prepare_for_spyre`` + ``prefill_embed``).
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
import torch.nn.functional as F
from conftest import EMBEDDING_MODELS
from transformers import AutoModel, AutoTokenizer

PROMPTS = [
    "The capital of France is Paris.",
    "Sentence embeddings are useful.",
]
COS_THRESHOLD = 0.9999

MODELS = {k: v for k, v in EMBEDDING_MODELS.items() if v["adapter"] is not None}


def _per_token_cosine(a, b, attention_mask):
    """Mean and min cosine similarity over real (unmasked) tokens.

    Args:
        a, b: ``[B, L, H]`` hidden states.
        attention_mask: ``[B, L]`` mask; ``1`` for real tokens.
    """
    a32 = a.float()
    b32 = b.float()
    cos = F.cosine_similarity(a32, b32, dim=-1)
    mask = attention_mask.bool()
    return cos[mask].mean().item(), cos[mask].min().item()


def _encode(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        PROMPTS, return_tensors="pt", padding=True, padding_side="right"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _torch_dtype(info):
    return torch.float32 if info.get("dtype") == "float32" else torch.float16


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_manual_path(model_key, load_adapter, unwrap_compiled_blocks, hf_common_mod):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    torch_dtype = _torch_dtype(info)

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    input_ids, attention_mask = _encode(tokenizer)

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
        adapter_hidden, _ = hf_common_mod.prefill_embed(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )
    del model
    gc.collect()

    assert (
        adapter_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: adapter {adapter_hidden.shape} vs ref {ref_hidden.shape}"
    _, min_cos = _per_token_cosine(adapter_hidden, ref_hidden, attention_mask)
    assert (
        min_cos >= COS_THRESHOLD
    ), f"min per-token cosine {min_cos:.6f} < threshold {COS_THRESHOLD}"


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_auto_loader(
    model_key, auto_spyre_model, unwrap_compiled_blocks, hf_common_mod
):
    info = MODELS[model_key]
    torch_dtype = _torch_dtype(info)

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    input_ids, attention_mask = _encode(tokenizer)

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
        adapter_hidden, _ = hf_common_mod.prefill_embed(
            adapter_module._run_backbone_forward, model, input_ids, attention_mask
        )
    del model
    gc.collect()

    assert (
        adapter_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: adapter {adapter_hidden.shape} vs ref {ref_hidden.shape}"
    _, min_cos = _per_token_cosine(adapter_hidden, ref_hidden, attention_mask)
    assert (
        min_cos >= COS_THRESHOLD
    ), f"min per-token cosine {min_cos:.6f} < threshold {COS_THRESHOLD}"
