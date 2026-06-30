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

"""Shared helpers for the CPU accuracy tests.

Kept as plain functions (not pytest fixtures) so each test file can import
exactly what it needs. Re-exported through ``conftest.py`` for convenience.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def torch_dtype_for(info):
    """Map a registry entry's ``dtype`` field to a torch dtype.

    Defaults to float16. ``"float32"`` (e.g. Granite 4 1B, where fp16 overflows
    on CPU) and ``"bfloat16"`` (e.g. EmbeddingGemma, which is bf16-native and
    overflows fp16) are recognized explicitly.
    """
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(info.get("dtype"), torch.float16)


def load_hf_causal_lm(info, torch_dtype, adapter_mod=None):
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


def load_hf_vlm(info, torch_dtype, adapter_mod=None):
    """Load the HF multimodal (image→text) reference, honoring ``load_fn``.

    Mirrors :func:`load_hf_causal_lm` for VLM adapters: when ``load_fn`` is set,
    the adapter module is expected to expose ``load_hf_model(path, dtype)`` (a
    non-standard loading path); otherwise the stock
    ``AutoModelForImageTextToText`` auto class is used.
    """
    if info.get("load_fn"):
        if adapter_mod is None:
            raise RuntimeError("load_fn=True requires adapter_mod")
        return adapter_mod.load_hf_model(info["path"], torch_dtype)
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        info["path"], dtype=torch_dtype, device_map="cpu"
    )


def encode_padded(tokenizer, prompts):
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


def min_cosine(a, b, attention_mask=None):
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


def cosine_per_row(a, b):
    """Per-row cosine similarity for ``[B, H]`` tensors. Returns a 1-D tensor."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1)
