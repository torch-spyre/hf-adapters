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
Model registries and programmatic selection for tests.

This module contains:
1. CAUSAL_LM_MODELS and EMBEDDING_MODELS registries (moved from conftest.py)
2. Programmatic selection of representative models based on CONFIG_TO_ADAPTER_MODULE_MAPPING

When new adapters are added to CONFIG_TO_ADAPTER_MODULE_MAPPING, tests will
automatically cover them by selecting one representative model per adapter.
"""

from __future__ import annotations

import os
import types

import pytest


def _include_gated() -> bool:
    """Whether gated models should be included in the parametrized test lists.

    Gated models (Llama, google/gemma-*, some Mistral repos) require HF auth.
    They are excluded by default so runs in environments without auth don't fail
    on collection. Set ``SPYRE_INCLUDE_GATED=1`` (e.g. on the Spyre pod, where the
    HF token is configured) to opt them in.
    """
    return os.getenv("SPYRE_INCLUDE_GATED", "0") == "1"


# Model registries - shared by all tests
CAUSAL_LM_MODELS = {
    # hf_gpt2.py
    "gpt2": {
        "name": "GPT-2 124M",
        "path": "gpt2",
        "adapter": "hf_gpt2.py",
        "size": "0.1b",
    },
    # hf_gpt_neo.py
    "gpt_neo": {
        "name": "GPT-Neo 125M",
        "path": "EleutherAI/gpt-neo-125m",
        "adapter": "hf_gpt_neo.py",
        "size": "0.1b",
    },
    # hf_gpt_neox.py
    "pythia_410m": {
        "name": "Pythia 410M",
        "path": "EleutherAI/pythia-410m",
        "adapter": "hf_gpt_neox.py",
        "size": "0.4b",
    },
    # hf_granite.py
    "granite8b": {
        "name": "Granite 3.3 8B",
        "path": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "hf_granite.py",
        "size": "8b",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
        "size": "2b",
    },
    # hf_granitemoehybrid.py
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_granitemoehybrid.py",
        "size": "1b",
    },
    # hf_granite_vision.py
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision.py",
        "size": "4b",
    },
    # hf_smollm3.py
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_smollm3.py",
        "size": "3b",
    },
    # hf_llama.py
    "tiny_llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
        "size": "1b",
    },
    "falcon3": {
        "name": "Falcon 3 1B",
        "path": "tiiuae/Falcon3-1B-Base",
        "adapter": "hf_llama.py",
        "size": "1b",
    },
    "deepseek-coder": {
        "name": "DeepSeek-Coder 1.3B",
        "path": "deepseek-ai/deepseek-coder-1.3b-base",
        "adapter": "hf_llama.py",
        "size": "1.3b",
    },
    "yi_6b": {
        "name": "Yi 1.5 6B",
        "path": "01-ai/Yi-1.5-6B",
        "adapter": "hf_llama.py",
        "size": "6b",
    },
    # hf_phi3.py
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_phi3.py",
        "size": "3.8b",
    },
    "phi3": {
        "name": "Phi-3.5 mini",
        "path": "microsoft/Phi-3.5-mini-instruct",
        "adapter": "hf_phi3.py",
        "size": "3.8b",
    },
    # hf_qwen2.py
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
        "size": "1.5b",
    },
    # hf_qwen3.py
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
        "size": "0.6b",
    },
    # hf_mistral.py
    "ministral": {
        "name": "Ministral 3B",
        "path": "ministral/Ministral-3B-Instruct",
        "adapter": "hf_mistral.py",
        "size": "3b",
    },
    # hf_ministral.py
    "ministral8b": {
        "name": "Ministral-8B-Instruct-2410",
        "path": "mistralai/Ministral-8B-Instruct-2410",
        "adapter": "hf_ministral.py",
        "size": "8b",
    },
    # hf_mistral3.py
    "mistral3": {
        "name": "Mistral-Small-3.2-24B-Instruct-2506",
        "path": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "adapter": "hf_mistral3.py",
        "size": "24b",
    },
    "ministral3": {
        "name": "Ministral-3-14B-Instruct-2512",
        "path": "mistralai/Ministral-3-14B-Instruct-2512",
        "adapter": "hf_mistral3.py",
        "size": "14b",
    },
    # hf_olmo.py
    "olmo1b": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_olmo.py",
        "size": "1b",
    },
    # hf_olmo2.py
    "olmo2_1b": {
        "name": "OLMo2 1B",
        "path": "allenai/OLMo-2-0425-1B",
        "adapter": "hf_olmo2.py",
        "size": "1b",
    },
    # hf_gemma3.py
    "gemma3_unsloth": {
        "name": "Gemma 3 1B",
        "path": "unsloth/gemma-3-1b-it",
        "adapter": "hf_gemma3.py",
        "size": "1b",
    },
    "gemma3_google": {
        "name": "Gemma 3 1B",
        "path": "google/gemma-3-1b-it",
        "adapter": "hf_gemma3.py",
        "is_gated": True,
        "size": "1b",
    },
    # hf_gemma4
    "gemma4_google": {
        "name": "Gemma 4 12B",
        "path": "google/gemma-4-12B-it",
        "adapter": "hf_gemma4.py",
        "size": "12b",
    },
}

EMBEDDING_MODELS = {
    # hf_gemma3.py
    "embeddinggemma": {
        "name": "EmbeddingGemma 300M",
        "path": "google/embeddinggemma-300m",
        "adapter": "hf_gemma3.py",
        "is_gated": True,
        "size": "0.3b",
    },
    # hf_qwen3.py
    "qwen3_embed": {
        "name": "Qwen3-Embedding 0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "adapter": "hf_qwen3.py",
        "size": "0.6b",
    },
    # hf_qwen2.py
    "qwen2_embed": {
        "name": "Jina-Qwen2-embed-0.5B",
        "path": "jinaai/jina-code-embeddings-0.5b",
        "adapter": "hf_qwen2.py",
        "size": "0.5b",
    },
    # hf_mistral.py
    "e5_mistral": {
        "name": "E5-Mistral-7B",
        "path": "intfloat/e5-mistral-7b-instruct",
        "adapter": "hf_mistral.py",
        "size": "7b",
    },
    "linq_embed_mistral": {
        "name": "Linq-Embed-Mistral",
        "path": "Linq-AI-Research/Linq-Embed-Mistral",
        "adapter": "hf_mistral.py",
        "size": "7b",
    },
    "sfr_embedding_mistral": {
        "name": "SFR-Embedding-Mistral",
        "path": "Salesforce/SFR-Embedding-Mistral",
        "adapter": "hf_mistral.py",
        "size": "7b",
    },
    # hf_bert.py
    "bge_base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
        "adapter": "hf_bert.py",
        "size": "0.1b",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "hf_bert.py",
        "size": "0.02b",
    },
    # hf_modernbert.py
    "modernbert": {
        "name": "ModernBERT-embed-base",
        "path": "nomic-ai/modernbert-embed-base",
        "adapter": "hf_modernbert.py",
        "size": "0.1b",
    },
    "gte_modernbert": {
        "name": "GTE-ModernBERT-base",
        "path": "Alibaba-NLP/gte-modernbert-base",
        "adapter": "hf_modernbert.py",
        "size": "0.1b",
    },
    "granite_embed": {
        "name": "Granite-Embedding-97m-multilingual-r2",
        "path": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "adapter": "hf_modernbert.py",
        "size": "0.1b",
    },
    # hf_xlm_roberta.py
    "bge_m3": {
        "name": "BGE-M3",
        "path": "BAAI/bge-m3",
        "adapter": "hf_xlm_roberta.py",
        "size": "0.5b",
    },
    "granite_125m": {
        "name": "Granite-Embedding-125m-English",
        "path": "ibm-granite/granite-embedding-125m-english",
        "adapter": "hf_xlm_roberta.py",
        "size": "0.1b",
    },
    "granite_278m": {
        "name": "Granite-Embedding-278m-Multilingual",
        "path": "ibm-granite/granite-embedding-278m-multilingual",
        "adapter": "hf_xlm_roberta.py",
        "size": "0.2b",
    },
    "granite_30m": {
        "name": "Granite-Embedding-30m-English",
        "path": "ibm-granite/granite-embedding-30m-english",
        "adapter": "hf_xlm_roberta.py",
        "size": "0.03B",
    },
    "granite_107m": {
        "name": "Granite-Embedding-107m-Multilingual",
        "path": "ibm-granite/granite-embedding-107m-multilingual",
        "adapter": "hf_xlm_roberta.py",
        "size": "0.1B",
    },
    # hf_mpnet.py
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "path": "sentence-transformers/all-mpnet-base-v2",
        "adapter": "hf_mpnet.py",
        "size": "0.1b",
    },
}


# Vision models. ``kind="tower"`` adapters are encoder-only; ``kind="vlm"`` adapters
# are full multimodal models with a causal text decoder, RoPE, KV caches, and ``generate``.
VISION_MODELS = {
    # hf_siglip_vision.py — SigLIP vision tower of Granite Vision 4.1
    "granite_vision_siglip": {
        "name": "Granite Vision 4.1 4B (SigLIP tower)",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_siglip_vision.py",
        "kind": "tower",  # bare vision tower: pixel_values -> patch hidden states
    },
    # hf_granite_vision_mm.py — combined two-tower (vision + text) forward
    "granite_vision_mm": {
        "name": "Granite Vision 4.1 4B (both towers)",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision_mm.py",
        "kind": "vlm",  # multimodal: image + text -> generated text
        "size": "4b",
    },
    # hf_pixtral_vision.py — Pixtral vision tower of Mistral3 Vision models
    "mistral3_vision_pixtral": {
        "name": "Mistral-Small-3.1-24B-Instruct-2503 (Pixtral tower)",
        "path": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "adapter": "hf_pixtral_vision.py",
        "kind": "tower",  # bare vision tower: pixel_values -> patch hidden states
    },
    # hf_mistral3_vision_mm.py — combined two-tower (Pixtral + Mistral text) forward
    "mistral3_vision_mm": {
        "name": "Mistral-Small-3.1-24B-Instruct-2503 (both towers)",
        "path": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "adapter": "hf_mistral3_vision_mm.py",
        "kind": "vlm",  # multimodal: image + text -> generated text
        "size": "24b",
    },
    # hf_mistral3_vision_mm.py — Ministral-3 14B (ministral3 text backbone variant)
    "ministral3_vision_mm": {
        "name": "Ministral-3-14B-Instruct-2512 (both towers)",
        "path": "mistralai/Ministral-3-14B-Instruct-2512",
        "adapter": "hf_mistral3_vision_mm.py",
        "kind": "vlm",  # multimodal: image + text -> generated text
        "dtype": "bfloat16",  # blocked-FP8 checkpoint, dequantized to bf16
        "size": "14b",
    },
    # hf_mistral3_vision_mm.py — Ministral-3 3B (smallest ministral3 vision variant)
    "ministral3_3b_vision_mm": {
        "name": "Ministral-3-3B-Instruct-2512 (both towers)",
        "path": "mistralai/Ministral-3-3B-Instruct-2512",
        "adapter": "hf_mistral3_vision_mm.py",
        "kind": "vlm",  # multimodal: image + text -> generated text
        "dtype": "bfloat16",  # blocked-FP8 checkpoint, dequantized to bf16
        "size": "3b",
    },
    # hf_gemma4_mm.py — unified encoder-free VLM (image + text -> text)
    "gemma4_mm": {
        "name": "Gemma 4 12B (unified VLM)",
        "path": "google/gemma-4-12B-it",
        "adapter": "hf_gemma4_mm.py",
        "kind": "vlm",  # multimodal: image + text -> generated text
        "size": "12b",
    },
}


def _get_adapter_module_name(adapter_module: types.ModuleType) -> str:
    """Extract module name from adapter module object (e.g., hf_qwen3)."""
    return adapter_module.__name__.split(".")[-1]


def _parse_size(size_str: str) -> float:
    """
    Parse size string (e.g., '2b', '0.3B', '1.5b') to float for comparison.

    Args:
        size_str: Size string with 'b' or 'B' suffix (case-insensitive)

    Returns:
        float: Size in billions
    """
    # Remove 'b' or 'B' suffix and convert to float
    return float(size_str.lower().rstrip("b"))


def _select_representative_paths(
    models: dict[str, dict],
    *,
    include_gated: bool,
    predicate=None,
) -> list[str]:
    """Select one representative model path per adapter module.

    Groups ``models`` by adapter and picks the smallest (by ``size``) model in
    each group, breaking ties by key name for determinism. Gated models are
    skipped unless ``include_gated``. An optional ``predicate(info) -> bool``
    filters which entries are eligible (e.g. ``kind == "vlm"`` for vision).
    """
    adapter_to_keys: dict[str, list[str]] = {}
    for key, info in models.items():
        if info.get("is_gated", False) and not include_gated:
            continue
        if predicate is not None and not predicate(info):
            continue
        adapter = info["adapter"].replace(".py", "")
        adapter_to_keys.setdefault(adapter, []).append(key)

    paths: list[str] = []
    for keys in adapter_to_keys.values():
        # Prefer smaller models (by size field) for faster tests; tie-break on
        # key name for consistency across runs.
        sorted_keys = sorted(
            keys,
            key=lambda k: (_parse_size(models[k]["size"]), k),
        )
        paths.append(models[sorted_keys[0]]["path"])
    return paths


# One representative model per adapter module (smallest by size), so tests
# automatically cover new adapters. A single ``_include_gated()`` snapshot is
# shared across all three selections. ``kind == "vlm"`` excludes bare vision towers.
_include_gated_flag = _include_gated()

CAUSAL_PATHS: list[str] = _select_representative_paths(
    CAUSAL_LM_MODELS, include_gated=_include_gated_flag
)
EMBED_PATHS: list[str] = _select_representative_paths(
    EMBEDDING_MODELS, include_gated=_include_gated_flag
)
VISION_PATHS: list[str] = _select_representative_paths(
    VISION_MODELS,
    include_gated=_include_gated_flag,
    predicate=lambda info: info.get("kind") == "vlm",
)

# Causal-LM models that just went green on torchs-spyre but aren't yet proven stable
# across repeated runs. Kept as a non-blocking signal (xfail, non-strict) for
# a trial period; remove an entry once it's been stably green so its Spyre
# tests go back to gating CI normally.

NON_BLOCKING_CAUSAL_MODELS: dict[str, str] = {
    CAUSAL_LM_MODELS[key]["path"]: (
        f"{key}: newly green on Spyre, non-blocking signal for a trial "
        "period before promoting to a blocking test"
    )
    for key in ("qwen3", "olmo2_1b", "gemma3_unsloth", "ministral8b", "gemma4_google")
}


def xfail_non_blocking(paths: list[str]) -> list[object]:
    """Wrap entries of ``paths`` found in NON_BLOCKING_CAUSAL_MODELS with xfail.

    The test still runs and its outcome (PASS/FAIL) is visible in the report,
    but a failure won't fail the pytest run or block CI.
    """
    return [
        (
            pytest.param(
                path,
                marks=pytest.mark.xfail(
                    reason=NON_BLOCKING_CAUSAL_MODELS[path], strict=False
                ),
                id=path,
            )
            if path in NON_BLOCKING_CAUSAL_MODELS
            else path
        )
        for path in paths
    ]
