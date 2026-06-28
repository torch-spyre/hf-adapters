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
        "load_fn": True,
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
    # hf_mistral3.py
    "mistral3": {
        "name": "Mistral-Small-3.2-24B-Instruct-2506",
        "path": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "adapter": "hf_mistral3.py",
        "load_fn": True,
        "size": "24b",
    },
    "ministral3": {
        "name": "Ministral-3-14B-Instruct-2512",
        "path": "mistralai/Ministral-3-14B-Instruct-2512",
        "adapter": "hf_mistral3.py",
        "load_fn": True,
        "dtype": "bfloat16",
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
        "is_gated": True,
        "size": "12b",
    },
}


EMBEDDING_MODELS = {
    # hf_gemma3.py
    "embeddinggemma": {
        "name": "EmbeddingGemma 300M",
        "path": "google/embeddinggemma-300m",
        "adapter": "hf_gemma3.py",
        "dtype": "bfloat16",  # bf16-native; fp16 overflows the residual stream
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


def _get_adapter_module_name(adapter_module):  # type: ignore[no-untyped-def]
    """Extract module name from adapter module object (e.g., hf_qwen3)."""
    return adapter_module.__name__.split(".")[-1]


def _parse_size(size_str):
    """
    Parse size string (e.g., '2b', '0.3B', '1.5b') to float for comparison.

    Args:
        size_str: Size string with 'b' or 'B' suffix (case-insensitive)

    Returns:
        float: Size in billions
    """
    # Remove 'b' or 'B' suffix and convert to float
    return float(size_str.lower().rstrip("b"))


def select_representative_models(config_mapping=None):
    """
    Programmatically select one representative model per adapter module.

    Analyzes CONFIG_TO_ADAPTER_MODULE_MAPPING and selects one model per adapter
    from the registries above. Prefers smaller models for faster test execution.

    Args:
        config_mapping: Optional CONFIG_TO_ADAPTER_MODULE_MAPPING to use.
                       If None, will be imported from hf_adapters.auto_spyre_model.

    Returns:
        tuple: (causal_keys, embed_keys) where each is a list of model keys
    """
    # Import here to avoid issues with conftest.py patching
    if config_mapping is None:
        from hf_adapters.auto_spyre_model import CONFIG_TO_ADAPTER_MODULE_MAPPING

        config_mapping = CONFIG_TO_ADAPTER_MODULE_MAPPING

    # Get set of adapter module names from CONFIG_TO_ADAPTER_MODULE_MAPPING
    adapter_modules_in_config = {
        _get_adapter_module_name(adapter_mod) for adapter_mod in config_mapping.values()
    }

    # Map adapter module names to model keys
    adapter_to_causal_keys = {}
    adapter_to_embed_keys = {}

    # Group causal LM models by adapter
    for key, info in CAUSAL_LM_MODELS.items():
        if info.get("is_gated", False):
            continue
        adapter = info["adapter"].replace(".py", "")
        # Only include if adapter is in CONFIG_TO_ADAPTER_MODULE_MAPPING
        if adapter in adapter_modules_in_config:
            if adapter not in adapter_to_causal_keys:
                adapter_to_causal_keys[adapter] = []
            adapter_to_causal_keys[adapter].append(key)

    # Group embedding models by adapter
    for key, info in EMBEDDING_MODELS.items():
        if info.get("is_gated", False):
            continue
        adapter = info["adapter"].replace(".py", "")
        # Only include if adapter is in CONFIG_TO_ADAPTER_MODULE_MAPPING
        if adapter in adapter_modules_in_config:
            if adapter not in adapter_to_embed_keys:
                adapter_to_embed_keys[adapter] = []
            adapter_to_embed_keys[adapter].append(key)

    # Select one representative per adapter for causal LM
    # Prefer smaller models (by size field) for faster tests
    causal_keys = []
    for adapter in sorted(adapter_modules_in_config):
        if adapter in adapter_to_causal_keys:
            keys = adapter_to_causal_keys[adapter]
            # Sort by size field (smallest first), then by key name for consistency
            sorted_keys = sorted(
                keys,
                key=lambda k: (
                    _parse_size(CAUSAL_LM_MODELS[k]["size"]),  # Sort by size
                    k,  # Then by key name for consistency
                ),
            )
            causal_keys.append(sorted_keys[0])

    # Select one representative per adapter for embeddings
    # Prefer smaller models (by size field) for faster tests
    embed_keys = []
    for adapter in sorted(adapter_modules_in_config):
        if adapter in adapter_to_embed_keys:
            keys = adapter_to_embed_keys[adapter]
            # Sort by size field (smallest first), then by key name for consistency
            sorted_keys = sorted(
                keys,
                key=lambda k: (
                    _parse_size(EMBEDDING_MODELS[k]["size"]),  # Sort by size
                    k,  # Then by key name for consistency
                ),
            )
            embed_keys.append(sorted_keys[0])

    return causal_keys, embed_keys


# Defer initialization until after conftest.py has patched hf_adapters
# These will be populated by conftest.py after it sets up the patched modules
CAUSAL_KEYS = []
EMBED_KEYS = []

# Made with Bob
