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
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "path": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_granitemoehybrid.py",
        "dtype": "float32",  # fp16 overflows on CPU due to multipliers
    },
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_smollm3.py",
    },
    "llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
    },
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_phi3.py",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
    },
    "mistral": {
        "name": "Mistral 7B v0.3",
        "path": "mistralai/Mistral-7B-v0.3",
        "adapter": "hf_mistral.py",
    },
    "olmo": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_olmo.py",
    },
    "olmo2": {
        "name": "OLMo2 1B",
        "path": "allenai/OLMo-2-0425-1B",
        "adapter": "hf_olmo2.py",
    },
    "falcon3": {
        "name": "Falcon 3 1B",
        "path": "tiiuae/Falcon3-1B-Base",
        "adapter": "hf_llama.py",
    },
    "deepseek-coder": {
        "name": "DeepSeek-Coder 1.3B",
        "path": "deepseek-ai/deepseek-coder-1.3b-base",
        "adapter": "hf_llama.py",
    },
    # Ministral 3B is gated — requires HF auth. Tested on Spyre pod only.
    # "ministral": {
    #     "name": "Ministral 3B",
    #     "path": "mistralai/Ministral-3B-Instruct",
    #     "adapter": "hf_mistral.py",
    # },
    "yi": {
        "name": "Yi 1.5 6B",
        "path": "01-ai/Yi-1.5-6B",
        "adapter": "hf_llama.py",
    },
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision.py",
        "load_fn": True,
    },
}


EMBEDDING_MODELS = {
    "qwen3_embed": {
        "name": "Qwen3-Embedding 0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "qwen2_embed": {
        "name": "GTE-Qwen2-1.5B",
        "path": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "adapter": "hf_qwen2.py",
    },
    "e5_mistral": {
        "name": "E5-Mistral-7B",
        "path": "intfloat/e5-mistral-7b-instruct",
        "adapter": "hf_mistral.py",
    },
    "bge_base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
        "adapter": "hf_bert.py",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "hf_bert.py",
    },
    "bge_m3": {
        "name": "BGE-M3",
        "path": "BAAI/bge-m3",
        "adapter": "hf_xlm_roberta.py",
    },
}


def _get_adapter_module_name(adapter_module):  # type: ignore[no-untyped-def]
    """Extract module name from adapter module object (e.g., hf_qwen3)."""
    return adapter_module.__name__.split(".")[-1]


def _select_representative_models(config_mapping=None):
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
        adapter = info["adapter"].replace(".py", "")
        # Only include if adapter is in CONFIG_TO_ADAPTER_MODULE_MAPPING
        if adapter in adapter_modules_in_config:
            if adapter not in adapter_to_causal_keys:
                adapter_to_causal_keys[adapter] = []
            adapter_to_causal_keys[adapter].append(key)

    # Group embedding models by adapter
    for key, info in EMBEDDING_MODELS.items():
        adapter = info["adapter"].replace(".py", "")
        # Only include if adapter is in CONFIG_TO_ADAPTER_MODULE_MAPPING
        if adapter in adapter_modules_in_config:
            if adapter not in adapter_to_embed_keys:
                adapter_to_embed_keys[adapter] = []
            adapter_to_embed_keys[adapter].append(key)

    # Select one representative per adapter for causal LM
    # Prefer smaller models (by key name heuristics) for faster tests
    causal_keys = []
    for adapter in sorted(adapter_modules_in_config):
        if adapter in adapter_to_causal_keys:
            keys = adapter_to_causal_keys[adapter]
            # Sort to get consistent selection; prefer keys with "2b", "1b", or shorter names
            sorted_keys = sorted(
                keys,
                key=lambda k: (
                    "2b" not in k and "1b" not in k,  # Prefer smaller models
                    len(CAUSAL_LM_MODELS[k]["path"]),  # Then by path length
                    k,  # Finally by key name for consistency
                ),
            )
            causal_keys.append(sorted_keys[0])

    # Select one representative per adapter for embeddings
    embed_keys = []
    for adapter in sorted(adapter_modules_in_config):
        if adapter in adapter_to_embed_keys:
            keys = adapter_to_embed_keys[adapter]
            # Sort for consistent selection
            sorted_keys = sorted(
                keys, key=lambda k: (len(EMBEDDING_MODELS[k]["path"]), k)
            )
            embed_keys.append(sorted_keys[0])

    return causal_keys, embed_keys


# Defer initialization until after conftest.py has patched hf_adapters
# These will be populated by conftest.py after it sets up the patched modules
CAUSAL_KEYS = []
EMBED_KEYS = []

# Made with Bob
