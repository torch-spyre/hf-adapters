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
Model registries shared by all tests.

Exports ``CAUSAL_LM_MODELS`` and ``EMBEDDING_MODELS``: dictionaries keyed by a
short model key, mapping to a small dict of metadata (``name``, ``path``,
``adapter``, optional ``dtype`` / ``load_fn``). Tests parametrise over these
directly; CI's ``-k`` filter selects which keys actually run.
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
    "tiny_llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
    },
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_phi3.py",
    },
    "phi3": {
        "name": "Phi-3.5 mini",
        "path": "microsoft/Phi-3.5-mini-instruct",
        "adapter": "hf_phi3.py",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
    },
    "ministral": {
        "name": "Ministral 3b instruct",
        "path": "ministral/Ministral-3b-instruct",
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
    # "yi_6b": {
    #     "name": "Yi 1.5 6B",
    #     "path": "01-ai/Yi-1.5-6B",
    #     "adapter": "hf_llama.py",
    # },
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision.py",
        "load_fn": True,
    },
    "gemma4": {
        "name": "Gemma 4 12B",
        "path": "google/gemma-4-12B-it",
        "adapter": "hf_gemma4.py",
    },
    # Gemma 3 is a gated checkpoint — requires HF auth. Tested on Spyre pod only.
    # "gemma3": {
    #     "name": "Gemma 3 1B",
    #     "path": "google/gemma-3-1b-it",
    #     "adapter": "hf_gemma3.py",
    # },
}


EMBEDDING_MODELS = {
    # EmbeddingGemma is a gated checkpoint — requires HF auth. Tested on Spyre pod only.
    # "embeddinggemma": {
    #     "name": "EmbeddingGemma 300M",
    #     "path": "google/embeddinggemma-300m",
    #     "adapter": "hf_gemma3.py",
    #     "dtype": "bfloat16",  # bf16-native; fp16 overflows the residual stream
    # },
    "qwen3_embed": {
        "name": "Qwen3-Embedding 0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "qwen2_embed": {
        "name": "Jina-Qwen2-embed-0.5B",
        "path": "jinaai/jina-code-embeddings-0.5b",
        "adapter": "hf_qwen2.py",
    },
    "e5_mistral": {
        "name": "E5-Mistral-7B",
        "path": "intfloat/e5-mistral-7b-instruct",
        "adapter": "hf_mistral.py",
    },
    "linq_embed_mistral": {
        "name": "Linq-Embed-Mistral",
        "path": "Linq-AI-Research/Linq-Embed-Mistral",
        "adapter": "hf_mistral.py",
    },
    "sfr_embedding_mistral": {
        "name": "SFR-Embedding-Mistral",
        "path": "Salesforce/SFR-Embedding-Mistral",
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
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "path": "sentence-transformers/all-mpnet-base-v2",
        "adapter": "hf_mpnet.py",
    },
    "modernbert": {
        "name": "ModernBERT-embed-base",
        "path": "nomic-ai/modernbert-embed-base",
        "adapter": "hf_modernbert.py",
    },
    "gte_modernbert": {
        "name": "GTE-ModernBERT-base",
        "path": "Alibaba-NLP/gte-modernbert-base",
        "adapter": "hf_modernbert.py",
    },
    "granite_embed": {
        "name": "Granite-Embedding-97m-multilingual-r2",
        "path": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "adapter": "hf_modernbert.py",
    },
}

# Made with Bob
