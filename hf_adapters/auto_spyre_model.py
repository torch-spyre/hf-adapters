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
Unified auto-loading interface for HuggingFace Transformers models on Spyre.

Provides a HuggingFace-style API that automatically selects the correct
adapter based on the model's config type.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)

The model is automatically prepared for Spyre (RoPE precomputation, RMSNorm
patching, LM head padding, compiled blocks) and moved to the Spyre device.
A `generate` method is attached to the model that handles the 64-block
padded decode generation loop.
"""

import types

import torch
from transformers import (
    AutoConfig,
    GemmaConfig,
    Granite4VisionConfig,
    GraniteConfig,
    GraniteMoeHybridConfig,
    LlamaConfig,
    MistralConfig,
    Olmo2Config,
    OlmoConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    SmolLM3Config,
)

from hf_adapters import (
    hf_gemma,
    hf_granite,
    hf_granite_vision,
    hf_granitemoehybrid,
    hf_llama,
    hf_mistral,
    hf_olmo,
    hf_olmo2,
    hf_phi3,
    hf_qwen2,
    hf_qwen3,
    hf_smollm3,
)
from hf_adapters.hf_common import load_model_common

CONFIG_TO_ADAPTER_MODULE_MAPPING = {
    GemmaConfig: hf_gemma,
    Granite4VisionConfig: hf_granite_vision,
    GraniteConfig: hf_granite,
    GraniteMoeHybridConfig: hf_granitemoehybrid,
    LlamaConfig: hf_llama,
    MistralConfig: hf_mistral,
    OlmoConfig: hf_olmo,
    Olmo2Config: hf_olmo2,
    Phi3Config: hf_phi3,
    Qwen2Config: hf_qwen2,
    Qwen3Config: hf_qwen3,
    SmolLM3Config: hf_smollm3,
}


class AutoSpyreModelForCausalLM:

    @staticmethod
    def from_pretrained(model_name_or_path, dtype=torch.float16):
        # Determine the appropriate Spyre adapter module for the model
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        if type(model_config) not in CONFIG_TO_ADAPTER_MODULE_MAPPING:
            raise Exception(
                f"Model {model_name_or_path} of type {type(model_config)} "
                "is not supported"
            )

        module = CONFIG_TO_ADAPTER_MODULE_MAPPING[type(model_config)]

        # Check if module has custom load_model function
        if hasattr(module, "load_model"):
            # Custom adapter loading method (e.g., Granite Vision)
            model = module.load_model(model_name_or_path, dtype)
        else:
            model = load_model_common(
                model_name_or_path, module.prepare_for_spyre, dtype
            )

        # Attach generate method using the module's forward function
        def model_generate(self, tokenizer, prompts, **kwargs):
            from hf_adapters.hf_common import generate

            return generate(module._run_forward, self, tokenizer, prompts, **kwargs)

        model.generate = types.MethodType(model_generate, model)

        return model
