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

from types import MethodType, ModuleType

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    BertConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Granite4VisionConfig,
    GraniteConfig,
    GraniteMoeHybridConfig,
    LlamaConfig,
    MistralConfig,
    ModernBertConfig,
    MPNetConfig,
    Olmo2Config,
    OlmoConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    SmolLM3Config,
    XLMRobertaConfig,
)
from transformers.configuration_utils import PretrainedConfig

from hf_adapters import (
    hf_bert,
    hf_gemma4,
    hf_granite,
    hf_granite_vision,
    hf_granitemoehybrid,
    hf_llama,
    hf_mistral,
    hf_modernbert,
    hf_mpnet,
    hf_olmo,
    hf_olmo2,
    hf_phi3,
    hf_qwen2,
    hf_qwen3,
    hf_smollm3,
    hf_xlm_roberta,
)
from hf_adapters.hf_common import load_model_common

CONFIG_TO_ADAPTER_MODULE_MAPPING: dict[type[PretrainedConfig], ModuleType] = {
    BertConfig: hf_bert,
    Gemma4Config: hf_gemma4,
    Gemma4TextConfig: hf_gemma4,
    Gemma4UnifiedConfig: hf_gemma4,
    Gemma4UnifiedTextConfig: hf_gemma4,
    Granite4VisionConfig: hf_granite_vision,
    GraniteConfig: hf_granite,
    GraniteMoeHybridConfig: hf_granitemoehybrid,
    LlamaConfig: hf_llama,
    MistralConfig: hf_mistral,
    ModernBertConfig: hf_modernbert,
    MPNetConfig: hf_mpnet,
    OlmoConfig: hf_olmo,
    Olmo2Config: hf_olmo2,
    Phi3Config: hf_phi3,
    Qwen2Config: hf_qwen2,
    Qwen3Config: hf_qwen3,
    SmolLM3Config: hf_smollm3,
    XLMRobertaConfig: hf_xlm_roberta,
}


def _resolve_adapter_module(model_name_or_path):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if type(model_config) not in CONFIG_TO_ADAPTER_MODULE_MAPPING:
        raise Exception(
            f"Model {model_name_or_path} of type {type(model_config)} "
            "is not supported"
        )
    return CONFIG_TO_ADAPTER_MODULE_MAPPING[type(model_config)]


class AutoSpyreModel:
    """Load an HF model via ``transformers.AutoModel`` and prepare it for Spyre.

    ``AutoModel`` is the generic auto-class: it dispatches based on the model
    config and may return any of several model classes (often, but not always,
    the bare backbone). Use a more specific ``AutoSpyreModelFor*`` subclass
    when the task is known.
    """

    _auto_model_cls = AutoModel

    @classmethod
    def from_pretrained(cls, model_name_or_path, dtype=torch.float16):
        module = _resolve_adapter_module(model_name_or_path)

        if hasattr(module, "load_model"):
            model = module.load_model(model_name_or_path, dtype)
        else:
            model = load_model_common(
                model_name_or_path,
                module.prepare_for_spyre,
                dtype,
                auto_model_cls=cls._auto_model_cls,
            )

        return model


class AutoSpyreModelForCausalLM(AutoSpyreModel):
    """Load an HF causal-LM model and prepare it for Spyre.

    Attaches a Spyre-aware ``generate`` method that runs the 64-block padded
    decode loop.
    """

    _auto_model_cls = AutoModelForCausalLM  # type: ignore[assignment]

    @classmethod
    def from_pretrained(cls, model_name_or_path, dtype=torch.float16):
        module = _resolve_adapter_module(model_name_or_path)
        model = super().from_pretrained(model_name_or_path, dtype=dtype)

        def model_generate(self, tokenizer, prompts, **kwargs):
            from hf_adapters.hf_common import generate

            return generate(module._run_forward, self, tokenizer, prompts, **kwargs)

        model.generate = MethodType(model_generate, model)

        return model
