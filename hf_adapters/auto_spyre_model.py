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

from __future__ import annotations

import os
from types import MethodType, ModuleType
from typing import Any, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BertConfig,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    GPT2Config,
    GPTNeoConfig,
    GPTNeoXConfig,
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
    RobertaConfig,
    SmolLM3Config,
    XLMRobertaConfig,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.ministral.configuration_ministral import MinistralConfig
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

from hf_adapters import (
    hf_bert,
    hf_gemma3,
    hf_gemma4,
    hf_gemma4_mm,
    hf_gpt2,
    hf_gpt_neo,
    hf_gpt_neox,
    hf_granite,
    hf_granite_vision,
    hf_granite_vision_mm,
    hf_granitemoehybrid,
    hf_llama,
    hf_ministral,
    hf_mistral,
    hf_mistral3,
    hf_mistral3_vision_mm,
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
from hf_adapters.hf_common import (
    SpyreNoAdapterError,
    assert_spyre_dimensions,
    load_model_common,
    move_model_to_spyre,
)

CONFIG_TO_ADAPTER_MODULE_MAPPING: dict[type[PretrainedConfig], ModuleType] = {
    BertConfig: hf_bert,
    Gemma3Config: hf_gemma3,
    Gemma3TextConfig: hf_gemma3,
    Gemma4Config: hf_gemma4,
    Gemma4TextConfig: hf_gemma4,
    Gemma4UnifiedConfig: hf_gemma4,
    Gemma4UnifiedTextConfig: hf_gemma4,
    GPT2Config: hf_gpt2,
    GPTNeoConfig: hf_gpt_neo,
    GPTNeoXConfig: hf_gpt_neox,
    Granite4VisionConfig: hf_granite_vision,
    GraniteConfig: hf_granite,
    GraniteMoeHybridConfig: hf_granitemoehybrid,
    LlamaConfig: hf_llama,
    MistralConfig: hf_mistral,
    MinistralConfig: hf_ministral,
    Mistral3Config: hf_mistral3,
    ModernBertConfig: hf_modernbert,
    MPNetConfig: hf_mpnet,
    OlmoConfig: hf_olmo,
    Olmo2Config: hf_olmo2,
    Phi3Config: hf_phi3,
    Qwen2Config: hf_qwen2,
    Qwen3Config: hf_qwen3,
    RobertaConfig: hf_xlm_roberta,
    SmolLM3Config: hf_smollm3,
    XLMRobertaConfig: hf_xlm_roberta,
}

# Multimodal (image-text-to-text) mapping — used by
# ``AutoSpyreModelForImageTextToText``. A multimodal checkpoint's config (e.g.
# Granite4VisionConfig) appears here mapped to the *combined* two-tower adapter,
# and in CONFIG_TO_ADAPTER_MODULE_MAPPING mapped to the *text-only* adapter
# (used by AutoSpyreModelForCausalLM). The auto class selects which.
IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING: dict[
    type[PretrainedConfig], ModuleType
] = {
    Gemma4UnifiedConfig: hf_gemma4_mm,
    Granite4VisionConfig: hf_granite_vision_mm,
    Mistral3Config: hf_mistral3_vision_mm,
}

MODEL_PATH_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "mistralai/Ministral-3-3B-Instruct-2512": torch.bfloat16,
    "mistralai/Ministral-3-8B-Instruct-2512": torch.bfloat16,
    "mistralai/Ministral-3-14B-Instruct-2512": torch.bfloat16,
    "google/embeddinggemma-300m": torch.bfloat16,
    "google/gemma-4-12b": torch.bfloat16,
    "google/gemma-4-12B-it": torch.bfloat16,
    "ibm-granite/granite-4.0-1b-base": torch.float32,
    "ibm-granite/granite-4.0-1b": torch.float32,
}


def resolve_adapter_module(
    model_name_or_path: Union[str, os.PathLike[str]],
    mapping: dict[
        type[PretrainedConfig], ModuleType
    ] = CONFIG_TO_ADAPTER_MODULE_MAPPING,
    trust_remote_code: bool | None = None,
) -> ModuleType:
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    if type(model_config) not in mapping:
        raise SpyreNoAdapterError(
            f"Model {model_name_or_path} of type {type(model_config)} "
            "is not supported"
        )
    assert_spyre_dimensions(model_config, model_name=str(model_name_or_path))
    return mapping[type(model_config)]


class AutoSpyreModel:
    """Load an HF model via ``transformers.AutoModel`` and prepare it for Spyre.

    ``AutoModel`` is the generic auto-class: it dispatches based on the model
    config and may return any of several model classes (often, but not always,
    the bare backbone). Use a more specific ``AutoSpyreModelFor*`` subclass
    when the task is known.
    """

    _auto_model_cls = AutoModel
    _module_mapping: dict[type[PretrainedConfig], ModuleType] = (
        CONFIG_TO_ADAPTER_MODULE_MAPPING
    )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ) -> torch.nn.Module:
        module: ModuleType = resolve_adapter_module(
            model_name_or_path=model_name_or_path, mapping=cls._module_mapping
        )

        model: torch.nn.Module = load_model_common(
            model_name_or_path,
            module,
            dtype,
            auto_model_cls=cls._auto_model_cls,
            tp_plan=tp_plan,
        )
        move_model_to_spyre(model, module, dtype)
        return model


class AutoSpyreModelForCausalLM(AutoSpyreModel):
    """Load an HF causal-LM model and prepare it for Spyre.

    Attaches a Spyre-aware ``generate`` method that runs the 64-block padded
    decode loop.
    """

    _auto_model_cls = AutoModelForCausalLM  # type: ignore[assignment]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ) -> torch.nn.Module:
        module: ModuleType = resolve_adapter_module(model_name_or_path)
        model: torch.nn.Module = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_generate(
            self: torch.nn.Module, tokenizer: Any, prompts: list[str], **kwargs: Any
        ):
            from hf_adapters.hf_common import generate

            return generate(module._run_forward, self, tokenizer, prompts, **kwargs)

        model.generate = MethodType(model_generate, model)  # type: ignore[assignment]

        return model


class AutoSpyreModelForImageTextToText(AutoSpyreModel):
    """Load a multimodal (image-text-to-text) model and prepare BOTH towers.

    Selects the combined two-tower adapter (vision tower + text decoder),
    loads the full VLM via ``AutoModelForImageTextToText``, and prepares both
    for Spyre. Attaches Spyre-aware ``prefill_logits`` (image + text → logits)
    and ``generate`` (full image→text decode) methods.
    """

    _auto_model_cls = AutoModelForImageTextToText  # type: ignore[assignment]
    _module_mapping: dict[type[PretrainedConfig], ModuleType] = (
        IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ):
        module: ModuleType = resolve_adapter_module(
            model_name_or_path,
            mapping=cls._module_mapping,
        )
        model: torch.nn.Module = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_prefill_logits(
            self: torch.nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            **kwargs: Any,
        ):
            # Extra multimodal inputs vary by model: Granite Vision needs
            # ``image_sizes`` (anyres tiling); Gemma 4 unified needs
            # ``image_position_ids`` + ``mm_token_type_ids``. Forward whatever
            # the processor produced as keyword args so each adapter takes its own.
            return module.prefill_logits(
                self, input_ids, attention_mask, pixel_values, **kwargs
            )

        def model_generate(
            self: torch.nn.Module,
            processor: Any,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            **kwargs: Any,
        ):
            return module.generate(
                self,
                processor,
                input_ids,
                attention_mask,
                pixel_values,
                **kwargs,
            )

        model.prefill_logits = MethodType(model_prefill_logits, model)  # type: ignore[assignment]
        model.generate = MethodType(model_generate, model)  # type: ignore[assignment]
        return model


def torch_dtype_for_model_path(model_path: str) -> torch.dtype:
    """Resolve the Spyre-safe torch dtype for *model_path*.

    Looks up *model_path* in ``MODEL_PATH_TO_TORCH_DTYPE``; defaults to
    ``torch.float16`` when no entry is found. Registry entries of
    ``torch.float32`` (e.g. Granite 4 1B, where fp16 overflows on CPU) are
    downcast to ``torch.float16`` because Spyre does not support float32;
    ``torch.bfloat16`` entries (e.g. EmbeddingGemma) are passed through
    unchanged.
    """
    dtype = MODEL_PATH_TO_TORCH_DTYPE.get(model_path, torch.float16)
    if dtype == torch.float32:
        return torch.float16
    return dtype
