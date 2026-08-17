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
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
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
    GraniteSWAConfig,
    LlamaConfig,
    MistralConfig,
    ModernBertConfig,
    MPNetConfig,
    Olmo2Config,
    OlmoConfig,
    Phi3Config,
    PreTrainedModel,
    Qwen2Config,
    Qwen3Config,
    RobertaConfig,
    SmolLM3Config,
    XLMRobertaConfig,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput
from transformers.models.ministral.configuration_ministral import MinistralConfig
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

from hf_adapters import (
    hf_bert,
    hf_dspark_gemma4,
    hf_dspark_granite,
    hf_dspark_qwen3,
    hf_gemma3,
    hf_gemma4,
    hf_gemma4_mm,
    hf_gpt2,
    hf_gpt_neo,
    hf_gpt_neox,
    hf_granite,
    hf_granite_swa,
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
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
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
    GraniteSWAConfig: hf_granite_swa,
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

# Architecture-name mapping — consulted BEFORE the config-class map. DSpark
# speculative-decoding *drafters* reuse their base model's config class
# (``Qwen3Config`` / ``Gemma4TextConfig`` / ``GraniteConfig``) but carry a
# distinct ``architectures`` entry (``*DSparkModel``). Config-class dispatch alone
# would route them to the *target* adapter; keying on the architecture name sends
# them to the drafter adapter instead. Normal targets have no entry here and fall
# through to ``CONFIG_TO_ADAPTER_MODULE_MAPPING`` unchanged.
ARCH_TO_ADAPTER_MODULE_MAPPING: dict[str, ModuleType] = {
    "Qwen3DSparkModel": hf_dspark_qwen3,
    "Gemma4DSparkModel": hf_dspark_gemma4,
    "GraniteDSparkModel": hf_dspark_granite,
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

# Sequence-classification (cross-encoder reranker) mapping — used by
# ``AutoSpyreModelForSequenceClassification``.
SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING: dict[
    type[PretrainedConfig], ModuleType
] = {
    XLMRobertaConfig: hf_xlm_roberta,
    RobertaConfig: hf_xlm_roberta,
}

MODEL_PATH_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "mistralai/Ministral-3-3B-Instruct-2512": torch.bfloat16,
    "mistralai/Ministral-3-8B-Instruct-2512": torch.bfloat16,
    "mistralai/Ministral-3-14B-Instruct-2512": torch.bfloat16,
    "google/embeddinggemma-300m": torch.bfloat16,
    "google/gemma-4-12b": torch.bfloat16,
    "google/gemma-4-12B-it": torch.bfloat16,
    "google/gemma-4-31b": torch.bfloat16,
    "ibm-granite/granite-4.0-1b-base": torch.float32,
    "ibm-granite/granite-4.0-1b": torch.float32,
    "ibm-research/granite-4.1-20b": torch.bfloat16,
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

    # Architecture-name dispatch first: DSpark drafters share their base model's
    # config class but carry a distinct ``*DSparkModel`` architecture, so route on
    # the architecture name before falling through to config-class dispatch.
    for arch in getattr(model_config, "architectures", None) or []:
        if arch in ARCH_TO_ADAPTER_MODULE_MAPPING:
            assert_spyre_dimensions(model_config, model_name=str(model_name_or_path))
            return ARCH_TO_ADAPTER_MODULE_MAPPING[arch]

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
    ) -> PreTrainedModel:
        module: ModuleType = resolve_adapter_module(
            model_name_or_path=model_name_or_path, mapping=cls._module_mapping
        )

        model: PreTrainedModel = load_model_common(
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
    ) -> PreTrainedModel:
        module: ModuleType = resolve_adapter_module(model_name_or_path)
        if getattr(module, "_is_encoder_only", False):
            raise SpyreUnsupportedModelError(
                "Generation is not currently supported for encoder-only architectures"
            )

        model: PreTrainedModel = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_generate(
            self: PreTrainedModel, tokenizer: Any, prompts: list[str], **kwargs: Any
        ):
            from hf_adapters.hf_common import generate

            return generate(module._run_forward, self, tokenizer, prompts, **kwargs)

        model.generate = MethodType(model_generate, model)  # type: ignore[assignment]

        return model


def _validate_encoder_task_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor | None,
    *,
    position_ids: torch.Tensor | None = None,
    head_mask: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
) -> None:
    """Validate the inference-only forward contract shared by encoder tasks."""
    if inputs_embeds is not None:
        raise SpyreUnsupportedFeatureError(
            "inputs_embeds is not currently supported on Spyre"
        )
    if input_ids is None:
        raise ValueError("input_ids must be provided")
    if position_ids is not None:
        raise SpyreUnsupportedFeatureError(
            "Custom position_ids are not currently supported on Spyre"
        )
    if head_mask is not None:
        raise SpyreUnsupportedFeatureError(
            "head_mask is not currently supported on Spyre"
        )
    if labels is not None or model.training:
        raise SpyreUnsupportedFeatureError(
            "Loss computation and training are not currently supported"
        )
    if output_attentions:
        raise SpyreUnsupportedFeatureError(
            "output_attentions is not currently supported on Spyre"
        )
    if output_hidden_states:
        raise SpyreUnsupportedFeatureError(
            "output_hidden_states is not currently supported on Spyre"
        )


class AutoSpyreModelForMaskedLM(AutoSpyreModel):
    """Load an HF masked-LM model with its encoder on Spyre.

    The complete masked-LM task head remains on CPU. The native forward returns
    a ``MaskedLMOutput`` whose logits are on CPU.
    """

    _auto_model_cls = AutoModelForMaskedLM  # type: ignore[assignment]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ) -> PreTrainedModel:
        module: ModuleType = resolve_adapter_module(
            model_name_or_path, mapping=cls._module_mapping
        )
        model: PreTrainedModel = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_forward(
            self: PreTrainedModel,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            head_mask: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs: Any,
        ):
            from hf_adapters.hf_common import prefill_masked_lm

            if encoder_hidden_states is not None or encoder_attention_mask is not None:
                raise SpyreUnsupportedFeatureError(
                    "Cross-attention inputs are not supported"
                )
            if kwargs:
                raise TypeError(f"Unsupported forward arguments: {sorted(kwargs)}")
            _validate_encoder_task_forward(
                self,
                input_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if attention_mask is None and input_ids is not None:
                attention_mask = torch.ones_like(input_ids)

            logits = prefill_masked_lm(
                module._run_backbone_forward,
                self,
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )
            use_return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            if use_return_dict:
                return MaskedLMOutput(logits=logits)  # type: ignore[arg-type]
            return (logits,)

        model.forward = MethodType(model_forward, model)  # type: ignore[assignment]
        return model


class AutoSpyreModelForQuestionAnswering(AutoSpyreModel):
    """Load an extractive-QA model with its encoder on Spyre and head on CPU."""

    _auto_model_cls = AutoModelForQuestionAnswering  # type: ignore[assignment]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ) -> PreTrainedModel:
        module: ModuleType = resolve_adapter_module(model_name_or_path)
        model: PreTrainedModel = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )
        if model.config.num_labels != 2:
            raise SpyreUnsupportedModelError(
                "Extractive question answering requires config.num_labels=2"
            )

        def model_forward(
            self: PreTrainedModel,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            head_mask: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            start_positions: torch.Tensor | None = None,
            end_positions: torch.Tensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs: Any,
        ):
            from hf_adapters.hf_common import prefill_question_answering

            if kwargs:
                raise TypeError(f"Unsupported forward arguments: {sorted(kwargs)}")
            positions = (
                start_positions if start_positions is not None else end_positions
            )
            _validate_encoder_task_forward(
                self,
                input_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if attention_mask is None and input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            start_logits, end_logits = prefill_question_answering(
                module._run_backbone_forward,
                self,
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )
            use_return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            if use_return_dict:
                return QuestionAnsweringModelOutput(
                    start_logits=start_logits, end_logits=end_logits
                )
            return start_logits, end_logits

        model.forward = MethodType(model_forward, model)  # type: ignore[assignment]
        return model


class AutoSpyreModelForSequenceClassification(AutoSpyreModel):
    """Load an XLM-RoBERTa cross-encoder reranker and prepare it for Spyre.

    Loads via ``AutoModelForSequenceClassification``, compiles the encoder
    backbone on Spyre, and attaches a ``rerank`` method that tokenizes
    query-document pairs and returns raw relevance logits.

    Example::

        model = AutoSpyreModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3"
        )
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        pairs = [("query text", "document text")]
        scores = model.rerank(tokenizer, pairs)          # raw logits
        probs  = torch.sigmoid(scores)                   # [0, 1] relevance
    """

    _auto_model_cls = AutoModelForSequenceClassification  # type: ignore[assignment]
    _module_mapping: dict[type[PretrainedConfig], ModuleType] = (
        SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike[str]],
        dtype: torch.dtype = torch.float16,
        tp_plan: Optional[Union[dict, str]] = None,
    ) -> PreTrainedModel:
        module: ModuleType = resolve_adapter_module(
            model_name_or_path, mapping=cls._module_mapping
        )
        model: PreTrainedModel = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_rerank(
            self: PreTrainedModel,
            tokenizer: Any,
            pairs: list[tuple[str, str]],
            **kwargs: Any,
        ):
            from hf_adapters.hf_common import prefill_reranker

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="right",
                return_attention_mask=True,
            )
            return prefill_reranker(
                module._run_backbone_forward,
                self,
                encoded["input_ids"],
                encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids", None),
            )

        model.rerank = MethodType(model_rerank, model)  # type: ignore[assignment]
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
        model: PreTrainedModel = super().from_pretrained(
            model_name_or_path, dtype=dtype, tp_plan=tp_plan
        )

        def model_prefill_logits(
            self: PreTrainedModel,
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
            self: PreTrainedModel,
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
