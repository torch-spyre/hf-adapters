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

"""HuggingFace Transformers adapter for the text backbone of Mistral-3-family models.

Extracts the language model from a Mistral-3-family multimodal checkpoint
(no trust_remote_code) and prepares it for Spyre. Vision encoder and
projection layers are discarded — this adapter handles text-only inference.

Supports two text backbone variants dispatched via ``text_config.model_type``:

- ``"mistral"`` — ``MistralForCausalLM`` (e.g. ``mistralai/Mistral-Small-3.2-24B-Instruct-2506``)
- ``"ministral3"`` — ``Ministral3ForCausalLM`` (e.g. ``mistralai/Ministral-3-14B-Instruct-2512``)

Both backbones are architecturally identical to Mistral 7B (standard GQA), so
compiled block, forward, and prepare logic are reused from ``hf_mistral``.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Ministral-3-14B-Instruct-2512")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Ministral-3-14B-Instruct-2512")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import json
from collections import defaultdict

from hf_adapters.hf_common import (
    _move_to_spyre_with_layout,
    _untie_embedding_and_lm_head,
    get_backbone,
    prepare_standard_gqa,
)


def _load_hf_model_mistral_small(model_path, dtype):
    """Load Mistral-Small text decoder via safetensor key-remap into MistralForCausalLM."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import MistralConfig, MistralForCausalLM

    cfg_path = hf_hub_download(model_path, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)

    mistral_cfg = MistralConfig(**full_cfg["text_config"])
    mistral_cfg.tie_word_embeddings = False
    model = MistralForCausalLM(mistral_cfg)

    idx_path = hf_hub_download(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    shard_keys = defaultdict(dict)
    for key, shard in idx["weight_map"].items():
        if key.startswith("language_model.model."):
            shard_keys[shard][key.replace("language_model.model.", "model.", 1)] = key
        elif key.startswith("language_model.lm_head."):
            shard_keys[shard][
                key.replace("language_model.lm_head.", "lm_head.", 1)
            ] = key
        elif key == "lm_head.weight":
            shard_keys[shard][key] = key

    state = {}
    for shard, key_map in shard_keys.items():
        shard_path = hf_hub_download(model_path, shard)
        data = load_file(shard_path)
        for new_key, old_key in key_map.items():
            state[new_key] = data[old_key]

    model.load_state_dict(state, strict=False)
    model.to(dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _load_hf_model_ministral(model_path, dtype):
    """Load Ministral-3 text decoder via AutoModelForCausalLM with automatic dequantization.

    The Ministral-3-14B checkpoint uses blocked FP8 quantization (weight +
    weight_scale_inv per projection). Loading via safetensor key-remap would
    bring in the raw quantized tensors with wrong shapes. AutoModelForCausalLM
    handles dequantization automatically and returns fp16/bf16 weights.

    The full Mistral3ForConditionalGeneration is loaded (vision tower + text
    decoder) and then the vision tower is deleted to free memory.  get_backbone()
    descends through model.model.language_model to the text backbone, and the
    top-level lm_head is used directly — matching the Gemma4 adapter pattern.
    """
    from transformers.models.mistral3.modeling_mistral3 import (
        Mistral3ForConditionalGeneration,
    )

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    )
    # Drop the vision tower and multi-modal projector — text-only inference.
    if hasattr(model, "model"):
        mm = model.model
        if hasattr(mm, "vision_tower"):
            del mm.vision_tower
        if hasattr(mm, "multi_modal_projector"):
            del mm.multi_modal_projector
    model.eval()
    model.requires_grad_(False)
    return model


def load_hf_model(model_path, dtype):
    """Load a Mistral-3-family text decoder.

    Dispatches to the correct HF text model class based on ``text_config.model_type``:
    - ``"mistral"``  → ``MistralForCausalLM`` (Mistral-Small-3.2)
    - ``"ministral3"`` → ``Ministral3ForCausalLM`` (Ministral-3-14B)
    """
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(model_path, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)

    text_model_type = full_cfg.get("text_config", {}).get("model_type", "mistral")
    if text_model_type == "ministral3":
        return _load_hf_model_ministral(model_path, dtype)
    return _load_hf_model_mistral_small(model_path, dtype)


def load_model(model_path, dtype):
    """Load a Mistral-3-family text decoder and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    # FP8 checkpoints (e.g. Ministral-3-14B) are dequantized to bf16 by
    # transformers regardless of the requested dtype. Use the model's actual
    # dtype for _move_to_spyre_with_layout so every parameter is cast
    # consistently and no bf16/fp16 mismatch arises.
    actual_dtype = next(model.parameters()).dtype
    _untie_embedding_and_lm_head(model)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, actual_dtype)
    print("Model ready.")
    return model


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Mistral-3-family model in-place."""
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    # Decide the correct RMSNorm class in one place by inspecting the first
    # decoder layer's norm — Ministral3 uses Ministral3RMSNorm, Mistral-Small
    # uses MistralRMSNorm.
    first_norm = get_backbone(model).layers[0].input_layernorm
    if isinstance(first_norm, MistralRMSNorm):
        rmsnorm_cls = MistralRMSNorm
    else:
        rmsnorm_cls = Ministral3RMSNorm

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "max_position_embeddings",
            "vocab_size",
        ):
            if hasattr(text_cfg, name) and not hasattr(cfg, name):
                setattr(cfg, name, getattr(text_cfg, name))

    prepare_standard_gqa(model, rmsnorm_cls)
