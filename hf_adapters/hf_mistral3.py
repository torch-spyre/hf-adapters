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
HuggingFace Transformers adapter for the text backbone of Mistral-Small models.

Extracts the Mistral language model from a Mistral-Small checkpoint
(no trust_remote_code) and prepares it for Spyre. Vision encoder and
projection layers are discarded — this adapter handles text-only inference.

The text backbone is architecturally identical to Mistral 7B, so
compiled block, forward, and prepare logic are reused from hf_mistral.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch

from hf_adapters.hf_common import (
    DEVICE,
    pad_lm_head,
    prepare_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def load_hf_model(model_path, dtype):
    """Load the text decoder as a plain MistralForCausalLM-compatible model."""
    import json
    from collections import defaultdict

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import MistralConfig, MistralForCausalLM

    cfg_path = hf_hub_download(model_path, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)

    mistral_cfg = MistralConfig(**full_cfg["text_config"])
    # The checkpoint ships both lm_head.weight and embed_tokens.weight even though
    # the config says they are tied. Match the checkpoint layout and silence the
    # tie warning by disabling tying on the extracted text-only config.
    mistral_cfg.tie_word_embeddings = False

    model = MistralForCausalLM(mistral_cfg)

    idx_path = hf_hub_download(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    shard_keys = defaultdict(dict)
    for k, shard in idx["weight_map"].items():
        if k.startswith("language_model.model."):
            new_key = k.replace("language_model.model.", "model.", 1)
            shard_keys[shard][new_key] = k
        elif k.startswith("language_model.lm_head."):
            new_key = k.replace("language_model.lm_head.", "lm_head.", 1)
            shard_keys[shard][new_key] = k
        elif k == "lm_head.weight":
            shard_keys[shard][k] = k

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


def load_model(model_path, dtype):
    """Load Mistral Small model and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    torch.nn.Module.to(model, DEVICE)
    print("Model ready.")
    return model


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Mistral Small model in-place."""
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    # Mistral3 is a multimodal wrapper config whose decoder lives in text_config
    # as a plain MistralConfig. The loaded text backbone is also the standard
    # Mistral decoder (`model.model.language_model`) using MistralRotaryEmbedding
    # and MistralRMSNorm. Mirror the decoder fields onto the top-level config so
    # shared standard-GQA helpers that read model.config.{hidden_size,...} work.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "max_position_embeddings",
            "vocab_size",
        ):
            if hasattr(text_cfg, name) and not hasattr(cfg, name):
                setattr(cfg, name, getattr(text_cfg, name))

    prepare_standard_gqa(model, MistralRMSNorm)

    # Mistral Small has a large vocab (131K+). Pad the LM head to a smooth stick
    # boundary to fit within Spyre's 256MB per-core limit.
    pad_lm_head(model)
