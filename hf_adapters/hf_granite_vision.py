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
HuggingFace Transformers adapter for the text backbone of Granite Vision models.

Extracts the Granite language model from a Granite Vision checkpoint
(no trust_remote_code) and prepares it for Spyre.  Vision encoder and
projection layers are discarded — this adapter handles text-only inference.

The text backbone is architecturally identical to Granite 3.3, so
compiled block, forward, and prepare logic are reused from hf_granite.

Usage::

    from hf_adapters.hf_granite_vision import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("ibm-granite/granite-vision-4.1-4b")
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-vision-4.1-4b")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import json
from collections import defaultdict

import torch

from hf_adapters.hf_common import DEVICE, generate as _generate
from hf_adapters.hf_granite import _run_forward, prepare_for_spyre


def _load_text_backbone(model_path, dtype=torch.float16):
    """Load just the Granite text backbone from a Granite Vision checkpoint.

    Remaps ``model.language_model.*`` keys to ``model.*`` and loads into
    a standard GraniteForCausalLM — no trust_remote_code needed.
    Uses strict=False because lm_head.weight is tied to embed_tokens.weight.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import GraniteConfig, GraniteForCausalLM

    cfg_path = hf_hub_download(model_path, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)

    granite_cfg = GraniteConfig(**full_cfg["text_config"])
    model = GraniteForCausalLM(granite_cfg)

    idx_path = hf_hub_download(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    shard_keys = defaultdict(dict)
    for k, shard in idx["weight_map"].items():
        if k.startswith("model.language_model."):
            new_key = k.replace("model.language_model.", "model.", 1)
            shard_keys[shard][new_key] = k

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


def load_hf_model(model_path, dtype=torch.float16):
    """Load the text backbone as a plain GraniteForCausalLM (for test harness)."""
    return _load_text_backbone(model_path, dtype)


def load_model(model_path, dtype=torch.float16):
    """Load Granite Vision text backbone for Spyre."""
    model = _load_text_backbone(model_path, dtype)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    model.to(DEVICE)
    print("Model ready.")
    return model


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Granite Vision text backbone on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
