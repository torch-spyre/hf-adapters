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
HuggingFace Transformers adapter for OLMo models on Spyre.

Covers model_type ``olmo``: OLMo 1B and 7B.
Standard pre-norm architecture like Llama, but uses a weight-free LayerNorm
(no learnable parameters) instead of RMSNorm.

Usage::

    from hf_adapters.hf_olmo import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("allenai/OLMo-1B-hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    load_model_common,
    make_standard_gqa_block,
    pad_attention_heads,
    pad_lm_head,
    standard_gqa_forward,
)
from hf_adapters.hf_common import (
    generate as _generate,
)

_run_forward = standard_gqa_forward


def _patch_olmo_layernorm(layernorm_cls):
    """Patch OlmoLayerNorm to stay in fp16 on Spyre (no dtype conversion).

    OlmoLayerNorm has no learnable weight — it's just functional layer_norm
    with eps=1e-5. The stock HF code casts to float32 which Spyre can't do.
    """

    def _forward_fp16(self, hidden_states):
        eps = getattr(self, "eps", 1e-5)
        if hidden_states.device.type == "spyre":
            return F.layer_norm(
                hidden_states,
                self.normalized_shape,
                None,
                None,
                eps=eps,
            )
        else:
            orig_dtype = hidden_states.dtype
            return F.layer_norm(
                hidden_states.float(),
                self.normalized_shape,
                None,
                None,
                eps=eps,
            ).to(orig_dtype)

    layernorm_cls.forward = _forward_fp16


def prepare_for_spyre(model):
    """Apply Spyre adaptations to OLMo model in-place."""
    from transformers.models.olmo.modeling_olmo import OlmoLayerNorm

    cfg = model.config
    orig_head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )

    padded_head_dim = None
    stick_aligned_head_dim = (
        (orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)
    ) * (2 * BLOCK_SIZE)
    if stick_aligned_head_dim > orig_head_dim:
        padded_head_dim = stick_aligned_head_dim
        pad_attention_heads(
            model,
            model.model.layers,
            orig_head_dim,
            padded_head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )

    model._spyre_rope = PrecomputedRotaryEmbedding(
        model.model.rotary_emb,
        padded_head_dim=padded_head_dim,
    )
    _patch_olmo_layernorm(OlmoLayerNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        make_standard_gqa_block(layer) for layer in model.model.layers
    ]


def load_model(model_path, dtype=torch.float16):
    """Load OLMo model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with OLMo on Spyre."""
    return _generate(standard_gqa_forward, model, tokenizer, prompts, **kwargs)
