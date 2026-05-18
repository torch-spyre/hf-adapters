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

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch.nn.functional as F

from hf_adapters.hf_common import (
    make_standard_gqa_block,
    pad_lm_head,
    prepare_rope_and_heads,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


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

    prepare_rope_and_heads(model)
    _patch_olmo_layernorm(OlmoLayerNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        make_standard_gqa_block(layer) for layer in model.model.layers
    ]
