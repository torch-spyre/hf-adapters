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

"""HuggingFace Transformers adapter for standalone Ministral causal-LM models.

Supports ``MinistralConfig`` (``model_type: ministral``) checkpoints, e.g.
``mistralai/Ministral-8B-Instruct-2410``.

The decoder is architecturally identical to Mistral 7B (standard GQA); compiled
block, forward, and prepare logic are reused from ``hf_common``.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    get_backbone,
    prepare_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def _patch_embed_tokens_for_spyre(model):
    """Replace embed_tokens with a CPU-resident wrapper to avoid Spyre device mismatch.

    Stores the embedding weight as a plain Python attribute (not ``nn.Parameter``)
    so ``move_to_spyre_with_layout`` does not move it to Spyre.  The gather runs
    on CPU; the float output is moved to the device of the first Q-projection.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = get_backbone(model)
    ref_weight = backbone.embed_tokens.weight.detach()

    class _EmbedCPU(nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "_w", ref_weight)

        def forward(self, input_ids):
            w = object.__getattribute__(self, "_w")
            h = F.embedding(input_ids.cpu(), w)
            target = backbone.layers[0].self_attn.q_proj.weight.device
            return h.to(target)

    backbone.embed_tokens = _EmbedCPU()


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Ministral causal-LM in-place."""
    from transformers.models.ministral.modeling_ministral import MinistralRMSNorm

    _patch_embed_tokens_for_spyre(model)
    prepare_standard_gqa(model, MinistralRMSNorm)
