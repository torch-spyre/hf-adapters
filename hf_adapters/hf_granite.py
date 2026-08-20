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
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    get_backbone,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
    text_config,
)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm."""
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    h = backbone.norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    logits = model.lm_head(h)
    return logits / text_config(model.config).logits_scaling


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    prepare_rope_and_heads(model)
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = prepare_standard_gqa_blocks(
        get_backbone(model).layers, True
    )
