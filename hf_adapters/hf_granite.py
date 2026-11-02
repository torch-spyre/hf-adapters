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
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from hf_adapters.execution import compile_block
from hf_adapters.hf_common import (
    _SDPA_MAX_SEQUENCE_TILE_SIZE,
    embed_text_tokens,
    get_backbone,
    prepare_lm_head_for_spyre,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
    run_lm_head,
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
    h = embed_text_tokens(model, input_ids)

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

    h = model._spyre_compiled_norm(h)
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
    return run_lm_head(model, h)


def prepare_for_spyre(model, *, runtime=None):
    """Prepare Granite in-place with optional external attention execution.

    The serving path prepares only the backbone. Its caller owns the LM head,
    KV cache, and the timing of compilation after device placement.
    """
    prepare_rope_and_heads(model)
    logits_scaling = text_config(model.config).logits_scaling
    if runtime is None:
        prepare_lm_head_for_spyre(
            model, logits_processor=lambda logits: logits / logits_scaling
        )
    else:
        runtime.logits_scale = 1.0 / logits_scaling
    backbone = get_backbone(model)
    model._spyre_compiled_blocks = prepare_standard_gqa_blocks(
        backbone.layers, True, runtime
    )
    model._spyre_compiled_norm = compile_block(backbone.norm, runtime)
    model._spyre_prefill_chunk_size = _SDPA_MAX_SEQUENCE_TILE_SIZE
