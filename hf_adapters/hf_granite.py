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

import types

import torch

from hf_adapters.fp8_linear import swap_linears_to_fp8

from hf_adapters.hf_common import (
    _SDPA_MAX_SEQUENCE_TILE_SIZE,
    get_backbone,
    pad_lm_head,
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
    logits = model.lm_head(h)
    return logits / text_config(model.config).logits_scaling

def _fp16_rmsnorm_forward(self, h):
    """RMSNorm with the variance reduction kept in fp16 on Spyre."""
    if h.device.type != "spyre":
        return type(self).forward(self, h)
    variance = (h * h).mean(-1, keepdim=True)
    return self.weight * (h * torch.rsqrt(variance + self.variance_epsilon))


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    backbone = get_backbone(model)
    # FP8 checkpoints only; runs before the blocks are built so they close over
    # FP8Linear.
    n_fp8, n_excluded = swap_linears_to_fp8(model)
    if n_fp8 or n_excluded:
        print(f"FP8: {n_fp8} module(s) -> FP8Linear, {n_excluded} -> fp16 nn.Linear")
    if n_fp8:
        # TODO: stock RMSNorm's fp32->fp16 cast leaves torch-spyre no feasible
        # layout for the FP8 scaled_mm that consumes it; keep the norms feeding
        # FP8Linear in fp16 until that is fixed.
        for layer in backbone.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.forward = types.MethodType(_fp16_rmsnorm_forward, norm)
    prepare_rope_and_heads(model)
    pad_lm_head(model)
    model._spyre_compiled_blocks = prepare_standard_gqa_blocks(backbone.layers, True)
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
    model._spyre_prefill_chunk_size = _SDPA_MAX_SEQUENCE_TILE_SIZE
