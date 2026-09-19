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

import torch

from hf_adapters.hf_common import (
    _SDPA_MAX_SEQUENCE_TILE_SIZE,
    get_backbone,
    prepare_lm_head_for_spyre,
    prepare_rope_and_heads,
    prepare_standard_gqa_h_only_blocks,
    prepare_standard_gqa_region_blocks,
    run_lm_head,
    text_config,
)


def _run_backbone_forward(
    model,
    input_ids,
    selected_freqs,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm.

    Takes ``selected_freqs`` (already gathered on the host by ``model._spyre_rope``)
    rather than ``position_ids``. The RoPE gather is intrinsically host-side
    (``.item()``-driven cache extend + CPU fancy-index) and must NOT be traced
    into the whole-forward graph — mirrors foundation-model-stack's ``eager_spyre``
    split, where the compiled forward consumes a ready freqs tensor.
    """
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        # Blocks on BOTH paths update key_caches[i]/value_caches[i] IN PLACE and
        # return only ``h``: the whole-forward compile turns a region block into
        # an ``invoke_subgraph`` HOP call that rejects a subgraph output aliasing
        # a subgraph input, so the cache buffers cannot be returned there. The
        # eager path matches that shape via prepare_standard_gqa_h_only_blocks
        # (not the raw 3-tuple StandardGQABlock) so this driver stays shared.
        h = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    h = model._spyre_compiled_norm(h)
    return h


def _run_forward_freqs(
    model,
    input_ids,
    selected_freqs,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling.

    Consumes ``selected_freqs`` directly, so the entire body is Spyre-traceable
    (no host-side RoPE gather). This is the callable wrapped by ``torch.compile``.
    """
    h = _run_backbone_forward(
        model,
        input_ids,
        selected_freqs,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    return run_lm_head(model, h)


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Eager Granite forward entry (position_ids based).

    Used by the stock token-compare test and any caller that still passes
    ``position_ids``. Performs the host-side RoPE gather here, then delegates to
    the freqs-based body. ``PrecomputedRotaryEmbedding.forward`` uses only its
    second arg for the gather, so ``input_ids`` as the first arg is a safe filler.
    """
    selected_freqs = model._spyre_rope(input_ids, position_ids)
    return _run_forward_freqs(
        model,
        input_ids,
        selected_freqs,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )


def _make_compiled_run_forward(model):
    """Bind Granite's whole-forward to ``model`` and torch.compile it once.

    The compiled callable owns the embed/mul prologue, the decoder-block loop
    (each block a nested_compile_region → compiled once), and the
    norm/head/scaling epilogue. It consumes ``selected_freqs`` as a graph
    INPUT — the host-side RoPE gather runs in the generate shim before this is
    called (see ``auto_spyre_model._resolve_run_forward_fn``). Signature matches
    that shim's call minus the leading ``model`` (which is closed over).
    """

    def _bound(
        input_ids,
        selected_freqs,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    ):
        return _run_forward_freqs(
            model,
            input_ids,
            selected_freqs,
            attn_mask,
            key_caches,
            value_caches,
            cache_index,
        )

    return torch.compile(_bound, dynamic=False)


def prepare_for_spyre(model, *, hier_compile: bool = False):
    """Apply Spyre adaptations to Granite 3.3 model in-place.

    Args:
        model: The HF Granite model to adapt (mutated in place).
        hier_compile: EXPERIMENTAL, opt-in. When False (the default), each
            decoder layer is compiled separately
            (``prepare_standard_gqa_h_only_blocks``) and generation runs through
            the eager ``_run_forward`` — the
            long-standing behavior every Spyre suite exercises today. When True,
            the layers become ``nested_compile_region`` blocks inside ONE
            ``torch.compile``d whole-forward (``_spyre_run_forward``), so the
            shared region is traced once and reused across all N layers instead
            of paying a separate compile per layer.

    Keyword-only so it can never be confused with the ``(model)``-only
    ``prepare_for_spyre`` signature every other adapter carries.
    """
    prepare_rope_and_heads(model)
    logits_scaling = text_config(model.config).logits_scaling
    prepare_lm_head_for_spyre(
        model, logits_processor=lambda logits: logits / logits_scaling
    )
    backbone = get_backbone(model)
    if hier_compile:
        model._spyre_compiled_blocks = prepare_standard_gqa_region_blocks(
            backbone.layers, True
        )
    else:
        # h-only (not the raw 3-tuple blocks): _run_backbone_forward is shared
        # with the hier path, where nested_compile_region forces a single-value
        # block return. Caches are mutated in place either way.
        model._spyre_compiled_blocks = prepare_standard_gqa_h_only_blocks(
            backbone.layers, True
        )
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
    model._spyre_prefill_chunk_size = _SDPA_MAX_SEQUENCE_TILE_SIZE
    # Only the hierarchical path attaches a compiled whole-forward. Leaving the
    # attribute unset on the default path is what keeps
    # ``auto_spyre_model._resolve_run_forward_fn`` inert (it falls back to the
    # eager ``_run_forward``), so default-path behavior is unchanged.
    if hier_compile:
        model._spyre_run_forward = _make_compiled_run_forward(model)
