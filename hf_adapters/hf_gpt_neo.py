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
HuggingFace Transformers adapter for GPT-Neo models on Spyre.

Covers model_type ``gpt_neo``: EleutherAI GPT-Neo (125M/1.3B/2.7B) and
GPT-J-style fine-tunes that share the same architecture.

Like GPT-2, GPT-Neo uses:
  - **learned absolute** position embeddings (``wpe``), no RoPE;
  - **LayerNorm** (weight + bias), pre-norm;
  - a GELU MLP (``c_fc`` → ``gelu_new`` → ``c_proj``);
  - the backbone at ``model.transformer``.

Unlike GPT-2, GPT-Neo uses:
  - ``nn.Linear`` for all projections (not ``Conv1D``), so no weight transpose;
  - **separate** ``q_proj``/``k_proj``/``v_proj`` (no fused ``c_attn``);
  - **alternating** global/local attention per layer (``config.attention_layers``).
    For Spyre we use full causal attention for every layer — the local window
    constraint is enforced by HF's stock forward but does not apply here.

``prepare_for_spyre`` patches the MLP activation, pads the LM head, and compiles
one block per layer via ``make_decoder_block``.

Head-dim padding
~~~~~~~~~~~~~~~~
The Spyre compiler requires every tensor dimension taking part in a matmul to be
a multiple of one stick (``BLOCK_SIZE = 64`` elements at fp16).  Real GPT-Neo
checkpoints (125 M, 1.3 B, 2.7 B) all have ``head_dim = 64``, which already
satisfies this.  Tiny fine-tunes such as ``roneneldan/TinyStories-*`` use
``head_dim = 4`` (``hidden_size=64, num_heads=16``), which does not.

When ``head_dim < BLOCK_SIZE`` ``prepare_for_spyre`` zero-pads all four
attention projections (Q, K, V, O) to the next stick boundary using the same
per-head end-padding as ``pad_attention_heads_simple`` (for non-RoPE encoders).
The padded rows are zero, so they contribute nothing to the dot-product — the
arithmetic is identical to the unpadded model.

GPT-Neo never applies ``1/sqrt(head_dim)`` scaling (the stock ``_attn`` does
raw ``Q·Kᵀ`` with no scale factor), so ``scale=1.0`` is preserved regardless
of padding.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    assert_spyre_dimensions,
    get_backbone,
    make_decoder_block,
    pad_lm_head,
    patch_new_gelu,
)


def _pad_attn_heads(layer, orig_head_dim, padded_head_dim, num_heads):
    """Zero-pad Q/K/V/O projections in a single GPT-Neo layer in-place.

    Uses the same per-head end-padding as ``pad_attention_heads_simple`` but
    targets GPT-Neo's module layout (``layer.attn.attention.{q,k,v}_proj`` /
    ``out_proj``) rather than BERT's.  ``attn.head_dim`` is updated so that
    ``_make_compiled_block`` always closes over the padded value.
    """
    attn = layer.attn.attention
    attn.q_proj = _pad_proj_output_simple(
        attn.q_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.k_proj = _pad_proj_output_simple(
        attn.k_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.v_proj = _pad_proj_output_simple(
        attn.v_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.out_proj = _pad_proj_input_simple(
        attn.out_proj, num_heads, orig_head_dim, padded_head_dim
    )
    attn.head_dim = padded_head_dim


def _make_compiled_block(layer):
    """Resolve GPT-Neo's module layout and hand off to ``make_decoder_block``.

    GPT-Neo uses separate ``q_proj``/``k_proj``/``v_proj`` (already ``nn.Linear``)
    and ``out_proj`` for the output projection. The attention submodule is nested
    at ``layer.attn.attention``. The MLP uses ``c_fc``/``act``/``c_proj`` like
    GPT-2.

    ``attn.head_dim`` may have been updated by ``_pad_attn_heads`` to a padded
    value; we always read it from the module so both padded and unpadded models
    use the correct dimension.  ``scale=1.0`` is unconditional — GPT-Neo omits
    the ``1/sqrt(head_dim)`` factor regardless of head size.
    """
    attn = layer.attn.attention
    mlp = layer.mlp
    num_heads = attn.num_heads
    head_dim = attn.head_dim  # padded if _pad_attn_heads ran, original otherwise
    return make_decoder_block(
        q_proj=attn.q_proj,
        k_proj=attn.k_proj,
        v_proj=attn.v_proj,
        o_proj=attn.out_proj,
        attn_ln=layer.ln_1,
        ffn_in=mlp.c_fc,
        act=mlp.act,
        ffn_out=mlp.c_proj,
        ffn_ln=layer.ln_2,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=1.0,  # GPT-Neo omits the 1/sqrt(head_dim) scale — padding doesn't change this
        pre_ln=True,
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
    """GPT-Neo backbone: token + learned position embeddings, compiled blocks, ln_f."""
    bb = get_backbone(model)
    h = bb.wte(input_ids) + bb.wpe(position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            None,  # selected_freqs — no RoPE
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    h = bb.ln_f(h)
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
    """GPT-Neo causal-LM forward: backbone + LM head (vocab-cropped)."""
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
    return logits[..., : model.config.vocab_size]


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a GPT-Neo model in-place.

    Patches the MLP activation (``NewGELUActivation``) to avoid ``torch.pow``
    on Spyre, pads the LM head vocab to a smooth stick count, and compiles one
    block per layer. No RoPE module is created — GPT-Neo uses learned absolute
    positions.

    When ``head_dim < BLOCK_SIZE`` (e.g. ``roneneldan/TinyStories-*`` with
    ``head_dim=4``), all four attention projections are zero-padded per-head to
    the next stick boundary before compilation so the SDPA Q·Kᵀ matmul lowers
    cleanly on Spyre.
    """
    cfg = model.config
    assert_spyre_dimensions(
        cfg, model_name=getattr(cfg, "name_or_path", "") or "gpt-neo"
    )

    bb = get_backbone(model)

    patch_new_gelu(type(bb.h[0].mlp.act))

    # Pad attention heads to a stick boundary when head_dim < BLOCK_SIZE.
    # Real GPT-Neo checkpoints (125M/1.3B/2.7B) have head_dim=64 and skip this.
    num_heads = cfg.num_heads
    embed_dim = cfg.hidden_size
    orig_head_dim = embed_dim // num_heads
    padded_head_dim = ((orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_head_dim > orig_head_dim:
        for layer in bb.h:
            _pad_attn_heads(layer, orig_head_dim, padded_head_dim, num_heads)
        model._spyre_head_dim = padded_head_dim
    else:
        padded_head_dim = orig_head_dim  # no-op; keep variable consistent below

    pad_lm_head(model)

    # GPT-Neo config uses num_heads / hidden_size rather than the standard
    # num_key_value_heads / head_dim. Set explicit KV shapes (with the
    # potentially-padded head_dim) so the cache allocator bypasses those lookups.
    model._spyre_kv_shapes = [
        (num_heads, padded_head_dim, padded_head_dim) for _ in range(cfg.num_layers)
    ]

    model._spyre_compiled_blocks = [_make_compiled_block(layer) for layer in bb.h]
