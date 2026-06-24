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
HuggingFace Transformers adapter for ModernBERT encoder-only models on Spyre.

Supports models with ``ModernBertConfig`` (e.g. answerdotai/ModernBERT-base,
nomic-ai/modernbert-embed-base).

ModernBERT departs from classic BERT (``hf_bert``) in almost every block
internal, so it gets a custom compiled block rather than reusing
``make_encoder_block``:

- **RoPE**, not absolute position embeddings. ModernBERT uses two RoPE
  frequency sets — one for global ("full_attention") layers (``rope_theta``
  160000) and one for local ("sliding_attention") layers (``rope_theta``
  10000). Each gets its own ``PrecomputedRotaryEmbedding``.
- **Pre-norm**: ``LayerNorm`` is applied *before* attention/MLP, with the
  residual added after (``h = h + attn(norm(h))``). BERT is post-norm.
- **Layer 0** has ``attn_norm = Identity`` (the embedding ``norm`` already
  normalized the input).
- **Fused QKV** (``attn.Wqkv``) — split into separate q/k/v at prepare time so
  the RoPE head-dim padding can use the interleaved ``[2, D/2]`` layout.
- **GeGLU MLP**: ``Wi`` projects to ``2 * intermediate``; the halves are
  ``input, gate`` and the block computes ``Wo(act(input) * gate)``.
- **Alternating global / local attention**: every ``global_attn_every_n_layers``-th
  layer (layer 0, 3, 6, ...) is global; the rest attend only within a
  ``±sliding_window`` band. The caller builds both masks and selects per layer.
- **No biases** by default (``attention_bias``/``mlp_bias``/``norm_bias`` False).
- **head_dim = 64** (``D/2 = 32 < BLOCK_SIZE``) so Q/K/V/O are always padded
  64→128 for stick alignment, exactly like all-MiniLM in ``hf_bert``. Every
  ModernBERT checkpoint (base, large, the embed fine-tunes) uses head_dim 64,
  so there is no unpadded path.

Usage mirrors the other encoder adapters — see ``hf_bert`` /
``prefill_encoder`` for the embedding-path entry point. ``_run_backbone_forward``
keeps the ``(model, input_ids, attn_mask, position_ids, token_type_ids)``
signature that ``prefill_encoder`` dispatches on; ``token_type_ids`` is unused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_sliding_window_band,
    apply_rope_matmul,
    get_backbone,
)


def _split_and_pad_qkv(attn, hidden_size, num_heads, orig_head_dim, padded_head_dim):
    """Split fused ``Wqkv`` into padded q/k/v linears and pad ``Wo``.

    ModernBERT packs Q, K, V into a single ``Wqkv`` projection whose output is
    viewed as ``[..., 3, num_heads, head_dim]``. We materialize three separate
    linears so the RoPE head-dim padding can use the interleaved ``[2, D/2]``
    layout that ``apply_rope_matmul`` expects (Q/K), and simple end-padding for
    V. ``Wo``'s input dim is padded to match (end-padding per head, zeros
    contribute nothing).

    Stores ``q_proj``/``k_proj``/``v_proj`` on ``attn`` and replaces ``Wo``.
    Returns nothing — mutates ``attn`` in place. ModernBERT has no attention
    bias by default, but bias is handled if present.
    """
    orig_half = orig_head_dim // 2
    padded_half = padded_head_dim // 2
    has_bias = attn.Wqkv.bias is not None

    w = attn.Wqkv.weight  # [3 * num_heads * head_dim, hidden]
    # [3, num_heads, head_dim, hidden]
    w = w.view(3, num_heads, orig_head_dim, hidden_size)
    b = None
    if has_bias:
        b = attn.Wqkv.bias.view(3, num_heads, orig_head_dim)

    def _make_qk(slice_w, slice_b):
        """Interleaved-padded linear for Q or K (RoPE-aware [2, D/2] layout)."""
        new_w = torch.zeros(num_heads * padded_head_dim, hidden_size, dtype=w.dtype)
        new_b = (
            torch.zeros(num_heads * padded_head_dim, dtype=slice_b.dtype)
            if slice_b is not None
            else None
        )
        for h in range(num_heads):
            d = h * padded_head_dim
            new_w[d : d + orig_half, :] = slice_w[h, :orig_half, :]
            new_w[d + padded_half : d + padded_half + orig_half, :] = slice_w[
                h, orig_half:orig_head_dim, :
            ]
            if slice_b is not None:
                new_b[d : d + orig_half] = slice_b[h, :orig_half]
                new_b[d + padded_half : d + padded_half + orig_half] = slice_b[
                    h, orig_half:orig_head_dim
                ]
        proj = nn.Linear(hidden_size, num_heads * padded_head_dim, bias=has_bias)
        proj.weight = nn.Parameter(new_w, requires_grad=False)
        if new_b is not None:
            proj.bias = nn.Parameter(new_b, requires_grad=False)
        return proj

    def _make_v(slice_w, slice_b):
        """End-padded linear for V (no RoPE — layout within a head is free)."""
        new_w = torch.zeros(num_heads * padded_head_dim, hidden_size, dtype=w.dtype)
        new_b = (
            torch.zeros(num_heads * padded_head_dim, dtype=slice_b.dtype)
            if slice_b is not None
            else None
        )
        for h in range(num_heads):
            d = h * padded_head_dim
            new_w[d : d + orig_head_dim, :] = slice_w[h, :, :]
            if slice_b is not None:
                new_b[d : d + orig_head_dim] = slice_b[h, :]
        proj = nn.Linear(hidden_size, num_heads * padded_head_dim, bias=has_bias)
        proj.weight = nn.Parameter(new_w, requires_grad=False)
        if new_b is not None:
            proj.bias = nn.Parameter(new_b, requires_grad=False)
        return proj

    attn.q_proj = _make_qk(w[0], b[0] if b is not None else None)
    attn.k_proj = _make_qk(w[1], b[1] if b is not None else None)
    attn.v_proj = _make_v(w[2], b[2] if b is not None else None)

    # Pad Wo input dim per head (end-padding; padded entries are zero).
    wo = attn.Wo.weight  # [hidden, num_heads * orig_head_dim]
    new_wo = torch.zeros(hidden_size, num_heads * padded_head_dim, dtype=wo.dtype)
    for h in range(num_heads):
        s = h * orig_head_dim
        d = h * padded_head_dim
        new_wo[:, d : d + orig_head_dim] = wo[:, s : s + orig_head_dim]
    new_o = nn.Linear(
        num_heads * padded_head_dim, hidden_size, bias=attn.Wo.bias is not None
    )
    new_o.weight = nn.Parameter(new_wo, requires_grad=False)
    if attn.Wo.bias is not None:
        new_o.bias = nn.Parameter(attn.Wo.bias.clone(), requires_grad=False)
    attn.Wo = new_o


def _make_compiled_block(layer, num_heads, head_dim, orig_head_dim):
    """Compile one ModernBERT encoder layer (pre-norm + RoPE + GeGLU).

    Block signature carries the per-layer mask and RoPE freqs (which differ
    between global and local layers), so the caller selects them:

        block_forward(hidden_states, attn_mask, selected_freqs) -> hidden_states

    SDPA scales by ``1/sqrt(orig_head_dim)`` because Q·Kᵀ only sums over the
    non-zero (unpadded) entries — the padded dims are zero on both sides.
    Dropout is skipped (eval-only).
    """
    attn_norm = layer.attn_norm
    q_proj = layer.attn.q_proj
    k_proj = layer.attn.k_proj
    v_proj = layer.attn.v_proj
    o_proj = layer.attn.Wo
    mlp_norm = layer.mlp_norm
    wi = layer.mlp.Wi
    act = layer.mlp.act
    wo = layer.mlp.Wo
    sdpa_scale = orig_head_dim**-0.5

    def block_forward(hidden_states, attn_mask, selected_freqs):
        bsz, seq_len, _ = hidden_states.shape

        normed = attn_norm(hidden_states)

        q = q_proj(normed).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(normed).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(normed).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=sdpa_scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)
        hidden_states = hidden_states + attn_out

        inp, gate = wi(mlp_norm(hidden_states)).chunk(2, dim=-1)
        hidden_states = hidden_states + wo(act(inp) * gate)

        return hidden_states

    return torch.compile(block_forward, dynamic=False)


_is_encoder_only = True


def _run_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """ModernBERT encoder backbone forward.

    Departs from ``encoder_backbone_forward``:

    - ``token_type_ids`` is unused — ModernBERT has no token-type table.
    - Embedding is a single token table followed by ``norm`` (LayerNorm).
    - Two RoPE freq sets and two attention masks (global + local) are built
      once and selected per layer by ``layer.attention_type``.
    - Final ``final_norm`` after the last block.

    ``attn_mask`` from ``prefill_encoder`` is the global bidirectional mask
    (zeros for real-token pairs, -inf for padding). The local mask is that mask
    restricted to a ``±sliding_window`` band.
    """
    backbone = get_backbone(model)
    emb = backbone.embeddings

    h = emb.norm(emb.tok_embeddings(input_ids))
    h = h.clone() if h.device.type == "spyre" else h

    # Per-layer-type RoPE freqs (global theta vs local theta).
    freqs = {
        layer_type: rope(h, position_ids)
        for layer_type, rope in model._spyre_rope.items()
    }

    # Local layers see the global mask intersected with the sliding-window band.
    local_mask = add_sliding_window_band(
        attn_mask.to("cpu"), model.config.sliding_window
    ).to(attn_mask.device)
    masks = {"full_attention": attn_mask, "sliding_attention": local_mask}

    for layer, compiled_block in zip(backbone.layers, model._spyre_compiled_blocks):
        lt = layer.attention_type
        h = compiled_block(h, masks[lt], freqs[lt])
        if h.device.type == "spyre":
            h = h.clone()

    h = backbone.final_norm(h)
    return h


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a ModernBERT encoder model in-place.

    1. Pad attention heads 64→128 (head_dim/2 = 32 < BLOCK_SIZE): split the
       fused ``Wqkv`` into padded q/k/v and pad ``Wo``. Every ModernBERT
       checkpoint has head_dim 64, so this always runs.
    2. Build one ``PrecomputedRotaryEmbedding`` per RoPE layer type (global +
       local thetas), padded to the new head_dim.
    3. Compile each encoder layer's block.
    """
    backbone = get_backbone(model)
    cfg = model.config
    num_heads = cfg.num_attention_heads
    orig_head_dim = cfg.hidden_size // num_heads

    # RoPE reshape needs head_dim/2 >= BLOCK_SIZE → head_dim >= 2 * BLOCK_SIZE.
    padded_head_dim = ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (
        2 * BLOCK_SIZE
    )
    assert padded_head_dim > orig_head_dim, (
        f"ModernBERT adapter expects head_dim ({orig_head_dim}) < 2*BLOCK_SIZE "
        f"({2 * BLOCK_SIZE}); a stick-aligned variant would need an unpadded path."
    )
    head_dim = padded_head_dim

    for layer in backbone.layers:
        _split_and_pad_qkv(
            layer.attn, cfg.hidden_size, num_heads, orig_head_dim, padded_head_dim
        )

    # One PrecomputedRotaryEmbedding per layer type, reusing ModernBERT's
    # per-type inv_freq buffers via a shim. padded_head_dim identity-pads the
    # rotation matrix so apply_rope_matmul passes the padded dims through.
    rope = backbone.rotary_emb
    model._spyre_rope = {}
    for layer_type in set(cfg.layer_types):
        inv_freq = getattr(rope, f"{layer_type}_inv_freq")
        model._spyre_rope[layer_type] = PrecomputedRotaryEmbedding(
            InvFreqShim(inv_freq),
            padded_head_dim=padded_head_dim,
        )

    model._spyre_head_dim = head_dim
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, num_heads, head_dim, orig_head_dim)
        for layer in backbone.layers
    ]
