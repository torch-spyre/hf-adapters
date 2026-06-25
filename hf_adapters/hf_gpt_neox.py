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
HuggingFace Transformers adapter for GPT-NeoX models on Spyre.

Covers model_type ``gpt_neox``: EleutherAI GPT-NeoX-20B, Pythia series
(70M–12B), and other GPT-NeoX-architecture models.

Key architecture details:
  - **Rotary position embeddings** (RoPE), partial: only the first
    ``partial_rotary_factor`` fraction of ``head_dim`` is rotated
    (default 0.25, so 25% of dims rotate, 75% pass through).
  - **Fused QKV** projection (``query_key_value``), split at prepare time.
  - **Parallel residual**: attention and MLP both read from the same
    pre-norm hidden states and their outputs are added to the residual
    together (not sequentially).
  - **LayerNorm** (not RMSNorm) — no patching required.
  - Backbone at ``model.gpt_neox`` (not ``model.model``), layers at
    ``layers``, final norm ``final_layer_norm``.
  - ``dense`` for the attention output projection.
  - ``dense_h_to_4h`` / ``act`` / ``dense_4h_to_h`` for the MLP.
  - ``gelu`` (not ``gelu_new``) activation — no ``torch.pow``, no patching.

``prepare_for_spyre`` splits the fused QKV, handles partial RoPE alignment
(same permutation technique as Phi-3), pads the LM head, and compiles one
block per layer.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    assert_spyre_dimensions,
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    pad_qk_proj_for_rope,
    permute_proj_for_rope,
    rope_dim_permutation,
)

# ---------------------------------------------------------------------------
# Weight splitting helpers
#
# Partial-RoPE alignment (rope_dim_permutation / permute_proj_for_rope /
# pad_qk_proj_for_rope) is shared with hf_phi3 and lives in hf_common; the
# QKV split below is GPT-NeoX-specific (per-head interleaved layout).
# ---------------------------------------------------------------------------


def _split_fused_qkv(layer, num_heads, head_dim):
    """Split the fused ``query_key_value`` into separate q/k/v linears.

    GPT-NeoX interleaves QKV per-head: the weight matrix, reshaped as
    ``[num_heads, 3*head_dim, hidden]``, has layout ``[q_h, k_h, v_h]`` for
    each head h — NOT the sequential ``[all_q, all_k, all_v]`` used by
    models like Phi-3. Each output ``nn.Linear`` has shape
    ``[num_heads*head_dim, hidden]`` with heads in the standard order.
    """
    w = layer.attention.query_key_value.weight  # [3*H*hd, hidden]
    b = layer.attention.query_key_value.bias  # [3*H*hd]
    hidden = w.shape[1]
    stride = 3 * head_dim

    w_q = torch.zeros(num_heads * head_dim, hidden, dtype=w.dtype)
    w_k = torch.zeros(num_heads * head_dim, hidden, dtype=w.dtype)
    w_v = torch.zeros(num_heads * head_dim, hidden, dtype=w.dtype)
    b_q = torch.zeros(num_heads * head_dim, dtype=b.dtype)
    b_k = torch.zeros(num_heads * head_dim, dtype=b.dtype)
    b_v = torch.zeros(num_heads * head_dim, dtype=b.dtype)

    for h in range(num_heads):
        src = h * stride
        dst = h * head_dim
        w_q[dst : dst + head_dim] = w[src : src + head_dim]
        w_k[dst : dst + head_dim] = w[src + head_dim : src + 2 * head_dim]
        w_v[dst : dst + head_dim] = w[src + 2 * head_dim : src + 3 * head_dim]
        b_q[dst : dst + head_dim] = b[src : src + head_dim]
        b_k[dst : dst + head_dim] = b[src + head_dim : src + 2 * head_dim]
        b_v[dst : dst + head_dim] = b[src + 2 * head_dim : src + 3 * head_dim]

    def _mk(w_data, b_data):
        p = nn.Linear(hidden, num_heads * head_dim, bias=True)
        p.weight = nn.Parameter(w_data.clone(), requires_grad=False)
        p.bias = nn.Parameter(b_data.detach().clone(), requires_grad=False)
        return p

    return _mk(w_q, b_q), _mk(w_k, b_k), _mk(w_v, b_v)


# ---------------------------------------------------------------------------
# Compiled block
# ---------------------------------------------------------------------------


def _make_compiled_block(layer, q_proj, k_proj, v_proj, head_dim, num_heads, scale):
    """Compiled block for GPT-NeoX with parallel residual.

    Parallel residual means attn and MLP both read from the same pre-norm
    state and are summed together into the residual (not chained through each
    other's output). This differs from the standard sequential residual used
    by most decoders.
    """
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    dense = layer.attention.dense  # output projection
    dense_h_to_4h = layer.mlp.dense_h_to_4h
    act = layer.mlp.act
    dense_4h_to_h = layer.mlp.dense_4h_to_h

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states

        # Parallel residual: both branches read from the same normed input.
        attn_input = input_ln(hidden_states)
        mlp_input = post_attn_ln(hidden_states)

        bsz, seq_len, _ = attn_input.shape
        q = q_proj(attn_input).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(attn_input).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(attn_input).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache, is_filling, token_index, cache_position
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = dense(attn_out)

        mlp_out = dense_4h_to_h(act(dense_h_to_4h(mlp_input)))

        h = residual + attn_out + mlp_out
        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Forward / prepare / load / generate
# ---------------------------------------------------------------------------


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """GPT-NeoX backbone: token embedding, RoPE, compiled blocks, final_layer_norm."""
    bb = get_backbone(model)
    h = bb.embed_in(input_ids)
    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = bb.final_layer_norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """GPT-NeoX causal-LM forward: backbone + embed_out (the LM head)."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )
    logits = model.embed_out(h)
    return logits[..., : model.config.vocab_size]


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a GPT-NeoX model in-place.

    Splits the fused QKV projection, applies the partial-RoPE permutation,
    pads head_dim if below a Spyre stick (``head_dim/2 < BLOCK_SIZE``),
    pads the LM head, and compiles one block per layer.
    """
    cfg = model.config
    assert_spyre_dimensions(
        cfg, model_name=getattr(cfg, "name_or_path", "") or "gpt-neox"
    )

    bb = get_backbone(model)
    num_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    hd = hidden // num_heads

    # RoPE: partial_rotary_factor lives in config.rope_parameters dict (not directly on config).
    rope_params = getattr(cfg, "rope_parameters", {}) or {}
    prf = rope_params.get(
        "partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 0.25)
    )
    rope_dim = int(prf * hd)

    # Align head_dim to Spyre stick: D/2 >= BLOCK_SIZE required for apply_rope_matmul.
    stick_aligned = ((hd + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    padded_head_dim = stick_aligned if stick_aligned > hd else None
    work_hd = padded_head_dim or hd

    model._spyre_head_dim = work_hd

    # Partial RoPE permutation: align HF rotate_half pairing with apply_rope_matmul.
    rope_perm = rope_dim_permutation(hd, rope_dim) if rope_dim != hd else None

    # Build PrecomputedRotaryEmbedding with identity padding for non-rotated dims.
    model._spyre_rope = PrecomputedRotaryEmbedding(
        bb.rotary_emb, padded_head_dim=work_hd
    )

    # GPT-NeoX names its output projection ``embed_out`` (not ``lm_head``);
    # pad_lm_head resolves it via _get_lm_head, so no aliasing is needed.
    pad_lm_head(model)

    # Split fused QKV, apply permutation, pad if needed; register as submodules.
    model._spyre_q_projs = nn.ModuleList()
    model._spyre_k_projs = nn.ModuleList()
    model._spyre_v_projs = nn.ModuleList()

    for layer in bb.layers:
        q, k, v = _split_fused_qkv(layer, num_heads, hd)

        # Order matters: permute on the original head_dim first, then pad. The
        # permutation lands both rotary halves in the pre-pad [0:half | half:hd]
        # positions that pad_qk_proj_for_rope splits on, so the two compose
        # correctly. Default Pythia (hd=64<2*BLOCK_SIZE, rope_dim=16!=hd) hits
        # both at once — covered by the gpt_neox CPU accuracy test.
        if rope_perm is not None:
            permute_proj_for_rope(q, num_heads, hd, rope_perm)
            permute_proj_for_rope(k, num_heads, hd, rope_perm)

        if padded_head_dim is not None:
            q = pad_qk_proj_for_rope(q, num_heads, hd, padded_head_dim)
            k = pad_qk_proj_for_rope(k, num_heads, hd, padded_head_dim)
            v = _pad_proj_output_simple(v, num_heads, hd, padded_head_dim)
            layer.attention.dense = _pad_proj_input_simple(
                layer.attention.dense, num_heads, hd, padded_head_dim
            )

        model._spyre_q_projs.append(q)
        model._spyre_k_projs.append(k)
        model._spyre_v_projs.append(v)

    # GPT-NeoX is MHA; KV shapes match Q shapes.
    model._spyre_kv_shapes = [
        (num_heads, work_hd, work_hd) for _ in range(cfg.num_hidden_layers)
    ]

    # Use the original (unpadded) head_dim for the attention scale. After
    # stick-padding Q/K/V, the zero-padded dims don't contribute to Q·K^T, so
    # the effective dot-product magnitude is still set by the original hd.
    orig_scale = bb.layers[0].attention.scaling

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, qp, kp, vp, work_hd, num_heads, orig_scale)
        for layer, qp, kp, vp in zip(
            bb.layers,
            model._spyre_q_projs,
            model._spyre_k_projs,
            model._spyre_v_projs,
        )
    ]
