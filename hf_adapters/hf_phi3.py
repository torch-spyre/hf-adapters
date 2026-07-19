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
HuggingFace Transformers adapter for Phi-3/Phi-4 models on Spyre.

Phi-3 differences from Granite:
- Combined QKV projection (split at prepare time)
- Combined gate+up MLP (split at prepare time)
- Partial RoPE: only ``partial_rotary_factor`` of head_dim is rotated.
  Handled by padding the rotation matrix with identity entries so that
  ``apply_rope_matmul`` on full head_dim passes through non-rotated dims.
- No embedding/residual/logits multipliers

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
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
    get_backbone,
    kv_cache_update,
    pad_lm_head,
    pad_qk_proj_for_rope,
    permute_proj_for_rope,
    rope_dim_permutation,
    split_fused_linear,
)

# ---------------------------------------------------------------------------
# Weight splitting
#
# Partial-RoPE alignment (rope_dim_permutation / permute_proj_for_rope /
# pad_qk_proj_for_rope) is shared with hf_gpt_neox and lives in hf_common.
# ---------------------------------------------------------------------------


def _split_fused_qkv(attn, num_q, num_kv, head_dim):
    """Split fused qkv_proj into separate q/k/v projections."""
    w = attn.qkv_proj.weight
    q_dim = num_q * head_dim
    k_dim = num_kv * head_dim
    hidden = w.shape[1]

    def _mk(w_data, out_dim):
        p = nn.Linear(hidden, out_dim, bias=False)
        p.weight = nn.Parameter(w_data.clone(), requires_grad=False)
        return p

    return (
        _mk(w[:q_dim], q_dim),
        _mk(w[q_dim : q_dim + k_dim], k_dim),
        _mk(w[q_dim + k_dim :], k_dim),
    )


# ---------------------------------------------------------------------------
# Compiled block
# ---------------------------------------------------------------------------


def _make_compiled_block(layer, q_proj, k_proj, v_proj, gate_proj, up_proj, head_dim):
    """Compiled block for Phi-3. Full head_dim RoPE via identity-padded freqs."""
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    down_proj = layer.mlp.down_proj
    act_fn = layer.mlp.activation_fn
    o_proj = layer.self_attn.o_proj
    scaling = layer.self_attn.scaling

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
        h = input_ln(hidden_states)
        bsz, seq_len, _ = h.shape

        # Separate Q/K/V projections (split at prepare time)
        q = q_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)

        # RoPE on full head_dim — identity-padded freqs pass through
        # non-rotated dims unchanged
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)

        h = residual + attn_out

        # MLP with pre-split gate/up
        residual = h
        h = post_attn_ln(h)
        h = down_proj(act_fn(gate_proj(h)) * up_proj(h))
        h = residual + h

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
    """Phi-3 backbone: embedding, blocks, norm."""
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
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

    h = model._spyre_compiled_norm(h)
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
    """Phi-3 causal-LM forward: backbone + LM head."""
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

    return model.lm_head(h)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Phi-3 model in-place."""
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads

    # RoPE reshape [B,L,H,2,D/2] needs D/2 >= BLOCK_SIZE (one Spyre stick).
    # head_dim < 2*BLOCK_SIZE (e.g. Phi-3.5-mini: 96) produces an unaligned
    # stick expression the compiler rejects. Pad Q/K/V/O up to the next
    # multiple of 2*BLOCK_SIZE (96 -> 128). Phi-4-mini (head_dim=128) needs
    # no padding. work_hd is the head_dim used everywhere downstream.
    stick_aligned = ((hd + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    padded_head_dim = stick_aligned if stick_aligned > hd else None
    work_hd = padded_head_dim or hd

    # KV caches are sized from _spyre_head_dim (see hf_common.kv_cache_shapes).
    # The compiled block writes work_hd-wide K/V, so the cache must match.
    model._spyre_head_dim = work_hd

    # RoPE with identity padding: pads the rotation matrix to work_hd//2 so the
    # zero-padded Q/K dims (partial rotary and/or stick padding) pass through.
    model._spyre_rope = PrecomputedRotaryEmbedding(
        get_backbone(model).rotary_emb, padded_head_dim=work_hd
    )

    # LM head: smooth-padded to a stick-aligned vocab whose per-core span fits
    # the 256 MB EAR limit (see hf_common.pad_lm_head).
    pad_lm_head(model)

    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads

    # Partial RoPE dimension permutation: HF pairs (j, j+rope_dim//2) but
    # apply_rope_matmul pairs (j, j+head_dim//2). Permute Q/K weights so
    # both agree. Q·K^T dot product is invariant under same permutation.
    # Computed on the original head_dim; stick padding (below) happens after.
    prf = getattr(cfg, "partial_rotary_factor", 1.0)
    rope_dim = int(prf * hd)
    rope_perm = rope_dim_permutation(hd, rope_dim) if rope_dim != hd else None

    # Split fused weights, register as submodules
    model._spyre_q_projs = nn.ModuleList()
    model._spyre_k_projs = nn.ModuleList()
    model._spyre_v_projs = nn.ModuleList()
    model._spyre_gate_projs = nn.ModuleList()
    model._spyre_up_projs = nn.ModuleList()

    for layer in get_backbone(model).layers:
        q, k, v = _split_fused_qkv(layer.self_attn, num_q, num_kv, hd)
        if rope_perm is not None:
            permute_proj_for_rope(q, num_q, hd, rope_perm)
            permute_proj_for_rope(k, num_kv, hd, rope_perm)
        if padded_head_dim is not None:
            # Interleave-pad Q/K (RoPE layout), end-pad V, input-pad O.
            q = pad_qk_proj_for_rope(q, num_q, hd, padded_head_dim)
            k = pad_qk_proj_for_rope(k, num_kv, hd, padded_head_dim)
            v = _pad_proj_output_simple(v, num_kv, hd, padded_head_dim)
            layer.self_attn.o_proj = _pad_proj_input_simple(
                layer.self_attn.o_proj, num_q, hd, padded_head_dim
            )
        model._spyre_q_projs.append(q)
        model._spyre_k_projs.append(k)
        model._spyre_v_projs.append(v)

        gate, up = split_fused_linear(layer.mlp.gate_up_proj.weight)
        model._spyre_gate_projs.append(gate)
        model._spyre_up_projs.append(up)

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, qp, kp, vp, gp, up, work_hd)
        for layer, qp, kp, vp, gp, up in zip(
            get_backbone(model).layers,
            model._spyre_q_projs,
            model._spyre_k_projs,
            model._spyre_v_projs,
            model._spyre_gate_projs,
            model._spyre_up_projs,
        )
    ]
    model._spyre_compiled_norm = torch.compile(get_backbone(model).norm, dynamic=False)
