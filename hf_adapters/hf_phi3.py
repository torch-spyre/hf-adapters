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
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    embed_text_tokens,
    get_backbone,
    kv_cache_update,
    pad_qk_proj_for_rope,
    permute_proj_for_rope,
    prepare_lm_head_for_spyre,
    rope_dim_permutation,
    run_lm_head,
)
from hf_adapters.spyre_tensor_parallel import spyre_compiled_all_reduce

# ---------------------------------------------------------------------------
# Weight splitting
#
# Partial-RoPE alignment (rope_dim_permutation / permute_proj_for_rope /
# pad_qk_proj_for_rope) is shared with hf_gpt_neox and lives in hf_common.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tensor parallelism
#
# Transformers' Phi-3 plan cuts the fused qkv_proj / gate_up_proj weights into
# equal row blocks and all-gathers their outputs, so a block can straddle the
# Q/K/V or gate/up boundary. This adapter splits those weights itself and never
# calls the fused modules, so it would see a shard with no gather. Instead the
# fused weights load whole on CPU and each rank keeps only its own heads and
# neurons, which leaves o_proj and down_proj with rank-local inputs.
# ---------------------------------------------------------------------------

_FUSED_MODULE_SUFFIXES = (".self_attn.qkv_proj", ".mlp.gate_up_proj")


@dataclass(frozen=True)
class _TpPlacement:
    """This process's position in the tensor-parallel group."""

    rank: int = 0
    size: int = 1
    group_name: str | None = None


def _tp_placement(model) -> _TpPlacement:
    """Read the TP position from the mesh Transformers left on ``o_proj``."""
    o_proj = get_backbone(model).layers[0].self_attn.o_proj
    mesh = getattr(o_proj, "_hf_device_mesh", None)
    if mesh is None or mesh.size() == 1:
        return _TpPlacement()
    return _TpPlacement(
        rank=mesh.get_local_rank(),
        size=mesh.size(),
        group_name=mesh.get_group().group_name,
    )


def _validate_tp_heads(num_q, num_kv, tp_size):
    """Reject head layouts that cannot give every rank whole heads."""
    if num_q % tp_size != 0:
        raise ValueError(
            f"Phi-3 TP cannot split {num_q} query heads across {tp_size} ranks"
        )
    if num_kv % tp_size != 0:
        raise ValueError(
            f"Phi-3 TP cannot split {num_kv} KV heads across {tp_size} ranks"
        )


def _validate_tp_intermediate(intermediate_size, tp_size):
    """Reject MLP widths that cannot give every rank equal gate/up rows."""
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"gate_up_proj's {intermediate_size} intermediate rows do not "
            f"divide across {tp_size} TP ranks"
        )


def spyre_tp_cpu_replicated_modules(model, tp_size):
    """Load the fused projections whole on CPU for per-rank slicing."""
    cfg = model.config
    _validate_tp_heads(cfg.num_attention_heads, cfg.num_key_value_heads, tp_size)
    _validate_tp_intermediate(cfg.intermediate_size, tp_size)
    return {
        name
        for name, _ in model.named_modules()
        if name.endswith(_FUSED_MODULE_SUFFIXES)
    }


def _check_rank(rank, tp_size):
    if not 0 <= rank < tp_size:
        raise ValueError(f"TP rank must be in [0, {tp_size}), got {rank}")


def _rank_rows(weight, rank, tp_size):
    """Return an independent copy of this rank's equal share of ``weight``."""
    rows = weight.shape[0] // tp_size
    return weight[rank * rows : (rank + 1) * rows].clone()


def _local_qkv_weights(w, num_q, num_kv, head_dim, rank, tp_size):
    """Split a full fused QKV weight and keep this rank's whole heads."""
    _check_rank(rank, tp_size)
    q_dim = num_q * head_dim
    k_dim = num_kv * head_dim
    expected_rows = q_dim + 2 * k_dim
    if w.shape[0] != expected_rows:
        raise ValueError(
            f"qkv_proj must hold all {expected_rows} fused rows, got "
            f"{w.shape[0]}; it may have been sharded before the Q/K/V split"
        )
    _validate_tp_heads(num_q, num_kv, tp_size)
    return (
        _rank_rows(w[:q_dim], rank, tp_size),
        _rank_rows(w[q_dim : q_dim + k_dim], rank, tp_size),
        _rank_rows(w[q_dim + k_dim :], rank, tp_size),
    )


def _local_gate_up_weights(w, rank, tp_size):
    """Split a full fused gate/up weight and keep this rank's matching rows."""
    _check_rank(rank, tp_size)
    intermediate_size = w.shape[0] // 2
    _validate_tp_intermediate(intermediate_size, tp_size)
    return (
        _rank_rows(w[:intermediate_size], rank, tp_size),
        _rank_rows(w[intermediate_size:], rank, tp_size),
    )


def _frozen_linear(weight):
    proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    proj.weight = nn.Parameter(weight, requires_grad=False)
    return proj


def _split_fused_qkv(attn, num_q, num_kv, head_dim, rank=0, tp_size=1):
    """Split fused qkv_proj into this rank's q/k/v projections."""
    weights = _local_qkv_weights(
        attn.qkv_proj.weight, num_q, num_kv, head_dim, rank, tp_size
    )
    return tuple(_frozen_linear(weight) for weight in weights)


def _split_fused_gate_up(mlp, rank=0, tp_size=1):
    """Split fused gate_up_proj into this rank's gate/up projections."""
    weights = _local_gate_up_weights(mlp.gate_up_proj.weight, rank, tp_size)
    return tuple(_frozen_linear(weight) for weight in weights)


def _row_parallel_linear(x, proj, tp_group_name):
    """Apply an ``o_proj``/``down_proj`` whose input is already rank-local.

    Under TP, ``F.linear`` skips the ``rowwise_split_input`` hooks, which would
    split the input a second time, and the partial sums reduce inside the graph.
    """
    if tp_group_name is None:
        return proj(x)
    return spyre_compiled_all_reduce(F.linear(x, proj.weight), tp_group_name)


# ---------------------------------------------------------------------------
# Compiled block
# ---------------------------------------------------------------------------


def _make_compiled_block(
    layer, q_proj, k_proj, v_proj, gate_proj, up_proj, head_dim, tp_group_name=None
):
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
        cache_index,
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
            cache_index,
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
        attn_out = _row_parallel_linear(attn_out, o_proj, tp_group_name)

        h = residual + attn_out

        # MLP with pre-split gate/up
        residual = h
        h = post_attn_ln(h)
        h = _row_parallel_linear(
            act_fn(gate_proj(h)) * up_proj(h), down_proj, tp_group_name
        )
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
    cache_index,
):
    """Phi-3 backbone: embedding, blocks, norm."""
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
    """Phi-3 causal-LM forward: backbone + LM head."""
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

    prepare_lm_head_for_spyre(model)

    tp = _tp_placement(model)
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    local_q = num_q // tp.size
    local_kv = num_kv // tp.size

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
        q, k, v = _split_fused_qkv(layer.self_attn, num_q, num_kv, hd, tp.rank, tp.size)
        if rope_perm is not None:
            permute_proj_for_rope(q, local_q, hd, rope_perm)
            permute_proj_for_rope(k, local_kv, hd, rope_perm)
        if padded_head_dim is not None:
            # Interleave-pad Q/K (RoPE layout), end-pad V, input-pad O.
            q = pad_qk_proj_for_rope(q, local_q, hd, padded_head_dim)
            k = pad_qk_proj_for_rope(k, local_kv, hd, padded_head_dim)
            v = _pad_proj_output_simple(v, local_kv, hd, padded_head_dim)
            # The TP loader may already have placed this shard on Spyre.
            layer.self_attn.o_proj = _pad_proj_input_simple(
                layer.self_attn.o_proj.to("cpu"), local_q, hd, padded_head_dim
            )
        model._spyre_q_projs.append(q)
        model._spyre_k_projs.append(k)
        model._spyre_v_projs.append(v)

        gate, up = _split_fused_gate_up(layer.mlp, tp.rank, tp.size)
        model._spyre_gate_projs.append(gate)
        model._spyre_up_projs.append(up)

        if tp.size > 1:
            # Whole on every rank and unused from here on: keep them off Spyre.
            layer.self_attn.qkv_proj = None
            layer.mlp.gate_up_proj = None

    if tp.size > 1:
        # kv_cache_shapes falls back to the global config count without a k_proj.
        model._spyre_kv_shapes = [
            (local_kv, work_hd, work_hd) for _ in get_backbone(model).layers
        ]

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, qp, kp, vp, gp, up, work_hd, tp.group_name)
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
