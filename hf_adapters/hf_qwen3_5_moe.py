# Copyright 2026 The Torch-Spyre Authors.
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

"""Spyre adapter for text-only Qwen3.5 MoE models.

The attention, Gated DeltaNet, cache, and model-forward implementations are
shared with the dense Qwen3.5 adapter. Prefill evaluates every routed expert;
one-token decode gathers only the selected experts. Both paths also evaluate
the sigmoid-gated shared expert present in every Qwen3.5 MoE layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters import hf_qwen3_5
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    moe_decode_selected_experts,
    moe_prefill_all_experts,
    optional_spyre_config_patch,
    prepare_lm_head_for_spyre,
    prepare_moe_expert_weights,
    text_config,
)
from hf_adapters.hf_qwen3_5 import (
    _allocate_caches,
    _rms_norm,
    _split_gated_q_projection,
)
from hf_adapters.hf_qwen3_5 import (
    _make_linear_attention_block as _make_dense_linear_attention_block,
)
from hf_adapters.spyre_tensor_parallel import spyre_compiled_all_reduce

# Keep local TP expert shards on CPU until ``prepare_moe_expert_weights`` performs
# their split/transpose/pad and final device transfer.
SPYRE_TP_CPU_STAGED_MODULES = {
    "model.layers.*.mlp.experts.gate_up_proj",
    "model.layers.*.mlp.experts.down_proj",
    "model.language_model.layers.*.mlp.experts.gate_up_proj",
    "model.language_model.layers.*.mlp.experts.down_proj",
}

_MOE_TILE = 32
_run_backbone_forward = hf_qwen3_5._run_backbone_forward
_run_forward = hf_qwen3_5._run_forward


def spyre_tp_replicated_linear_modules(model, tp_size):
    """Replicate dense branches; only routed experts are sharded and reduced."""
    del tp_size
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (
            ".mlp.shared_expert." in name
            or ".self_attn." in name
            or ".linear_attn." in name
        )
    }


def _router_topk_cpu(x, router_weight, top_k):
    """Apply the stock Qwen3.5 router exactly on CPU.

    Routing is deliberately a graph boundary. Besides preserving HF's fp32
    softmax, this avoids the keep_by_index -> packed-BMM layout permutation in
    current torch-spyre builds.
    """
    if x.device.type != "cpu" or router_weight.device.type != "cpu":
        raise RuntimeError("Qwen3.5 MoE routing must be CPU staged")
    router_logits = F.linear(x, router_weight)
    probabilities = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
    weights, expert_indices = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(x.dtype), expert_indices


def _route_decode_on_cpu(x, mlp, top_k, stick_size):
    weights, expert_indices = _router_topk_cpu(x.to("cpu"), mlp.gate.weight, top_k)
    weights = weights[..., None].expand(-1, -1, stick_size).contiguous()
    expert_indices = (
        expert_indices.to(x.dtype)[..., None].expand(-1, -1, stick_size).contiguous()
    )
    return weights.to(x.device), expert_indices.to(x.device)


def _moe_decode(x, mlp, weights, expert_indices, top_k, stick_size):
    experts = mlp.experts
    return moe_decode_selected_experts(
        x,
        weights[..., 0],
        expert_indices[..., 0],
        experts.gate_proj,
        experts.up_proj,
        experts.down_proj,
        top_k,
        _MOE_TILE,
        stick_size,
        "silu",
    )


def _moe_prefill_routing(x, mlp, top_k):
    weights, selected = _router_topk_cpu(x.to("cpu"), mlp.gate.weight, top_k)
    routing = torch.zeros(
        x.shape[0], mlp.gate.weight.shape[0], dtype=x.dtype, device="cpu"
    ).scatter(-1, selected, weights)
    return routing.to(x.device)


def _shared_expert(x, mlp):
    shared = mlp.shared_expert
    output = shared.down_proj(F.silu(shared.gate_proj(x)) * shared.up_proj(x))
    return torch.sigmoid(mlp.shared_expert_gate(x)) * output


def _reduce_expert_output_fp32(output, tp_group_name):
    """Reduce a materialized routed-expert output through an FP32 graph."""
    if tp_group_name is None:
        return output
    output_dtype = output.dtype
    return spyre_compiled_all_reduce(output.float(), tp_group_name).to(output_dtype)


def _normalize_ffn_input(residual, post_attention_norm):
    hidden_size = residual.shape[-1]
    return _rms_norm(residual, post_attention_norm).reshape(-1, hidden_size)


def _permute_proj_for_rope_cpu(proj, num_heads, head_dim, perm):
    """Permute Q/K outputs on CPU, including TP-loaded Spyre parameters."""
    weight_device = proj.weight.device
    weight = proj.weight.detach().cpu().view(num_heads, head_dim, -1)
    permuted_weight = weight[:, perm, :].contiguous().view(num_heads * head_dim, -1)
    if weight_device.type == "spyre":
        from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped

        permuted_weight = _dma_to_spyre_dim_order_swapped(
            permuted_weight,
            target_dtype=proj.weight.dtype,
            device=weight_device,
        )
    else:
        permuted_weight = permuted_weight.to(weight_device)
    proj.weight = nn.Parameter(permuted_weight, requires_grad=False)
    if proj.bias is not None:
        bias_device = proj.bias.device
        bias = proj.bias.detach().cpu().view(num_heads, head_dim)
        permuted_bias = bias[:, perm].contiguous().view(-1).to(bias_device)
        proj.bias = nn.Parameter(permuted_bias, requires_grad=False)


def _decode_routed_experts(
    x,
    weights,
    expert_indices,
    mlp,
    top_k,
    stick_size,
):
    return _moe_decode(x, mlp, weights, expert_indices, top_k, stick_size)


def _prefill_routed_experts(x, routing_weight, mlp):
    experts = mlp.experts
    return moe_prefill_all_experts(
        x,
        routing_weight[..., None],
        experts.gate_proj,
        experts.up_proj,
        experts.down_proj,
        "silu",
    )


def _finish_ffn(residual, x, routed, mlp, tp_group_name=None):
    output = routed + _shared_expert(x, mlp)
    output = _reduce_expert_output_fp32(output, tp_group_name)
    return residual + output.to(residual.dtype).reshape_as(residual)


def _make_attention_block(
    layer, q_proj, gate_proj, head_dim, top_k, stick_size, tp_group_name=None
):
    attn = layer.self_attn
    input_norm = layer.input_layernorm
    post_attention_norm = layer.post_attention_layernorm
    mlp = layer.mlp

    def project_forward(hidden_states, selected_freqs):
        residual = hidden_states
        h = _rms_norm(hidden_states, input_norm)
        batch_size, sequence_length, _ = h.shape

        query = q_proj(h).view(batch_size, sequence_length, -1, head_dim)
        gate = gate_proj(h)
        key = attn.k_proj(h).view(batch_size, sequence_length, -1, head_dim)
        value = attn.v_proj(h).view(batch_size, sequence_length, -1, head_dim)
        query = _rms_norm(query, attn.q_norm).transpose(1, 2)
        key = _rms_norm(key, attn.k_norm).transpose(1, 2)
        value = value.transpose(1, 2)
        query = apply_rope_matmul(query, selected_freqs)
        key = apply_rope_matmul(key, selected_freqs)
        return residual, query, key, value, gate

    def update_cache(key, value, key_cache, value_cache, cache_index):
        return kv_cache_update(key, value, key_cache, value_cache, cache_index)

    def attention_forward(query, key_cache, value_cache, attention_mask):
        return F.scaled_dot_product_attention(
            query,
            key_cache,
            value_cache,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )

    def finish_attention(residual, attention_output, gate):
        batch_size, sequence_length = residual.shape[:2]
        h = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return residual + attn.o_proj(h * torch.sigmoid(gate))

    def normalize_ffn_input(hidden_states):
        return _normalize_ffn_input(hidden_states, post_attention_norm)

    def prefill_routed_experts(x, routing_weight):
        return _prefill_routed_experts(x, routing_weight, mlp)

    def decode_routed_experts(x, weights, expert_indices):
        return _decode_routed_experts(
            x,
            weights,
            expert_indices,
            mlp,
            top_k,
            stick_size,
        )

    def finish_ffn(hidden_states, x, routed):
        return _finish_ffn(hidden_states, x, routed, mlp, tp_group_name)

    compiled_normalize_prefill = torch.compile(normalize_ffn_input, dynamic=False)
    compiled_normalize_decode = torch.compile(normalize_ffn_input, dynamic=False)
    compiled_prefill_routed = torch.compile(
        prefill_routed_experts, dynamic=False, fullgraph=True
    )

    def named_prefill_routed(x, routing_weight):
        with optional_spyre_config_patch({"allow_all_ops_in_lx_planning": True}):
            return compiled_prefill_routed(x, routing_weight)

    compiled_finish_prefill = torch.compile(finish_attention, dynamic=False)
    compiled_finish_decode = torch.compile(finish_attention, dynamic=False)
    compiled_decode_routed = torch.compile(
        decode_routed_experts, dynamic=False, fullgraph=True
    )
    compiled_finish_prefill_ffn = torch.compile(
        finish_ffn, dynamic=False, fullgraph=True
    )
    compiled_finish_decode_ffn = torch.compile(
        finish_ffn, dynamic=False, fullgraph=True
    )

    def finish_prefill(residual, attention_output, gate):
        hidden_states = compiled_finish_prefill(residual, attention_output, gate)
        x = compiled_normalize_prefill(hidden_states)
        routing_weight = _moe_prefill_routing(x, mlp, top_k)
        routed = named_prefill_routed(x, routing_weight)
        return compiled_finish_prefill_ffn(hidden_states, x, routed)

    def finish_decode(residual, attention_output, gate):
        hidden_states = compiled_finish_decode(residual, attention_output, gate)
        x = compiled_normalize_decode(hidden_states)
        weights, expert_indices = _route_decode_on_cpu(x, mlp, top_k, stick_size)
        routed = compiled_decode_routed(x, weights, expert_indices)
        return compiled_finish_decode_ffn(hidden_states, x, routed)

    # Preserve the dense runner's eight-item tuple. The two finish entries are
    # lightweight Python dispatchers over separate compiled attention and MoE
    # stages, so the MoE is no longer fused into the attention graph.
    return (
        torch.compile(project_forward, dynamic=False),
        torch.compile(project_forward, dynamic=False),
        torch.compile(update_cache, dynamic=False),
        torch.compile(update_cache, dynamic=False),
        torch.compile(attention_forward, dynamic=False),
        torch.compile(attention_forward, dynamic=False),
        finish_prefill,
        finish_decode,
    )


def _make_linear_attention_block(layer, top_k, stick_size, tp_group_name=None):
    """Reuse dense's split mixer/CPU recurrence protocol and replace its FFN."""
    dense_block = _make_dense_linear_attention_block(layer)
    post_attention_norm = layer.post_attention_layernorm
    mlp = layer.mlp

    def normalize_ffn_input(hidden_states):
        return _normalize_ffn_input(hidden_states, post_attention_norm)

    def prefill_routed_experts(x, routing_weight):
        return _prefill_routed_experts(x, routing_weight, mlp)

    def decode_routed_experts(x, weights, expert_indices):
        return _decode_routed_experts(
            x,
            weights,
            expert_indices,
            mlp,
            top_k,
            stick_size,
        )

    def finish_ffn(hidden_states, x, routed):
        return _finish_ffn(hidden_states, x, routed, mlp, tp_group_name)

    compiled_normalize_prefill = torch.compile(normalize_ffn_input, dynamic=False)
    compiled_normalize_decode = torch.compile(normalize_ffn_input, dynamic=False)
    compiled_prefill_routed = torch.compile(
        prefill_routed_experts, dynamic=False, fullgraph=True
    )
    compiled_decode_routed = torch.compile(
        decode_routed_experts, dynamic=False, fullgraph=True
    )
    compiled_finish_prefill_ffn = torch.compile(
        finish_ffn, dynamic=False, fullgraph=True
    )
    compiled_finish_decode_ffn = torch.compile(
        finish_ffn, dynamic=False, fullgraph=True
    )

    def named_prefill_routed(x, routing_weight):
        with optional_spyre_config_patch({"allow_all_ops_in_lx_planning": True}):
            return compiled_prefill_routed(x, routing_weight)

    def finish_prefill(hidden_states):
        x = compiled_normalize_prefill(hidden_states)
        routing_weight = _moe_prefill_routing(x, mlp, top_k)
        routed = named_prefill_routed(x, routing_weight)
        return compiled_finish_prefill_ffn(hidden_states, x, routed)

    def finish_decode(hidden_states):
        x = compiled_normalize_decode(hidden_states)
        weights, expert_indices = _route_decode_on_cpu(x, mlp, top_k, stick_size)
        routed = compiled_decode_routed(x, weights, expert_indices)
        return compiled_finish_decode_ffn(hidden_states, x, routed)

    return (*dense_block[:14], finish_prefill, finish_decode)


def prepare_for_spyre(model):
    """Prepare a text-only Qwen3.5 MoE causal LM for Spyre in place."""
    cfg = text_config(model.config)
    unsupported = set(cfg.layer_types) - {"full_attention", "linear_attention"}
    if unsupported:
        raise ValueError(f"Unsupported Qwen3.5 MoE layer types: {sorted(unsupported)}")
    if cfg.hidden_act != "silu":
        raise ValueError(f"Unsupported Qwen3.5 MoE activation: {cfg.hidden_act}")
    if cfg.linear_num_value_heads % cfg.linear_num_key_heads:
        raise ValueError(
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        )
    if cfg.head_dim // 2 < BLOCK_SIZE:
        raise ValueError(
            f"Qwen3.5 MoE head_dim={cfg.head_dim} is too small for Spyre RoPE; "
            "head padding is not implemented"
        )
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError(
            "num_experts_per_tok must be positive and no greater than num_experts"
        )

    backbone = get_backbone(model)
    # Keep the small router projections on CPU. Each layer crosses an explicit
    # graph boundary for exact fp32-softmax/top-k routing, then sends only the
    # selected indices and normalized weights back to Spyre.
    model._spyre_cpu_submodules = [
        name for name, _ in model.named_modules() if name.endswith(".mlp.gate")
    ]
    rope_dim = int(cfg.partial_rotary_factor * cfg.head_dim)
    if rope_dim % 2:
        raise ValueError(f"Qwen3.5 MoE rotary dimension must be even, got {rope_dim}")

    model._spyre_rope = PrecomputedRotaryEmbedding(
        backbone.rotary_emb, padded_head_dim=cfg.head_dim
    )
    model._spyre_head_dim = cfg.head_dim
    model._spyre_cache_allocator = _allocate_caches
    model._spyre_prefill_chunk_size = BLOCK_SIZE
    prepare_lm_head_for_spyre(model)

    try:
        from torch_spyre._C import get_elem_in_stick

        stick_size = get_elem_in_stick(next(model.parameters()).dtype)
    except ImportError:
        stick_size = BLOCK_SIZE

    rope_half = rope_dim // 2
    pass_half = (cfg.head_dim - rope_dim) // 2
    rope_permutation = (
        torch.cat(
            [
                torch.arange(0, rope_half),
                torch.arange(rope_dim, rope_dim + pass_half),
                torch.arange(rope_half, rope_dim),
                torch.arange(rope_dim + pass_half, cfg.head_dim),
            ]
        )
        if rope_dim != cfg.head_dim
        else None
    )
    model._spyre_q_projs = nn.ModuleList()
    model._spyre_gate_projs = nn.ModuleList()
    compiled_blocks = []

    for layer_type, layer in zip(cfg.layer_types, backbone.layers):
        mlp = layer.mlp
        expert_mesh = getattr(mlp.experts, "_hf_device_mesh", None)
        tp_group_name = (
            expert_mesh.get_group().group_name if expert_mesh is not None else None
        )
        prepare_moe_expert_weights(mlp.experts)

        if layer_type == "full_attention":
            local_query_heads = layer.self_attn.q_proj.weight.shape[0] // (
                2 * cfg.head_dim
            )
            local_kv_heads = layer.self_attn.k_proj.weight.shape[0] // cfg.head_dim
            query_projection, gate_projection = _split_gated_q_projection(
                layer.self_attn, local_query_heads, cfg.head_dim
            )
            layer.self_attn.q_proj = nn.Identity()
            if rope_permutation is not None:
                _permute_proj_for_rope_cpu(
                    query_projection,
                    local_query_heads,
                    cfg.head_dim,
                    rope_permutation,
                )
                _permute_proj_for_rope_cpu(
                    layer.self_attn.k_proj,
                    local_kv_heads,
                    cfg.head_dim,
                    rope_permutation,
                )
                q_norm_device = layer.self_attn.q_norm.weight.device
                k_norm_device = layer.self_attn.k_norm.weight.device
                layer.self_attn.q_norm.weight.data = (
                    layer.self_attn.q_norm.weight.data.to("cpu")[rope_permutation]
                    .contiguous()
                    .to(q_norm_device)
                )
                layer.self_attn.k_norm.weight.data = (
                    layer.self_attn.k_norm.weight.data.to("cpu")[rope_permutation]
                    .contiguous()
                    .to(k_norm_device)
                )
            model._spyre_q_projs.append(query_projection)
            model._spyre_gate_projs.append(gate_projection)
            compiled_blocks.append(
                _make_attention_block(
                    layer,
                    query_projection,
                    gate_projection,
                    cfg.head_dim,
                    int(cfg.num_experts_per_tok),
                    stick_size,
                    tp_group_name,
                )
            )
        else:
            conv_weight = torch.zeros(
                1,
                layer.linear_attn.conv_dim,
                BLOCK_SIZE,
                dtype=layer.linear_attn.conv1d.weight.dtype,
                device="cpu",
            )
            conv_weight[..., -layer.linear_attn.conv_kernel_size :] = (
                layer.linear_attn.conv1d.weight[:, 0, :].to("cpu")
            )
            layer.linear_attn._spyre_conv_weight = nn.Parameter(
                conv_weight, requires_grad=False
            )
            layer.linear_attn._spyre_conv_taps = nn.ParameterList(
                nn.Parameter(conv_weight[..., index, None], requires_grad=False)
                for index in range(
                    BLOCK_SIZE - layer.linear_attn.conv_kernel_size, BLOCK_SIZE
                )
            )
            shift_matrices = torch.zeros(
                layer.linear_attn.conv_kernel_size - 1,
                2 * BLOCK_SIZE,
                BLOCK_SIZE,
                dtype=conv_weight.dtype,
            )
            for lag in range(1, layer.linear_attn.conv_kernel_size):
                shift_matrices[
                    lag - 1,
                    BLOCK_SIZE - lag : 2 * BLOCK_SIZE - lag,
                    :,
                ] = torch.eye(BLOCK_SIZE, dtype=conv_weight.dtype)
            layer.linear_attn._spyre_conv_shift_matrices = nn.Parameter(
                shift_matrices, requires_grad=False
            )
            identity = torch.eye(BLOCK_SIZE, dtype=conv_weight.dtype)
            decode_matrices = []
            for lag in range(1, layer.linear_attn.conv_kernel_size):
                selector = torch.zeros(BLOCK_SIZE, 1, dtype=conv_weight.dtype)
                selector[-lag, 0] = 1
                decode_matrices.append(selector)
            shift_state = torch.roll(identity, -1, dims=1)
            shift_state[0, -1] = 0
            append_token = torch.zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=conv_weight.dtype)
            append_token[0, -1] = 1
            decode_matrices.extend((shift_state, append_token))
            layer.linear_attn._spyre_conv_decode_matrices = nn.ParameterList(
                nn.Parameter(matrix, requires_grad=False) for matrix in decode_matrices
            )
            layer.linear_attn.conv1d = nn.Identity()
            compiled_blocks.append(
                _make_linear_attention_block(
                    layer,
                    int(cfg.num_experts_per_tok),
                    stick_size,
                    tp_group_name,
                )
            )

    model._spyre_compiled_blocks = compiled_blocks
    model._spyre_compiled_norm = torch.compile(
        lambda hidden_states: _rms_norm(hidden_states, backbone.norm), dynamic=False
    )
