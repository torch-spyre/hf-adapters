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
    apply_rope_matmul,
    kv_cache_update,
    moe_decode_selected_experts,
    moe_prefill_all_experts,
    named_moe_prefill_inputs,
    optional_spyre_config_patch,
    prepare_moe_expert_weights,
    text_config,
)
from hf_adapters.hf_qwen3_5 import (
    _make_linear_attention_block as _make_dense_linear_attention_block,
)
from hf_adapters.hf_qwen3_5 import (
    _prepare_attention_projections,
    _prepare_linear_attention_constants,
    _rms_norm,
    _setup_qwen3_5_text_decoder,
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
    # Keep every routed expert index exact during CPU-to-Spyre transport. Model
    # dtypes such as bfloat16 cannot represent every integer above 256.
    expert_indices = (
        expert_indices.to(torch.float32)[..., None]
        .expand(-1, -1, stick_size)
        .contiguous()
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
    shared = _shared_expert(x, mlp)
    routed = _reduce_expert_output_fp32(routed, tp_group_name)
    output = routed + shared
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
        experts = mlp.experts
        with named_moe_prefill_inputs(
            x, experts.gate_proj, experts.up_proj, experts.down_proj
        ):
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
        experts = mlp.experts
        with named_moe_prefill_inputs(
            x, experts.gate_proj, experts.up_proj, experts.down_proj
        ):
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
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError(
            "num_experts_per_tok must be positive and no greater than num_experts"
        )

    cfg, backbone, rope_permutation = _setup_qwen3_5_text_decoder(model)
    # Keep the small router projections on CPU. Each layer crosses an explicit
    # graph boundary for exact fp32-softmax/top-k routing, then sends only the
    # selected indices and normalized weights back to Spyre.
    model._spyre_cpu_submodules = [
        name for name, _ in model.named_modules() if name.endswith(".mlp.gate")
    ]

    try:
        from torch_spyre._C import get_elem_in_stick

        stick_size = get_elem_in_stick(next(model.parameters()).dtype)
    except ImportError:
        stick_size = BLOCK_SIZE

    compiled_blocks = []
    for layer_type, layer in zip(cfg.layer_types, backbone.layers):
        mlp = layer.mlp
        expert_mesh = getattr(mlp.experts, "_hf_device_mesh", None)
        tp_group_name = (
            expert_mesh.get_group().group_name if expert_mesh is not None else None
        )
        prepare_moe_expert_weights(mlp.experts)

        if layer_type == "full_attention":
            query_projection, gate_projection = _prepare_attention_projections(
                model, layer, cfg.head_dim, rope_permutation
            )
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
            _prepare_linear_attention_constants(layer.linear_attn)
            compiled_blocks.append(
                _make_linear_attention_block(
                    layer,
                    int(cfg.num_experts_per_tok),
                    stick_size,
                    tp_group_name,
                )
            )

    model._spyre_compiled_blocks = compiled_blocks
