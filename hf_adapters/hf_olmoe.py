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

"""Spyre adapter for OLMoE causal language models.

Prefill routes tokens before evaluating every expert; single-token decode
instead gathers only the selected experts. Both paths use the same persistent,
device-resident expert weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import hf_adapters.hf_common as hf_common
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    optional_spyre_config_patch,
    pad_lm_head,
    prepare_rope_and_heads,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward
_MOE_TILE = 32


def _name_prefill_inputs(x, gate, up, down):
    if x.device.type != "spyre":
        return

    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    tokens = x.shape[0] * x.shape[1]
    experts, hidden, intermediate = gate.shape
    for name, extent in (
        ("E", experts),
        ("T", tokens),
        ("H", hidden),
        ("M", intermediate),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)
    name_tensor_dims(x, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])


def _reset_named_dims(x):
    if x.device.type != "spyre":
        return

    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _router_topk(x, weight, top_k, norm_topk_prob):
    router_logits = F.linear(x, weight)
    softmax_dtype = None if x.device.type == "spyre" else torch.float32
    probs = torch.softmax(router_logits, dtype=softmax_dtype, dim=-1)
    tokens = probs.shape[0]
    topk_input = probs.expand(2, -1).contiguous() if tokens == 1 else probs
    weights, expert_indices = torch.topk(topk_input, top_k, dim=-1)
    weights = weights[:tokens]
    expert_indices = expert_indices[:tokens]
    if norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    return (
        probs.to(router_logits.dtype),
        weights.to(router_logits.dtype),
        expert_indices,
    )


def _moe_decode(
    x,
    router_weight,
    gate_dev,
    up_dev,
    down_dev,
    top_k,
    norm_topk_prob,
    tile,
    stick_size,
):
    """Run the selected-expert decode FFN and combine outputs on device."""
    T, H = x.shape
    _, weights, expert_indices = _router_topk(x, router_weight, top_k, norm_topk_prob)

    if x.device.type == "spyre":
        from torch_spyre._inductor.propagate_hints import spyre_hint

        # Widen topk's fp16 indices onto a stick before converting them to the
        # device's int32 gather indices. The layout pass inserts the restickify.
        index_stick = (
            expert_indices[..., None].expand(T, top_k, stick_size).contiguous()
        )
        index_stick = index_stick.to(torch.float32)
        index_address = index_stick[..., : stick_size // 2].to(torch.int32)
        expert_indices = index_address[..., 0]
        hint = spyre_hint(tiles={"row": tile})
    else:
        from contextlib import nullcontext

        hint = nullcontext()

    with hint:
        rows = T * top_k
        intermediate = gate_dev.shape[-1]
        inputs = x[:, None, :].expand(T, top_k, H).contiguous().reshape(rows, 1, H)
        gate = gate_dev[expert_indices].reshape(rows, H, intermediate)
        up = up_dev[expert_indices].reshape(rows, H, intermediate)
        down = down_dev[expert_indices].reshape(rows, intermediate, H)

        gate_out = torch.bmm(inputs, gate)
        up_out = torch.bmm(inputs, up)
        expert_out = torch.bmm(F.silu(gate_out) * up_out, down).reshape(T, top_k, H)
        return (expert_out * weights[..., None]).sum(dim=1)


def _moe_prefill_route_packed(
    x,
    router_weight,
    top_k,
    norm_topk_prob,
    stick_size,
    route_identity,
):
    """Compute packed prefill routing weights."""
    probs, weights, selected = _router_topk(x, router_weight, top_k, norm_topk_prob)
    if x.device.type == "spyre":
        routing = torch.ops.spyre.keep_by_index(probs, selected, -1, 0.0)
        if norm_topk_prob:
            routing = routing / routing.sum(-1, keepdim=True)
        routing = routing.to(x.dtype)
    else:
        routing = x.new_zeros(probs.shape).scatter(-1, selected, weights)

    # ReLU materializes the expansion; the identity BMM puts it on a stick.
    packed = torch.relu(routing.unsqueeze(-1).expand(-1, -1, stick_size))
    return packed @ route_identity


def _moe_prefill_experts(x, routing_weight, gate, up, down):
    """Evaluate every expert and sum their routed outputs."""
    if x.device.type == "spyre":
        from torch_spyre._inductor.propagate_hints import spyre_hint

        experts = gate.shape[0]
        with spyre_hint(named_dims=["E", "T", "ONE"]):
            route = routing_weight.permute(1, 0, 2).contiguous().clone()
        with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
            gate_out = torch.matmul(x.unsqueeze(0), gate)
            up_out = torch.matmul(x.unsqueeze(0), up)
            down_out = torch.matmul(F.silu(gate_out) * up_out, down)
            return (down_out * route).sum(dim=0)

    gate_out = torch.matmul(x.unsqueeze(0), gate)
    up_out = torch.matmul(x.unsqueeze(0), up)
    down_out = torch.matmul(F.silu(gate_out) * up_out, down)
    route = routing_weight.permute(1, 0, 2)
    return (down_out * route).sum(dim=0)


class OlmoeAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.head_dim = attention.head_dim
        self.scaling = attention.scaling

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states))
        k = self.k_norm(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache, cache_index
        )
        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), key_cache, value_cache


class OlmoeMoEBlock(nn.Module):
    def __init__(self, layer, top_k, norm_topk_prob, stick_size):
        super().__init__()
        self.self_attn = OlmoeAttention(layer.self_attn)
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.gate = layer.mlp.gate
        self.experts = layer.mlp.experts
        self._top_k = top_k
        self._norm_topk_prob = norm_topk_prob
        self._stick_size = stick_size
        self._compiled_decode = torch.compile(
            self._full_decode_forward, dynamic=False, fullgraph=True
        )
        self._compiled_prefill_attn = torch.compile(
            self._attn_forward, dynamic=False, fullgraph=True
        )
        self._compiled_prefill_ffn = torch.compile(
            self._prefill_ffn, dynamic=False, fullgraph=True
        )
        self.train(layer.training)

    def _attn_forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        return residual + attn_out, key_cache, value_cache

    def _decode_ffn(self, residual):
        hidden_size = residual.shape[-1]
        x = self.post_attention_layernorm(residual).reshape(-1, hidden_size)
        experts = self.experts
        moe_out = _moe_decode(
            x,
            self.gate.weight,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
            self._top_k,
            self._norm_topk_prob,
            _MOE_TILE,
            self._stick_size,
        )
        return residual + moe_out.to(residual.dtype).reshape_as(residual)

    def _full_decode_forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        hidden_states, key_cache, value_cache = self._attn_forward(
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        return self._decode_ffn(hidden_states), key_cache, value_cache

    def _prefill_ffn(self, residual):
        hidden_size = residual.shape[-1]
        x = self.post_attention_layernorm(residual).reshape(-1, hidden_size)
        routing_weight = _moe_prefill_route_packed(
            x,
            self.gate.weight,
            self._top_k,
            self._norm_topk_prob,
            self._stick_size,
            self.gate.route_identity,
        )[..., :1]
        experts = self.experts
        moe_out = _moe_prefill_experts(
            x,
            routing_weight,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
        )
        return residual + moe_out.to(residual.dtype).reshape_as(residual)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        if hidden_states.shape[1] > 1:
            hidden_states, key_cache, value_cache = self._compiled_prefill_attn(
                hidden_states,
                selected_freqs,
                attn_mask,
                key_cache,
                value_cache,
                cache_index,
            )
            experts = self.experts
            try:
                _name_prefill_inputs(
                    hidden_states,
                    experts.gate_proj,
                    experts.up_proj,
                    experts.down_proj,
                )
                with optional_spyre_config_patch(
                    {"allow_all_ops_in_lx_planning": True}
                ):
                    hidden_states = self._compiled_prefill_ffn(hidden_states)
            finally:
                _reset_named_dims(hidden_states)
        else:
            hidden_states, key_cache, value_cache = self._compiled_decode(
                hidden_states,
                selected_freqs,
                attn_mask,
                key_cache,
                value_cache,
                cache_index,
            )
        return hidden_states, key_cache, value_cache


def _move_expert_weight(weight):
    if not str(hf_common.DEVICE).startswith("spyre"):
        return weight

    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.to(hf_common.DEVICE)


def _prepare_experts(experts):
    gate_up = experts.gate_up_proj.detach()
    del experts.gate_up_proj

    intermediate_size = gate_up.shape[1] // 2
    gate = gate_up[:, :intermediate_size].transpose(1, 2).contiguous()
    experts.gate_proj = _move_expert_weight(gate)
    del gate

    up = gate_up[:, intermediate_size:].transpose(1, 2).contiguous()
    experts.up_proj = _move_expert_weight(up)
    del up
    del gate_up

    down = experts.down_proj.detach().transpose(1, 2).contiguous()
    del experts.down_proj
    experts.down_proj = _move_expert_weight(down)


def prepare_for_spyre(model):
    """Prepare an OLMoE causal LM for Spyre in place."""
    backbone = get_backbone(model)
    cfg = model.config
    head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    if cfg.hidden_act != "silu":
        raise SpyreUnsupportedFeatureError(
            f"OLMoE activation {cfg.hidden_act!r} is not supported; expected 'silu'"
        )
    if cfg.clip_qkv is not None:
        raise SpyreUnsupportedFeatureError("OLMoE clip_qkv is not supported")
    if getattr(cfg, "sliding_window", None) is not None:
        raise SpyreUnsupportedFeatureError(
            "OLMoE sliding-window attention is not supported"
        )
    if head_dim % (2 * BLOCK_SIZE) != 0:
        raise SpyreUnsupportedModelError(
            "OLMoE head padding is not supported because Q/K RMSNorm spans the "
            f"native projections; head_dim must be a multiple of {2 * BLOCK_SIZE}, "
            f"got head_dim={head_dim}"
        )

    prepare_rope_and_heads(model)
    pad_lm_head(model)

    try:
        from torch_spyre._C import get_elem_in_stick

        stick_size = get_elem_in_stick(next(model.parameters()).dtype)
    except ImportError:
        stick_size = BLOCK_SIZE

    blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        block = OlmoeMoEBlock(
            layer,
            int(cfg.num_experts_per_tok),
            bool(cfg.norm_topk_prob),
            stick_size,
        )
        route_dtype = block.gate.weight.dtype
        block.gate.route_identity = torch.eye(stick_size, dtype=route_dtype).to(
            hf_common.DEVICE
        )
        _prepare_experts(block.experts)
        backbone.layers[i] = block
        blocks.append(block)

    model._spyre_compiled_blocks = blocks
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
