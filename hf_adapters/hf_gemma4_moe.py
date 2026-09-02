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

"""Spyre adapter for the sparse Gemma 4 MoE causal LM.

The attention path comes from :mod:`hf_gemma4`. Prefill routes tokens before
evaluating every expert; single-token decode gathers only the selected experts.
Both paths share one device-resident expert-weight set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import optional_spyre_config_patch, text_config
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _gemma4_rms_norm,
    _run_backbone_forward,
    _run_forward,
    _setup_gemma4_text_decoder,
)

__all__ = ["prepare_for_spyre", "_run_forward", "_run_backbone_forward"]

_MOE_TILE = 32  # Decode gather requires tiles with at least two rows.


def _name_prefill_inputs(x, gate, up, down):
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


def _reset_named_dims():
    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _router_probs(x, weight, scale, root_size, eps):
    x = _gemma4_rms_norm(x, None, eps)
    return torch.softmax(F.linear(x * scale * root_size, weight), dim=-1)


def _topk(probs, top_k):
    tokens = probs.shape[0]
    topk_input = probs.expand(2, -1).contiguous() if tokens == 1 else probs
    weights, expert_indices = torch.topk(topk_input, top_k, dim=-1)
    return weights[:tokens], expert_indices[:tokens]


def _compiled_moe_loop_region(
    x_router,
    x_expert,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    gate_dev,
    up_dev,
    down_dev,
    top_k,
    tile,
    stick_size,
    eps,
):
    """Run the routed decode FFN and combine its expert outputs on device."""
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape
    probs = _router_probs(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        eps,
    )
    weights, expert_indices = _topk(probs, top_k)
    weights = weights / weights.sum(-1, keepdim=True)

    # Widen topk's fp16 indices onto a stick before converting them to the
    # device's int32 gather indices. The layout pass inserts the restickify.
    index_stick = expert_indices[..., None].expand(T, top_k, stick_size).contiguous()
    index_stick = index_stick.to(torch.float32)
    index_address = index_stick[..., : stick_size // 2].to(torch.int32)
    expert_indices = index_address[..., 0]

    with spyre_hint(tiles={"row": tile}):
        rows = T * top_k
        intermediate = gate_dev.shape[-1]
        inputs = (
            x_expert[:, None, :].expand(T, top_k, H).contiguous().reshape(rows, 1, H)
        )
        gate = gate_dev[expert_indices].reshape(rows, H, intermediate)
        up = up_dev[expert_indices].reshape(rows, H, intermediate)
        down = down_dev[expert_indices].reshape(rows, intermediate, H)

        gate_out = torch.bmm(inputs, gate)
        up_out = torch.bmm(inputs, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        expert_out = torch.bmm(activated, down).reshape(T, top_k, H)

        # Scale on the H-carrying tensor because bare [T,K] products have no
        # legal layout. The widened source gives the gather a physical stick.
        expert_scale = per_expert_scale[expert_indices][..., :1]
        expert_out = expert_out * weights[..., None] * expert_scale
        return expert_out.sum(dim=1)


def _moe_route_persistent_packed(
    x_router,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    top_k,
    stick_size,
    eps,
    route_identity,
):
    """Compute packed prefill routing weights on device."""
    probs = _router_probs(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        eps,
    )
    _, selected = _topk(probs, top_k)
    weights = torch.ops.spyre.keep_by_index(probs, selected, -1, 0.0)
    weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * per_expert_scale

    # ReLU materializes the expansion; the identity BMM puts it on a stick.
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, stick_size))
    return packed @ route_identity


def _moe_expert_persistent(x_expert, routing_weight, gate, up, down):
    """Evaluate every expert and sum their routed outputs on device."""
    from torch_spyre._inductor.propagate_hints import spyre_hint

    experts, hidden, intermediate = gate.shape

    x = x_expert.unsqueeze(0)
    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = routing_weight.permute(1, 0, 2).contiguous().clone()

    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
        gate_out = torch.matmul(x, gate)
        up_out = torch.matmul(x, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        down_out = torch.matmul(activated, down)
        return (down_out * route).sum(dim=0)


class Gemma4MoEBlock(nn.Module):
    """Gemma 4 decoder block with parallel dense and sparse FFNs."""

    def __init__(
        self,
        layer,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        moe_k,
        stick_size,
    ):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.experts = layer.experts
        self.router = layer.router
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
        self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        self._moe_k = moe_k
        self._stick_size = stick_size
        self._moe_rms_eps = self.router.eps
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
        hidden_states = _gemma4_rms_norm(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.eps,
        )
        attn_out, key_cache, value_cache = self.self_attn(
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        hidden_states = residual + _gemma4_rms_norm(
            attn_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return hidden_states, key_cache, value_cache

    def _dense_forward(self, residual):
        dense_input = _gemma4_rms_norm(
            residual,
            self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps,
        )
        return _gemma4_rms_norm(
            self.mlp(dense_input),
            self.post_feedforward_layernorm_1.weight,
            self.post_feedforward_layernorm_1.eps,
        )

    def _full_decode_forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
    ):
        hidden_states, key_cache, value_cache = self._attn_forward(
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        return self._decode_ffn(hidden_states, layer_scalar), key_cache, value_cache

    def _decode_ffn(self, residual, layer_scalar):
        hidden_size = residual.shape[-1]
        dense_out = self._dense_forward(residual)
        router_input = residual.reshape(-1, hidden_size)
        expert_input = _gemma4_rms_norm(
            router_input,
            self.pre_feedforward_layernorm_2.weight,
            self.pre_feedforward_layernorm_2.eps,
        )
        experts = self.experts
        router = self.router
        moe_out = _compiled_moe_loop_region(
            router_input,
            expert_input,
            router.proj.weight,
            router.scale,
            router.scalar_root_size,
            router.per_expert_scale_stick,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
            self._moe_k,
            _MOE_TILE,
            self._stick_size,
            self._moe_rms_eps,
        )
        moe_out = moe_out.to(expert_input.dtype).reshape_as(residual)
        moe_out = _gemma4_rms_norm(
            moe_out,
            self.post_feedforward_layernorm_2.weight,
            self.post_feedforward_layernorm_2.eps,
        )
        ffn_out = _gemma4_rms_norm(
            dense_out + moe_out,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
        )
        return (residual + ffn_out) * layer_scalar

    def _prefill_ffn(self, residual, layer_scalar):
        router_input = residual.reshape(-1, residual.shape[-1])
        dense_out = self._dense_forward(residual)
        expert_input = _gemma4_rms_norm(
            router_input,
            self.pre_feedforward_layernorm_2.weight,
            self.pre_feedforward_layernorm_2.eps,
        )
        router = self.router
        routing_weight = _moe_route_persistent_packed(
            router_input,
            router.proj.weight,
            router.scale,
            router.scalar_root_size,
            router.per_expert_scale,
            self._moe_k,
            self._stick_size,
            self._moe_rms_eps,
            router.route_identity,
        )[..., :1]
        experts = self.experts
        moe_out = _moe_expert_persistent(
            expert_input,
            routing_weight,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
        )
        moe_out = _gemma4_rms_norm(
            moe_out.to(residual.dtype).reshape_as(residual),
            self.post_feedforward_layernorm_2.weight,
            self.post_feedforward_layernorm_2.eps,
        )
        ffn_input = dense_out + moe_out
        ffn_out = _gemma4_rms_norm(
            ffn_input,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
        )
        return (residual + ffn_out) * layer_scalar

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
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
            _name_prefill_inputs(
                hidden_states,
                experts.gate_proj,
                experts.up_proj,
                experts.down_proj,
            )
            with optional_spyre_config_patch({"allow_all_ops_in_lx_planning": True}):
                hidden_states = self._compiled_prefill_ffn(hidden_states, layer_scalar)
            _reset_named_dims()
        else:
            hidden_states, key_cache, value_cache = self._compiled_decode(
                hidden_states,
                selected_freqs,
                attn_mask,
                key_cache,
                value_cache,
                cache_index,
                layer_scalar,
            )

        return hidden_states, key_cache, value_cache


def _move_expert_weight(weight):
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.to("spyre")


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
    """Prepare a Gemma 4 MoE causal LM for Spyre in place."""
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.model_utils import dma_moe_per_expert_scale_to_spyre

    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)
    stick_size = get_elem_in_stick(torch.float16)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    moe_k = int(cfg.top_k_experts)
    num_q_heads, kv_shapes, kv_equals_v = _setup_gemma4_text_decoder(
        model, allow_moe=True
    )

    blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        block = Gemma4MoEBlock(
            layer,
            num_q_heads[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            kv_equals_v[i],
            moe_k,
            stick_size,
        )
        expert_scale = block.router.per_expert_scale.detach()
        block.router.route_identity = torch.eye(
            stick_size, dtype=expert_scale.dtype
        ).to("spyre")
        block.router.per_expert_scale_stick = dma_moe_per_expert_scale_to_spyre(
            expert_scale
        )
        _prepare_experts(block.experts)
        backbone.layers[i] = block
        blocks.append(block)

    model._spyre_compiled_blocks = blocks
