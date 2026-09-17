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

from hf_adapters.hf_common import (
    moe_decode_selected_experts,
    moe_prefill_all_experts,
    moe_topk,
    named_moe_prefill_inputs,
    optional_spyre_config_patch,
    prepare_moe_expert_weights,
    text_config,
)
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _gemma4_rms_norm,
    _run_backbone_forward,
    _run_forward,
    _setup_gemma4_text_decoder,
    spyre_tp_grouped_colwise_modules,
)
from hf_adapters.spyre_tensor_parallel import spyre_compiled_all_reduce

# The HF checkpoint conversion first merges each expert's 2D gate/up tensors
# into these 3D parameters.  Keep the local TP shards on CPU until
# ``_prepare_experts`` performs its split/transpose/pad and one final custom
# DMA, avoiding a Spyre -> CPU -> Spyre round trip.
SPYRE_TP_CPU_STAGED_MODULES = {
    "model.language_model.layers.*.experts.gate_up_proj",
    "model.language_model.layers.*.experts.down_proj",
}

__all__ = [
    "prepare_for_spyre",
    "prepare_text_decoder_for_spyre",
    "_run_forward",
    "_run_backbone_forward",
    "spyre_tp_grouped_colwise_modules",
]

_MOE_TILE = 32  # Decode gather requires tiles with at least two rows.


def _router_probs(x, weight, scale, root_size, eps):
    x = _gemma4_rms_norm(x, None, eps)
    return torch.softmax(F.linear(x * scale * root_size, weight), dim=-1)


def _compiled_moe_loop_region(
    x_router,
    x_expert,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale_stick,
    gate_dev,
    up_dev,
    down_dev,
    top_k,
    tile,
    stick_size,
    eps,
):
    """Run the routed decode FFN and combine its expert outputs on device."""
    probs = _router_probs(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        eps,
    )
    weights, expert_indices = moe_topk(probs, top_k)
    weights = weights / weights.sum(-1, keepdim=True)
    return moe_decode_selected_experts(
        x_expert,
        weights,
        expert_indices,
        gate_dev,
        up_dev,
        down_dev,
        top_k,
        tile,
        stick_size,
        "gelu_tanh",
        per_expert_scale_stick=per_expert_scale_stick,
    )


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
    _, selected = moe_topk(probs, top_k)
    weights = torch.ops.spyre.keep_by_index(probs, selected, -1, 0.0)
    weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * per_expert_scale

    # ReLU materializes the expansion; the identity BMM puts it on a stick.
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, stick_size))
    return packed @ route_identity


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
        self._tp_device_mesh = getattr(self.experts, "_hf_device_mesh", None)
        self._tp_group_name = (
            self._tp_device_mesh.get_group().group_name
            if self._tp_device_mesh is not None
            else None
        )
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
        decode_forward = (
            self._full_tp_decode_forward
            if self._tp_device_mesh is not None
            else self._full_decode_forward
        )
        prefill_ffn = (
            self._tp_prefill_ffn
            if self._tp_device_mesh is not None
            else self._prefill_ffn
        )
        self._compiled_decode = torch.compile(
            decode_forward, dynamic=False, fullgraph=True
        )
        self._compiled_prefill_attn = torch.compile(
            self._attn_forward, dynamic=False, fullgraph=True
        )
        self._compiled_prefill_ffn = torch.compile(
            prefill_ffn, dynamic=False, fullgraph=True
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
        dense_out, moe_out = self._decode_ffn_local(residual)
        return self._finish_ffn(residual, dense_out, moe_out, layer_scalar)

    def _tp_decode_ffn(self, residual, layer_scalar):
        dense_out, moe_out = self._decode_ffn_local(residual)
        moe_out = spyre_compiled_all_reduce(moe_out, self._tp_group_name)
        return self._finish_ffn(residual, dense_out, moe_out, layer_scalar)

    def _full_tp_decode_forward(
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
        hidden_states = self._tp_decode_ffn(hidden_states, layer_scalar)
        return hidden_states, key_cache, value_cache

    def _decode_ffn_local(self, residual):
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
        moe_out = moe_out.to(expert_input.dtype)
        return dense_out, moe_out

    def _finish_ffn(self, residual, dense_out, moe_out, layer_scalar):
        moe_out = _gemma4_rms_norm(
            moe_out.reshape_as(residual),
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
        dense_out, moe_out = self._prefill_ffn_local(residual)
        return self._finish_ffn(residual, dense_out, moe_out, layer_scalar)

    def _tp_prefill_ffn(self, residual, layer_scalar):
        dense_out, moe_out = self._prefill_ffn_local(residual)
        moe_out = spyre_compiled_all_reduce(moe_out, self._tp_group_name)
        return self._finish_ffn(residual, dense_out, moe_out, layer_scalar)

    def _prefill_ffn_local(self, residual):
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
        moe_out = moe_prefill_all_experts(
            expert_input,
            routing_weight,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
            "gelu_tanh",
        )
        moe_out = moe_out.to(residual.dtype)
        return dense_out, moe_out

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
            with named_moe_prefill_inputs(
                hidden_states,
                experts.gate_proj,
                experts.up_proj,
                experts.down_proj,
            ):
                with optional_spyre_config_patch(
                    {"allow_all_ops_in_lx_planning": True}
                ):
                    hidden_states = self._compiled_prefill_ffn(
                        hidden_states, layer_scalar
                    )
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


def prepare_text_decoder_for_spyre(model):
    """Prepare only the Gemma 4 MoE text decoder for Spyre in place."""
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
        # Under TP this parameter was loaded directly on Spyre.  Widening the
        # per-expert scalar to one stick is intentionally a host operation.
        expert_scale = block.router.per_expert_scale.detach().cpu()
        block.router.route_identity = torch.eye(
            stick_size, dtype=expert_scale.dtype
        ).to("spyre")
        block.router.per_expert_scale_stick = dma_moe_per_expert_scale_to_spyre(
            expert_scale
        )
        prepare_moe_expert_weights(block.experts, pad_to_multiple=stick_size)
        backbone.layers[i] = block
        blocks.append(block)

    model._spyre_compiled_blocks = blocks


def prepare_for_spyre(model):
    """Prepare a Gemma 4 MoE causal LM for Spyre in place."""
    prepare_text_decoder_for_spyre(model)
