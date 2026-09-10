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

from contextlib import nullcontext

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

# Automatic for supported inputs and compilers. None selects automatically,
# False is a comparison opt-out, True requires support before cache writes.
# Requires the LX stack plus reader-compatible staging. The recorded performance
# also used rewrite-preserved weight-copy proofs; those are not a safety gate.
_PREFILL_EXPERT_DIVISIONS = None


def _prefill_expert_config():
    """Check supported compiler controls, scoped by the caller."""
    options = {"allow_all_ops_in_lx_planning": True}
    if _PREFILL_EXPERT_DIVISIONS is False:
        return options
    from torch_spyre._inductor import config

    if not hasattr(config, "consumer_compatible_input_staging"):
        if _PREFILL_EXPERT_DIVISIONS is True:
            raise RuntimeError(
                "Prefill divisions require reader-compatible input staging"
            )
        return options
    # This checks the control, not which rewrites preserve elision candidates.
    # Missing preservation keeps the copy; it loses performance, not safety.
    if not hasattr(config, "read_copy_elision"):
        if _PREFILL_EXPERT_DIVISIONS is True:
            raise RuntimeError("Read-copy elision is not available")
        return options
    if not hasattr(config, "lx_planner_relayout"):
        if _PREFILL_EXPERT_DIVISIONS is True:
            raise RuntimeError("Prefill divisions require LX relayout support")
        return options
    if (
        config.sencores != 32
        or config.layout_solver != "greedy"
        or config.co_optimizing_lx_planning
        or config.ktir_emitter
        or config.ignore_work_division_hints
        or config.ignore_wsr_hints
        or not config.lx_planning
    ):
        if _PREFILL_EXPERT_DIVISIONS is True:
            raise RuntimeError(
                "Prefill divisions require 32 cores, greedy LX planning, "
                "honored hints and the SDSC path"
            )
        return options
    options.update(
        lx_planner_relayout=True,
        consumer_compatible_input_staging=True,
        read_copy_elision=True,
    )
    return options


def _validate_prefill_expert_inputs(x, gate, up, down, routing_weight=None):
    """Select supported inputs; reject unsupported explicit requests early.

    Forward sees [batch, tokens, hidden]; the expert region sees its flattened
    [rows, hidden] form. Check both without copying or reshaping device data.
    Routing is produced later, so its shape is checked again inside the region.
    """
    if _PREFILL_EXPERT_DIVISIONS is False:
        return False
    shape = tuple(x.shape)
    input_ok = shape == (512, 2816) or (
        routing_weight is None
        and len(shape) == 3
        and shape[0] * shape[1] == 512
        and shape[2] == 2816
    )
    if (
        not input_ok
        or tuple(gate.shape) != (128, 2816, 704)
        or tuple(up.shape) != tuple(gate.shape)
        or tuple(down.shape) != (128, 704, 2816)
        # Both host 16-bit formats map to SEN169_FP16 on Spyre. The production
        # checkpoint uses bfloat16, unlike the isolated float16 microbenchmark.
        or x.dtype not in (torch.float16, torch.bfloat16)
        or any(t.dtype != x.dtype for t in (gate, up, down))
        or (
            routing_weight is not None
            and (
                tuple(routing_weight.shape) != (512, 128, 1)
                or routing_weight.dtype != x.dtype
            )
        )
    ):
        if _PREFILL_EXPERT_DIVISIONS is True:
            raise ValueError(
                "Prefill divisions require matching FP16/BF16 E128/T512/H2816/F704"
            )
        return False
    return True


# Enabled automatically for supported decode calls. The route assignment and
# the compiler's indexed-selection layout are a measured pair: either alone
# regressed this workload. Keep the compiler option scoped to decode calls.
_DECODE_ROUTE_SCHEDULE = True


def _decode_route_schedule_enabled(tokens, top_k):
    """Pair R8 with its compiler capability; other shapes use ordinary decode."""
    if not _DECODE_ROUTE_SCHEDULE or tokens != 1 or top_k != 8:
        return False
    from torch_spyre._inductor import config

    return (
        hasattr(config, "indexed_selection_consumer_layout")
        and config.sencores == 32
        and not config.ignore_work_division_hints
        and not config.ignore_wsr_hints
    )


# Independent of gate/up reduction blocking. Enabled at width 1024;
# smaller widths and intermediate-retention experiments are not shipped.
_DECODE_DOWN_OUTPUT_PANEL = 1024


def _decode_down_output_blocks(activated, down_bank, expert_indices, block_size):
    """Select output-column blocks before indexing the original expert bank.

    Every output still sums its full reduction dimension in one BMM. Concatenate
    columns in order; never concatenate weights into a full selected slab.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    rows, _, intermediate = activated.shape
    hidden = down_bank.shape[-1]
    outputs = []
    for start in range(0, hidden, block_size):
        width = min(block_size, hidden - start)
        selected = down_bank[:, :, start : start + width][expert_indices].reshape(
            rows, intermediate, width
        )
        # The indexed load keeps data columns unsplit on this compiler. H:4
        # needs a proven distributed load or explicit transfer, not a new hint.
        with spyre_hint(named_dims=["R", "ONE", "H"], work_div={"R": rows, "H": 1}):
            outputs.append(torch.bmm(activated, selected))
    return torch.cat(outputs, dim=-1)


def _decode_down_panel(hidden, intermediate, dtypes, route_schedule):
    """Choose the measured block width only for its supported decode shape."""
    if _DECODE_DOWN_OUTPUT_PANEL not in (None, 1024):
        raise ValueError("Unsupported decode block width; expected 1024 or None")
    if (
        route_schedule
        and hidden == 2816
        and intermediate == 704
        # Both host formats use SEN169_FP16 device arithmetic/storage.
        # Gemma's checkpoint uses bfloat16; float32 is a different device path.
        and dtypes[0] in (torch.float16, torch.bfloat16)
        and all(dtype == dtypes[0] for dtype in dtypes)
    ):
        return _DECODE_DOWN_OUTPUT_PANEL
    return None


# Enabled for supported decode: four 704-term partial sums replace a 2816-term dot product.
# This changes addition grouping and requires separate numerical acceptance.
_DECODE_GATE_UP_K_PANEL = 704


def _decode_gate_up_blocks(inputs, gate_bank, up_bank, expert_indices, block_size):
    """Load and consume one gate/up block at a time from unchanged banks."""
    from torch_spyre._inductor.propagate_hints import spyre_hint

    rows, _, hidden = inputs.shape
    intermediate = gate_bank.shape[-1]
    gate_out = up_out = None
    for start in range(0, hidden, block_size):
        width = min(block_size, hidden - start)
        x_slice = inputs[:, :, start : start + width]
        # Keep the measured gate-BMM-add, then up-BMM-add order so the two
        # large selected panels need not be live together. Add in start order.
        gate_panel = gate_bank[:, start : start + width, :][expert_indices].reshape(
            rows, width, intermediate
        )
        with spyre_hint(named_dims=["R", "ONE", "F"], work_div={"R": 8}):
            gate_part = torch.bmm(x_slice, gate_panel)
        gate_out = gate_part if gate_out is None else gate_out + gate_part
        up_panel = up_bank[:, start : start + width, :][expert_indices].reshape(
            rows, width, intermediate
        )
        with spyre_hint(named_dims=["R", "ONE", "F"], work_div={"R": 8}):
            up_part = torch.bmm(x_slice, up_panel)
        up_out = up_part if up_out is None else up_out + up_part
    return gate_out, up_out


def _decode_gate_up_panel(hidden, intermediate, dtypes, route_schedule):
    """Choose the measured block width only for its supported decode shape."""
    if _DECODE_GATE_UP_K_PANEL not in (None, 704):
        raise ValueError("Unsupported decode block width; expected 704 or None")
    if (
        route_schedule
        and hidden == 2816
        and intermediate == 704
        # Both host formats use SEN169_FP16 device arithmetic/storage.
        # Gemma's checkpoint uses bfloat16; float32 is a different device path.
        and dtypes[0] in (torch.float16, torch.bfloat16)
        and all(dtype == dtypes[0] for dtype in dtypes)
    ):
        return _DECODE_GATE_UP_K_PANEL
    return None


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
    route_schedule = _decode_route_schedule_enabled(T, top_k)
    down_panel = _decode_down_panel(
        H,
        gate_dev.shape[-1],
        (x_expert.dtype, gate_dev.dtype, up_dev.dtype, down_dev.dtype),
        route_schedule,
    )
    gate_up_panel = _decode_gate_up_panel(
        H,
        gate_dev.shape[-1],
        (x_expert.dtype, gate_dev.dtype, up_dev.dtype, down_dev.dtype),
        route_schedule,
    )
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
        if down_panel is None:
            down = down_dev[expert_indices].reshape(rows, intermediate, H)
        if gate_up_panel is not None:
            gate_out, up_out = _decode_gate_up_blocks(
                inputs, gate_dev, up_dev, expert_indices, gate_up_panel
            )
        else:
            gate = gate_dev[expert_indices].reshape(rows, H, intermediate)
            up = up_dev[expert_indices].reshape(rows, H, intermediate)
            if route_schedule:
                with spyre_hint(named_dims=["R", "ONE", "F"], work_div={"R": 8}):
                    gate_out = torch.bmm(inputs, gate)
                    up_out = torch.bmm(inputs, up)
            else:
                gate_out = torch.bmm(inputs, gate)
                up_out = torch.bmm(inputs, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        if down_panel is not None:
            expert_out = _decode_down_output_blocks(
                activated, down_dev, expert_indices, down_panel
            )
        else:
            expert_out = torch.bmm(activated, down)
        expert_out = expert_out.reshape(T, top_k, H)

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

    use_divisions = _validate_prefill_expert_inputs(
        x_expert, gate, up, down, routing_weight
    ) and _prefill_expert_config().get("consumer_compatible_input_staging", False)

    x = x_expert.unsqueeze(0)
    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = routing_weight.permute(1, 0, 2).contiguous().clone()

    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
        with spyre_hint(work_div={"T": 8, "H": 4}) if use_divisions else nullcontext():
            gate_out = torch.matmul(x, gate)
            up_out = torch.matmul(x, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        with spyre_hint(work_div={"T": 16, "H": 2}) if use_divisions else nullcontext():
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
        # Check explicit opt-in requirements before attention/cache mutation.
        prefill_options = (
            _prefill_expert_config() if hidden_states.shape[1] > 1 else None
        )
        if hidden_states.shape[1] > 1:
            experts = self.experts
            if not _validate_prefill_expert_inputs(
                hidden_states, experts.gate_proj, experts.up_proj, experts.down_proj
            ):
                prefill_options = {"allow_all_ops_in_lx_planning": True}
            hidden_states, key_cache, value_cache = self._compiled_prefill_attn(
                hidden_states,
                selected_freqs,
                attn_mask,
                key_cache,
                value_cache,
                cache_index,
            )
            try:
                _name_prefill_inputs(
                    hidden_states,
                    experts.gate_proj,
                    experts.up_proj,
                    experts.down_proj,
                )
                with optional_spyre_config_patch(prefill_options):
                    hidden_states = self._compiled_prefill_ffn(
                        hidden_states, layer_scalar
                    )
            finally:
                _reset_named_dims()
        else:
            # Do not enable the slower route-only configuration on an older
            # compiler. Both the wrapper and region use this same eligibility.
            decode_config = (
                optional_spyre_config_patch({"indexed_selection_consumer_layout": True})
                if _decode_route_schedule_enabled(
                    hidden_states.shape[0] * hidden_states.shape[1], self._moe_k
                )
                else nullcontext()
            )
            with decode_config:
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
