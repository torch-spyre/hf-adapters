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

"""HuggingFace Transformers adapter for dense text-only Qwen3.5 models.

Qwen3.5 is a hybrid decoder.  Layers selected by ``config.layer_types`` use
one of gated full attention or stateful Gated DeltaNet linear attention.  The
latter has independent convolution and recurrent states, so this adapter uses
heterogeneous entries in the generic generation cache lists.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    allocate_kv_cache_tensor,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    permute_proj_for_rope,
    prepare_lm_head_for_spyre,
    rope_dim_permutation,
    run_lm_head,
    text_config,
)


def _rms_norm(x, norm):
    """Qwen3.5 RMSNorm, avoiding ``pow`` in compiled Spyre regions."""
    input_dtype = x.dtype
    x_float = x.float()
    x_float = x_float * torch.rsqrt(
        (x_float * x_float).mean(-1, keepdim=True) + norm.eps
    )
    return (x_float * (1.0 + norm.weight.float())).to(input_dtype)


def _gated_rms_norm(x, gate, norm):
    """Qwen3.5's norm-before-SiLU-gate output normalization."""
    input_dtype = x.dtype
    x_float = x.float()
    x_float = x_float * torch.rsqrt(
        (x_float * x_float).mean(-1, keepdim=True) + norm.variance_epsilon
    )
    # Preserve HF's cast boundary: the normalized value is rounded to the model
    # dtype before the learned weight and fp32 gate are applied.
    normalized = norm.weight * x_float.to(input_dtype)
    return (normalized * F.silu(gate.float())).to(input_dtype)


def _dense_mlp(x, mlp):
    return mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


def _split_gated_q_projection(attn, num_heads, head_dim):
    """Split per-head ``[query, gate]`` rows from Qwen3.5's doubled q_proj."""
    source = attn.q_proj
    hidden_size = source.weight.shape[1]
    source_weight = source.weight.detach().to("cpu")
    weights = source_weight.view(num_heads, 2, head_dim, hidden_size)

    query = nn.Linear(hidden_size, num_heads * head_dim, bias=source.bias is not None)
    gate = nn.Linear(hidden_size, num_heads * head_dim, bias=source.bias is not None)
    query.weight = nn.Parameter(
        weights[:, 0].contiguous().view(-1, hidden_size), requires_grad=False
    )
    gate.weight = nn.Parameter(
        weights[:, 1].contiguous().view(-1, hidden_size), requires_grad=False
    )
    if source.bias is not None:
        biases = source.bias.detach().to("cpu").view(num_heads, 2, head_dim)
        query.bias = nn.Parameter(
            biases[:, 0].contiguous().view(-1), requires_grad=False
        )
        gate.bias = nn.Parameter(
            biases[:, 1].contiguous().view(-1), requires_grad=False
        )
    return query, gate


def _make_attention_block(layer, q_proj, gate_proj, head_dim):
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

    def finish_forward(residual, attention_output, gate):
        batch_size, sequence_length = residual.shape[:2]
        h = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        h = residual + attn.o_proj(h * torch.sigmoid(gate))
        residual = h
        h = _rms_norm(h, post_attention_norm)
        return residual + _dense_mlp(h, mlp)

    # Keep separate graph caches for the fixed prefill and one-token decode shapes.
    return (
        torch.compile(project_forward, dynamic=False),
        torch.compile(project_forward, dynamic=False),
        torch.compile(update_cache, dynamic=False),
        torch.compile(update_cache, dynamic=False),
        torch.compile(attention_forward, dynamic=False),
        torch.compile(attention_forward, dynamic=False),
        torch.compile(finish_forward, dynamic=False),
        torch.compile(finish_forward, dynamic=False),
    )


def _causal_conv_prefill(x, conv_state, taps, shift_matrices):
    """Depthwise causal convolution over one fixed, stick-sized prefill chunk."""
    # The history boundary is stick-aligned. A single 128x64 matmul per lag avoids
    # the partial-stick slices and output-stack mutations rejected by Spyre layout
    # propagation, as well as adding matmuls with incompatible element arrangements.
    history = torch.cat((conv_state, x), dim=-1)
    output = x * taps[-1]
    for lag in range(len(taps) - 1):
        shifted = torch.matmul(history, shift_matrices[lag])
        output = output + shifted * taps[-2 - lag]
    return F.silu(output), x


def _causal_conv_decode(x, conv_state, taps, decode_matrices):
    """Update the convolution state using only stick-aligned operands."""
    x_stick = F.pad(x, (0, BLOCK_SIZE - x.shape[-1]))
    output = x_stick * taps[-1]
    for lag in range(1, len(taps)):
        output = output + (conv_state @ decode_matrices[lag - 1]) * taps[-1 - lag]
    new_state = conv_state @ decode_matrices[-2] + x_stick @ decode_matrices[-1]
    return F.silu(output[..., : x.shape[-1]]), new_state


def _delta_recurrence(query, key, value, a, beta, recurrent_state, a_log, dt_bias):
    """Run the stateful delta rule on CPU, outside Spyre compiled graphs."""
    output_device = value.device
    output_dtype = value.dtype
    query = query.to("cpu", torch.float32)
    key = key.to("cpu", torch.float32)
    value = value.to("cpu", torch.float32)
    a = a.to("cpu", torch.float32)
    decay = -a_log.to("cpu", torch.float32).exp() * F.softplus(
        a + dt_bias.to("cpu", torch.float32)
    )
    beta = beta.to("cpu", torch.float32)
    state = recurrent_state.to("cpu", torch.float32)
    batch_size, sequence_length, num_heads, key_dim = query.shape
    state = state.view(batch_size, num_heads, key_dim, value.shape[-1])
    query = query * (key_dim**-0.5)
    outputs = []
    for index in range(sequence_length):
        q_t = query[:, index]
        k_t = key[:, index]
        v_t = value[:, index]
        state = state * decay[:, index, :, None, None].exp()
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * beta[:, index, :, None]
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    return (
        torch.stack(outputs, dim=1).to(output_device, output_dtype),
        state.reshape_as(recurrent_state).to(
            recurrent_state.device, recurrent_state.dtype
        ),
    )


def _linear_projection(linear_attn, hidden_states):
    batch_size, sequence_length, _ = hidden_states.shape
    mixed_qkv = linear_attn.in_proj_qkv(hidden_states).transpose(1, 2)
    z = linear_attn.in_proj_z(hidden_states).view(
        batch_size, sequence_length, linear_attn.num_v_heads, linear_attn.head_v_dim
    )
    beta = torch.sigmoid(linear_attn.in_proj_b(hidden_states))
    a = linear_attn.in_proj_a(hidden_states)
    return mixed_qkv, z, beta, a


def _linear_split_inputs(linear_attn, mixed_qkv):
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
        dim=-1,
    )
    return query.clone(), key.clone(), value.clone()


def _linear_normalize_inputs(linear_attn, query, key, value):
    batch_size, sequence_length = query.shape[:2]
    query_heads = query.view(
        batch_size, sequence_length, linear_attn.num_k_heads, linear_attn.head_k_dim
    )
    key_heads = key.view(
        batch_size, sequence_length, linear_attn.num_k_heads, linear_attn.head_k_dim
    )
    query_scale = torch.rsqrt(
        (query_heads * query_heads).sum(dim=-1, keepdim=True) + 1e-6
    ).repeat_interleave(linear_attn.head_k_dim, dim=-1)
    key_scale = torch.rsqrt(
        (key_heads * key_heads).sum(dim=-1, keepdim=True) + 1e-6
    ).repeat_interleave(linear_attn.head_k_dim, dim=-1)
    query = (query * query_scale.flatten(2)).view_as(query_heads)
    key = (key * key_scale.flatten(2)).view_as(key_heads)
    value = value.view(
        batch_size, sequence_length, linear_attn.num_v_heads, linear_attn.head_v_dim
    )
    repeats = linear_attn.num_v_heads // linear_attn.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    return query, key, value


def _make_linear_attention_block(layer):
    linear_attn = layer.linear_attn
    input_norm = layer.input_layernorm
    post_attention_norm = layer.post_attention_layernorm
    mlp = layer.mlp

    def finish_norm(z, core_output):
        batch_size, sequence_length = z.shape[:2]
        return _gated_rms_norm(
            core_output.reshape(-1, linear_attn.head_v_dim),
            z.reshape(-1, linear_attn.head_v_dim),
            linear_attn.norm,
        ).view(batch_size, sequence_length, -1)

    def finish_projection(residual, core_output):
        return residual + linear_attn.out_proj(core_output)

    def split_inputs(conv_output):
        return _linear_split_inputs(linear_attn, conv_output)

    def normalize_inputs(query, key, value):
        return _linear_normalize_inputs(linear_attn, query, key, value)

    def ffn_forward(hidden_states):
        residual = hidden_states
        h = _rms_norm(hidden_states, post_attention_norm)
        return residual + _dense_mlp(h, mlp)

    def project_hidden(hidden_states, padding_mask):
        h = _rms_norm(hidden_states, input_norm)
        return h * padding_mask[:, :, None]

    def project_values(h):
        return _linear_projection(linear_attn, h)

    def conv_prefill(mixed_qkv, conv_state):
        conv_output, _ = _causal_conv_prefill(
            mixed_qkv,
            conv_state,
            linear_attn._spyre_conv_taps,
            linear_attn._spyre_conv_shift_matrices,
        )
        return conv_output

    def conv_decode(mixed_qkv, conv_state):
        output_device = mixed_qkv.device
        output_dtype = mixed_qkv.dtype
        output, new_state = _causal_conv_decode(
            mixed_qkv.to("cpu"),
            conv_state.to("cpu"),
            [tap.to("cpu") for tap in linear_attn._spyre_conv_taps],
            [matrix.to("cpu") for matrix in linear_attn._spyre_conv_decode_matrices],
        )
        return (
            output.to(output_device, output_dtype),
            new_state.to(output_device, output_dtype),
        )

    return (
        torch.compile(project_hidden, dynamic=False),
        torch.compile(project_hidden, dynamic=False),
        torch.compile(project_values, dynamic=False),
        torch.compile(project_values, dynamic=False),
        conv_prefill,
        conv_decode,
        split_inputs,
        split_inputs,
        normalize_inputs,
        normalize_inputs,
        torch.compile(finish_norm, dynamic=False),
        torch.compile(finish_norm, dynamic=False),
        torch.compile(finish_projection, dynamic=False),
        torch.compile(finish_projection, dynamic=False),
        torch.compile(ffn_forward, dynamic=False),
        torch.compile(ffn_forward, dynamic=False),
    )


def _padding_mask(attention_mask, sequence_length, cache_index):
    """Recover the per-token validity mask from the additive causal mask."""
    block_start = int(cache_index[0].to("cpu"))
    rows = torch.arange(sequence_length)
    diagonal = attention_mask.to("cpu")[:, 0, rows, block_start + rows]
    return (diagonal == 0).to(dtype=attention_mask.dtype, device=attention_mask.device)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    backbone = get_backbone(model)
    hidden_states = backbone.embed_tokens(input_ids)
    selected_freqs = model._spyre_rope(hidden_states, position_ids)
    padding_mask = _padding_mask(attn_mask, hidden_states.shape[1], cache_index)
    decode = hidden_states.shape[1] == 1

    for index, layer_type in enumerate(text_config(model.config).layer_types):
        compiled_block = model._spyre_compiled_blocks[index]
        if layer_type == "full_attention":
            if len(compiled_block) == 2:
                # The MoE sibling shares this runner and retains its fused
                # attention-plus-experts protocol.
                block = compiled_block[1] if decode else compiled_block[0]
                hidden_states, key_caches[index], value_caches[index] = block(
                    hidden_states,
                    selected_freqs,
                    attn_mask,
                    key_caches[index],
                    value_caches[index],
                    cache_index,
                )
                continue
            (
                project_prefill,
                project_decode,
                cache_prefill,
                cache_decode,
                attention_prefill,
                attention_decode,
                finish_prefill,
                finish_decode,
            ) = compiled_block
            project = project_decode if decode else project_prefill
            update_cache = cache_decode if decode else cache_prefill
            attention = attention_decode if decode else attention_prefill
            finish = finish_decode if decode else finish_prefill
            residual, query, key, value, gate = project(hidden_states, selected_freqs)
            key_caches[index], value_caches[index] = update_cache(
                key,
                value,
                key_caches[index],
                value_caches[index],
                cache_index,
            )
            attention_output = attention(
                query, key_caches[index], value_caches[index], attn_mask
            )
            hidden_states = finish(residual, attention_output, gate)
        else:
            (
                hidden_prefill,
                hidden_decode,
                values_prefill,
                values_decode,
                conv_prefill,
                conv_decode,
                split_prefill,
                split_decode,
                normalize_prefill,
                normalize_decode,
                norm_prefill,
                norm_decode,
                output_prefill,
                output_decode,
                prefill_ffn,
                decode_ffn,
            ) = compiled_block
            project_hidden = hidden_decode if decode else hidden_prefill
            project_values = values_decode if decode else values_prefill
            conv = conv_decode if decode else conv_prefill
            split_inputs = split_decode if decode else split_prefill
            normalize_inputs = normalize_decode if decode else normalize_prefill
            finish_norm = norm_decode if decode else norm_prefill
            finish_projection = output_decode if decode else output_prefill
            ffn = decode_ffn if decode else prefill_ffn
            conv_state = key_caches[index]
            recurrent_state = value_caches[index]
            residual = hidden_states
            h = project_hidden(hidden_states, padding_mask)
            mixed_qkv, z, beta, a = project_values(h)
            if decode:
                conv_output, new_conv_state = conv(mixed_qkv, conv_state)
            else:
                conv_output = conv(mixed_qkv, conv_state)
                new_conv_state = mixed_qkv
            query, key, value = split_inputs(conv_output.to("cpu"))
            query, key, value = normalize_inputs(query, key, value)
            linear_attn = get_backbone(model).layers[index].linear_attn
            core_output, new_recurrent_state = _delta_recurrence(
                query,
                key,
                value,
                a,
                beta,
                recurrent_state,
                linear_attn.A_log,
                linear_attn.dt_bias,
            )
            core_output = finish_norm(z, core_output.to(z.device, z.dtype))
            hidden_states = finish_projection(residual, core_output)
            hidden_states = ffn(hidden_states)
            # Keep mutation outside the compiled graph. Generic chunked prefill
            # passes a shallow cache-list copy, so replacing its entries would not
            # seed the original state tensors later consumed by decode.
            conv_state.copy_(new_conv_state)
            recurrent_state.copy_(new_recurrent_state)

    return model._spyre_compiled_norm(hidden_states)


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    hidden_states = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    return run_lm_head(model, hidden_states)


def _allocate_caches(model, batch_size, max_cache_len, dtype, device):
    cfg = text_config(model.config)
    backbone = get_backbone(model)
    key_caches = []
    value_caches = []
    for layer_type, layer in zip(cfg.layer_types, backbone.layers):
        if layer_type == "full_attention":
            local_kv_heads = layer.self_attn.k_proj.weight.shape[0] // cfg.head_dim
            key_caches.append(
                allocate_kv_cache_tensor(
                    batch_size,
                    local_kv_heads,
                    max_cache_len,
                    cfg.head_dim,
                    dtype,
                    device,
                )
            )
            value_caches.append(
                allocate_kv_cache_tensor(
                    batch_size,
                    local_kv_heads,
                    max_cache_len,
                    cfg.head_dim,
                    dtype,
                    device,
                )
            )
        else:
            conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            conv_dim += cfg.linear_num_value_heads * cfg.linear_value_head_dim
            key_caches.append(
                torch.zeros(
                    batch_size,
                    conv_dim,
                    BLOCK_SIZE,
                    dtype=dtype,
                    device=device,
                )
            )
            # The reference DeltaNet recurrence accumulates and caches this
            # matrix in fp32. Keep it on CPU because the recurrence is staged
            # there, avoiding a bf16 round-trip after every layer/token.
            value_caches.append(
                torch.zeros(
                    batch_size * cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                    dtype=torch.float32,
                    device="cpu",
                )
            )
    return key_caches, value_caches


def prepare_for_spyre(model):
    """Apply dense text-only Qwen3.5 adaptations in-place."""
    cfg = text_config(model.config)
    unsupported = set(cfg.layer_types) - {"full_attention", "linear_attention"}
    if unsupported:
        raise ValueError(f"Unsupported Qwen3.5 layer types: {sorted(unsupported)}")
    if cfg.hidden_act != "silu":
        raise ValueError(f"Unsupported Qwen3.5 activation: {cfg.hidden_act}")
    if cfg.linear_num_value_heads % cfg.linear_num_key_heads:
        raise ValueError(
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        )
    if cfg.head_dim // 2 < BLOCK_SIZE:
        raise ValueError(
            f"Qwen3.5 head_dim={cfg.head_dim} is too small for Spyre RoPE; head padding is not implemented"
        )

    backbone = get_backbone(model)
    rope_dim = int(cfg.partial_rotary_factor * cfg.head_dim)
    if rope_dim % 2:
        raise ValueError(f"Qwen3.5 rotary dimension must be even, got {rope_dim}")
    model._spyre_rope = PrecomputedRotaryEmbedding(
        backbone.rotary_emb, padded_head_dim=cfg.head_dim
    )
    model._spyre_head_dim = cfg.head_dim
    model._spyre_cache_allocator = _allocate_caches
    model._spyre_prefill_chunk_size = BLOCK_SIZE
    prepare_lm_head_for_spyre(model)

    rope_permutation = (
        rope_dim_permutation(cfg.head_dim, rope_dim)
        if rope_dim != cfg.head_dim
        else None
    )
    model._spyre_q_projs = nn.ModuleList()
    model._spyre_gate_projs = nn.ModuleList()
    compiled_blocks = []

    for layer_type, layer in zip(cfg.layer_types, backbone.layers):
        if layer_type == "full_attention":
            local_query_heads = layer.self_attn.q_proj.weight.shape[0] // (
                2 * cfg.head_dim
            )
            local_kv_heads = layer.self_attn.k_proj.weight.shape[0] // cfg.head_dim
            query_projection, gate_projection = _split_gated_q_projection(
                layer.self_attn, local_query_heads, cfg.head_dim
            )
            # The split projections contain all original rows; do not carry the
            # doubled source parameter onto the accelerator as a third copy.
            layer.self_attn.q_proj = nn.Identity()
            if rope_permutation is not None:
                permute_proj_for_rope(
                    query_projection,
                    local_query_heads,
                    cfg.head_dim,
                    rope_permutation,
                )
                permute_proj_for_rope(
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
                    layer, query_projection, gate_projection, cfg.head_dim
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
                selector = torch.zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=conv_weight.dtype)
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
            compiled_blocks.append(_make_linear_attention_block(layer))

    model._spyre_compiled_blocks = compiled_blocks
    model._spyre_compiled_norm = torch.compile(
        lambda hidden_states: _rms_norm(hidden_states, backbone.norm), dynamic=False
    )
