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

"""Full Muse Glimmer image-to-text adapter for Spyre.

Muse Glimmer combines a DINO-like vision encoder, a two-layer perception
projector, and a custom sandwich-norm GQA decoder.  Dynamic vision metadata
(interpolation, window permutation, masks and two-dimensional positions) is
built on CPU; the large linear/attention blocks run as fixed-shape compiled
Spyre graphs.  Video is intentionally outside this initial adapter.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    get_vision_bilinear_indices_and_weights,
)
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    PrecomputedRotaryEmbedding,
    _mask_fill_value,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    get_backbone,
    get_model_dtype,
    kv_cache_update,
    pad_lm_head,
    pad_qk_proj_for_rope,
    patch_layernorm,
    text_config,
)

_GENERATION_INPUT_NAMES: tuple = ("pixel_values", "image_grid_thw")
_GENERATION_REQUIRED_INPUT_NAMES: tuple = ()
_GENERATION_TOKEN_ALIGNED_INPUTS: dict = {}


def _scale_free_rmsnorm(x, eps):
    """Muse scale-free RMSNorm, preserving stock fp32 accumulation/output cast."""
    xf = x.float()
    variance = (xf * xf).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps)).to(x.dtype)


def _centered_rmsnorm(x, weight, eps):
    """Muse centered RMSNorm with the checkpoint's ``1 + weight`` convention."""
    xf = x.float()
    variance = (xf * xf).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(variance + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def _patch_muse_norm_classes(model):
    """Replace Muse norm forwards with ``x*x`` implementations (no torch.pow)."""
    backbone = get_backbone(model)
    scale_free_cls = type(backbone.embed_tokens.embed_norm)
    centered_cls = type(backbone.layers[0].input_layernorm)

    def scale_free_forward(self, hidden_states):
        if not self.with_scale:
            return _scale_free_rmsnorm(hidden_states, self.eps)
        hidden_states_float = hidden_states.float()
        variance = (hidden_states_float * hidden_states_float).mean(-1, keepdim=True)
        out = hidden_states_float * torch.rsqrt(variance + self.eps)
        return (out * self.weight.float()).to(hidden_states.dtype)

    def centered_forward(self, hidden_states):
        return _centered_rmsnorm(hidden_states, self.weight, self.eps)

    scale_free_cls.forward = scale_free_forward
    centered_cls.forward = centered_forward


@dataclass(frozen=True)
class _TextBlockSpec:
    uses_rope: bool
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    scale: float
    qk_scale: float
    qk_norm_eps: float
    input_norm_eps: float
    post_attention_norm_eps: float
    pre_feedforward_norm_eps: float
    post_feedforward_norm_eps: float


def _text_block_spec(layer, cfg, uses_rope):
    attn = layer.self_attn
    projections = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)
    attention_bias = projections[0].bias is not None
    if any((proj.bias is not None) != attention_bias for proj in projections):
        raise ValueError("Muse attention projections must use a consistent bias layout")
    if attn.gate_proj.bias is not None:
        raise ValueError("Muse attention gate projection must be bias-free")
    if any(
        proj.bias is not None
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
    ):
        raise ValueError("Muse MLP projections must be bias-free")
    if cfg.hidden_activation != "silu":
        raise ValueError(
            f"Muse text activation {cfg.hidden_activation!r} is not supported"
        )
    return _TextBlockSpec(
        uses_rope=uses_rope,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        attention_bias=attention_bias,
        scale=attn.scaling,
        qk_scale=attn.qk_scale_factor,
        qk_norm_eps=attn.qk_norm.eps,
        input_norm_eps=layer.input_layernorm.eps,
        post_attention_norm_eps=layer.post_attention_layernorm.eps,
        pre_feedforward_norm_eps=layer.pre_feedforward_layernorm.eps,
        post_feedforward_norm_eps=layer.post_feedforward_layernorm.eps,
    )


def _text_block_state(layer, spec):
    attn = layer.self_attn
    state = (
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        attn.gate_proj.weight,
        attn.o_proj.weight,
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        layer.pre_feedforward_layernorm.weight,
        layer.post_feedforward_layernorm.weight,
        layer.mlp.gate_proj.weight,
        layer.mlp.up_proj.weight,
        layer.mlp.down_proj.weight,
    )
    if spec.attention_bias:
        state += (
            attn.q_proj.bias,
            attn.k_proj.bias,
            attn.v_proj.bias,
            attn.o_proj.bias,
        )
    return state


def _make_text_forward(spec):
    """Build a parameter-explicit text block shared by all matching layers."""

    def body(
        state,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        (
            q_weight,
            k_weight,
            v_weight,
            gate_weight,
            o_weight,
            input_norm_weight,
            post_attn_norm_weight,
            pre_ff_norm_weight,
            post_ff_norm_weight,
            mlp_gate_weight,
            mlp_up_weight,
            mlp_down_weight,
            *biases,
        ) = state
        if spec.attention_bias:
            q_bias, k_bias, v_bias, o_bias = biases
        else:
            q_bias = k_bias = v_bias = o_bias = None

        residual = hidden_states
        h = _centered_rmsnorm(hidden_states, input_norm_weight, spec.input_norm_eps)
        bsz, seq_len, _ = h.shape
        q = (
            F.linear(h, q_weight, q_bias)
            .view(bsz, seq_len, spec.num_attention_heads, spec.head_dim)
            .transpose(1, 2)
        )
        k = (
            F.linear(h, k_weight, k_bias)
            .view(bsz, seq_len, spec.num_key_value_heads, spec.head_dim)
            .transpose(1, 2)
        )
        v = (
            F.linear(h, v_weight, v_bias)
            .view(bsz, seq_len, spec.num_key_value_heads, spec.head_dim)
            .transpose(1, 2)
        )
        q = _scale_free_rmsnorm(q, spec.qk_norm_eps) * spec.qk_scale
        k = _scale_free_rmsnorm(k, spec.qk_norm_eps)
        if spec.uses_rope:
            q = apply_rope_matmul(q, selected_freqs)
            k = apply_rope_matmul(k, selected_freqs)
        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache, cache_index
        )
        out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=spec.scale,
            enable_gqa=True,
        )
        gate = torch.sigmoid(F.linear(h, gate_weight)).view(
            bsz, seq_len, spec.num_attention_heads, spec.head_dim
        )
        out = out * gate.transpose(1, 2)
        out = out.transpose(1, 2).reshape(
            bsz, seq_len, spec.num_attention_heads * spec.head_dim
        )
        out = F.linear(out, o_weight, o_bias)
        out = _centered_rmsnorm(
            out, post_attn_norm_weight, spec.post_attention_norm_eps
        )
        h = residual + out
        residual = h
        h = _centered_rmsnorm(h, pre_ff_norm_weight, spec.pre_feedforward_norm_eps)
        h = F.linear(
            F.silu(F.linear(h, mlp_gate_weight)) * F.linear(h, mlp_up_weight),
            mlp_down_weight,
        )
        h = _centered_rmsnorm(h, post_ff_norm_weight, spec.post_feedforward_norm_eps)
        return residual + h, key_cache, value_cache

    if spec.uses_rope:
        return torch.compile(body, dynamic=False, fullgraph=True)

    # Keep the NoPE graph free of both rotary operations and a rotary tensor input.
    def nope_body(
        state,
        hidden_states,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        return body(
            state,
            hidden_states,
            None,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )

    return torch.compile(nope_body, dynamic=False, fullgraph=True)


def _state_signature(state):
    return tuple((tuple(t.shape), t.dtype) for t in state)


def _finite_text_masks(attn_mask, cache_index, window, dtype):
    """Convert the generic mask to finite full/sliding Muse masks on CPU."""
    base = attn_mask.to("cpu")
    blocked = ~torch.isfinite(base) | (base < 0)
    fill = _mask_fill_value(dtype)

    full = torch.zeros(blocked.shape, dtype=dtype)
    full.masked_fill_(blocked, fill)

    if window is None:
        return full, full

    q = cache_index.to("cpu")[:, None]
    k = torch.arange(base.shape[-1])[None, :]
    outside = (q - k) >= window
    local = torch.zeros(blocked.shape, dtype=dtype)
    local.masked_fill_(blocked | outside[None, None], fill)
    return full, local


def _run_text(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    cfg = text_config(model.config)
    h = inputs_embeds
    selected_freqs = model._spyre_rope(h, position_ids)
    full_mask, local_mask = _finite_text_masks(
        attn_mask, cache_index, cfg.sliding_window, h.dtype
    )
    full_mask = full_mask.to(h.device)
    local_mask = local_mask.to(h.device)
    layers = get_backbone(model).layers
    for i, (block, spec) in enumerate(
        zip(model._spyre_text_blocks, model._spyre_text_block_specs)
    ):
        mask = local_mask if cfg.layer_types[i] == "sliding_attention" else full_mask
        state = _text_block_state(layers[i], spec)
        if spec.uses_rope:
            h, key_caches[i], value_caches[i] = block(
                state,
                h,
                selected_freqs,
                mask,
                key_caches[i],
                value_caches[i],
                cache_index,
            )
        else:
            h, key_caches[i], value_caches[i] = block(
                state,
                h,
                mask,
                key_caches[i],
                value_caches[i],
                cache_index,
            )
    return model._spyre_compiled_text_norm(h)


def _logits_from_embeds(
    model,
    inputs_embeds,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    cfg = text_config(model.config)
    h = _run_text(
        model,
        inputs_embeds,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    h = h[:, -1:, :]
    logits = model.lm_head(h)[..., : model._spyre_original_vocab_size]
    logits = logits * cfg.output_multiplier
    cap = cfg.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _segment_mask(cu_seqlens, total, padded, dtype):
    """Dense finite block-diagonal vision mask from cumulative boundaries."""
    mask = torch.full((padded, padded), _mask_fill_value(dtype), dtype=dtype)
    bounds = cu_seqlens.to("cpu").tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask[start:end, start:end] = 0
    # Padded query rows are irrelevant and cropped, but let them attend to self to
    # avoid a softmax over an entirely penalized row.
    if padded > total:
        idx = torch.arange(total, padded)
        mask[idx, idx] = 0
    return mask[None, None]


def _vision_rope_matrices(inv_freq, position_ids, padded_head_dim, dtype):
    """Build Muse `[w,h,w,h]` 2D rotation matrices, including the +1 offset."""
    pos = position_ids.to("cpu").flip(-1) + 1  # columns become [w, h]
    w, h = pos[:, 0].float(), pos[:, 1].float()
    inv = inv_freq.to("cpu").float()
    freq_w = torch.outer(w, inv)
    freq_h = torch.outer(h, inv)
    half_freq = torch.cat([freq_w, freq_h], dim=-1)
    cos, sin = half_freq.cos(), half_freq.sin()
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(
        pos.shape[0], 2, 2, half_freq.shape[-1]
    )
    padded_half = padded_head_dim // 2
    if padded_half > half_freq.shape[-1]:
        n = padded_half - half_freq.shape[-1]
        ident = torch.zeros(pos.shape[0], 2, 2, n)
        ident[:, 0, 0] = 1
        ident[:, 1, 1] = 1
        rot = torch.cat([rot, ident], dim=-1)
    return rot.contiguous().to(dtype)


@dataclass(frozen=True)
class _VisionBlockSpec:
    num_heads: int
    head_dim: int
    scale: float
    norm1_eps: float
    norm2_eps: float


def _vision_block_spec(layer, num_heads, head_dim, scale, hidden_act):
    if hidden_act != "gelu":
        raise ValueError(f"Muse vision activation {hidden_act!r} is not supported")
    modules = (
        layer.norm1,
        layer.norm2,
        layer.attn.q_proj,
        layer.attn.k_proj,
        layer.attn.v_proj,
        layer.attn.proj,
        layer.mlp.fc1,
        layer.mlp.fc2,
    )
    if any(module.weight is None or module.bias is None for module in modules):
        raise ValueError("Muse vision blocks require affine norms and biased linears")
    return _VisionBlockSpec(
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        norm1_eps=layer.norm1.eps,
        norm2_eps=layer.norm2.eps,
    )


def _vision_block_state(layer):
    return (
        layer.norm1.weight,
        layer.norm1.bias,
        layer.attn.q_proj.weight,
        layer.attn.q_proj.bias,
        layer.attn.k_proj.weight,
        layer.attn.k_proj.bias,
        layer.attn.v_proj.weight,
        layer.attn.v_proj.bias,
        layer.attn.proj.weight,
        layer.attn.proj.bias,
        layer.norm2.weight,
        layer.norm2.bias,
        layer.mlp.fc1.weight,
        layer.mlp.fc1.bias,
        layer.mlp.fc2.weight,
        layer.mlp.fc2.bias,
    )


def _layernorm(x, weight, bias, eps):
    if x.device.type != "spyre":
        return F.layer_norm(x, weight.shape, weight, bias, eps)
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    centered = xf - mean
    inv = torch.rsqrt((centered * centered).mean(-1, keepdim=True) + eps)
    h = (x - mean.to(x.dtype)) * inv.to(x.dtype)
    return h * weight + bias


def _make_vision_forward(spec):
    """Build a parameter-explicit vision block shared by matching layers."""

    def block(state, hidden_states, rope, mask):
        (
            norm1_weight,
            norm1_bias,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            norm2_weight,
            norm2_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ) = state
        residual = hidden_states
        h = _layernorm(hidden_states, norm1_weight, norm1_bias, spec.norm1_eps)
        seq = h.shape[0]
        q = (
            F.linear(h, q_weight, q_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        k = (
            F.linear(h, k_weight, k_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        v = (
            F.linear(h, v_weight, v_bias)
            .view(1, seq, spec.num_heads, spec.head_dim)
            .transpose(1, 2)
        )
        q = apply_rope_matmul(q, rope).contiguous()
        k = apply_rope_matmul(k, rope).contiguous()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=spec.scale
        )
        out = out.transpose(1, 2).reshape(seq, spec.num_heads * spec.head_dim)
        h = residual + F.linear(out, out_weight, out_bias)
        mlp_input = _layernorm(h, norm2_weight, norm2_bias, spec.norm2_eps)
        mlp_output = F.linear(
            F.gelu(F.linear(mlp_input, fc1_weight, fc1_bias)),
            fc2_weight,
            fc2_bias,
        )
        return h + mlp_output

    return torch.compile(block, dynamic=False, fullgraph=True)


def _make_vision_projector_core(model):
    """Compile the post-merge perception adapter, projection and RMSNorm."""
    adapter = model.model.vision_adapter
    projection = model.model.vision_projection
    perception_norm = model.model.perception_emb_norm

    def core(hidden_states):
        h = adapter(hidden_states)
        h = projection(h)
        return perception_norm(h)

    return torch.compile(core, dynamic=False)


def _pixel_shuffle_cpu(hidden_states, grid_thw, merge_size):
    """Native per-frame 2x2 merge, performed after graph-boundary readback."""
    output, offset = [], 0
    dim = hidden_states.shape[-1]
    for t, h, w in grid_thw.to("cpu").tolist():
        n = t * h * w
        chunk = hidden_states[offset : offset + n]
        perm = torch.arange(h * w).view(
            h // merge_size, merge_size, w // merge_size, merge_size
        )
        perm = perm.permute(0, 2, 1, 3).reshape(-1)
        if t > 1:
            perm = (perm[None] + torch.arange(t)[:, None] * h * w).reshape(-1)
        chunk = chunk[perm].view(-1, merge_size * merge_size, dim)
        output.append(chunk.permute(0, 2, 1).contiguous().view(-1, dim * merge_size**2))
        offset += n
    return torch.cat(output)


def _vision_features(model, pixel_values, image_grid_thw):
    """Run the prepared Muse vision tower and projector once for prefill."""
    tower = model.model.vision_tower
    cfg = tower.config
    dtype = get_model_dtype(model)
    grid = image_grid_thw.to("cpu")
    total = int(grid.prod(-1).sum().item())
    padded = math.ceil(total / BLOCK_SIZE) * BLOCK_SIZE

    # CPU interpolation from the small 32x32 positional table; large patch
    # projection stays on Spyre.
    indices, weights = get_vision_bilinear_indices_and_weights(
        grid, cfg.pos_emb_height, spatial_merge_size=1
    )
    pos_table = tower.patch_embedder.position_embedding_table.weight.detach().cpu()
    pos = (pos_table[indices] * weights[:, :, None]).sum(0).to(dtype)

    window_index, window_cu = get_vision_window_index(
        grid,
        spatial_merge_size=1,
        window_size=cfg.pos_emb_height * cfg.patch_size,
        patch_size=cfg.patch_size,
    )
    full_cu = get_vision_cu_seqlens(grid)
    position_ids = get_vision_position_ids(grid, spatial_merge_size=1)
    # The positional values are already emitted in the patch embedder's native
    # block-major order. Apply the same window permutation to pixels, positions,
    # and RoPE metadata before entering the compiled blocks.
    # The flattened patch width is 1176, which is not stick-aligned and cannot
    # lower as a Spyre matmul. Keep this one projection on CPU, then move its
    # stick-aligned 1536-wide output to Spyre for the encoder blocks.
    pv = pixel_values.to("cpu", dtype)[window_index]
    h = tower.patch_embedder.patch_embedding(pv).reshape(total, -1)
    h = tower.ln_pre(h + pos[window_index]).to(DEVICE)
    position_ids = position_ids[window_index]

    if padded > total:
        h = F.pad(h, (0, 0, 0, padded - total))
    rope = _vision_rope_matrices(
        model._spyre_vision_inv_freq, position_ids, model._spyre_vision_head_dim, dtype
    )
    if padded > total:
        identity = torch.zeros(padded - total, 2, 2, model._spyre_vision_head_dim // 2)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        rope = torch.cat([rope, identity.to(dtype)])
    rope = rope[None].to(DEVICE)
    masks = {
        "full_attention": _segment_mask(full_cu, total, padded, dtype).to(DEVICE),
        "window_attention": _segment_mask(window_cu, total, padded, dtype).to(DEVICE),
    }
    for i, block in enumerate(model._spyre_vision_blocks):
        state = _vision_block_state(tower.layers[i])
        h = block(state, h, rope, masks[cfg.layer_types[i]])

    # Final vision LayerNorm is tokenwise, so it commutes with unpermutation and
    # can stay in a compiled device graph. Integer unpermutation and ragged pixel
    # shuffle intentionally remain on CPU after that graph boundary.
    h = model._spyre_compiled_vision_norm(h[:total]).to("cpu")
    reverse = torch.argsort(window_index)
    h = h[reverse]
    h = _pixel_shuffle_cpu(h, grid, cfg.merge_size).to(dtype).to(DEVICE)

    return model._spyre_compiled_vision_projector(h)


def _embed_and_scatter(model, input_ids, image_features=None):
    """Normalize token embeddings and inject image features via a CPU additive."""
    backbone = get_backbone(model)
    ids = input_ids.clone()
    image_mask = ids == model.config.image_token_id
    video_mask = ids == model.config.video_token_id
    if bool(video_mask.any()):
        raise NotImplementedError("Muse Glimmer video inputs are not supported")
    ids[image_mask] = 0
    h = backbone.embed_tokens(ids.to(backbone.embed_tokens.weight.device))
    if image_features is None:
        return h
    n_tokens = int(image_mask.sum())
    hidden = h.shape[-1]
    if image_features.numel() != n_tokens * hidden:
        raise ValueError(
            "Image features and image tokens do not match: "
            f"tokens {n_tokens}, features {tuple(image_features.shape)}"
        )
    keep = (~image_mask).unsqueeze(-1).to(h.dtype).to(h.device)
    additive = torch.zeros(h.shape, dtype=h.dtype)
    additive[image_mask] = image_features.to("cpu", h.dtype).reshape(n_tokens, hidden)
    return h * keep + additive.to(h.device)


def prepare_for_spyre(model):
    """Prepare Muse Glimmer vision, projector and text decoder in-place."""
    if getattr(model, "_spyre_prepared", False):
        return

    cfg = text_config(model.config)
    tower = model.model.vision_tower
    vcfg = tower.config
    _patch_muse_norm_classes(model)

    assert (
        cfg.head_dim // 2 >= BLOCK_SIZE
    ), "Muse text head_dim must satisfy D/2 >= one stick"
    model._spyre_original_vocab_size = cfg.vocab_size
    model._spyre_rope = PrecomputedRotaryEmbedding(get_backbone(model).rotary_emb)
    model._spyre_kv_shapes = [
        (cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    layers = get_backbone(model).layers
    if len(cfg.layer_rope_theta) != len(layers) or len(cfg.layer_types) != len(layers):
        raise ValueError("Muse text layer metadata does not match the decoder depth")
    model._spyre_text_block_specs = [
        _text_block_spec(layer, cfg, bool(cfg.layer_rope_theta[i]))
        for i, layer in enumerate(layers)
    ]
    compiled_text_by_spec = {}
    text_signature_by_spec = {}
    model._spyre_text_blocks = []
    for i, (layer, spec) in enumerate(zip(layers, model._spyre_text_block_specs)):
        signature = _state_signature(_text_block_state(layer, spec))
        if spec in text_signature_by_spec and text_signature_by_spec[spec] != signature:
            raise ValueError(f"Muse text layer {i} has an incompatible state signature")
        text_signature_by_spec.setdefault(spec, signature)
        if spec not in compiled_text_by_spec:
            compiled_text_by_spec[spec] = _make_text_forward(spec)
        model._spyre_text_blocks.append(compiled_text_by_spec[spec])
    model._spyre_compiled_text_norm = torch.compile(
        get_backbone(model).norm, dynamic=False
    )
    pad_lm_head(model)

    orig_head_dim = vcfg.hidden_size // vcfg.num_attention_heads
    padded_head_dim = math.ceil(orig_head_dim / (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    scale = orig_head_dim**-0.5
    for layer in tower.layers:
        layer.attn.q_proj = pad_qk_proj_for_rope(
            layer.attn.q_proj, vcfg.num_attention_heads, orig_head_dim, padded_head_dim
        )
        layer.attn.k_proj = pad_qk_proj_for_rope(
            layer.attn.k_proj, vcfg.num_attention_heads, orig_head_dim, padded_head_dim
        )
        layer.attn.v_proj = _pad_proj_output_simple(
            layer.attn.v_proj, vcfg.num_attention_heads, orig_head_dim, padded_head_dim
        )
        layer.attn.proj = _pad_proj_input_simple(
            layer.attn.proj, vcfg.num_attention_heads, orig_head_dim, padded_head_dim
        )
    patch_layernorm(tower.ln_pre, tower.ln_post)
    model._spyre_vision_inv_freq = tower.rotary_emb.inv_freq.detach().cpu()
    model._spyre_vision_head_dim = padded_head_dim
    vision_specs = [
        _vision_block_spec(
            layer,
            vcfg.num_attention_heads,
            padded_head_dim,
            scale,
            vcfg.hidden_act,
        )
        for layer in tower.layers
    ]
    compiled_vision_by_spec = {}
    vision_signature_by_spec = {}
    model._spyre_vision_blocks = []
    for i, (layer, spec) in enumerate(zip(tower.layers, vision_specs)):
        signature = _state_signature(_vision_block_state(layer))
        if (
            spec in vision_signature_by_spec
            and vision_signature_by_spec[spec] != signature
        ):
            raise ValueError(
                f"Muse vision layer {i} has an incompatible state signature"
            )
        vision_signature_by_spec.setdefault(spec, signature)
        if spec not in compiled_vision_by_spec:
            compiled_vision_by_spec[spec] = _make_vision_forward(spec)
        model._spyre_vision_blocks.append(compiled_vision_by_spec[spec])
    model._spyre_compiled_vision_norm = torch.compile(tower.ln_post, dynamic=False)
    model._spyre_compiled_vision_projector = _make_vision_projector_core(model)
    model._spyre_cpu_submodules = [
        "model.vision_tower.patch_embedder.patch_embedding",
        "model.vision_tower.patch_embedder.position_embedding_table",
        "model.vision_tower.ln_pre",
    ]
    model._spyre_prepared = True


def _prefill_forward(
    *,
    model,
    input_ids,
    position_ids,
    attention_mask,
    key_caches,
    value_caches,
    cache_index,
    pixel_values=None,
    image_grid_thw=None,
):
    has_pixels = pixel_values is not None
    has_grid = image_grid_thw is not None
    if has_pixels != has_grid:
        raise ValueError("pixel_values and image_grid_thw must be supplied together")
    features = (
        _vision_features(model, pixel_values, image_grid_thw) if has_pixels else None
    )
    embeds = _embed_and_scatter(model, input_ids, features)
    return _logits_from_embeds(
        model,
        embeds.to(DEVICE),
        position_ids.to(DEVICE),
        attention_mask.to(DEVICE),
        key_caches,
        value_caches,
        cache_index,
    )
