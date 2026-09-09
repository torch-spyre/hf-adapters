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

# A finite mask penalty is deliberate. Spyre's 16-bit arithmetic must never see
# infinities, and finfo.min can overflow in downstream additions.
_MASK_PENALTY = -10_000.0


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


def _make_text_block(layer, cfg, uses_rope):
    """Compile one Muse decoder layer; NoPE closures omit RoPE from their graph."""
    attn = layer.self_attn
    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
    gate_proj, o_proj = attn.gate_proj, attn.o_proj
    input_norm = layer.input_layernorm
    post_attn_norm = layer.post_attention_layernorm
    pre_ff_norm = layer.pre_feedforward_layernorm
    post_ff_norm = layer.post_feedforward_layernorm
    mlp = layer.mlp
    n_q, n_kv, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    scale = head_dim**-0.5
    qk_scale = cfg.qk_scale_factor

    def body(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        h = input_norm(hidden_states)
        bsz, seq_len, _ = h.shape
        q = q_proj(h).view(bsz, seq_len, n_q, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, n_kv, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, n_kv, head_dim).transpose(1, 2)
        q = _scale_free_rmsnorm(q, attn.qk_norm.eps) * qk_scale
        k = _scale_free_rmsnorm(k, attn.qk_norm.eps)
        if uses_rope:
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
            scale=scale,
            enable_gqa=True,
        )
        gate = torch.sigmoid(gate_proj(h)).view(bsz, seq_len, n_q, head_dim)
        out = out * gate.transpose(1, 2)
        out = out.transpose(1, 2).reshape(bsz, seq_len, n_q * head_dim)
        out = post_attn_norm(o_proj(out))
        h = residual + out
        residual = h
        h = post_ff_norm(mlp(pre_ff_norm(h)))
        return residual + h, key_cache, value_cache

    if uses_rope:
        return torch.compile(body, dynamic=False)

    # Keep the NoPE graph free of both rotary operations and a rotary tensor input.
    def nope_body(
        hidden_states,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        return body(
            hidden_states,
            None,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )

    return torch.compile(nope_body, dynamic=False)


def _finite_text_masks(attn_mask, cache_index, window, dtype):
    """Convert the generic mask to finite full/sliding Muse masks on CPU."""
    base = attn_mask.to("cpu")
    blocked = ~torch.isfinite(base) | (base < 0)

    full = torch.zeros(blocked.shape, dtype=dtype)
    full.masked_fill_(blocked, _MASK_PENALTY)

    if window is None:
        return full, full

    q = cache_index.to("cpu")[:, None]
    k = torch.arange(base.shape[-1])[None, :]
    outside = (q - k) >= window
    local = torch.zeros(blocked.shape, dtype=dtype)
    local.masked_fill_(blocked | outside[None, None], _MASK_PENALTY)
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
    for i, block in enumerate(model._spyre_text_blocks):
        is_local = cfg.layer_types[i] == "sliding_attention"
        mask = local_mask if is_local else full_mask
        if model._spyre_text_uses_rope[i]:
            h, key_caches[i], value_caches[i] = block(
                h,
                selected_freqs,
                mask,
                key_caches[i],
                value_caches[i],
                cache_index,
            )
        else:
            h, key_caches[i], value_caches[i] = block(
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
    mask = torch.full((padded, padded), _MASK_PENALTY, dtype=dtype)
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


def _make_vision_block(layer, num_heads, head_dim, scale):
    norm1, norm2 = layer.norm1, layer.norm2
    attn, mlp = layer.attn, layer.mlp

    def block(hidden_states, rope, mask):
        residual = hidden_states
        h = norm1(hidden_states)
        seq = h.shape[0]
        q = attn.q_proj(h).view(1, seq, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(1, seq, num_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(1, seq, num_heads, head_dim).transpose(1, 2)
        q = apply_rope_matmul(q, rope).contiguous()
        k = apply_rope_matmul(k, rope).contiguous()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale
        )
        out = out.transpose(1, 2).reshape(seq, num_heads * head_dim)
        h = residual + attn.proj(out)
        return h + mlp(norm2(h))

    return torch.compile(block, dynamic=False)


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
        h = block(h, rope, masks[cfg.layer_types[i]])

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
    model._spyre_text_uses_rope = [bool(theta) for theta in cfg.layer_rope_theta]
    model._spyre_text_blocks = [
        _make_text_block(layer, cfg, model._spyre_text_uses_rope[i])
        for i, layer in enumerate(get_backbone(model).layers)
    ]
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
        patch_layernorm(layer.norm1, layer.norm2)
    patch_layernorm(tower.ln_pre, tower.ln_post)
    model._spyre_vision_inv_freq = tower.rotary_emb.inv_freq.detach().cpu()
    model._spyre_vision_head_dim = padded_head_dim
    model._spyre_vision_blocks = [
        _make_vision_block(layer, vcfg.num_attention_heads, padded_head_dim, scale)
        for layer in tower.layers
    ]
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
