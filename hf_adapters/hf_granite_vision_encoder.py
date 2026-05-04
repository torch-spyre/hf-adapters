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
HuggingFace Transformers adapter for Granite Vision encoder + projectors.

Treats the vision encoder pipeline (SiglipVisionModel + WindowQFormerDownsampler
projectors) as a standalone compiled model for Spyre debugging.  The language
model backbone is discarded — this adapter handles vision-only forward.

Architecture:
  pixel_values [B, num_patches, 3, 384, 384]
    → SiglipVisionModel (27 ViT encoder layers)
    → extract hidden states at multiple layers
    → 4 deepstack WindowQFormerDownsampler projectors
    → 4 spatial WindowQFormerDownsampler projectors
    → projected features [B, seq, llm_hidden_size]

Each SiglipVisionModel encoder layer:
  LayerNorm → MultiHeadAttention (QKV, no bias, GELU) → residual
  LayerNorm → MLP (fc1→gelu→fc2) → residual

Each WindowQFormerDownsampler:
  LayerNorm → window → InterpolateDownsampler/SpatialOffsetDownsampler
  → QFormer cross-attention (1 layer Blip2QFormer) → unwindow → Linear

Usage::

    from hf_adapters.hf_granite_vision_encoder import load_model, run_forward
    model = load_model("ibm-granite/granite-vision-4.1-4b")
    # pixel_values: [B, num_patches, 3, 384, 384]
    features = run_forward(model, pixel_values)
"""

import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import DEVICE, BLOCK_SIZE, pad_mlp


# ---------------------------------------------------------------------------
# Head-dim padding (no RoPE — simple end-padding for all projections)
# ---------------------------------------------------------------------------

def _pad_proj(proj, n_heads, orig_head_dim, padded_head_dim, is_output=False):
    """Zero-pad a single Q/K/V/O projection from orig_head_dim to padded_head_dim."""
    w = proj.weight
    if is_output:
        hidden = w.shape[0]
        new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[:, d:d + orig_head_dim] = w[:, s:s + orig_head_dim]
        new_proj = nn.Linear(n_heads * padded_head_dim, hidden, bias=proj.bias is not None)
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_proj.bias = nn.Parameter(proj.bias.clone(), requires_grad=False)
    else:
        hidden = w.shape[1]
        new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[d:d + orig_head_dim, :] = w[s:s + orig_head_dim, :]
        new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=proj.bias is not None)
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
            for h in range(n_heads):
                s = h * orig_head_dim
                d = h * padded_head_dim
                new_b[d:d + orig_head_dim] = proj.bias[s:s + orig_head_dim]
            new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def _pad_vision_attention(layers, orig_head_dim, padded_head_dim, num_heads):
    """Zero-pad Q/K/V/O projections in vision encoder layers."""
    for layer in layers:
        attn = layer.self_attn
        attn.q_proj = _pad_proj(attn.q_proj, num_heads, orig_head_dim, padded_head_dim)
        attn.k_proj = _pad_proj(attn.k_proj, num_heads, orig_head_dim, padded_head_dim)
        attn.v_proj = _pad_proj(attn.v_proj, num_heads, orig_head_dim, padded_head_dim)
        attn.out_proj = _pad_proj(attn.out_proj, num_heads, orig_head_dim, padded_head_dim, is_output=True)
        attn.head_dim = padded_head_dim


def _pad_qformer_attention(projectors, orig_head_dim, padded_head_dim, num_heads):
    """Zero-pad Q/K/V/O in QFormer self-attention and cross-attention layers."""
    for projector in projectors:
        for qformer_layer in projector.qformer.encoder.layer:
            # Self-attention
            sa = qformer_layer.attention.attention
            sa.query = _pad_proj(sa.query, num_heads, orig_head_dim, padded_head_dim)
            sa.key = _pad_proj(sa.key, num_heads, orig_head_dim, padded_head_dim)
            sa.value = _pad_proj(sa.value, num_heads, orig_head_dim, padded_head_dim)
            qformer_layer.attention.output.dense = _pad_proj(
                qformer_layer.attention.output.dense, num_heads,
                orig_head_dim, padded_head_dim, is_output=True,
            )

            # Cross-attention
            ca = qformer_layer.crossattention.attention
            ca.query = _pad_proj(ca.query, num_heads, orig_head_dim, padded_head_dim)
            ca.key = _pad_proj(ca.key, num_heads, orig_head_dim, padded_head_dim)
            ca.value = _pad_proj(ca.value, num_heads, orig_head_dim, padded_head_dim)
            qformer_layer.crossattention.output.dense = _pad_proj(
                qformer_layer.crossattention.output.dense, num_heads,
                orig_head_dim, padded_head_dim, is_output=True,
            )


# ---------------------------------------------------------------------------
# Compiled blocks for vision encoder layers (SiglipEncoderLayer)
# ---------------------------------------------------------------------------

def _make_vision_block(layer, padded_head_dim):
    """Compiled block for a single Siglip encoder layer."""
    ln1 = layer.layer_norm1
    attn = layer.self_attn
    ln2 = layer.layer_norm2
    mlp = layer.mlp
    num_heads = attn.num_heads
    head_dim = padded_head_dim

    def block_forward(hidden_states):
        residual = hidden_states
        h = ln1(hidden_states)

        bsz, seq_len, _ = h.shape

        q = attn.q_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
        attn_out = attn.out_proj(attn_out)

        h = residual + attn_out

        residual = h
        h = ln2(h)
        h = mlp(h)
        h = residual + h

        return h

    return torch.compile(block_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Compiled block for WindowQFormerDownsampler projector
# ---------------------------------------------------------------------------

def _make_projector_block(projector):
    """Compiled block for a single WindowQFormerDownsampler."""
    norm = projector.norm
    qformer = projector.qformer
    out_linear = projector.out_linear
    downsampler = projector.downsampler
    # Capture the module itself; access .query/.image_positions at call time
    # so that model.to(DEVICE) updates are reflected.
    proj_ref = projector
    image_side = projector.image_side
    window_side = projector.window_side
    query_side = projector.query_side

    def _win(x, side, win):
        B, _, C = x.shape
        n = side // win
        return (
            x.view(B, side, side, C)
            .view(B, n, win, n, win, C)
            .transpose(2, 3)
            .flatten(0, 2)
            .flatten(1, 2)
        )

    def _unwin(xw, n, win):
        Bnn, _, C = xw.shape
        B = Bnn // (n * n)
        side = n * win
        return (
            xw.view(B, n, n, win, win, C)
            .transpose(2, 3)
            .contiguous()
            .view(B, side, side, C)
            .flatten(1, 2)
        )

    def projector_forward(image_features):
        B, HW, C = image_features.shape
        n = image_side // window_side

        image_features = norm(image_features)
        enc = _win(image_features, image_side, window_side)

        downsampled = downsampler(image_features)

        new_side = n * query_side
        downsampled_w = _win(downsampled, new_side, query_side)

        query_embeds = proj_ref.query + downsampled_w
        encoder_embeds = enc + proj_ref.image_positions

        out_w = qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_embeds,
            return_dict=True,
        ).last_hidden_state

        out = _unwin(out_w, n=n, win=query_side)
        return out_linear(out)

    return torch.compile(projector_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Patch embedding workaround (Conv2d → unfold + linear)
# ---------------------------------------------------------------------------

def _patch_embed_as_linear(pixel_values, patch_weight, patch_bias,
                           position_embedding, patch_size, num_patches,
                           target_device):
    """Replace Conv2d patch embedding with unfold+linear.

    Conv2d(3, hidden, k=patch_size, s=patch_size) on [B, 3, H, W] is
    equivalent to unfolding into non-overlapping patches and projecting
    each with a linear layer.

    Runs entirely on CPU (unfold + matmul + position add), then moves
    the result to the target device.
    """
    B = pixel_values.shape[0]

    pv_cpu = pixel_values.cpu() if pixel_values.device.type != "cpu" else pixel_values
    patches = pv_cpu.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, -1)

    embeddings = F.linear(patches, patch_weight, patch_bias)
    embeddings = embeddings + position_embedding

    return embeddings.to(target_device)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _run_vision_tower(model, pixel_values):
    """Run SiglipVisionModel via compiled blocks, return all hidden states.

    Returns hidden_states list matching HF's output_hidden_states format:
    [embeddings, after_layer_0, ..., after_layer_N-1] (length = num_layers + 1).
    post_layernorm is NOT included — it's applied separately if needed.
    """
    embeddings = _patch_embed_as_linear(
        pixel_values,
        model._spyre_patch_weight,
        model._spyre_patch_bias,
        model._spyre_position_embedding,
        model._spyre_patch_size,
        model._spyre_num_patches,
        model._spyre_target_device,
    )
    hidden_states = embeddings

    all_hidden_states = [hidden_states]
    for compiled_block in model._spyre_vision_blocks:
        hidden_states = compiled_block(hidden_states)
        all_hidden_states.append(hidden_states)

    return hidden_states, all_hidden_states


def _run_forward(model, pixel_values):
    """Full vision encoder forward: vision tower + all projectors.

    Args:
        model: Prepared model with compiled blocks attached.
        pixel_values: [B, 3, 384, 384] image patches (single-patch for now).

    Returns:
        List of (llm_layer_idx, projected_features) tuples.
    """
    _, all_hidden_states = _run_vision_tower(model, pixel_values)

    all_features = []

    for proj_idx, (vision_layer, llm_layer) in enumerate(model._deepstack_layer_map):
        selected = all_hidden_states[vision_layer]
        if model._vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        proj_block = model._spyre_deepstack_blocks[proj_idx]
        projected = proj_block(selected)
        all_features.append((llm_layer, projected))

    if model._spyre_spatial_blocks is not None:
        spatial_feature = all_hidden_states[model._spatial_vision_layer]
        if model._vision_feature_select_strategy == "default":
            spatial_feature = spatial_feature[:, 1:]
        for group_idx, llm_layer in enumerate(model._spatial_target_layers):
            proj_block = model._spyre_spatial_blocks[group_idx]
            projected = proj_block(spatial_feature)
            all_features.append((llm_layer, projected))

    return all_features


# ---------------------------------------------------------------------------
# Prepare for Spyre
# ---------------------------------------------------------------------------

def prepare_for_spyre(model):
    """Apply Spyre adaptations to the vision encoder model in-place."""
    vision_tower = model.vision_tower.vision_model

    # Pad attention heads to stick-aligned size (72 → 128)
    vision_config = vision_tower.config
    orig_head_dim = vision_config.hidden_size // vision_config.num_attention_heads
    padded_head_dim = (
        ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    )
    if padded_head_dim > orig_head_dim:
        _pad_vision_attention(
            vision_tower.encoder.layers,
            orig_head_dim, padded_head_dim,
            vision_config.num_attention_heads,
        )

    # Pad MLP intermediate size to stick-aligned (4304 → 4352)
    pad_mlp(
        vision_tower.encoder.layers,
        vision_config.intermediate_size,
        lambda layer: (layer.mlp.fc1, layer.mlp.fc2),
    )

    # Pad QFormer attention heads (head_dim=64 → 128)
    # QFormer config: hidden_size=1152, num_attention_heads=1152//64=18, head_dim=64
    qformer_num_heads = vision_config.hidden_size // 64
    qformer_head_dim = 64
    qformer_padded = (
        ((qformer_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    )
    all_projectors = list(model.layerwise_projectors)
    if model.spatial_projectors is not None:
        all_projectors += list(model.spatial_projectors)
    _pad_qformer_attention(all_projectors, qformer_head_dim, qformer_padded, qformer_num_heads)

    # Replace Conv2d patch embedding with CPU tensors for F.linear
    # (Conv2d and unfold not supported on Spyre; runs entirely on CPU)
    embeddings_mod = vision_tower.embeddings
    patch_conv = embeddings_mod.patch_embedding
    patch_size = patch_conv.kernel_size[0]
    in_features = patch_conv.in_channels * patch_size * patch_size
    out_features = patch_conv.out_channels

    model._spyre_patch_weight = patch_conv.weight.reshape(out_features, in_features).clone()
    model._spyre_patch_bias = patch_conv.bias.clone() if patch_conv.bias is not None else None
    model._spyre_position_embedding = embeddings_mod.position_embedding.weight.clone()
    model._spyre_patch_size = patch_size
    model._spyre_num_patches = embeddings_mod.position_embedding.num_embeddings
    model._spyre_target_device = DEVICE

    model._vision_post_layernorm = vision_tower.post_layernorm
    model._padded_head_dim = padded_head_dim

    model._spyre_vision_blocks = [
        _make_vision_block(layer, padded_head_dim)
        for layer in vision_tower.encoder.layers
    ]

    model._spyre_deepstack_blocks = [
        _make_projector_block(proj)
        for proj in model.layerwise_projectors
    ]

    if model.spatial_projectors is not None:
        model._spyre_spatial_blocks = [
            _make_projector_block(proj)
            for proj in model.spatial_projectors
        ]
    else:
        model._spyre_spatial_blocks = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_vision_encoder(model_path, dtype=torch.float16):
    """Load vision tower + projectors from a Granite Vision checkpoint.

    Discards the language model backbone. Uses trust_remote_code because
    the projector classes (WindowQFormerDownsampler) are custom.
    """
    from transformers import AutoModel, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype,
    )

    # model is Granite4VisionForConditionalGeneration wrapping Granite4VisionModel.
    # Extract the inner model which has vision_tower, layerwise_projectors, etc.
    # Then strip the language model to save memory.
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "language_model"):
        inner.language_model = None
    model = inner

    model._deepstack_layer_map = config.deepstack_layer_map
    model._spatial_vision_layer = getattr(config, "spatial_vision_layer", -1)
    model._spatial_target_layers = getattr(config, "spatial_target_layers", [])
    model._vision_feature_select_strategy = getattr(
        config, "vision_feature_select_strategy", "default"
    )

    model.eval()
    model.requires_grad_(False)
    return model


def load_hf_model(model_path, dtype=torch.float16):
    """Load the vision encoder as-is for CPU reference testing."""
    return _load_vision_encoder(model_path, dtype)


def load_model(model_path, dtype=torch.float16):
    """Load vision encoder prepared for Spyre."""
    model = _load_vision_encoder(model_path, dtype)
    prepare_for_spyre(model)
    print("Moving vision encoder to Spyre ...")
    model.to(DEVICE)
    print("Vision encoder ready.")
    return model
