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
HuggingFace Transformers adapter for Pixtral vision towers on Spyre.

Targets the ``PixtralVisionModel`` extracted from a Mistral3 multimodal
checkpoint. A Pixtral tower is a **pre-LN, bidirectional, 2D-RoPE**
transformer encoder over a variable-length patch sequence — one image at a
time, with block-diagonal attention so patches only attend within the same
image when batching.

What this adapter handles (the vision *tower* only):

    pixel_values [B, 3, H, W]
      → Conv2d patch embed + 2D-RoPE position table   (CPU)
      → N pre-LN encoder blocks (SDPA, SwiGLU MLP)     (Spyre)
      → last_hidden_state [B, P, hidden]

The multimodal projector and the merge of image features into the text stream
are a separate concern handled by ``hf_mistral3_vision_mm``.

Spyre adaptations:

- ``head_dim`` (Pixtral default: 64) is equal to one stick so the
  ``D/2 = 32 < 64`` sub-stick rule requires padding to 128.  Q/K/V/O
  projections are zero-padded with the SDPA scale held at ``1/sqrt(64)``.
- The 2D RoPE uses a pre-computed ``[max_patches, head_dim]`` ``inv_freq``
  table. Instead of the stock ``rotate_half`` (which slices along the head
  dim and fails on Spyre), we pre-build ``[P, 2, 2, D/2]`` rotation matrices
  in ``_build_pixtral_rope_matrices`` and apply them via ``apply_rope_matmul``
  — the same approach used by the text-decoder adapters.
- The Conv2d patch embed and position-id meshgrid run on CPU; the encoded
  patch embeddings are moved to Spyre before the compiled blocks run.
- The block-diagonal attention mask (each image sees only its own patches) is
  built on CPU as an additive fp16 mask and moved to Spyre.
- ``PixtralRMSNorm`` is patched via ``patch_rmsnorm`` (same as text decoders).

Usage::

    from hf_adapters.hf_pixtral_vision import load_model, prefill_vision_tower
    model = load_model("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    # pixel_values: [B, 3, H, W] on CPU; image_sizes: list of (H, W)
    last_hidden = prefill_vision_tower(model, pixel_values, image_sizes)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    apply_rope_matmul,
    patch_rmsnorm,
)

_is_vision_tower = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_pixtral_tower(model):
    """Return the PixtralVisionModel regardless of how the model was loaded.

    Accepts the bare tower (has ``transformer`` + ``ln_pre`` + ``patch_conv``)
    or a multimodal wrapper with it at ``model.model.vision_tower``.
    """
    if hasattr(model, "transformer") and hasattr(model, "patch_conv"):
        return model
    for attr_path in ("vision_tower", "model.vision_tower"):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and obj is not None:
            return obj
    raise ValueError(f"Could not locate a Pixtral vision tower on {type(model)}")


def _pad_pixtral_heads(layers, num_heads, orig_head_dim, padded_head_dim):
    """Zero-pad Pixtral per-layer Q/K/V/O projections to a stick boundary.

    Pixtral attention lives at ``layer.attention.{q,k,v,o}_proj``.
    """
    for layer in layers:
        attn = layer.attention
        attn.q_proj = _pad_proj_output_simple(
            attn.q_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.k_proj = _pad_proj_output_simple(
            attn.k_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.v_proj = _pad_proj_output_simple(
            attn.v_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.o_proj = _pad_proj_input_simple(
            attn.o_proj, num_heads, orig_head_dim, padded_head_dim
        )


def _build_pixtral_rope_matrices(
    inv_freq_table, num_positions, head_dim, padded_head_dim, dtype
):
    """Convert Pixtral's ``inv_freq`` table to ``[P, 2, 2, D/2]`` rotation matrices.

    Pixtral's ``PixtralRotaryEmbedding`` pre-computes a ``[max_patches_per_side**2,
    head_dim]`` ``inv_freq`` table indexed by 2D position id.  The stock forward
    does ``cos/sin → rotate_half`` which slices along head_dim (Spyre-incompatible).

    Here we take the pre-computed ``inv_freq`` rows corresponding to the current
    image's position ids and build the ``[P, 2, 2, D/2]`` rotation matrices that
    ``apply_rope_matmul`` expects — identical to ``PrecomputedRotaryEmbedding``
    but built on-the-fly per image (the shape ``P`` varies per image).

    Args:
        inv_freq_table: ``[max_patches, head_dim]`` pre-computed inv_freq (CPU).
        num_positions: number of patches P for this image (shape of position_ids).
        head_dim: original (unpadded) head dim (e.g. 64).
        padded_head_dim: padded head dim (e.g. 128).
        dtype: target dtype (fp16).

    Returns:
        ``[P, 2, 2, padded_head_dim // 2]`` rotation matrices on CPU.
    """
    rope_half = head_dim // 2
    freqs = inv_freq_table[:num_positions]  # [P, head_dim]
    # cos/sin over the first rope_half entries (inv_freq is already [P, head_dim])
    cos = torch.cos(freqs[:, :rope_half])  # [P, rope_half]
    sin = torch.sin(freqs[:, :rope_half])  # [P, rope_half]
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(num_positions, 2, 2, rope_half)

    if padded_head_dim > head_dim:
        pad_half = padded_head_dim // 2 - rope_half
        ident = torch.zeros(num_positions, 2, 2, pad_half)
        ident[:, 0, 0, :] = 1.0
        ident[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, ident], dim=-1)

    return rot.contiguous().to(dtype)


def _make_patch_embed_fn(tower):
    """Build a CPU callable: pixel_values → patch_embeds [total_patches, hidden].

    Pixtral uses a Conv2d patch embedding (``tower.patch_conv``) followed by
    a pre-LN (``tower.ln_pre``). Unlike SigLIP there is no position embedding
    added here — 2D RoPE is applied inside each attention layer. The Conv2d
    weight is snapshot at prepare time so it keeps running on CPU after the
    tower is moved to Spyre.

    Returns a function that accepts ``pixel_values`` (single image, CPU) and
    ``image_size`` (H, W tuple) and returns ``[1, P, hidden]`` patch embeddings
    on CPU.
    """
    conv_weight = tower.patch_conv.weight.detach().cpu()
    conv_bias = (
        tower.patch_conv.bias.detach().cpu()
        if tower.patch_conv.bias is not None
        else None
    )
    stride = tower.patch_conv.stride
    padding = tower.patch_conv.padding
    patch_size = tower.patch_size
    ln_pre_weight = tower.ln_pre.weight.detach().cpu()
    ln_pre_bias = getattr(tower.ln_pre, "bias", None)
    ln_pre_bias = ln_pre_bias.detach().cpu() if ln_pre_bias is not None else None
    ln_pre_eps = tower.ln_pre.variance_epsilon

    def patch_embed(pixel_values, image_size):
        """pixel_values: [1, C, H, W] CPU → [1, P, hidden] CPU."""
        pv = pixel_values.to(conv_weight.dtype).cpu()
        # Conv2d over the padded image; crop to actual patches
        patches = F.conv2d(pv, conv_weight, conv_bias, stride=stride, padding=padding)
        H, W = image_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        patches = patches[:, :, :h_patches, :w_patches]  # [1, hidden, h, w]
        patches = patches.flatten(2).transpose(1, 2)  # [1, P, hidden]
        # Pre-LN (ln_pre is an RMSNorm): apply via F.layer_norm approximation
        # PixtralRMSNorm: hidden = hidden / rms(hidden) * weight
        h = patches.float()
        rms = (h * h).mean(dim=-1, keepdim=True).add(ln_pre_eps).sqrt()
        h = (h / rms).to(conv_weight.dtype) * ln_pre_weight
        if ln_pre_bias is not None:
            h = h + ln_pre_bias
        return h

    return patch_embed


def _build_block_attn_mask(patch_counts, dtype=torch.float16):
    """Build a block-diagonal additive attention mask on CPU.

    Each image in the batch attends only to its own patches. Returns a
    ``[1, 1, total_patches, total_patches]`` additive mask (0 = attend,
    ``-inf`` = block), matching the convention used by Spyre text adapters.
    """
    total = sum(patch_counts)
    mask = torch.full((total, total), fill_value=float("-inf"), dtype=dtype)
    start = 0
    for n in patch_counts:
        mask[start : start + n, start : start + n] = 0.0
        start += n
    return mask[None, None, :, :]  # [1, 1, total, total]


def _make_compiled_pixtral_block(layer, num_heads, head_dim, scale):
    """Compiled pre-LN Pixtral encoder block (RMSNorm + RoPE SDPA + SwiGLU MLP).

    Unlike SigLIP (``make_vision_encoder_block``), Pixtral uses:
    - RMSNorm instead of LayerNorm (patched by ``patch_rmsnorm``)
    - 2D RoPE via ``apply_rope_matmul`` (``selected_freqs`` passed in)
    - SwiGLU MLP (``gate_proj``, ``up_proj``, ``down_proj``)
    - An additive attention mask (block-diagonal; passed in)

    The compiled block signature is:
        block_forward(hidden_states, selected_freqs, attn_mask) -> hidden_states
    """
    attn_norm = layer.attention_norm
    ffn_norm = layer.ffn_norm
    q_proj = layer.attention.q_proj
    k_proj = layer.attention.k_proj
    v_proj = layer.attention.v_proj
    o_proj = layer.attention.o_proj
    gate_proj = layer.feed_forward.gate_proj
    up_proj = layer.feed_forward.up_proj
    down_proj = layer.feed_forward.down_proj
    act_fn = layer.feed_forward.act_fn

    def block_forward(hidden_states, selected_freqs, attn_mask):
        bsz, seq_len, _ = hidden_states.shape

        # --- Attention ---
        residual = hidden_states
        h = attn_norm(hidden_states)
        q = q_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        # Apply 2D RoPE via matmul (Spyre-safe: no slicing)
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)
        hidden_states = residual + attn_out

        # --- MLP (SwiGLU) ---
        residual = hidden_states
        h = ffn_norm(hidden_states)
        h = down_proj(act_fn(gate_proj(h)) * up_proj(h))
        hidden_states = residual + h

        return hidden_states

    return torch.compile(block_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Pixtral vision tower in-place.

    Pads attention heads to a stick boundary (64 → 128), patches
    ``PixtralRMSNorm``, builds the CPU patch-embed closure and the compiled
    pre-LN encoder blocks, and stashes the 2D RoPE ``inv_freq`` table and
    max-patch-count for per-image RoPE matrix construction.
    """
    from transformers.models.pixtral.modeling_pixtral import PixtralRMSNorm

    tower = _get_pixtral_tower(model)
    cfg = tower.config

    num_heads = cfg.num_attention_heads
    orig_head_dim = cfg.hidden_size // num_heads  # typically 64
    scale = orig_head_dim**-0.5

    layers = tower.transformer.layers

    # Pad heads to next stick boundary (64 → 128 for the default config).
    padded_head_dim = ((orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_head_dim > orig_head_dim:
        _pad_pixtral_heads(layers, num_heads, orig_head_dim, padded_head_dim)
    head_dim = padded_head_dim

    # Patch PixtralRMSNorm in-place (fp16 on Spyre, float32 on CPU).
    patch_rmsnorm(PixtralRMSNorm)

    # Snapshot the CPU patch-embed closure (Conv2d + ln_pre; keeps working on
    # CPU after _move_to_spyre_with_layout relocates the tower's params).
    model._spyre_pixtral_patch_embed = _make_patch_embed_fn(tower)

    # Snapshot the 2D RoPE inv_freq table on CPU — used in prefill_vision_tower
    # to build per-image rotation matrices without a forward through the RoPE module.
    inv_freq = tower.patch_positional_embedding.inv_freq.detach().cpu()
    model._spyre_pixtral_inv_freq = inv_freq
    model._spyre_pixtral_orig_head_dim = orig_head_dim
    model._spyre_pixtral_padded_head_dim = head_dim

    # Compile encoder blocks.
    model._spyre_compiled_blocks = [
        _make_compiled_pixtral_block(layer, num_heads, head_dim, scale)
        for layer in layers
    ]


def load_hf_model(model_path, dtype=torch.float16):
    """Load the bare PixtralVisionModel (stock HF, for tests).

    Pulls just the vision tower out of the multimodal checkpoint by remapping
    ``model.vision_tower.*`` keys into a fresh ``PixtralVisionModel`` — no
    ``trust_remote_code`` and without materializing the full VLM.
    """
    import json
    from collections import defaultdict

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    cfg = AutoConfig.from_pretrained(model_path)
    tower = PixtralVisionModel(cfg.vision_config)

    idx_path = hf_hub_download(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    # Mistral3ForConditionalGeneration stores vision weights at "model.vision_tower.*"
    prefix = "model.vision_tower."
    shard_keys = defaultdict(dict)
    for k, shard in idx["weight_map"].items():
        if k.startswith(prefix):
            shard_keys[shard][k[len(prefix) :]] = k

    state = {}
    for shard, key_map in shard_keys.items():
        data = load_file(hf_hub_download(model_path, shard))
        for new_key, old_key in key_map.items():
            state[new_key] = data[old_key]

    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected pixtral-tower keys: {unexpected[:5]}"
    tower.to(dtype)
    tower.eval()
    tower.requires_grad_(False)
    return tower


def load_model(model_path, dtype=torch.float16):
    """Load a Pixtral vision tower and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    prepare_for_spyre(model)
    print("Moving Pixtral vision tower to Spyre ...")
    torch.nn.Module.to(model, DEVICE)
    print("Pixtral vision tower ready.")
    return model


def prefill_vision_tower(model, pixel_values, image_sizes, output_hidden_states=False):
    """Run the prepared Pixtral tower: pixel_values → last_hidden_state.

    Args:
        model: The multimodal model (or bare tower) prepared by ``prepare_for_spyre``.
        pixel_values: ``[B, 3, H, W]`` fp16 tensor on CPU.
        image_sizes: list of ``(H, W)`` tuples, one per image.
        output_hidden_states: if True, return ``(last_hidden, all_hidden_states)``
            where ``all_hidden_states`` is a tuple of per-layer outputs (HF convention).

    Returns:
        ``last_hidden_state`` ``[1, total_patches, hidden]`` on Spyre (or
        ``(last_hidden_state, tuple_of_per_layer_states)`` when
        ``output_hidden_states=True``).
    """
    inv_freq = model._spyre_pixtral_inv_freq  # [max_patches, head_dim] on CPU
    orig_head_dim = model._spyre_pixtral_orig_head_dim
    padded_head_dim = model._spyre_pixtral_padded_head_dim
    patch_embed_fn = model._spyre_pixtral_patch_embed
    compiled_blocks = model._spyre_compiled_blocks

    # Step 1: CPU — patch embed each image and collect per-image patch counts
    # and 2D RoPE position ids.
    all_embeds = []
    patch_counts = []
    all_rope = []

    for i, (pv, isize) in enumerate(zip(pixel_values, image_sizes)):
        pv_single = pv.unsqueeze(0)  # [1, C, H, W]
        emb = patch_embed_fn(pv_single, isize)  # [1, P, hidden]
        P = emb.shape[1]
        all_embeds.append(emb.squeeze(0))  # [P, hidden]
        patch_counts.append(P)

        # Build 2D RoPE rotation matrices for this image's patches [P, 2, 2, D/2]
        dtype = emb.dtype
        rope_mats = _build_pixtral_rope_matrices(
            inv_freq, P, orig_head_dim, padded_head_dim, dtype
        )  # [P, 2, 2, padded_head_dim//2] on CPU
        all_rope.append(rope_mats)

    # Concatenate all images into a single sequence [1, total_patches, hidden]
    hidden = torch.cat(all_embeds, dim=0).unsqueeze(0)  # [1, total_patches, hidden]

    # Build the block-diagonal attention mask on CPU [1, 1, total, total]
    attn_mask = _build_block_attn_mask(patch_counts, dtype=hidden.dtype)

    # Concatenate per-image RoPE matrices [1, total_patches, 2, 2, D/2]
    rope_mats_all = torch.cat(all_rope, dim=0).unsqueeze(0)  # [1, P_total, 2, 2, D/2]

    # Move inputs to Spyre
    hidden = hidden.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)
    rope_mats_all = rope_mats_all.to(DEVICE)

    # Step 2: Spyre — run compiled blocks
    all_hidden_states = (hidden,) if output_hidden_states else None
    for compiled_block in compiled_blocks:
        hidden = compiled_block(hidden, rope_mats_all, attn_mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden,)

    if output_hidden_states:
        return hidden, all_hidden_states
    return hidden
