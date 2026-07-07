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
HuggingFace Transformers adapter for SigLIP vision towers on Spyre.

Targets the ``SiglipVisionModel`` extracted from a multimodal checkpoint
(e.g. the vision tower of Granite Vision 4.1 and Gemma 3). A SigLIP tower is
a **pre-LN, bidirectional, no-RoPE, no-KV-cache** transformer encoder over a
fixed-length patch sequence — the same shape as the BERT-family embedders,
but pre-LN and fed by a Conv2d patch embedding instead of a token table.

What this adapter handles (the vision *tower* only):

    pixel_values [B, 3, H, W]
      → Conv2d patch embed + learned position embedding   (CPU)
      → N pre-LN encoder blocks (SDPA, GELU-tanh MLP)      (Spyre)
      → post-LayerNorm
      → last_hidden_state [B, P, hidden]

The multimodal projector and the merge of image features into the text stream
are a separate concern (VLM integration) and are not handled here.

Spyre adaptations:
- ``head_dim`` (SigLIP: 1152/16 = 72) is below a stick (BLOCK_SIZE=64) for the
  reshape, so Q/K/V/O are zero-padded to the next stick multiple (72 → 128)
  with the SDPA scale held at ``1/sqrt(72)``.
- The Conv2d patch embed + ``nn.Embedding`` position table run on CPU (their
  output is moved to Spyre), matching the embedding-lookup CPU fallback used
  by the decoder adapters. ``nn.Conv2d`` lowering on Spyre is not assumed; see
  docs/siglip_vision_spyre_findings.md.
Usage::

    from hf_adapters.hf_siglip_vision import load_model, prefill_vision_tower
    model = load_model("ibm-granite/granite-vision-4.1-4b")
    # pixel_values: [B, 3, 384, 384] on CPU
    last_hidden = prefill_vision_tower(model, pixel_values)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    make_vision_encoder_block,
    prefill_vision,
    vision_backbone_forward,
)

_run_backbone_forward = vision_backbone_forward
_is_vision_tower = True


def _get_vision_tower(model):
    """Return the ``SiglipVisionModel`` regardless of how it was loaded.

    Accepts the bare tower (``SiglipVisionModel`` — has ``encoder`` directly) or
    a multimodal wrapper exposing it at ``model.vision_tower`` /
    ``model.model.vision_tower``.
    """
    if hasattr(model, "encoder") and hasattr(model, "embeddings"):
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
    raise ValueError(f"Could not locate a SigLIP vision tower on {type(model)}")


def _get_inner(tower):
    """Resolve the module holding ``embeddings``/``encoder``/``post_layernorm``.

    In transformers 5.x ``SiglipVisionModel`` holds these directly; older
    layouts nest them under ``.vision_model``.
    """
    if hasattr(tower, "encoder") and hasattr(tower, "embeddings"):
        return tower
    return getattr(tower, "vision_model", tower)


def _pad_vision_heads(layers, num_heads, orig_head_dim, padded_head_dim):
    """Zero-pad SigLIP per-layer Q/K/V/O projections to a stick boundary.

    SigLIP attention lives at ``layer.self_attn.{q,k,v,out}_proj`` (vs BERT's
    ``attention.self.*``), so we pad with the low-level helpers directly
    instead of ``pad_attention_heads_simple`` (which is wired to BERT's layout).
    """
    for layer in layers:
        attn = layer.self_attn
        attn.q_proj = _pad_proj_output_simple(
            attn.q_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.k_proj = _pad_proj_output_simple(
            attn.k_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.v_proj = _pad_proj_output_simple(
            attn.v_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.out_proj = _pad_proj_input_simple(
            attn.out_proj, num_heads, orig_head_dim, padded_head_dim
        )


def _pad_vision_mlp(layers, orig_inter, padded_inter):
    """Zero-pad each SigLIP MLP's intermediate dim to a stick boundary.

    SigLIP's ``intermediate_size`` (e.g. 4304 for Granite-Vision-4.1) is not a
    multiple of ``BLOCK_SIZE``. The Spyre compiler lays matmul operands out in
    64-element sticks and cannot identify/pad the contraction (K) dim of an
    fc2 matmul over a stick-misaligned intermediate.
    """
    for layer in layers:
        mlp = layer.mlp
        mlp.fc1 = _pad_proj_output_simple(mlp.fc1, 1, orig_inter, padded_inter)
        mlp.fc2 = _pad_proj_input_simple(mlp.fc2, 1, orig_inter, padded_inter)


def _make_patch_embed(inner):
    """Build a CPU callable: pixel_values [B,C,H,W] -> patch_embeds [B,P,hidden].

    Wraps the tower's own ``embeddings`` (Conv2d + position table). Runs on CPU;
    ``prefill_vision`` moves the result to Spyre. We don't reuse
    ``embeddings.forward`` directly because it may guard on dtype/interpolation
    flags; the patch path is small and fixed, so we inline it.

    The Conv2d weight/bias and position table are captured as CPU copies here, at
    prepare time, so this closure keeps running on CPU after
    ``move_to_spyre`` relocates the module's own params to Spyre
    (``nn.Conv2d`` is not assumed to lower on Spyre — see
    docs/siglip_vision_spyre_findings.md). The Spyre layout monkey patch cannot
    exclude these (the conv weight is 4-D, the position table is reached via a
    tower-specific path), so we snapshot on CPU rather than skip-the-move.
    """
    emb = inner.embeddings
    weight = emb.patch_embedding.weight.detach().cpu()
    bias = emb.patch_embedding.bias
    bias = bias.detach().cpu() if bias is not None else None
    stride = emb.patch_embedding.stride
    padding = emb.patch_embedding.padding
    pos_embed = emb.position_embedding.weight.detach().cpu()
    pos_ids = emb.position_ids.detach().cpu()

    def patch_embed(pixel_values):
        pv = pixel_values.to(weight.dtype).cpu()
        patches = F.conv2d(pv, weight, bias, stride=stride, padding=padding)
        patches = patches.flatten(2).transpose(1, 2)  # [B, P, hidden]
        patches = patches + pos_embed[pos_ids]
        return patches

    return patch_embed


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a SigLIP vision tower in-place.

    Pads attention heads and MLP intermediate dim to stick boundaries, builds
    the CPU patch-embed closure and the compiled pre-LN encoder blocks, and
    stashes the tower's final LayerNorm. After this the tower runs via
    ``prefill_vision`` / ``vision_backbone_forward``.
    """
    tower = _get_vision_tower(model)
    inner = _get_inner(tower)
    cfg = inner.config

    num_heads = cfg.num_attention_heads
    orig_head_dim = cfg.hidden_size // num_heads
    scale = orig_head_dim**-0.5

    layers = inner.encoder.layers

    padded_head_dim = ((orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_head_dim > orig_head_dim:
        _pad_vision_heads(layers, num_heads, orig_head_dim, padded_head_dim)
    head_dim = padded_head_dim

    # SigLIP's intermediate_size (e.g. 4304) is often not stick-aligned; the
    # Spyre compiler can't lower the fc2 matmul over a misaligned K dim. Zero-pad
    # the FFN intermediate to a stick boundary (bit-exact — see _pad_vision_mlp).
    orig_inter = cfg.intermediate_size
    padded_inter = ((orig_inter + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_inter > orig_inter:
        _pad_vision_mlp(layers, orig_inter, padded_inter)

    model._spyre_patch_embed = _make_patch_embed(inner)
    model._spyre_post_layernorm = inner.post_layernorm
    model._spyre_compiled_blocks = [
        make_vision_encoder_block(
            q_proj=layer.self_attn.q_proj,
            k_proj=layer.self_attn.k_proj,
            v_proj=layer.self_attn.v_proj,
            o_proj=layer.self_attn.out_proj,
            layer_norm1=layer.layer_norm1,
            layer_norm2=layer.layer_norm2,
            ffn_in=layer.mlp.fc1,
            act=layer.mlp.activation_fn,
            ffn_out=layer.mlp.fc2,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
        )
        for layer in layers
    ]


def load_hf_model(model_path, dtype=torch.float16):
    """Load the bare ``SiglipVisionModel`` reference (stock HF, for tests).

    Pulls just the vision tower out of the multimodal checkpoint by remapping
    ``model.vision_tower.*`` → ``*`` and loading into a fresh
    ``SiglipVisionModel`` — no ``trust_remote_code`` and without materializing
    the full VLM (mirrors ``hf_granite_vision``'s text-backbone extraction).
    """
    import json
    from collections import defaultdict

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, SiglipVisionModel

    cfg = AutoConfig.from_pretrained(model_path)
    tower = SiglipVisionModel(cfg.vision_config)

    idx_path = hf_hub_download(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    prefix = "model.vision_tower.vision_model."
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
    assert not unexpected, f"unexpected vision-tower keys: {unexpected[:5]}"
    tower.to(dtype)
    tower.eval()
    tower.requires_grad_(False)
    return tower


def load_model(model_path, dtype=torch.float16):
    """Load a SigLIP vision tower and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    prepare_for_spyre(model)
    print("Moving vision tower to Spyre ...")
    torch.nn.Module.to(model, DEVICE)
    print("Vision tower ready.")
    return model


def prefill_vision_tower(model, pixel_values, output_hidden_states=False):
    """Run the prepared tower: pixel_values -> last_hidden_state [B, P, hidden].

    With ``output_hidden_states=True`` also returns the per-layer hidden states
    tuple (HF convention), so a VLM can select ``vision_feature_layer``.
    """
    return prefill_vision(
        _run_backbone_forward,
        model,
        pixel_values,
        output_hidden_states=output_hidden_states,
    )
