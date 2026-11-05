#!/usr/bin/env python3
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

"""Convert an OpenCLIP safetensors checkpoint to HuggingFace CLIPModel format.

Why: ``hf_adapters.hf_clip`` loads HF ``CLIPModel`` checkpoints. Some CLIP
models (e.g. ``Marqo/marqo-fashionCLIP``) ship OpenCLIP-format weights
instead, which ``hf_clip`` cannot load directly. This script converts them
offline, so the adapter needs no ``open_clip`` dependency.

What it does:
  * renames OpenCLIP keys to HF ``CLIPModel`` keys,
  * splits each fused ``attn.in_proj_{weight,bias}`` into ``q/k/v_proj``,
  * transposes BOTH ``visual.proj`` and ``text_projection`` (OpenCLIP computes
    ``x @ proj``; HF ``nn.Linear`` computes ``x @ W.T``),
  * writes ``model.safetensors`` + ``config.json`` to ``--output``.

Usage (CPU only, no GPU/Spyre needed)::

    python scripts/convert_openclip_to_hf.py \\
        --input  /path/to/openclip/snapshot_dir \\
        --output /path/to/hf_output_dir

``--input`` must contain ``model.safetensors``. Options:
  --text-layers / --vision-layers  transformer depth (default 12 / 12).
  --quick-gelu                     use QuickGELU. Default is plain GELU, which is
                                   what LAION / Marqo OpenCLIP models use. Pass
                                   this only for OpenAI-style CLIP weights.

After conversion, load with::

    from hf_adapters.hf_clip import load_hf_model
    model = load_hf_model("/path/to/hf_output_dir")

Always validate a new conversion against the native OpenCLIP model with
``scripts/verify_clip_conversion.py`` (expect cosine ~1.0 on CPU fp32).

Assumptions / limitations:
  * Source keys must carry the ``model.`` prefix (as in the Marqo
    ``trust_remote_code`` wrapper checkpoints). Plain ``open_clip`` checkpoints
    without that prefix are not handled.
  * head_dim is assumed to be 64 (ViT-B style); ``num_attention_heads`` in the
    generated config is ``hidden_size // 64``.
  * No tokenizer or image preprocessor is written. Use the ones from
    ``openai/clip-vit-base-patch16`` (same BPE vocab, mean and std) when the
    source model follows the OpenAI CLIP recipe.
"""

import argparse
import json
import math
import os
import struct

# ---------------------------------------------------------------------------
# Minimal safetensors reader/writer (no external deps beyond stdlib + numpy)
# ---------------------------------------------------------------------------
import numpy as np

_DTYPE_MAP = {
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (None, 2),  # handled specially
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
}


def _read_safetensors(path):
    """Return (metadata_dict, {key: numpy_array}) from a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read()

    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header = json.loads(raw[8 : 8 + header_len])
    data_offset = 8 + header_len

    metadata = header.pop("__metadata__", {})
    tensors = {}
    for key, info in header.items():
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        blob = raw[data_offset + start : data_offset + end]

        if dtype_str == "BF16":
            # numpy has no bfloat16; load as uint16 and reinterpret later
            arr = np.frombuffer(blob, dtype=np.uint16).reshape(shape)
        else:
            np_dtype, _ = _DTYPE_MAP[dtype_str]
            arr = np.frombuffer(blob, dtype=np_dtype).reshape(shape)

        tensors[key] = (dtype_str, arr)

    return metadata, tensors


def _write_safetensors(path, tensors):
    """Write {key: (dtype_str, numpy_array)} to a safetensors file."""
    # Build contiguous byte blobs and header
    header = {}
    blobs = []
    offset = 0
    for key in sorted(tensors.keys()):
        dtype_str, arr = tensors[key]
        blob = arr.tobytes()
        header[key] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires header length to be 8-byte aligned
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


# ---------------------------------------------------------------------------
# Key remapping tables
# ---------------------------------------------------------------------------


def _text_remap(n):
    return {
        f"model.transformer.resblocks.{n}.ln_1.weight": f"text_model.encoder.layers.{n}.layer_norm1.weight",
        f"model.transformer.resblocks.{n}.ln_1.bias": f"text_model.encoder.layers.{n}.layer_norm1.bias",
        f"model.transformer.resblocks.{n}.ln_2.weight": f"text_model.encoder.layers.{n}.layer_norm2.weight",
        f"model.transformer.resblocks.{n}.ln_2.bias": f"text_model.encoder.layers.{n}.layer_norm2.bias",
        f"model.transformer.resblocks.{n}.attn.out_proj.weight": f"text_model.encoder.layers.{n}.self_attn.out_proj.weight",
        f"model.transformer.resblocks.{n}.attn.out_proj.bias": f"text_model.encoder.layers.{n}.self_attn.out_proj.bias",
        f"model.transformer.resblocks.{n}.mlp.c_fc.weight": f"text_model.encoder.layers.{n}.mlp.fc1.weight",
        f"model.transformer.resblocks.{n}.mlp.c_fc.bias": f"text_model.encoder.layers.{n}.mlp.fc1.bias",
        f"model.transformer.resblocks.{n}.mlp.c_proj.weight": f"text_model.encoder.layers.{n}.mlp.fc2.weight",
        f"model.transformer.resblocks.{n}.mlp.c_proj.bias": f"text_model.encoder.layers.{n}.mlp.fc2.bias",
    }


def _vision_remap(n):
    return {
        f"model.visual.transformer.resblocks.{n}.ln_1.weight": f"vision_model.encoder.layers.{n}.layer_norm1.weight",
        f"model.visual.transformer.resblocks.{n}.ln_1.bias": f"vision_model.encoder.layers.{n}.layer_norm1.bias",
        f"model.visual.transformer.resblocks.{n}.ln_2.weight": f"vision_model.encoder.layers.{n}.layer_norm2.weight",
        f"model.visual.transformer.resblocks.{n}.ln_2.bias": f"vision_model.encoder.layers.{n}.layer_norm2.bias",
        f"model.visual.transformer.resblocks.{n}.attn.out_proj.weight": f"vision_model.encoder.layers.{n}.self_attn.out_proj.weight",
        f"model.visual.transformer.resblocks.{n}.attn.out_proj.bias": f"vision_model.encoder.layers.{n}.self_attn.out_proj.bias",
        f"model.visual.transformer.resblocks.{n}.mlp.c_fc.weight": f"vision_model.encoder.layers.{n}.mlp.fc1.weight",
        f"model.visual.transformer.resblocks.{n}.mlp.c_fc.bias": f"vision_model.encoder.layers.{n}.mlp.fc1.bias",
        f"model.visual.transformer.resblocks.{n}.mlp.c_proj.weight": f"vision_model.encoder.layers.{n}.mlp.fc2.weight",
        f"model.visual.transformer.resblocks.{n}.mlp.c_proj.bias": f"vision_model.encoder.layers.{n}.mlp.fc2.bias",
    }


_SCALAR_REMAP = {
    "model.token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "model.positional_embedding": "text_model.embeddings.position_embedding.weight",
    "model.ln_final.weight": "text_model.final_layer_norm.weight",
    "model.ln_final.bias": "text_model.final_layer_norm.bias",
    "model.logit_scale": "logit_scale",
    "model.visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "model.visual.class_embedding": "vision_model.embeddings.class_embedding",
    "model.visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
    "model.visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "model.visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "model.visual.ln_post.weight": "vision_model.post_layernorm.weight",
    "model.visual.ln_post.bias": "vision_model.post_layernorm.bias",
}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir, output_dir, text_layers, vision_layers, quick_gelu=False):
    src_path = os.path.join(input_dir, "model.safetensors")
    print(f"Reading {src_path} ...")
    _, src = _read_safetensors(src_path)
    print(f"  {len(src)} source keys")

    out = {}
    remap = {}

    # Per-layer renames for text and vision towers
    for n in range(text_layers):
        remap.update(_text_remap(n))
    for n in range(vision_layers):
        remap.update(_vision_remap(n))
    remap.update(_SCALAR_REMAP)

    consumed = set()

    # 1. Direct renames
    for src_key, hf_key in remap.items():
        if src_key not in src:
            print(f"  WARNING: expected key not found in checkpoint: {src_key}")
            continue
        out[hf_key] = src[src_key]
        consumed.add(src_key)

    # 2. Fused QKV split — text tower
    for n in range(text_layers):
        for proj_key, q_key, k_key, v_key in [
            (
                f"model.transformer.resblocks.{n}.attn.in_proj_weight",
                f"text_model.encoder.layers.{n}.self_attn.q_proj.weight",
                f"text_model.encoder.layers.{n}.self_attn.k_proj.weight",
                f"text_model.encoder.layers.{n}.self_attn.v_proj.weight",
            ),
            (
                f"model.transformer.resblocks.{n}.attn.in_proj_bias",
                f"text_model.encoder.layers.{n}.self_attn.q_proj.bias",
                f"text_model.encoder.layers.{n}.self_attn.k_proj.bias",
                f"text_model.encoder.layers.{n}.self_attn.v_proj.bias",
            ),
        ]:
            dtype_str, arr = src[proj_key]
            chunk = arr.shape[0] // 3
            q, k, v = arr[:chunk], arr[chunk : 2 * chunk], arr[2 * chunk :]
            out[q_key] = (dtype_str, q.copy())
            out[k_key] = (dtype_str, k.copy())
            out[v_key] = (dtype_str, v.copy())
            consumed.add(proj_key)

    # 3. Fused QKV split — vision tower
    for n in range(vision_layers):
        for proj_key, q_key, k_key, v_key in [
            (
                f"model.visual.transformer.resblocks.{n}.attn.in_proj_weight",
                f"vision_model.encoder.layers.{n}.self_attn.q_proj.weight",
                f"vision_model.encoder.layers.{n}.self_attn.k_proj.weight",
                f"vision_model.encoder.layers.{n}.self_attn.v_proj.weight",
            ),
            (
                f"model.visual.transformer.resblocks.{n}.attn.in_proj_bias",
                f"vision_model.encoder.layers.{n}.self_attn.q_proj.bias",
                f"vision_model.encoder.layers.{n}.self_attn.k_proj.bias",
                f"vision_model.encoder.layers.{n}.self_attn.v_proj.bias",
            ),
        ]:
            dtype_str, arr = src[proj_key]
            chunk = arr.shape[0] // 3
            q, k, v = arr[:chunk], arr[chunk : 2 * chunk], arr[2 * chunk :]
            out[q_key] = (dtype_str, q.copy())
            out[k_key] = (dtype_str, k.copy())
            out[v_key] = (dtype_str, v.copy())
            consumed.add(proj_key)

    # 4. Projections: OpenCLIP computes ``x @ proj`` with proj stored [in, out];
    #    HF nn.Linear computes ``x @ W.T`` with W stored [out, in]. Both must be
    #    transposed (text_projection is square, so a shape check won't catch it).
    for src_key, hf_key in (
        ("model.visual.proj", "visual_projection.weight"),
        ("model.text_projection", "text_projection.weight"),
    ):
        if src_key in src:
            dtype_str, arr = src[src_key]
            out[hf_key] = (dtype_str, arr.T.copy())
            consumed.add(src_key)

    # Report any unconsumed keys (should be none)
    leftover = set(src.keys()) - consumed
    if leftover:
        print(f"  WARNING: {len(leftover)} unconsumed source keys:")
        for k in sorted(leftover):
            print(f"    {k}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"Writing {out_path} ({len(out)} keys) ...")
    _write_safetensors(out_path, out)

    # Infer config dims from checkpoint shapes
    text_hidden = src["model.token_embedding.weight"][1].shape[1]
    text_inter = src["model.transformer.resblocks.0.mlp.c_fc.weight"][1].shape[0]
    # in_proj_weight shape [3*hidden, hidden]; head_dim = 64 for ViT-B/16
    text_heads = text_hidden // 64

    vision_hidden = src["model.visual.conv1.weight"][1].shape[0]
    vision_inter = src["model.visual.transformer.resblocks.0.mlp.c_fc.weight"][1].shape[
        0
    ]
    patch_size = src["model.visual.conv1.weight"][1].shape[2]
    num_pos = src["model.visual.positional_embedding"][1].shape[0]  # 197 = 1 + 14*14
    image_size = int(math.sqrt(num_pos - 1)) * patch_size
    vision_heads = vision_hidden // 64

    hidden_act = "quick_gelu" if quick_gelu else "gelu"
    text_vocab = src["model.token_embedding.weight"][1].shape[0]
    text_seq_len = src["model.positional_embedding"][1].shape[0]

    config = {
        "model_type": "clip",
        "text_config": {
            "model_type": "clip_text_model",
            "hidden_size": text_hidden,
            "intermediate_size": text_inter,
            "num_hidden_layers": text_layers,
            "num_attention_heads": text_heads,
            "vocab_size": text_vocab,
            "max_position_embeddings": text_seq_len,
            "hidden_act": hidden_act,
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": vision_hidden,
            "intermediate_size": vision_inter,
            "num_hidden_layers": vision_layers,
            "num_attention_heads": vision_heads,
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_act": hidden_act,
        },
        "projection_dim": src["model.text_projection"][1].shape[1],
    }

    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {cfg_path}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input", required=True, help="Directory containing source model.safetensors"
    )
    ap.add_argument(
        "--output", required=True, help="Output directory for converted checkpoint"
    )
    ap.add_argument("--text-layers", type=int, default=12)
    ap.add_argument("--vision-layers", type=int, default=12)
    ap.add_argument(
        "--quick-gelu",
        action="store_true",
        help="Use QuickGELU (OpenAI CLIP). Default is GELU (LAION/Marqo).",
    )
    args = ap.parse_args()
    convert(
        args.input, args.output, args.text_layers, args.vision_layers, args.quick_gelu
    )
