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

"""Convert an OpenCLIP checkpoint to a ready-to-serve HuggingFace CLIPModel dir.

Why: ``hf_adapters.hf_clip`` and vLLM (``spyre-inference``) load HF ``CLIPModel``
checkpoints. Some CLIP models (e.g. ``Marqo/marqo-fashionCLIP``) ship OpenCLIP
weights instead. This script converts them offline, so neither runtime needs
``open_clip``.

The output directory is self-contained and works with ``CLIPModel.from_pretrained``
(hf-adapters) and ``vllm`` (``runner="pooling"``) without further steps:

  * ``model.safetensors``      renamed weights, fused QKV split, projections
                               transposed (OpenCLIP computes ``x @ proj``, HF
                               ``nn.Linear`` computes ``x @ W.T``)
  * ``config.json``            with ``"architectures": ["CLIPModel"]``
  * tokenizer + ``preprocessor_config.json`` (copied from ``--processor-source``)

Everything that depends on the model is inferred from the checkpoint: the key
prefix (``model.`` or none), the number of layers per tower, the hidden / MLP
sizes, the patch and image size. Attention head counts come from
``open_clip_config.json`` next to the weights (``text_cfg.heads``,
``vision_cfg.heads`` or ``vision_cfg.head_width``) or from ``--text-heads`` /
``--vision-heads``. Without either, open_clip's default ``head_width`` of 64 is
assumed and a warning is printed. The script fails instead of writing a
truncated or mislabelled model when keys are missing or left over.

Usage (CPU only, no Spyre needed)::

    python scripts/convert_openclip_to_hf.py \\
        --input  /path/to/openclip/snapshot_dir \\
        --output /path/to/hf_output_dir

``--input`` is a directory containing ``open_clip_model.safetensors`` or
``model.safetensors``, or the path of such a file.

Options:
  --text-heads / --vision-heads  override the head counts.
  --quick-gelu                   use QuickGELU (OpenAI CLIP). Default is plain GELU
                                 (LAION / Marqo), unless ``open_clip_config.json``
                                 sets ``quick_gelu``.
  --processor-source             tokenizer + image processor to copy into the
                                 output (default ``openai/clip-vit-base-patch16``;
                                 same BPE vocab, mean and std as Marqo B/16).
                                 Verify this for a different model family.
  --no-processor                 skip the tokenizer / preprocessor files.

Always validate a new conversion against the native OpenCLIP model with
``scripts/verify_clip_conversion.py`` (expect cosine ~1.0 on CPU fp32).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SOURCE_FILES: tuple[str, ...] = ("open_clip_model.safetensors", "model.safetensors")
DEFAULT_HEAD_WIDTH = 64
DEFAULT_PROCESSOR_SOURCE = "openai/clip-vit-base-patch16"

# OpenCLIP residual-block suffix -> HF CLIP encoder-layer suffix. Same for both towers.
BLOCK_RENAMES: dict[str, str] = {
    "ln_1.weight": "layer_norm1.weight",
    "ln_1.bias": "layer_norm1.bias",
    "ln_2.weight": "layer_norm2.weight",
    "ln_2.bias": "layer_norm2.bias",
    "attn.out_proj.weight": "self_attn.out_proj.weight",
    "attn.out_proj.bias": "self_attn.out_proj.bias",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
}

# Tower-independent tensors (names without the optional ``model.`` prefix).
GLOBAL_RENAMES: dict[str, str] = {
    "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "positional_embedding": "text_model.embeddings.position_embedding.weight",
    "ln_final.weight": "text_model.final_layer_norm.weight",
    "ln_final.bias": "text_model.final_layer_norm.bias",
    "logit_scale": "logit_scale",
    "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "visual.class_embedding": "vision_model.embeddings.class_embedding",
    "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
    "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "visual.ln_post.weight": "vision_model.post_layernorm.weight",
    "visual.ln_post.bias": "vision_model.post_layernorm.bias",
}

# OpenCLIP projection name -> HF Linear weight name (both are transposed).
PROJECTION_RENAMES: dict[str, str] = {
    "visual.proj": "visual_projection.weight",
    "text_projection": "text_projection.weight",
}


class ConversionError(Exception):
    """The checkpoint cannot be converted faithfully."""


def resolve_source(path: Path) -> Path:
    """Return the safetensors file to convert from a file or snapshot directory."""
    if path.is_file():
        return path
    for name in SOURCE_FILES:
        if (path / name).is_file():
            return path / name
    raise ConversionError(f"no {' or '.join(SOURCE_FILES)} found in {path}")


def detect_prefix(keys: Iterable[str]) -> str:
    """Return ``"model."`` (wrapper checkpoints) or ``""`` (plain OpenCLIP)."""
    keyset = set(keys)
    for prefix in ("", "model."):
        if f"{prefix}token_embedding.weight" in keyset:
            return prefix
    raise ConversionError(
        "not an OpenCLIP checkpoint: 'token_embedding.weight' (with or without a "
        "'model.' prefix) not found. Is this already an HF CLIP checkpoint?"
    )


def count_layers(keys: Iterable[str], base: str) -> int:
    """Number of ``<base>.resblocks.N`` layers, which must be contiguous from 0."""
    pattern = re.compile(re.escape(base) + r"\.resblocks\.(\d+)\.")
    indices = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    if not indices:
        raise ConversionError(f"no '{base}.resblocks.N.*' keys found in checkpoint")
    count = max(indices) + 1
    missing = sorted(set(range(count)) - indices)
    if missing:
        raise ConversionError(f"'{base}.resblocks' is missing layers {missing}")
    return count


def take(
    src: dict[str, torch.Tensor], consumed: set[str], key: str, hint: str
) -> torch.Tensor:
    """Fetch ``key`` from the checkpoint and mark it consumed, or fail clearly."""
    if key not in src:
        raise ConversionError(f"key '{key}' not found in checkpoint ({hint})")
    consumed.add(key)
    return src[key]


def convert_tower(
    src: dict[str, torch.Tensor],
    consumed: set[str],
    out: dict[str, torch.Tensor],
    src_base: str,
    hf_base: str,
    num_layers: int,
) -> None:
    """Rename one tower's residual blocks and split each fused QKV projection."""
    for n in range(num_layers):
        s = f"{src_base}.resblocks.{n}"
        h = f"{hf_base}.encoder.layers.{n}"
        hint = f"{src_base} layer {n}; check the key prefix and layer count"
        for src_suffix, hf_suffix in BLOCK_RENAMES.items():
            out[f"{h}.{hf_suffix}"] = take(src, consumed, f"{s}.{src_suffix}", hint)
        for kind in ("weight", "bias"):
            fused = take(src, consumed, f"{s}.attn.in_proj_{kind}", hint)
            if fused.shape[0] % 3:
                raise ConversionError(
                    f"'{s}.attn.in_proj_{kind}' dim 0 ({fused.shape[0]}) is not "
                    "divisible by 3, so it is not a fused QKV tensor"
                )
            for name, part in zip("qkv", fused.chunk(3, dim=0)):
                out[f"{h}.self_attn.{name}_proj.{kind}"] = part.contiguous().clone()


def load_open_clip_config(src_file: Path) -> dict[str, object]:
    """Return ``model_cfg`` from ``open_clip_config.json`` beside the weights, or {}."""
    cfg_path = src_file.parent / "open_clip_config.json"
    if not cfg_path.is_file():
        return {}
    model_cfg = json.loads(cfg_path.read_text()).get("model_cfg", {})
    return model_cfg if isinstance(model_cfg, dict) else {}


def infer_heads(
    tower: str,
    hidden: int,
    cli_heads: int | None,
    cfg: dict[str, object],
) -> int:
    """Attention heads for ``tower`` ("text" or "vision"), never silently guessed.

    Order: CLI flag, ``open_clip_config.json`` (``heads`` or ``head_width``), then
    open_clip's own default head width of 64 (noted when the config file exists,
    warned about when there is no config at all).
    """
    tower_cfg = cfg.get(f"{tower}_cfg")
    heads = cli_heads
    source = "command line"
    if heads is None and isinstance(tower_cfg, dict):
        if isinstance(tower_cfg.get("heads"), int):
            heads, source = tower_cfg["heads"], "open_clip_config.json heads"
        elif isinstance(tower_cfg.get("head_width"), int):
            head_width = tower_cfg["head_width"]
            heads, source = hidden // head_width, "open_clip_config.json head_width"
    if heads is None:
        heads = hidden // DEFAULT_HEAD_WIDTH
        source = f"open_clip default head_width {DEFAULT_HEAD_WIDTH}"
        if not isinstance(tower_cfg, dict):
            print(
                f"  WARNING: no open_clip_config.json and no --{tower}-heads; "
                f"assuming head_width {DEFAULT_HEAD_WIDTH} -> {heads} heads. "
                "Wrong for other head widths; embeddings would be silently wrong."
            )
    if heads <= 0 or hidden % heads:
        raise ConversionError(
            f"{tower} hidden size {hidden} is not divisible by {heads} heads "
            f"(from {source}); pass --{tower}-heads"
        )
    print(
        f"  {tower}: hidden {hidden}, {heads} heads, head_dim {hidden // heads} ({source})"
    )
    return heads


def build_config(
    src: dict[str, torch.Tensor],
    cfg: dict[str, object],
    text_layers: int,
    vision_layers: int,
    text_heads: int | None,
    vision_heads: int | None,
    quick_gelu: bool,
) -> dict[str, object]:
    """HF ``CLIPConfig`` dict from tensor shapes (``src`` has no ``model.`` prefix)."""
    text_hidden = src["token_embedding.weight"].shape[1]
    vision_hidden = src["visual.conv1.weight"].shape[0]
    patch_size = src["visual.conv1.weight"].shape[2]
    grid = math.isqrt(src["visual.positional_embedding"].shape[0] - 1)
    if grid * grid != src["visual.positional_embedding"].shape[0] - 1:
        raise ConversionError(
            "visual.positional_embedding length is not 1 + a square number"
        )
    use_quick_gelu = quick_gelu or bool(cfg.get("quick_gelu", False))
    hidden_act = "quick_gelu" if use_quick_gelu else "gelu"
    print(f"  activation: {hidden_act}")
    config = {
        "architectures": ["CLIPModel"],
        "model_type": "clip",
        "projection_dim": src["text_projection"].shape[1],
        "text_config": {
            "model_type": "clip_text_model",
            "hidden_size": text_hidden,
            "intermediate_size": src["transformer.resblocks.0.mlp.c_fc.weight"].shape[
                0
            ],
            "num_hidden_layers": text_layers,
            "num_attention_heads": infer_heads("text", text_hidden, text_heads, cfg),
            "vocab_size": src["token_embedding.weight"].shape[0],
            "max_position_embeddings": src["positional_embedding"].shape[0],
            # OpenCLIP pools the text at the highest token id (the EOT token);
            # HF pools at the first eos_token_id, whose default only matches a
            # 49408-entry vocab.
            "eos_token_id": src["token_embedding.weight"].shape[0] - 1,
            "hidden_act": hidden_act,
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": vision_hidden,
            "intermediate_size": src[
                "visual.transformer.resblocks.0.mlp.c_fc.weight"
            ].shape[0],
            "num_hidden_layers": vision_layers,
            "num_attention_heads": infer_heads(
                "vision", vision_hidden, vision_heads, cfg
            ),
            "image_size": grid * patch_size,
            "patch_size": patch_size,
            "hidden_act": hidden_act,
        },
    }
    if src["visual.proj"].shape[1] != config["projection_dim"]:
        raise ConversionError(
            f"visual.proj output dim {src['visual.proj'].shape[1]} != "
            f"text_projection output dim {config['projection_dim']}"
        )
    return config


def check_against_hf(config: dict[str, object], out: dict[str, torch.Tensor]) -> None:
    """Fail if the converted tensors do not match what HF ``CLIPModel`` expects."""
    from transformers import CLIPConfig, CLIPModel

    with torch.device("meta"):
        expected = {
            k: tuple(v.shape)
            for k, v in CLIPModel(CLIPConfig(**config)).state_dict().items()
            if not k.endswith("position_ids")
        }
    actual = {k: tuple(v.shape) for k, v in out.items()}
    problems = [f"missing {k}" for k in sorted(expected.keys() - actual.keys())]
    problems += [f"unexpected {k}" for k in sorted(actual.keys() - expected.keys())]
    problems += [
        f"shape {k}: got {actual[k]}, HF expects {expected[k]}"
        for k in sorted(expected.keys() & actual.keys())
        if expected[k] != actual[k]
    ]
    if problems:
        raise ConversionError(
            "converted weights do not match HF CLIPModel:\n  " + "\n  ".join(problems)
        )


def write_processor(source: str, output: Path) -> None:
    """Copy the tokenizer and image preprocessor files vLLM / HF need."""
    from transformers import CLIPImageProcessor, CLIPTokenizerFast

    CLIPTokenizerFast.from_pretrained(source).save_pretrained(output)
    CLIPImageProcessor.from_pretrained(source).save_pretrained(output)


def convert(
    input_path: Path,
    output_dir: Path,
    text_heads: int | None = None,
    vision_heads: int | None = None,
    quick_gelu: bool = False,
    processor_source: str | None = DEFAULT_PROCESSOR_SOURCE,
) -> None:
    src_file = resolve_source(input_path)
    print(f"Reading {src_file} ...")
    raw = load_file(str(src_file))
    prefix = detect_prefix(raw)
    src = {k.removeprefix(prefix): v for k, v in raw.items()}
    print(f"  {len(src)} tensors, key prefix {prefix!r}")

    text_layers = count_layers(src, "transformer")
    vision_layers = count_layers(src, "visual.transformer")
    print(f"  layers: text {text_layers}, vision {vision_layers}")

    consumed: set[str] = set()
    out: dict[str, torch.Tensor] = {}
    hint = "is this an OpenCLIP checkpoint?"
    for src_key, hf_key in GLOBAL_RENAMES.items():
        out[hf_key] = take(src, consumed, src_key, hint)
    for src_key, hf_key in PROJECTION_RENAMES.items():
        # Both are transposed. text_projection is square, so no shape check
        # would catch a missing transpose.
        out[hf_key] = take(src, consumed, src_key, hint).t().contiguous()
    convert_tower(src, consumed, out, "transformer", "text_model", text_layers)
    convert_tower(
        src, consumed, out, "visual.transformer", "vision_model", vision_layers
    )

    leftover = sorted(src.keys() - consumed)
    if leftover:
        raise ConversionError(
            f"{len(leftover)} checkpoint tensors were not converted (unsupported "
            "architecture?):\n  " + "\n  ".join(leftover)
        )

    config = build_config(
        src,
        load_open_clip_config(src_file),
        text_layers,
        vision_layers,
        text_heads,
        vision_heads,
        quick_gelu,
    )
    check_against_hf(config, out)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    print(f"Writing {weights_path} ({len(out)} tensors) ...")
    save_file(out, str(weights_path), metadata={"format": "pt"})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if processor_source:
        print(f"Writing tokenizer + preprocessor from {processor_source} ...")
        write_processor(processor_source, output_dir)
    print(f"Done: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory with open_clip_model.safetensors / model.safetensors, or the file",
    )
    ap.add_argument(
        "--output", required=True, type=Path, help="Output directory (created)"
    )
    ap.add_argument("--text-heads", type=int, help="Text tower attention heads")
    ap.add_argument("--vision-heads", type=int, help="Vision tower attention heads")
    ap.add_argument(
        "--quick-gelu",
        action="store_true",
        help="Use QuickGELU (OpenAI CLIP). Default is GELU (LAION/Marqo).",
    )
    ap.add_argument(
        "--processor-source",
        default=DEFAULT_PROCESSOR_SOURCE,
        help="Hub id or path to copy the tokenizer + image processor from "
        f"(default {DEFAULT_PROCESSOR_SOURCE})",
    )
    ap.add_argument(
        "--no-processor",
        action="store_true",
        help="Do not write tokenizer / preprocessor files",
    )
    args = ap.parse_args(argv)
    try:
        convert(
            args.input,
            args.output,
            args.text_heads,
            args.vision_heads,
            args.quick_gelu,
            None if args.no_processor else args.processor_source,
        )
    except ConversionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
