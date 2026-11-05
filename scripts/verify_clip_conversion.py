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

"""Verify a converted OpenCLIP -> HF CLIP checkpoint on Spyre.

Run this after ``scripts/convert_openclip_to_hf.py`` to prove that the converted
model, running through ``hf_adapters.hf_clip`` on Spyre (fp16), produces the same
embeddings as the original OpenCLIP model (the ground truth).

The same image pixels and token ids are fed to both models. The script prints:
  1. per-input cosine similarity between the two embeddings (text and image),
  2. zero-shot image -> label top-1 for both models, side by side.

Pass criteria on Spyre fp16: every cosine >= ~0.999 and identical top-1 labels.

Requirements:
  * A Spyre device (``export SPYRE_DEVICES=0``).
  * ``open_clip_torch`` (+ ``ftfy``, ``wcwidth``). It is NOT a dependency of
    hf-adapters and is only needed by this script. If missing:
    ``pip install --no-deps open_clip_torch ftfy wcwidth``.
  * ``--converted``: the folder written by ``scripts/convert_openclip_to_hf.py``.
  * ``--images``: a folder of your own .png/.jpg/.jpeg test images, plus
    ``--labels`` text prompts that describe them (the defaults are generic
    clothing / fruit labels and will not suit other image sets).
  * Network or a populated ``HF_HOME`` so ``--openclip-model`` can be loaded
    (default ``hf-hub:Marqo/marqo-fashionCLIP``).

Usage on Spyre (run from the hf-adapters repo root)::

    export HF_HOME=/models/hf_cache
    export SPYRE_DEVICES=0
    PYTHONPATH=. python scripts/verify_clip_conversion.py --spyre \\
        --converted /path/to/converted_model \\
        --images /path/to/images \\
        --labels "a red apple" "a black suit" "a pair of shoes"

Arguments:
  --converted      HF-format CLIP dir written by convert_openclip_to_hf.py.
  --images         Directory of .png/.jpg/.jpeg test images.
  --labels         Text prompts for zero-shot ranking.
  --openclip-model Native OpenCLIP model id used as ground truth.
  --spyre          Run the converted model on Spyre fp16. Use this for real
                   validation. Without it the converted model runs on CPU fp32.

Note (CPU): omitting ``--spyre`` runs the converted model on CPU fp32, where the
result should be cosine ~1.00000. Run it first if a Spyre run looks wrong: it
shows whether the conversion itself is correct, which separates a bad
conversion from a Spyre problem.
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_LABELS = [
    "a black suit",
    "a blue suit",
    "a tan suit",
    "an apple",
    "a dress",
    "a pair of shoes",
]


def _emb(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--openclip-model", default="hf-hub:Marqo/marqo-fashionCLIP")
    ap.add_argument("--converted", required=True)
    ap.add_argument("--images", required=True, help="Directory of .png/.jpg images")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--spyre", action="store_true")
    args = ap.parse_args()

    import open_clip

    import hf_adapters.hf_clip as hf_clip

    paths = sorted(
        p
        for ext in ("png", "jpg", "jpeg")
        for p in glob.glob(os.path.join(args.images, f"*.{ext}"))
    )
    names = [os.path.basename(p) for p in paths]

    # Ground truth: native OpenCLIP, same tokenizer + preprocessing for both.
    ref, _, preprocess = open_clip.create_model_and_transforms(args.openclip_model)
    ref.eval()
    ids = open_clip.get_tokenizer(args.openclip_model)(args.labels)
    pix = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
    with torch.no_grad():
        ref_t = F.normalize(ref.encode_text(ids).float(), dim=-1)
        ref_i = F.normalize(ref.encode_image(pix).float(), dim=-1)

    dtype = torch.float16 if args.spyre else torch.float32
    model = hf_clip.load_hf_model(args.converted, dtype=dtype)
    if args.spyre:
        from hf_adapters.hf_common import move_model_to_spyre

        move_model_to_spyre(model, hf_clip, dtype)
    with torch.no_grad():
        hf_t = F.normalize(_emb(model.get_text_features(input_ids=ids)).float(), dim=-1)
        hf_i = F.normalize(
            _emb(model.get_image_features(pixel_values=pix.to(dtype))).float(), dim=-1
        )

    print("\nParity vs native OpenCLIP (cosine, 1.0 = identical):")
    for label, c in zip(args.labels, (ref_t * hf_t).sum(-1).tolist()):
        print(f"  text   {label!r:<22} {c:.5f}")
    for name, c in zip(names, (ref_i * hf_i).sum(-1).tolist()):
        print(f"  image  {name:<22} {c:.5f}")

    print("\nZero-shot top-1 (image -> best label):")
    print(f"  {'image':<22} {'native OpenCLIP':<18} {'converted':<18} sim")
    for k, name in enumerate(names):
        r = (ref_i[k] @ ref_t.T).argmax().item()
        h = (hf_i[k] @ hf_t.T).argmax().item()
        sim = (hf_i[k] @ hf_t.T)[h].item()
        print(f"  {name:<22} {args.labels[r]:<18} {args.labels[h]:<18} {sim:.3f}")


if __name__ == "__main__":
    main()
