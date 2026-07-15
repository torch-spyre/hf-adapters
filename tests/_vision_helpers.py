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

"""Shared helpers for the vision/multimodal CPU accuracy tests.

The reference for a vision-tower test is the stock-HF tower run on a
deterministic ``pixel_values`` input. We synthesize ``pixel_values`` directly
(seeded ``torch.randn`` at the tower's native resolution) rather than running an
image through the processor: the tower test certifies the tower, and a fixed
canonical tensor keeps the test fast (no image file, no processor tiling) and
fully deterministic across runs.
"""

from __future__ import annotations

import inspect

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

from tests.conftest import load_ref_model

# ── VLM (image→text) end-to-end helpers ──────────────────────────────────────
#
# These drive a full multimodal adapter (both towers) the way an application
# would: processor → adapter.generate → decoded text, compared against stock's
# real ``model.generate``. They are model-agnostic given a model path — the only
# convention they bake in is the modern single-call chat-template path, which
# every current HF VLM processor supports and which (for anyres VLMs like Granite
# Vision) produces correct tiling + image-token expansion. Per-model inputs
# (e.g. ``image_sizes``, ``image_grid_thw``) ride along in the returned ``batch``
# dict, so a new VLM needs no change here.

SAMPLE_IMAGE = {
    "repo_id": "huggingface/documentation-images",
    "filename": "pipeline-cat-chonk.jpeg",
    "repo_type": "dataset",
}


def _load_sample_image() -> Image.Image:
    """A real, recognizable hub image (a chonky cat) so a caption is judgeable.

    Downloaded at test time — no committed fixture. Human-eyeballable output is
    a deliberate secondary signal on top of the token-exact assertion.
    """
    path = hf_hub_download(**SAMPLE_IMAGE)
    return Image.open(path).convert("RGB")


def extra_image_inputs(fn, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Batch tensors, beyond the standard three, that ``fn`` declares as params.

    VLM adapters take ``input_ids, attention_mask, pixel_values`` and then
    whatever extra image inputs their model needs — Granite Vision / Mistral 3:
    ``image_sizes``; Gemma 4 unified: ``image_position_ids`` +
    ``mm_token_type_ids``. Matching a processor ``batch`` against the callee's
    signature (``adapter.generate`` / ``adapter._prefill_forward``) keeps the CPU
    and Spyre e2e harnesses signature-agnostic across those adapters; the extra
    tensors are forwarded by keyword.
    """
    accepted = set(inspect.signature(fn).parameters)
    standard = {"input_ids", "attention_mask", "pixel_values"}
    return {
        k: v
        for k, v in batch.items()
        if k in accepted and k not in standard and isinstance(v, torch.Tensor)
    }


def build_vlm_batch(
    model_path: str,
    prompt: str,
    image: Image.Image | None = None,
) -> tuple[AutoProcessor, dict[str, torch.Tensor]]:
    """Processor + tokenized (image + prompt) batch, the official VLM way.

    Embeds the image in the conversation and lets ``apply_chat_template`` tokenize
    and expand image tokens in one call (``tokenize=True, return_dict=True``). The
    two-step ``processor(text=..., images=...)`` path under-tiles anyres images and
    mis-aligns image tokens, so the documented single-call path is used instead.

    Sets ``padding_side='left'`` to match the adapters' right-aligned decode
    convention. Returns ``(processor, batch)``; ``batch`` carries whatever image
    inputs the model needs (``pixel_values``, ``image_sizes``, …).
    """
    if "mistral" in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
    # processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    if image is None:
        image = _load_sample_image()
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return processor, batch


def stock_vlm_generate(
    model_path: str,
    processor: AutoProcessor,
    batch: dict[str, torch.Tensor],
    adapter_mod,
    max_new_tokens: int,
    ref_model=None,
) -> str:
    """Reference: stock ``AutoModelForImageTextToText.generate`` on ``batch``.

    Loaded via stock HF directly so the reference stays independent of the code
    under test. Returns the decoded **new** text (prompt tokens sliced off).

    Pass ``ref_model`` to reuse an already-loaded stock model.
    """
    from transformers import AutoModelForImageTextToText

    if ref_model is None:
        ref_model = load_ref_model(
            model_path=model_path,
            adapter_mod=adapter_mod,
            auto_model_cls=AutoModelForImageTextToText,
        )

    prompt_len = batch["input_ids"].shape[1]
    with torch.no_grad():
        gen = ref_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
    return text
