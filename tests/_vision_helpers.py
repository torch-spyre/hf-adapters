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

import torch


def make_pixel_values(num_channels, image_size, batch=1, seed=0):
    """Deterministic ``pixel_values`` ``[batch, num_channels, image_size, image_size]``.

    Float32 in roughly the normalized pixel range a processor would emit. The
    exact distribution is irrelevant — accuracy is measured as adapter-vs-stock
    agreement on the *same* input, so any fixed tensor works as long as it is
    reproducible.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch, num_channels, image_size, image_size, generator=g, dtype=torch.float32
    )


def pixel_values_for_config(vision_config, batch=1, seed=0):
    """Build ``pixel_values`` matching a vision config's channels/resolution."""
    return make_pixel_values(
        vision_config.num_channels, vision_config.image_size, batch=batch, seed=seed
    )


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


def load_sample_image():
    """A real, recognizable hub image (a chonky cat) so a caption is judgeable.

    Downloaded at test time — no committed fixture. Human-eyeballable output is
    a deliberate secondary signal on top of the token-exact assertion.
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image

    path = hf_hub_download(**SAMPLE_IMAGE)
    return Image.open(path).convert("RGB")


def build_vlm_batch(model_path, prompt, image=None):
    """Processor + tokenized (image + prompt) batch, the official VLM way.

    Embeds the image in the conversation and lets ``apply_chat_template`` tokenize
    and expand image tokens in one call (``tokenize=True, return_dict=True``). The
    two-step ``processor(text=..., images=...)`` path under-tiles anyres images and
    mis-aligns image tokens, so the documented single-call path is used instead.

    Sets ``padding_side='left'`` to match the adapters' right-aligned decode
    convention. Returns ``(processor, batch)``; ``batch`` carries whatever image
    inputs the model needs (``pixel_values``, ``image_sizes``, …).
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    if image is None:
        image = load_sample_image()
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


def stock_vlm_generate(model_path, processor, batch, dtype, max_new_tokens):
    """Reference: stock ``AutoModelForImageTextToText.generate`` on ``batch``.

    Loaded via stock HF directly so the reference stays independent of the code
    under test. Returns the decoded **new** text (prompt tokens sliced off).
    """
    import torch
    from transformers import AutoModelForImageTextToText

    ref_model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="cpu"
    ).eval()
    prompt_len = batch["input_ids"].shape[1]
    with torch.no_grad():
        gen = ref_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
    del ref_model
    return text
