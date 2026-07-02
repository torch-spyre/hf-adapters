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

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

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


def load_sample_image() -> Image.Image:
    """A real, recognizable hub image (a chonky cat) so a caption is judgeable.

    Downloaded at test time — no committed fixture. Human-eyeballable output is
    a deliberate secondary signal on top of the token-exact assertion.
    """
    path = hf_hub_download(**SAMPLE_IMAGE)
    return Image.open(path).convert("RGB")


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


def stock_vlm_generate(
    model_path: str,
    processor: AutoProcessor,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    """Reference: stock ``AutoModelForImageTextToText.generate`` on ``batch``.

    Loaded via stock HF directly so the reference stays independent of the code
    under test. Returns the decoded **new** text (prompt tokens sliced off).
    """
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


def stock_vlm_greedy_steps(
    model_path: str,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
    num_steps: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """Stock HF per-step greedy logits + token ids over prefill + decode.

    Runs ``AutoModelForImageTextToText.generate`` greedily for ``num_steps``
    tokens with ``output_logits=True`` and returns ``(logits, token_ids)``:

    - ``logits``: list of ``num_steps`` fp32 ``[vocab]`` tensors — the
      distribution stock greedily picked each generated token from (step 0 =
      prefill / first token, step k = after k generated tokens).
    - ``token_ids``: the ``num_steps`` greedily chosen ids (``logits[i].argmax()``).

    The Spyre e2e test uses ``token_ids`` as the teacher-forcing sequence and
    ``logits`` as the per-step top-1 reference, so the adapter is compared on the
    *same* prefix at every step (no greedy-fork amplification).
    """
    from transformers import AutoModelForImageTextToText

    ref_model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="cpu"
    ).eval()
    with torch.no_grad():
        gen = ref_model.generate(
            **batch,
            max_new_tokens=num_steps,
            do_sample=False,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
    logits = [step[0].float().clone() for step in gen.logits]
    prompt_len = batch["input_ids"].shape[1]
    token_ids = gen.sequences[0, prompt_len : prompt_len + num_steps].tolist()
    del ref_model
    return logits, token_ids


def stock_vlm_greedy_hidden_steps(
    model_path: str,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
    num_steps: int,
) -> tuple[list[torch.Tensor], list[int], torch.Tensor]:
    """Stock HF per-step greedy last-layer hidden state + token ids.

    Sister of ``stock_vlm_greedy_steps`` — same greedy ``generate`` call, but
    returns the text decoder's post-final-norm hidden state (what ``lm_head``
    reads) instead of the post-``lm_head`` logits. Used by the VLM embedding
    e2e test to teacher-force the adapter against the same greedy prefix and
    compare last_hidden_state per step.

    Returns
      - ``hidden_steps``: list of ``num_steps`` fp32 tensors.
          Step 0  : ``[prompt_len, H]`` — full prefill last-layer hidden state,
                    one row per input token (post-final-norm).
          Step k>0: ``[H]``             — hidden state at the newly-decoded token.
      - ``token_ids``: ``num_steps`` greedily chosen ids (same sequence
        ``stock_vlm_greedy_steps`` returns), used as the teacher-forcing prefix.
      - ``prefill_attention_mask``: ``[prompt_len]`` bool, so the step-0
        per-token cosine can be restricted to real (non-padded) positions.
    """
    from transformers import AutoModelForImageTextToText

    ref_model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="cpu"
    ).eval()
    with torch.no_grad():
        gen = ref_model.generate(
            **batch,
            max_new_tokens=num_steps,
            do_sample=False,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    # gen.hidden_states: tuple of len num_steps; each step is a tuple of per-layer
    # tensors, [-1] being post-final-norm. Step 0: [B, prompt_len, H]; step k>0: [B, 1, H].
    hidden_steps: list[torch.Tensor] = []
    for step_i, layers in enumerate(gen.hidden_states):
        last_layer = layers[-1][0].float().clone()
        hidden_steps.append(last_layer if step_i == 0 else last_layer[0])
    prompt_len = batch["input_ids"].shape[1]
    token_ids = gen.sequences[0, prompt_len : prompt_len + num_steps].tolist()
    prefill_amask = batch["attention_mask"][0].bool().clone()
    del ref_model
    return hidden_steps, token_ids, prefill_amask
