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
End-to-end CPU test for multimodal (image→text) VLM adapters' generate loop.

For each registered multimodal VLM adapter, follows the documented VLM usage
shape (processor → generate → decode) on CPU and compares the adapter's
generated text against stock's REAL ``model.generate(**inputs)``.

  test_vlm_generate[<key>]
    Runs the adapter's multimodal generate (image + prompt → autoregressive
    decode → decoded string) on a real hub sample image and asserts the
    generated token sequence matches stock ``model.generate`` token-for-token.

Parametrized off ``model_registry.VISION_MODELS``; selects the multimodal
two-tower adapters.

Marked ``slow``: loads the full VLM twice and runs an autoregressive decode on
CPU. Run with ``--run-slow``.
"""

import gc

import pytest
import torch
from _vision_helpers import build_vlm_batch, stock_vlm_generate
from conftest import torch_dtype_for
from model_registry import VISION_MODELS

MAX_NEW_TOKENS = 16
PROMPT = "Briefly describe this image."

# Multimodal (image→text) vlm adapters load both towers and run a full generate.
MODELS = {k: v for k, v in VISION_MODELS.items() if v.get("kind") == "vlm"}


def _adapter_generate(adapter, model, processor, batch, max_new_tokens):
    """Drive an adapter's multimodal ``generate`` from a processor batch.

    Adapters take image inputs positionally (``input_ids, attention_mask,
    pixel_values, image_sizes``); map the batch onto that signature here so the
    shared harness stays signature-agnostic.
    """
    return adapter.generate(
        model,
        processor,
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_vlm_generate(model_key, load_adapter, unwrap_compiled_blocks):
    info = MODELS[model_key]
    adapter = load_adapter(info["adapter"])
    dtype = torch_dtype_for(info)

    processor, batch = build_vlm_batch(info["path"], PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    # --- Adapter generate (greedy) ---
    model = adapter.load_hf_model(info["path"], dtype)
    adapter.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)
    with torch.no_grad():
        adapter_text = _adapter_generate(
            adapter, model, processor, batch, MAX_NEW_TOKENS
        )
    del model
    gc.collect()

    # --- Stock reference: the FULL model.generate() (real deepstack) ---
    ref_text = stock_vlm_generate(info["path"], processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()

    # Print both captions so a human can eyeball the result (visible with -s).
    print(f"\n[{model_key} e2e] prompt: {PROMPT!r}")
    print(f"[{model_key} e2e] adapter: {adapter_text[0]!r}")
    print(f"[{model_key} e2e] stock:   {ref_text!r}")

    # The adapter must match stock's REAL multimodal generate token-for-token.
    assert adapter_text[0] == ref_text, (
        f"adapter generate diverged from stock generate:\n"
        f"  adapter: {adapter_text[0]!r}\n"
        f"  stock:   {ref_text!r}"
    )
    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
