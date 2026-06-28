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
End-to-end Spyre test for multimodal (image→text) VLM adapters' generate loop.

For each registered ``kind="vlm"`` adapter in ``VISION_MODELS``, loads the
full VLM, prepares it for Spyre (both towers compiled), moves it to Spyre, and
runs the adapter's ``generate`` against a real hub image.  The adapter's decoded
output is compared token-for-token against the stock ``model.generate`` run on
CPU.

  test_vlm_generate_spyre[<key>]
    Adapter generate (image + prompt → Spyre decode → decoded string) must
    match stock ``model.generate`` on CPU token-for-token.

Parametrized off ``VISION_MODELS``; selects ``kind="vlm"`` entries.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py
    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm
"""

import gc
import importlib

import pytest
import torch
from _helpers import torch_dtype_for
from _vision_helpers import build_vlm_batch, stock_vlm_generate
from model_registry import VISION_MODELS

from hf_adapters.hf_common import _move_to_spyre_with_layout

MAX_NEW_TOKENS = 16
PROMPT = "Briefly describe this image."

MODELS = {k: v for k, v in VISION_MODELS.items() if v.get("kind") == "vlm"}


def _adapter_generate(adapter, model, processor, batch, max_new_tokens):
    """Drive an adapter's multimodal ``generate`` from a processor batch."""
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


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_vlm_generate_spyre(model_key):
    info = MODELS[model_key]
    adapter_module_name = info["adapter"].replace(".py", "")
    adapter = importlib.import_module(f"hf_adapters.{adapter_module_name}")
    dtype = torch_dtype_for(info)

    processor, batch = build_vlm_batch(info["path"], PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    print(f"\n{'=' * 70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'=' * 70}")

    # --- Stock CPU reference (run first, before prepare_for_spyre patches RMSNorm) ---
    print("  Running stock CPU reference ...")
    ref_text = stock_vlm_generate(info["path"], processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()
    print(f"  stock:   {ref_text!r}")

    # --- Adapter generate on Spyre ---
    print("  Loading model for Spyre ...")
    model = adapter.load_hf_model(info["path"], dtype)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, dtype)
    print("  Running adapter generate on Spyre ...")
    with torch.no_grad():
        adapter_text = _adapter_generate(
            adapter, model, processor, batch, MAX_NEW_TOKENS
        )
    del model
    gc.collect()

    print(f"  adapter: {adapter_text[0]!r}")
    print(f"  prompt:  {PROMPT!r}")

    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
    assert adapter_text[0] == ref_text, (
        f"adapter generate diverged from stock generate:\n"
        f"  adapter: {adapter_text[0]!r}\n"
        f"  stock:   {ref_text!r}"
    )
