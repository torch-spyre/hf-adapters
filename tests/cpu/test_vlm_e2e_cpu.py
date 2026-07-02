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

Parametrized off ``model_registry.VISION_PATHS`` (the multimodal two-tower
adapters that load both towers and run a full generate).

Marked ``slow``: loads the full VLM twice and runs an autoregressive decode on
CPU. Run with ``--run-slow``.

pytest -s -vvv tests/cpu/test_vlm_e2e_cpu.py -k granite_vision_mm

"""

import gc
import types

import pytest
import torch

from hf_adapters.auto_spyre_model import (
    IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)
from tests._vision_helpers import build_vlm_batch, stock_vlm_generate
from tests.conftest import load_hf_vlm, torch_dtype_for_model_path
from tests.model_registry import VISION_PATHS

MAX_NEW_TOKENS: int = 16
PROMPT: str = "Briefly describe this image."


def _adapter_generate(
    adapter: types.ModuleType,
    model: torch.nn.Module,
    processor,
    batch: dict,
    max_new_tokens: int,
) -> list[str]:
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
@pytest.mark.parametrize("model_path", VISION_PATHS, ids=VISION_PATHS)
def test_vlm_generate(model_path: str, unwrap_compiled_blocks) -> None:
    adapter = resolve_adapter_module(
        model_path, mapping=IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )
    dtype = torch_dtype_for_model_path(model_path)

    processor, batch = build_vlm_batch(model_path, PROMPT)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    # --- Adapter generate (greedy) ---
    model = load_hf_vlm(model_path, dtype, adapter_mod=adapter)
    adapter.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)
    with torch.no_grad():
        adapter_text = _adapter_generate(
            adapter, model, processor, batch, MAX_NEW_TOKENS
        )
    del model
    gc.collect()

    # --- Stock reference: the FULL model.generate() (real deepstack) ---
    ref_text = stock_vlm_generate(model_path, processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()

    # Print both captions so a human can eyeball the result (visible with -s).
    print(f"\n[{model_path} e2e] prompt: {PROMPT!r}")
    print(f"[{model_path} e2e] adapter: {adapter_text[0]!r}")
    print(f"[{model_path} e2e] stock:   {ref_text!r}")

    # The adapter must match stock's REAL multimodal generate token-for-token.
    assert adapter_text[0] == ref_text, (
        f"adapter generate diverged from stock generate:\n"
        f"  adapter: {adapter_text[0]!r}\n"
        f"  stock:   {ref_text!r}"
    )
    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
