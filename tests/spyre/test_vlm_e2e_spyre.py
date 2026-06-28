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
End-to-end Spyre test for multimodal (image→text) VLM adapters.

For each registered ``kind="vlm"`` adapter in ``VISION_MODELS``, loads the full
VLM, prepares it for Spyre (both towers compiled), moves it to Spyre, and checks
the adapter against the stock model on a real hub image.

  test_vlm_generate_spyre[<key>]
    1. PREFILL agreement (the assertion): the adapter's first-token logits on
       Spyre must top-1-match stock HF's on CPU and stay close (cosine ≥
       ``MIN_COSINE``). This is the one comparison point free of greedy-fork
       amplification — see below.
    2. Free-run caption (a printed, non-degeneracy-checked diagnostic): the
       adapter's full ``generate`` output is decoded and printed next to stock's,
       and asserted non-empty. It is NOT required to match token-for-token.

Why not exact-string match? Both paths run the same fp16 dtype, so any
divergence is Spyre's native accumulation + the adapter's op decomposition
(``x*x*x`` gelu-tanh, matmul-RoPE, padded SDPA) versus CPU's fp16 — not a
precision tier. A 16-token greedy free-run can't be bit-stable across two
fp16 substrates; the prefill top-1 + cosine check catches a real regression
(wrong first token, large diff, NaN) without failing on benign drift.

Parametrized off ``VISION_MODELS``; selects ``kind="vlm"`` entries.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py
    pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm
"""

import gc
import importlib

import pytest
import torch
import torch.nn.functional as F
from _helpers import torch_dtype_for
from _vision_helpers import (
    build_vlm_batch,
    stock_vlm_first_token_logits,
    stock_vlm_generate,
)
from model_registry import VISION_MODELS

from hf_adapters.hf_common import _move_to_spyre_with_layout

MAX_NEW_TOKENS = 16
PROMPT = "Briefly describe this image."
# Prefill first-token logit agreement floor. granite-vision-4.1 measures 0.9999;
# 0.999 leaves margin for fp16 drift while still flagging a real regression.
MIN_COSINE = 0.999

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

    tokenizer = processor.tokenizer

    print(f"\n{'=' * 70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'=' * 70}")

    # --- Stock CPU references (run first, before prepare_for_spyre patches RMSNorm).
    # Prefill first-token logits are the assertion target; the free-run caption is a
    # printed diagnostic. ---
    print("  Running stock CPU reference (prefill logits + generate) ...")
    ref_logits = stock_vlm_first_token_logits(info["path"], batch, dtype)
    ref_text = stock_vlm_generate(info["path"], processor, batch, dtype, MAX_NEW_TOKENS)
    gc.collect()
    print(f"  stock:   {ref_text!r}")

    # --- Adapter on Spyre ---
    print("  Loading model for Spyre ...")
    model = adapter.load_hf_model(info["path"], dtype)
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, dtype)

    # Prefill first-token logits on Spyre (the assertion point — no greedy fork).
    print("  Running adapter prefill on Spyre ...")
    with torch.no_grad():
        logits, _, _ = adapter.prefill_logits(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["pixel_values"],
            batch["image_sizes"],
        )
    sp_first = logits.to("cpu")[0, -1, : ref_logits.shape[-1]].float()

    # Free-run caption — printed diagnostic + non-degeneracy check, not asserted
    # token-for-token (benign fp16 drift forks the greedy path; see module docstring).
    print("  Running adapter generate on Spyre ...")
    with torch.no_grad():
        adapter_text = _adapter_generate(
            adapter, model, processor, batch, MAX_NEW_TOKENS
        )
    del model
    gc.collect()

    # --- Prefill agreement (the assertion) ---
    ref_top1 = int(ref_logits.argmax())
    sp_top1 = int(sp_first.argmax())
    cosine = F.cosine_similarity(ref_logits, sp_first, dim=-1).item()
    max_diff = (ref_logits - sp_first).abs().max().item()
    print(
        f"  prefill top1: stock={ref_top1} {tokenizer.decode([ref_top1])!r}  "
        f"spyre={sp_top1} {tokenizer.decode([sp_top1])!r}  "
        f"cosine={cosine:.6f}  max|diff|={max_diff:.4f}"
    )
    print(f"  adapter: {adapter_text[0]!r}")
    print(f"  prompt:  {PROMPT!r}")

    assert len(adapter_text[0]) > 0, "adapter generated an empty string"
    assert sp_top1 == ref_top1, (
        f"adapter prefill first token diverged from stock:\n"
        f"  stock: {ref_top1} {tokenizer.decode([ref_top1])!r}\n"
        f"  spyre: {sp_top1} {tokenizer.decode([sp_top1])!r}"
    )
    assert cosine >= MIN_COSINE, (
        f"adapter prefill logits diverged from stock: cosine {cosine:.6f} "
        f"< {MIN_COSINE} (max|diff|={max_diff:.4f})"
    )
