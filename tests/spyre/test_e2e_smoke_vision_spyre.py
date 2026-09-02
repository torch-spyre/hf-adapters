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
E2E smoke test: load HF vision model on Spyre, generate tokens across multiple
images, verify non-trivial output for each.

Verifies (per image):
  - generate() produces non-empty output
  - Generated tokens are not all-zero or all-same

Note: This test does NOT check the accuracy or factual correctness of inference
results — generated text must be reviewed manually to assess answer quality.
A status of PASS means inference completed successfully and produced non-degenerate
output; it does not indicate the model answered correctly.

Reports (per image):
  - TTFT — Time to First Token (ms): measured from within the adapter's generate
    loop (prefill + first decode step in a single pass). Adapters that do not yet
    support native timing show ``n/a``.
  - ITL  — Inter-Token Latency (ms): average time per token after the first.
    Adapters that do not yet support native timing show ``n/a``.
  - Total generation time (s) for MAX_NEW_TOKENS tokens

The model is loaded once per parametrized model_path, then all 5 sample images
from SMOKE_TEST_IMAGES (_vision_helpers.py) are run through it in sequence.
Images are sourced from the public ``huggingface/documentation-images`` dataset —
no local files required.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_smoke_vision_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_smoke_vision_spyre.py -k granite
"""

import inspect
import time
import types
from typing import Any

import pytest
import torch
from _vision_helpers import (
    build_vlm_batch,
    extra_image_inputs,
    load_smoke_test_images,
)
from model_registry import NON_BLOCKING_VISION_MODELS, VISION_PATHS, xfail_non_blocking

from hf_adapters import AutoSpyreModelForImageTextToText
from hf_adapters.auto_spyre_model import (
    IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    dtype_for_model_path,
    resolve_adapter_module,
)

pytestmark = pytest.mark.model_harness("vision")

# Number of tokens to generate per image. Must be ≥2 so that ITL
# (avg of tokens 2..N) is well-defined.
MAX_NEW_TOKENS = 16


def _adapter_generate(
    adapter: types.ModuleType,
    model: Any,
    processor: Any,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> list[str]:
    """Drive an adapter's multimodal ``generate`` from a processor batch.

    ``timing=True`` is forwarded to adapters that declare a ``timing`` parameter
    in their ``generate`` signature. Adapters that do not support it yet receive
    no timing argument.
    """
    extra = extra_image_inputs(adapter.generate, batch)
    sig = inspect.signature(adapter.generate)
    timing_kwarg = {"timing": True} if "timing" in sig.parameters else {}
    return adapter.generate(
        model,
        processor,
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **extra,
        **timing_kwarg,
    )


def _run_single_image(
    adapter: types.ModuleType,
    model: Any,
    dtype: torch.dtype,
    model_path: str,
    label: str,
    prompt: str,
    image: Any,
) -> dict[str, Any]:
    """Run generate for one image and return a per-image result dict.

    The model is already loaded and on-device — this only handles batch
    preparation, the timed generate call, and validation.

    TTFT and ITL are measured natively from within the adapter's generate loop
    for adapters that support ``timing=True``. For adapters that do not yet
    support it, ``ttft_ms`` and ``itl_ms`` are returned as ``None``.
    """
    processor_i, batch = build_vlm_batch(model_path, prompt, image)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    # --- Full generation (timing printed by adapter if supported) -------------
    t0 = time.time()
    with torch.no_grad():
        outputs = _adapter_generate(
            adapter, model, processor_i, batch, max_new_tokens=MAX_NEW_TOKENS
        )
    gen_time_s = time.time() - t0

    output_text = outputs[0] if outputs else ""

    # --- Validation -----------------------------------------------------------
    tokenizer = processor_i.tokenizer
    checks: dict[str, Any] = {
        "non_empty": len(output_text.strip()) > 0,
        "not_all_spaces": output_text.strip() != "",
    }
    if output_text:
        gen_ids = tokenizer.encode(output_text, add_special_tokens=False)
        checks["has_tokens"] = len(gen_ids) > 0
        checks["not_all_zero"] = not all(t == 0 for t in gen_ids)
        checks["not_all_same"] = len(set(gen_ids)) > 1 or len(gen_ids) <= 1
        checks["token_ids"] = gen_ids
    else:
        checks["has_tokens"] = False
        checks["not_all_zero"] = False
        checks["not_all_same"] = False
        checks["token_ids"] = []

    passed = all(v for k, v in checks.items() if k != "token_ids")
    return {
        "label": label,
        "prompt": prompt,
        "status": "PASS" if passed else "FAIL",
        "tokens": len(checks.get("token_ids", [])),
        "text": output_text[:50],
        "gen_s": gen_time_s,
        "checks": checks,
    }


def run_vision_smoke_test(model_path: str) -> dict[str, Any]:
    """Load model once, then run all SMOKE_TEST_IMAGES through it in sequence.

    Returns a result dict with per-image results and overall load time.
    TTFT and ITL are printed to stdout by the adapter during generation for
    adapters that support native timing.
    """
    print(f"\n{'=' * 70}")
    print(f"  loading from {model_path}")
    print(f"{'=' * 70}")

    dtype = dtype_for_model_path(model_path, target_device="spyre")
    adapter = resolve_adapter_module(
        model_path, mapping=IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING
    )

    t0 = time.time()
    model = AutoSpyreModelForImageTextToText.from_pretrained(
        model_name_or_path=model_path, dtype=dtype
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Download all sample images once (cached after first run).
    smoke_images = load_smoke_test_images()
    print(f"  Running {len(smoke_images)} images through model ...")

    image_results = []
    for label, prompt, image in smoke_images:
        print(f"\n  [{label}] prompt: {prompt!r}")
        result = _run_single_image(
            adapter, model, dtype, model_path, label, prompt, image
        )
        print(f"  [{label}] output: {result['text']!r}")
        print(f"  [{label}] gen: {result['gen_s']:.1f}s  status: {result['status']}")
        image_results.append(result)

    overall = "PASS" if all(r["status"] == "PASS" for r in image_results) else "FAIL"
    return {
        "model": model_path,
        "status": overall,
        "load_s": load_time,
        "images": image_results,
    }


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(VISION_PATHS, table=NON_BLOCKING_VISION_MODELS)
)
def test_e2e_smoke_vision_spyre(model_path: str) -> None:
    result = run_vision_smoke_test(model_path)

    print("\n## E2E Vision Smoke Test Results\n")
    print("| Image | Status | Tokens | Generated Text | Gen (s) |")
    print("|-------|--------|--------|----------------|---------|")
    for r in result["images"]:
        print(
            f"| {r['label']} | {r['status']} | {r['tokens']} "
            f"| {r['text']!r} | {r['gen_s']:.1f} |"
        )
    print(f"\nModel load: {result['load_s']:.1f}s  Overall: {result['status']}")

    assert result["status"] == "PASS", result
