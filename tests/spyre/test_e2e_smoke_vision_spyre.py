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
  - TTFT — Time to First Token (ms): measured from within the shared generate
    loop (prefill + first decode step in a single pass).
  - ITL  — Inter-Token Latency (ms): average time per token after the first.
  - Total generation time (s) for MAX_NEW_TOKENS tokens

The model is loaded once per parametrized model_path, then all 5 sample images
from SMOKE_TEST_IMAGES (_vision_helpers.py) are run through it in sequence.
Images are sourced from the public ``huggingface/documentation-images`` dataset —
no local files required.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_smoke_vision_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_smoke_vision_spyre.py -k granite
"""

import time
from typing import Any

import pytest
import torch

from hf_adapters import AutoSpyreModelForImageTextToText
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests._vision_helpers import build_vlm_batch, load_smoke_test_images
from tests.model_registry import (
    NON_BLOCKING_VISION_MODELS,
    VISION_PATHS,
    xfail_non_blocking,
)

pytestmark = pytest.mark.model_harness("vision")

# Number of tokens to generate per image. Must be ≥2 so that ITL
# (avg of tokens 2..N) is well-defined.
MAX_NEW_TOKENS = 16


def _run_single_image(
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

    TTFT and ITL are printed by the shared generate loop via ``timing=True``.
    """
    processor_i, batch = build_vlm_batch(model_path, prompt, image)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)

    # --- Full generation (timing printed by the shared generate loop) ---------
    prompt_len = batch["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        sequences = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            timing=True,
        )
    gen_time_s = time.time() - t0

    generated_ids = sequences[0, prompt_len:]
    output_text = processor_i.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- Validation -----------------------------------------------------------
    gen_ids = generated_ids.tolist()
    checks: dict[str, Any] = {
        "non_empty": len(output_text.strip()) > 0,
        "not_all_spaces": output_text.strip() != "",
    }
    if gen_ids:
        checks["has_tokens"] = True
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
    TTFT and ITL are printed to stdout by the shared generate loop.
    """
    print(f"\n{'=' * 70}")
    print(f"  loading from {model_path}")
    print(f"{'=' * 70}")

    dtype = dtype_for_model_path(model_path, target_device="spyre")

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
        result = _run_single_image(model, dtype, model_path, label, prompt, image)
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
