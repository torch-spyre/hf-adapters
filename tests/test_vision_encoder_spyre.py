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
Spyre compilation test for the Granite Vision encoder adapter.

Tests each component individually to identify what fails in torch-spyre/deeptools:
  1. Single vision encoder layer (SiglipEncoderLayer)
  2. Full vision tower (27 layers)
  3. Single WindowQFormerDownsampler projector
  4. Full pipeline (tower + projectors)

Usage:
    python tests/test_vision_encoder_spyre.py [layer|tower|projector|full]
"""

import sys
import time
import traceback

import torch

from hf_adapters.hf_granite_vision_encoder import (
    _load_vision_encoder,
    _make_vision_block,
    _make_projector_block,
    _pad_vision_attention,
    _run_vision_tower,
    _run_forward,
    prepare_for_spyre,
)
from hf_adapters.hf_common import DEVICE, BLOCK_SIZE, pad_mlp

MODEL_PATH = "ibm-granite/granite-vision-4.1-4b"


def test_single_vision_layer():
    """Test compiling and running a single SiglipEncoderLayer on Spyre."""
    print(f"\n{'='*70}")
    print("  TEST: Single vision encoder layer on Spyre")
    print(f"{'='*70}\n")

    model = _load_vision_encoder(MODEL_PATH, dtype=torch.float16)
    vision_config = model.vision_tower.vision_model.config
    orig_head_dim = vision_config.hidden_size // vision_config.num_attention_heads
    padded_head_dim = (
        ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)
    )

    layer = model.vision_tower.vision_model.encoder.layers[0]

    num_patches = (384 // 16) ** 2  # 576
    hidden_size = 1152

    # Pad attention heads before compiling
    if padded_head_dim > orig_head_dim:
        print(f"  Padding head_dim: {orig_head_dim} → {padded_head_dim}")
        _pad_vision_attention([layer], orig_head_dim, padded_head_dim,
                             vision_config.num_attention_heads)

    # Pad MLP intermediate size
    padded_mlp = pad_mlp(
        [layer], vision_config.intermediate_size,
        lambda l: (l.mlp.fc1, l.mlp.fc2),
    )
    print(f"  Padding MLP: {vision_config.intermediate_size} → {padded_mlp}")

    print(f"  Creating compiled block...")
    compiled_block = _make_vision_block(layer, padded_head_dim)

    print(f"  Moving layer to {DEVICE}...")
    layer.to(DEVICE)

    print(f"  Input: hidden_states [{1}, {num_patches}, {hidden_size}]")
    hidden_states = torch.randn(1, num_patches, hidden_size, dtype=torch.float16, device=DEVICE)

    print(f"  Running compiled block...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled_block(hidden_states)
    t1 = time.time()

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Time: {t1-t0:.3f}s")
    print(f"  Has NaN: {output.cpu().isnan().any().item()}")
    print(f"  PASS")
    return True


def test_full_vision_tower():
    """Test compiling and running the full 27-layer vision tower on Spyre."""
    print(f"\n{'='*70}")
    print("  TEST: Full vision tower (27 layers) on Spyre")
    print(f"{'='*70}\n")

    model = _load_vision_encoder(MODEL_PATH, dtype=torch.float16)
    prepare_for_spyre(model)

    print(f"  Moving model to {DEVICE}...")
    model.to(DEVICE)

    num_patches = (384 // 16) ** 2
    pixel_values = torch.randn(1, 3, 384, 384, dtype=torch.float16, device=DEVICE)

    print(f"  Input: pixel_values [{1}, 3, 384, 384]")
    print(f"  Running vision tower ({len(model._spyre_vision_blocks)} compiled blocks)...")

    t0 = time.time()
    with torch.no_grad():
        final_hidden, all_hidden = _run_vision_tower(model, pixel_values)
    t1 = time.time()

    print(f"  Final hidden shape: {list(final_hidden.shape)}")
    print(f"  Num hidden states: {len(all_hidden)}")
    print(f"  Time: {t1-t0:.3f}s")
    print(f"  Has NaN: {final_hidden.cpu().isnan().any().item()}")
    print(f"  PASS")
    return True


def test_single_projector():
    """Test compiling and running a single WindowQFormerDownsampler on Spyre."""
    print(f"\n{'='*70}")
    print("  TEST: Single WindowQFormerDownsampler projector on Spyre")
    print(f"{'='*70}\n")

    model = _load_vision_encoder(MODEL_PATH, dtype=torch.float16)
    prepare_for_spyre(model)
    projector = model.layerwise_projectors[0]

    num_patches = (384 // 16) ** 2  # 576
    hidden_size = 1152

    print(f"  Creating compiled projector block...")
    compiled_proj = _make_projector_block(projector)

    print(f"  Moving projector to {DEVICE}...")
    projector.to(DEVICE)

    print(f"  Input: image_features [{1}, {num_patches}, {hidden_size}]")
    image_features = torch.randn(1, num_patches, hidden_size, dtype=torch.float16, device=DEVICE)

    print(f"  Running compiled projector...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled_proj(image_features)
    t1 = time.time()

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Time: {t1-t0:.3f}s")
    print(f"  Has NaN: {output.cpu().isnan().any().item()}")
    print(f"  PASS")
    return True


def test_full_pipeline():
    """Test the complete vision pipeline on Spyre."""
    print(f"\n{'='*70}")
    print("  TEST: Full vision pipeline (tower + projectors) on Spyre")
    print(f"{'='*70}\n")

    model = _load_vision_encoder(MODEL_PATH, dtype=torch.float16)
    prepare_for_spyre(model)

    print(f"  Moving model to {DEVICE}...")
    model.to(DEVICE)

    pixel_values = torch.randn(1, 3, 384, 384, dtype=torch.float16, device=DEVICE)
    print(f"  Input: pixel_values [{1}, 3, 384, 384]")

    print(f"  Running full pipeline...")
    t0 = time.time()
    with torch.no_grad():
        all_features = _run_forward(model, pixel_values)
    t1 = time.time()

    print(f"  Produced {len(all_features)} feature groups:")
    for i, (llm_layer, feat) in enumerate(all_features):
        has_nan = feat.cpu().isnan().any().item()
        print(f"    [{i}] llm_layer={llm_layer}, shape={list(feat.shape)}, nan={has_nan}")

    print(f"  Time: {t1-t0:.3f}s")
    print(f"  PASS")
    return True


TESTS = {
    "layer": ("Single vision layer", test_single_vision_layer),
    "tower": ("Full vision tower", test_full_vision_tower),
    "projector": ("Single projector", test_single_projector),
    "full": ("Full pipeline", test_full_pipeline),
}


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(TESTS.keys())

    results = {}
    for key in which:
        if key not in TESTS:
            print(f"Unknown test: {key}. Options: {list(TESTS.keys())}")
            continue
        name, fn = TESTS[key]
        try:
            ok = fn()
            results[key] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"\n  !!! FAILED: {e}")
            traceback.print_exc()
            results[key] = f"ERROR: {e}"

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for key in which:
        if key in results:
            name, _ = TESTS[key]
            print(f"  {name:<30} {results[key]}")
    print(f"{'='*70}\n")
