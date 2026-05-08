"""
Accuracy test: compare vision encoder output on Spyre vs stock HF on CPU.

Runs the stock HuggingFace forward on CPU as reference, then runs the
Spyre adapter forward, and compares output feature values.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hf_adapters.hf_granite_vision_encoder import (
    _load_vision_encoder,
    _run_forward,
    prepare_for_spyre,
)

MODEL_PATH = "ibm-granite/granite-vision-4.1-4b"
DEVICE = "spyre"


def hf_reference_forward(model, pixel_values):
    """Run stock HF vision tower + projectors forward (no adapter patches)."""
    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values, output_hidden_states=True)

    all_features = []

    for proj_idx, (vision_layer, llm_layer) in enumerate(model._deepstack_layer_map):
        selected = vision_outputs.hidden_states[vision_layer]
        if model._vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        with torch.no_grad():
            projected = model.layerwise_projectors[proj_idx](selected)
        all_features.append((llm_layer, projected))

    if model.spatial_projectors is not None:
        spatial_feature = vision_outputs.hidden_states[model._spatial_vision_layer]
        if model._vision_feature_select_strategy == "default":
            spatial_feature = spatial_feature[:, 1:]
        for group_idx, llm_layer in enumerate(model._spatial_target_layers):
            with torch.no_grad():
                projected = model.spatial_projectors[group_idx](spatial_feature)
            all_features.append((llm_layer, projected))

    return all_features


def main():
    print(f"\n{'='*70}")
    print(f"  Vision Encoder Accuracy: Spyre vs CPU (stock HF)")
    print(f"  Model: {MODEL_PATH}")
    print(f"{'='*70}\n")

    dtype = torch.float16
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 384, 384, dtype=dtype)

    # --- CPU reference (stock HF, no adapter patches) ---
    print("  Loading model for CPU reference...")
    model_ref = _load_vision_encoder(MODEL_PATH, dtype=dtype)
    model_ref.eval()
    print("  Running stock HF forward on CPU...")
    ref_results = hf_reference_forward(model_ref, pixel_values)
    print(f"  Produced {len(ref_results)} feature groups")
    del model_ref

    # --- Spyre (adapter forward) ---
    print(f"\n  Loading model for Spyre...")
    model = _load_vision_encoder(MODEL_PATH, dtype=dtype)
    prepare_for_spyre(model)
    model.to(DEVICE)
    print(f"  Running adapter forward on Spyre...")
    with torch.no_grad():
        spyre_results = _run_forward(model, pixel_values.clone(), )

    print(f"  Produced {len(spyre_results)} feature groups")

    # --- Compare ---
    print(f"\n{'='*70}")
    print(f"  {'Group':<8} {'LLM Layer':<12} {'Shape':<20} {'Max Diff':<12} {'Mean Diff':<12} {'Status'}")
    print(f"  {'-'*68}")

    all_match = True
    for i, ((ref_layer, ref_feat), (sp_layer, sp_feat)) in enumerate(
        zip(ref_results, spyre_results)
    ):
        assert ref_layer == sp_layer, f"Layer mismatch: {ref_layer} vs {sp_layer}"

        ref_t = ref_feat.float()
        sp_t = sp_feat.cpu().float()

        diff = (sp_t - ref_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Tolerance: fp16 through 27 ViT layers + padded attention (72->128) +
        # QFormer + downsampler. Head padding causes systematic drift.
        # CPU adapter test uses max_diff < 3.0; Spyre adds hardware numerics.
        match = max_diff < 5.0
        if not match:
            all_match = False

        status = "PASS" if match else "FAIL"
        print(f"  {i:<8} {ref_layer:<12} {str(list(sp_t.shape)):<20} {max_diff:<12.4f} {mean_diff:<12.6f} {status}")

    print(f"  {'-'*68}")
    if all_match:
        print("  RESULT: ALL PASS")
    else:
        print("  RESULT: MISMATCH")
    print(f"{'='*70}\n")

    return all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
