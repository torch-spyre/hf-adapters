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
Gate 1: Gemma 4 MoE grouped/batched matmul on Spyre.

Tests that both Option 4A row-batched and dense expert-batched matmul geometries
compile on Spyre and produce numerically correct results.

Correctness criterion: Compare device fp16 output against fp32 CPU ground truth.
For a dot product of reduction length H=2816, fp16 accumulation error is expected
to scale as O(H * eps_fp16) ≈ O(1e-2) relative to the output scale. With outputs
O(100–250), this translates to absolute error O(1–2.5). We use relative error
vs. a clipped denominator to avoid noise from near-zero elements:

  denom = ref32.abs().clamp_min(1.0)
  rel_err = |got_fp32 - ref32| / denom
  Tolerance: max_rel < 0.5 (conservative), mean_rel < 0.02 (tight bound on
  systematic fp16 accumulation error; normal is ~0.007).

Outcome (2026-07-31):
---------------------
PASS. Both geometries compile without abort and are numerically correct.

Row-batched [256,1,2816]×[256,2816,704]:
  - Compilation: OK, no out_reuse_dim abort
  - Mean relative error (vs fp32): 0.73%
  - Max relative error: 0.45 (clipped denom)
  - Mean absolute error: 0.146
  - Verdict: PASS

Expert-batched [8,32,2816]×[8,2816,704]:
  - Compilation: OK, no out_reuse_dim abort
  - Mean relative error (vs fp32): 0.74%
  - Max relative error: 0.42 (clipped denom)
  - Mean absolute error: 0.145
  - Verdict: PASS

Both Option 4A row-batched and dense expert-batched geometries are viable for MoE.
Measured errors are within expected fp16 accumulation bounds for H=2816.
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch_spyre
torch_spyre._autoload()

import torch


H, F, E, T, K = 2816, 704, 8, 32, 8  # small E/T to iterate fast; F=moe_intermediate_size


def row_batched(a, w):           # 4A geometry: one weight per row
    return torch.bmm(a, w)       # [N,1,H] x [N,H,F] -> [N,1,F]


def expert_batched(a, w):        # dense geometry: all experts
    return torch.bmm(a, w)       # [E,T,H] x [E,H,F] -> [E,T,F]


def check(fn, a_shape, w_shape):
    """
    Verify one matmul geometry compiles on Spyre and matches fp32 ground truth.
    Uses relative error with clipped denominator to avoid near-zero noise.
    """
    print(f"\nTesting {a_shape} x {w_shape}")
    a_fp16 = torch.randn(*a_shape, dtype=torch.float16)
    w_fp16 = torch.randn(*w_shape, dtype=torch.float16)

    # fp32 ground truth on CPU
    a_fp32 = a_fp16.float()
    w_fp32 = w_fp16.float()
    ref_fp32 = fn(a_fp32, w_fp32)
    print(f"  ref_fp32 shape: {ref_fp32.shape}, min/max: {ref_fp32.min():.4f} / {ref_fp32.max():.4f}")

    # Compile and run on Spyre (fp16)
    cfn = torch.compile(fn, dynamic=False)
    a_spyre = a_fp16.to("spyre")
    w_spyre = w_fp16.to("spyre")
    got_fp16 = cfn(a_spyre, w_spyre).cpu()
    got_fp32 = got_fp16.float()
    print(f"  got_fp32 shape: {got_fp32.shape}, min/max: {got_fp32.min():.4f} / {got_fp32.max():.4f}")

    # Relative error with clipped denominator to avoid near-zero noise
    denom = ref_fp32.abs().clamp_min(1.0)
    rel_err = (got_fp32 - ref_fp32).abs() / denom
    abs_err = (got_fp32 - ref_fp32).abs()

    print(f"  Mean relative error: {rel_err.mean():.4%}")
    print(f"  Max relative error: {rel_err.max():.4f}")
    print(f"  Mean absolute error: {abs_err.mean():.6f}")
    print(f"  Max absolute error: {abs_err.max():.6f}")

    # Tolerance: mean_rel < 2% (normal fp16 accumulation), max_rel < 50% (conservative)
    assert rel_err.mean() < 0.02, f"Mean rel error {rel_err.mean():.4%} exceeds 2% limit"
    assert rel_err.max() < 0.5, f"Max rel error {rel_err.max():.4f} exceeds 0.5 limit"
    print(f"OK {a_shape} x {w_shape}")


if __name__ == "__main__":
    results = []

    try:
        check(row_batched, (T * K, 1, H), (T * K, H, F))
        results.append(("row_batched", "PASS"))
    except Exception as e:
        print(f"Row-batched FAILED: {e}")
        results.append(("row_batched", f"FAIL: {e}"))

    try:
        check(expert_batched, (E, T, H), (E, H, F))
        results.append(("expert_batched", "PASS"))
    except Exception as e:
        print(f"Expert-batched FAILED: {e}")
        results.append(("expert_batched", f"FAIL: {e}"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name}: {status}")
