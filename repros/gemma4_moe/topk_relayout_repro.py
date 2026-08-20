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

"""Repro: RUN topk, then FAIL to relayout its [T,K] output to [N,1].

This closes the gap left by ``rowweight_reshape_probe.py``, which used a
``randn(T, K)`` STAND-IN for the router weights and therefore never actually
ran topk. Here the ``[T,K]`` tensor whose ``reshape(N, 1)`` aborts is a
GENUINE ``torch.topk`` output: a cheap on-device softmax feeds
``w, idx = torch.topk(probs, K, dim=-1)``, and it is ``w`` (the topk VALUES)
that is reshaped to ``[N,1]`` and broadcast-multiplied against an ``[N,H]``
activation under the region's ``spyre_hint(tiles={"row": TILE})``. That is
the tail of the Gemma-4 26B-A4B MoE loop-on-topk region in
``hf_adapters/hf_gemma4_moe.py`` ``_compiled_moe_loop_region`` (lines ~419-435):

    probs  = torch.softmax(logits, dim=-1)      # [T,E]
    w, idx = torch.topk(probs, K, dim=-1)        # w=[T,K] fp16, idx=[T,K] int64
    w      = w / w.sum(-1, keepdim=True)         # [T,K]
    w      = w * per_expert_scale[idx]           # [T,K]  (dropped here; see note)
    row_w  = w.reshape(N, 1)                       # [N,1]  <-- reshape trigger
    ...
    row_out = row_out * row_w                     # [N,H] * [N,1]  buf1 ABORTS

This minimal bridge drops the ``w * per_expert_scale[idx]`` step: that step is
an indirect gather (a SEPARATE backend ask), and the reshape of ANY ``[T,K]``
fp16 buffer to ``[N,1]`` is what aborts regardless of whether its provenance is
topkvalue/sum or a gather output. Both are ``[T,K]`` fp16 partial-stick buffers
with the same layout and the same trigger.

On the Spyre backend ``torch.topk`` lowers via the ``spyre_topk`` decomp
(``torch_spyre/_inductor/decompositions.py:326``) to
``spyre.topkvalue`` / ``spyre.topkindex(...).to(int64)``; ``k > 4`` raises
``Unsupported("Topk is not supported for this config")``, so K must be <= 4.
(``idx`` is unused below, so ``topkindex`` may be DCE'd and its int64->int32
downcast warning may not fire -- harmless either way.)

EXACT ABORT (from the on-card layout gate, verbatim):

    InductorError: Unsupported: Spyre backend does not support:
      Multi-arg pointwise (buf1): no supported output layout found
      with size=[256, 2816] and coordinates=[d0, d1]
    propagate_layouts.py:1099  _multi_arg_pointwise_layouts

ROOT CAUSE (one line): the topk VALUE tensor is ``[T=64, K=4]`` fp16, so K=4
lives in a PARTIAL stick padded to 64 (a stick is 64 fp16 elements); reshaping
that ``[T,K]`` to ``[N=256,1]`` makes the layout pass read the partial-stick
buffer with a flat ``[256]`` index, giving the cross-stick expr
``Mod(d0,4)+floor(d0/4)`` (mod 4, not mod 64) that ``is_stick_expr_offset_free``
rejects, so ``_multi_arg_pointwise_layouts`` finds no output layout and raises
at ``propagate_layouts.py:1099`` (gate at :985). It is an UNREPRESENTABLE
cross-stick read, not a runtime bug and not the topk index dtype (already fixed).

CONTRAST: a device-native ``[N,1]`` operand that never passed through a ``[T,K]``
shape compiles fine (``native`` path below), pinning the cause to the topk-output
reshape and NOT to the ``[N,H]*[N,1]`` broadcast shape itself. The exit code
REQUIRES both (topk_reshape aborts AND native compiles), so a repro that stops
isolating -- e.g. if the broadcast itself became unsupported -- fails loudly.

The fix is a backend restickify (HBM round-trip) at the reshape, laying the
buffer N-outermost before any flat consumer -- see the backend-asks doc ask #5.
Sibling repros: ``repros/gemma4_moe/rowweight_reshape_probe.py`` (7-way
isolation with a randn stand-in) and ``repros/gemma4_moe/gatherbcast_layout_repro.py``
(the faithful full region). This file is the minimal "topk actually runs" bridge.

Run (Spyre host, NO model load, NO real weights):
    PYTHONPATH=<hf-adapters> python3 -u repros/gemma4_moe/topk_relayout_repro.py
"""
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device + backend
from torch_spyre._inductor.propagate_hints import spyre_hint

T = 64  # tokens
K = 4  # top-K (Spyre topk cap: k <= 4)
N = T * K  # 256 routed rows
H = 2816  # hidden_size
E = 128  # num_experts
TILE = 32  # row tile (matches the region's spyre_hint)

_LAYOUT_MSG = "no supported output layout found"


def _classify(name, fn, args):
    """Compile+run fn for spyre; classify OK / LAYOUT-ABORT / OTHER-ABORT."""
    try:
        out = torch.compile(fn, dynamic=False)(*args)
        out.cpu()
        print(f"  {name:<8}: OK  (out {tuple(out.shape)})")
        return "OK"
    except Exception as e:  # noqa: BLE001 - repro: classify, don't crash
        # The top InductorError.__str__ is a generic hint; the real layout
        # reason is on a chained cause. Walk __cause__/__context__ and join
        # every message so the classifier sees the nested layout text.
        parts, seen, cur = [], set(), e
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            parts.append(f"{type(cur).__name__}: {cur}")
            cur = cur.__cause__ or cur.__context__
        msg = "\n".join(parts)
        if _LAYOUT_MSG in msg:
            reason = next(
                (ln.strip() for ln in msg.splitlines() if _LAYOUT_MSG in ln), msg
            )
            print(f"  {name:<8}: LAYOUT-ABORT  {reason}")
            return "LAYOUT-ABORT"
        print(f"  {name:<8}: OTHER-ABORT   {msg.splitlines()[-1]}")
        return "OTHER-ABORT"


def main():
    torch.manual_seed(0)
    print(f"shapes: T={T} K={K} N={N} H={H} E={E} TILE={TILE}")

    # A small [T,E] logits tensor on device. We skip the full router (RMSNorm +
    # F.linear) -- the point is that topk RUNS; a cheap softmax of a device
    # tensor gives a valid probability distribution to select from.
    logits = torch.randn(T, E, dtype=torch.float16).to("spyre")
    x_nh = torch.randn(N, H, dtype=torch.float16).to("spyre")

    # PATH topk_reshape: RUN topk, reshape its VALUES [T,K] -> [N,1], broadcast.
    # This is the faithful region tail and the case under investigation.
    def path_topk_reshape(logits_te, x):
        with spyre_hint(tiles={"row": TILE}):
            probs = torch.softmax(logits_te, dim=-1)  # [T,E]
            w, idx = torch.topk(probs, K, dim=-1)  # w=[T,K] fp16, idx int64
            w = w / w.sum(-1, keepdim=True)  # [T,K] normalize
            row_w = w.reshape(N, 1)  # [T,K] -> [N,1]  <-- reshape trigger
            return x * row_w  # [N,H] * [N,1]  buf1 ABORTS

    # PATH native: identical broadcast multiply, but the [N,1] operand is a
    # device-native tensor that NEVER passed through a [T,K] shape. If this
    # compiles while path_topk_reshape aborts, the trigger is provably the
    # topk-output reshape across the K<64 partial stick, not the broadcast.
    w_n1_native = torch.randn(N, 1, dtype=torch.float16).to("spyre")

    def path_native(w_n1, x):
        with spyre_hint(tiles={"row": TILE}):
            return x * w_n1

    results = {}
    results["topk_reshape"] = _classify(
        "topk", path_topk_reshape, (logits, x_nh)
    )
    results["native"] = _classify("native", path_native, (w_n1_native, x_nh))

    print("\nsummary:")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")
    # The repro is only doing its job if the genuine topk-output reshape aborts
    # in the layout pass AND the device-native [N,1] contrast still compiles --
    # the pairing is what isolates the reshape as the trigger rather than the
    # [N,H]*[N,1] broadcast. Require BOTH so a repro that stops isolating (e.g.
    # if the broadcast itself became unsupported) fails loudly with exit 1.
    if results["topk_reshape"] == "LAYOUT-ABORT" and results["native"] != "OK":
        print("NOTE: native [N,1] contrast did NOT compile -- broadcast itself"
              " may now be unsupported; re-examine the isolation.")
    ok = (
        results["topk_reshape"] == "LAYOUT-ABORT"
        and results["native"] == "OK"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
