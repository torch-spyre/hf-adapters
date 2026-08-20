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

"""Isolated topk repro on the Gemma-4 26B-A4B MoE router shapes.

The MoE router (``hf_adapters/hf_gemma4_moe.py`` ``_moe_route`` /
``_compiled_moe_loop_region``) selects the top-K experts per token with:

    logits = F.linear(x, W_router)   # [T, E]
    probs  = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)   # values [T,K], indices [T,K]

The op under test here is *only* that ``torch.topk`` step, compiled on Spyre,
on the exact real-model shape so we can see how it behaves in isolation
(nothing else in the graph to confuse the diagnosis).

Real ``google/gemma-4-26B-A4B-it`` router facts (verified against the HF config):
  * hidden_size H          = 2816   (irrelevant to topk; here for provenance)
  * num_experts E          = 128    <-- topk operates over this axis (dim=-1)
  * top_k_experts          = 8      <-- what the MODEL actually wants
  * moe_intermediate_size  = 704
  * bring-up K             = 4      <-- Spyre topk is documented to cap at k<=4

Token count T is the number of rows fed through the router at once; the gates
use T=64, so probs is a [64, 128] tensor.  We sweep T over {1, 64} to cover the
single-token decode row and the multi-token gate shape.

For each (T, K, dtype) case this script:
  1. builds a probability-like input the same way the router does
     (softmax over random logits), so ties are effectively absent and the
     top-K is well defined;
  2. runs eager CPU topk (fp32) as ground truth;
  3. compiles ``torch.topk(x, K, dim=-1)`` for Spyre and runs it on device;
  4. compares BOTH returned values and returned indices, and reports the exact
     failure category:
       - COMPILE-ABORT ...... device compile raised (e.g. the k>4 Unsupported)
       - INDEX-DTYPE ........ indices came back non-integer (fp16) -> see the
                              ``idx.to(torch.int64)`` workaround in the adapter
       - INDEX-WRONG ........ index VALUES disagree with the CPU top-K set
       - VALUE-WRONG ........ top-K VALUES disagree beyond fp16 tolerance
       - OK ................. values + index-set both match

After the sweep it runs three targeted single-mode probes (verified against the
torch-spyre source on 2026-08-04):
  * probe_rank3 .................... Spyre topk is 2-D only; a 3-D input raises
                                     "topk only implemented for 2-D tensors"
                                     (``customops.py``). The router must flatten
                                     [B,T,E] -> [T,E] before topk.
  * probe_largest_false ........... the decomp drops the ``largest``/``sorted``
                                     kwargs (``decompositions.py``), so
                                     ``largest=False`` silently returns the
                                     LARGEST K -- a wrong-answer trap.
  * probe_index_gather_consequence  the downstream effect of the fp16-index
                                     dtype: ``per_expert_scale[idx]`` with vs
                                     without the ``.to(int64)`` cast (the exact
                                     line the adapter needed).

Known-good baseline (from torch-spyre's own test suite): the closest existing
unit test covers ``(64, 256) k=4 dim=-1``; the halved reduction width used here
(128 = 2 sticks) is NOT directly tested, which is part of why an isolated repro
on the real router shape is worth having.

The point is to give the MoE agent a single, dependency-free artifact that
exercises topk on the shapes that matter and says *precisely* what breaks.

Run (on the Spyre test host, when the card is free):
  cd hf-adapters
  python3 -u repros/gemma4_moe/topk_router_shapes.py

Optionally restrict to one K:  ``TOPK_ONLY_K=8 python3 -u repros/...``
(single-K mode runs only the sweep, not the targeted probes).

NOTE: the Spyre VFIO device is single-tenant. If another process holds it (e.g.
a concurrent gate run) every case reports DEVICE-UNAVAIL, which is an
environment condition, NOT a topk result. Check ``ps aux | grep spyre`` first.

OBSERVED ON-DEVICE (aviros-spyre-test, 2026-08-04) -- richer than the
source-read predicted:
  * T=64 K=4 fp16 ....... FAIL: INDEX-DTYPE. topk LOWERS and RUNS; indices come
                          back torch.float16 (values fine). This is the one
                          case that produced usable output.
  * T=1  K=4 fp16/fp32 .. COMPILE-ABORT "AllSameNode.from_args: out_layouts is
                          empty". A single-row [1,128] topk does NOT lower --
                          a SHAPE limit the source read did not predict. The
                          router's decode step (one token = one row) hits this.
  * T=64 K=4 fp32 ....... COMPILE-ABORT "ReStickifyOpHBM on IEEE_FP32". The
                          fp32 topk path can't restickify; fp16 is the path.
  * K=8 (all) ........... COMPILE-ABORT (the k>4 cap), as predicted.
  * rank-3 probe ........ COMPILE-ABORT, as predicted (2-D only).

  * gather probe (THE KEY RESULT) -- BOTH the no-cast AND the ``.to(int64)``
    forms abort identically:
        "aten.index.Tensor: indices must be int64, byte or bool.
         Got [torch.float16]"
    The ``spyre::topkindex`` FAKE declares dtype=int64 (customops.py:162), so
    at the FX-graph level the index tensor is ALREADY typed int64 -- Inductor
    therefore ELIMINATES the user's ``idx.to(torch.int64)`` as a no-op. But the
    real device buffer topkindex builds is fp16 (lowering uses x.get_dtype()),
    and the generic ``aten.index`` lowering checks the REAL buffer dtype and
    asserts. => the adapter's ``idx = idx.to(torch.int64)`` workaround CANNOT
    fix this in a pure-device graph (the cast has nothing to bite on). The fix
    must be in torch-spyre: make ``topkindex`` produce an int32/int64 device
    buffer (or add a real fp16->int convert the lowering won't fold). This is
    the single most actionable finding for the MoE router.

  * INDEX SEMANTICS (verified on-device 2026-08-04): the fp16 buffer holds the
    NUMERIC value of the index, NOT reinterpreted int bits. Device returned
    [44.0, 10.0, 92.0, 104.0] vs CPU top-4 [44, 10, 92, 104] -- 100% match
    under ``didx.to(int64)`` (value round), 1.17% under ``view(int16)`` (bits).
    fp16 is exact for ints <= 2048, so indices 0-127 are lossless. IMPLICATION
    for the torch-spyre fix: it must be a NUMERIC fp16->int convert (round the
    value), NOT a bitcast/view -- a raw reinterpret would yield garbage
    (e.g. 44.0 = 0x5180 -> 20864). Host-side ``.to(int64)`` on the returned
    buffer is already correct; the gap is purely that the IN-GRAPH cast is
    elided so no convert runs before aten.index.
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

import torch  # noqa: E402

# ---- real Gemma-4 26B-A4B MoE router dimensions -----------------------------
E = 128  # num_experts  -> the topk axis length
K_REAL = 8  # top_k_experts the model actually uses
K_BRINGUP = 4  # Spyre topk cap the adapter currently pins to

# Token-row counts to exercise: 1 = single decode row, 64 = the gate/prefill tile.
T_CASES = [1, 64]
# K values to exercise: 4 (bring-up, expected to lower) and 8 (real, expected
# to hit the documented k<=4 cap).  Both are interesting: the whole reason the
# adapter pins K=4 is that K=8 does not lower.
K_CASES = [K_BRINGUP, K_REAL]
# Router probs are the softmax output; that is fp16 on device (compute dtype).
# fp32 is included to separate "topk itself is wrong" from "fp16 rounding".
DTYPE_CASES = [torch.float16, torch.float32]


def make_probs(T, dtype, seed):
    """Router-like input: softmax over random logits -> [T, E] probabilities.

    Softmax makes ties vanishingly unlikely, so the top-K *set* is unambiguous
    and index mismatches are real disagreements, not tie-break noise.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g, dtype=torch.float32)
    return torch.softmax(logits, dim=-1).to(dtype)


def topk_fn(x, k):
    # Exactly the router's call: largest K along the expert axis.
    return torch.topk(x, k, dim=-1)


def _is_device_unavailable(msg):
    """True if the error is a Spyre card open/contention failure, not a topk
    result. The VFIO device is single-tenant, so a concurrent run (e.g. the
    other agent's gate) makes every case fail at device-open before topk
    lowers -- do NOT report that as a topk abort."""
    blob = msg.lower()
    return any(s in blob for s in (
        "vfio", "0x332b", "device or resource busy",
        "failed to open the ibm spyre",
    ))


def classify(dev_vals, dev_idx, ref_vals, ref_idx, dtype):
    """Return (status, detail) comparing device topk output to CPU truth."""
    problems = []

    # (1) index dtype: aten.index downstream needs an integer index tensor.
    if not (dev_idx.dtype in (torch.int32, torch.int64)):
        problems.append(
            f"INDEX-DTYPE: indices returned as {dev_idx.dtype} "
            f"(need int32/int64 for aten.index; router casts .to(int64))"
        )

    # Compare on CPU in fp32 / int64 for a clean diff.
    di = dev_idx.detach().cpu().to(torch.int64)
    dv = dev_vals.detach().cpu().to(torch.float32)
    ri = ref_idx.to(torch.int64)
    rv = ref_vals.to(torch.float32)

    # (2) index SET agreement per row (order within the top-K may differ for
    # near-equal probs; the router renormalizes so the *set* is what matters).
    set_mismatch_rows = 0
    for t in range(di.shape[0]):
        if set(di[t].tolist()) != set(ri[t].tolist()):
            set_mismatch_rows += 1
    if set_mismatch_rows:
        problems.append(
            f"INDEX-WRONG: {set_mismatch_rows}/{di.shape[0]} rows selected a "
            f"different top-K expert set than CPU"
        )

    # (3) value agreement (sort each row so order differences don't count).
    dv_s, _ = torch.sort(dv, dim=-1)
    rv_s, _ = torch.sort(rv, dim=-1)
    tol = 2e-2 if dtype == torch.float16 else 1e-5
    max_abs = (dv_s - rv_s).abs().max().item() if dv_s.numel() else 0.0
    if max_abs > tol:
        problems.append(
            f"VALUE-WRONG: top-K values differ by up to {max_abs:.3e} "
            f"(tol {tol:.0e})"
        )

    if problems:
        return "FAIL", "; ".join(problems)
    return "OK", f"index-set match, max value diff {max_abs:.3e}"


def run_case(T, k, dtype):
    tag = f"T={T:<3} K={k} E={E} dtype={str(dtype).replace('torch.', ''):<8}"
    probs = make_probs(T, dtype, seed=1234 + T * 100 + k)

    ref_vals, ref_idx = topk_fn(probs.float(), k)  # CPU fp32 ground truth

    compiled = torch.compile(topk_fn, dynamic=False)
    try:
        dev_vals, dev_idx = compiled(probs.to("spyre"), k)
    except Exception as exc:  # noqa: BLE001 - we want the full category here
        msg = f"{type(exc).__name__}: {exc}"
        first = msg.splitlines()[0][:200]
        if _is_device_unavailable(msg):
            print(f"[DEVICE-UNAVAIL] {tag} :: Spyre card unavailable/busy "
                  f"(not a topk result) -- {first}")
            return "DEVICE-UNAVAIL"
        # A genuine lowering abort. For K>4 this is EXPECTED: the k<=4 cap at
        # decompositions.py (guard `if k > 4`) raises
        # "Spyre backend does not support: Topk is not supported for this
        # config".
        print(f"[COMPILE-ABORT] {tag} :: {first}")
        return "COMPILE-ABORT"

    status, detail = classify(dev_vals, dev_idx, ref_vals, ref_idx, dtype)
    print(f"[{status:<12}] {tag} :: {detail}")
    return status


# --- targeted single-shot probes for the specific documented failure modes ---
# These complement the shape sweep above. Each isolates ONE known constraint of
# the Spyre topk lowering and prints what it demonstrates. They are the fastest
# way to reproduce a specific mode without reading the whole sweep.

def probe_rank3():
    """Mode: rank != 2. Spyre topk is 2-D only; a 3-D input (e.g. router probs
    left as [B,T,E]) raises 'topk only implemented for 2-D tensors'. The MoE
    router must flatten to [T,E] first. K=4 so the k-cap does not mask this."""
    print("\n-- probe: rank-3 input (expect COMPILE-ABORT: 2-D only) --")
    g = torch.Generator().manual_seed(7)
    x = torch.softmax(torch.randn(2, 8, E, generator=g), dim=-1).half()
    try:
        torch.compile(topk_fn, dynamic=False)(x.to("spyre"), 4)
        print("   [UNEXPECTED-OK] rank-3 topk lowered (2-D constraint gone?)")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if _is_device_unavailable(msg):
            print("   [DEVICE-UNAVAIL] card busy -- rerun when free")
        else:
            print(f"   [COMPILE-ABORT as expected] {msg.splitlines()[0][:160]}")


def probe_largest_false():
    """Mode: largest=False unmodeled. The Spyre decomp drops the largest/sorted
    kwargs (implicit largest=True), so topk(..., largest=False) silently returns
    the LARGEST K instead of the smallest -> a wrong-answer trap for any code
    that assumed smallest-K. Demonstrated by comparing to CPU smallest-K."""
    print("\n-- probe: largest=False (expect device returns LARGEST, "
          "mismatching CPU smallest) --")
    g = torch.Generator().manual_seed(9)
    x = torch.softmax(torch.randn(64, E, generator=g), dim=-1).half()
    ref_small_v, _ = torch.topk(x.float(), 4, dim=-1, largest=False)
    try:
        dv, _ = torch.compile(
            lambda t, k: torch.topk(t, k, dim=-1, largest=False), dynamic=False
        )(x.to("spyre"), 4)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        tag = "DEVICE-UNAVAIL" if _is_device_unavailable(msg) else "COMPILE-ABORT"
        print(f"   [{tag}] {msg.splitlines()[0][:160]}")
        return
    dv = dv.detach().cpu().float()
    # If largest=False were honored, dv would match ref_small_v. If the kwarg is
    # dropped, dv is the LARGEST-K instead (much bigger values).
    ref_big_v, _ = torch.topk(x.float(), 4, dim=-1, largest=True)
    matches_small = (dv.sort(-1).values - ref_small_v.sort(-1).values).abs().max().item()
    matches_big = (dv.sort(-1).values - ref_big_v.sort(-1).values).abs().max().item()
    if matches_big < 2e-2 and matches_small > 2e-2:
        print(f"   [CONFIRMED] device ignored largest=False -> returned "
              f"LARGEST (diff vs smallest={matches_small:.3e})")
    elif matches_small < 2e-2:
        print(f"   [OK] device honored largest=False (diff vs smallest="
              f"{matches_small:.3e})")
    else:
        print(f"   [UNCLEAR] diff vs smallest={matches_small:.3e}, "
              f"vs largest={matches_big:.3e}")


def probe_index_gather_consequence():
    """Mode: fp16 index dtype. topk returns indices in the compute dtype (fp16)
    on device, not int64. This probe shows the DOWNSTREAM effect the router
    cares about: feeding topk indices into a gather (per_expert_scale[idx]) --
    the exact operation at hf_gemma4_moe.py that needed the .to(int64) cast.
    Runs the router's `w * per_expert_scale[idx]` step end-to-end on device."""
    print("\n-- probe: index-dtype downstream gather (per_expert_scale[idx]) --")
    g = torch.Generator().manual_seed(11)
    probs = torch.softmax(torch.randn(64, E, generator=g), dim=-1).half()
    scale = (torch.rand(E, generator=g) + 0.5).half()  # per_expert_scale [E]

    def router_tail_nocast(p, s):
        _, idx = torch.topk(p, 4, dim=-1)   # idx fp16 on device
        return s[idx]                        # gather WITHOUT .to(int64)

    def router_tail_cast(p, s):
        _, idx = torch.topk(p, 4, dim=-1)
        return s[idx.to(torch.int64)]        # the adapter's fix

    _, ref_idx = torch.topk(probs.float(), 4, dim=-1)
    ref = scale.float()[ref_idx]
    for name, fn in (("no-cast", router_tail_nocast), ("with .to(int64)", router_tail_cast)):
        try:
            out = torch.compile(fn, dynamic=False)(
                probs.to("spyre"), scale.to("spyre")
            ).detach().cpu().float()
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            tag = "DEVICE-UNAVAIL" if _is_device_unavailable(msg) else "ABORT"
            print(f"   {name}: [{tag}] {msg.splitlines()[0][:150]}")
            continue
        rel = (out - ref).abs().max().item()
        verdict = "matches CPU" if rel < 2e-2 else f"WRONG (max diff {rel:.3e})"
        print(f"   {name}: gather {verdict}")


def main():
    only_k = os.environ.get("TOPK_ONLY_K")
    k_cases = [int(only_k)] if only_k else K_CASES

    print("=" * 78)
    print("Gemma-4 26B-A4B MoE router topk repro")
    print(f"  expert axis E={E}, K in {k_cases} (real model top_k_experts=8),")
    print(f"  T in {T_CASES}, dtype in "
          f"{[str(d).replace('torch.', '') for d in DTYPE_CASES]}")
    print("  op under test: torch.topk(probs[T,E], K, dim=-1)")
    print("=" * 78)

    results = {}
    for k in k_cases:
        for dtype in DTYPE_CASES:
            for T in T_CASES:
                results[(T, k, dtype)] = run_case(T, k, dtype)

    print("-" * 78)
    n_ok = sum(1 for v in results.values() if v == "OK")
    n_abort = sum(1 for v in results.values() if v == "COMPILE-ABORT")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_unavail = sum(1 for v in results.values() if v == "DEVICE-UNAVAIL")
    print(f"summary: {n_ok} OK / {n_fail} FAIL / {n_abort} COMPILE-ABORT / "
          f"{n_unavail} DEVICE-UNAVAIL (of {len(results)} cases)")
    if n_unavail:
        print("NOTE: DEVICE-UNAVAIL means the Spyre card was busy/absent -- "
              "these are NOT topk results. Re-run when the card is free "
              "(the VFIO device is single-tenant; check `ps aux | grep spyre`).")

    if not only_k:  # run the targeted single-mode probes on a full sweep
        print("\n" + "=" * 78)
        print("targeted probes (one documented failure mode each)")
        print("=" * 78)
        probe_rank3()
        probe_largest_false()
        probe_index_gather_consequence()

    print("\nexpected (per torch-spyre source, verified 2026-08-04):")
    print("  * K=8 cases -> COMPILE-ABORT: 'Topk is not supported for this "
          "config' (k>4 cap, decompositions.py).")
    print("  * K=4 cases -> lower OK; watch for INDEX-DTYPE (topk indices come "
          "back fp16, not int64).")
    print("  * rank-3 probe -> COMPILE-ABORT ('2-D tensors' only).")
    print("  * largest=False probe -> device returns LARGEST (kwarg dropped).")
    print("  * gather probe -> shows why the .to(int64) cast is required.")


if __name__ == "__main__":
    main()
