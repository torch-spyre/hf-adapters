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

"""Probe: is the [T,K]->[N,1] router-weight RESHAPE the layout-abort trigger?

The MoE loop-on-topk region (``hf_gemma4_moe.py`` ``_compiled_moe_loop_region``)
aborts in the torch-spyre layout pass on the final ``row_out * row_w`` multiply
(``Multi-arg pointwise buf1: no supported output layout found``). The
layout-gap investigation's hypothesis: the unstickable operand is ``row_w``,
which is ``w.reshape(N, 1)`` where ``w`` is ``[T=64, K=4]``. A ``[64,4]`` fp16
tensor puts K=4 in a partial stick padded to 64, so the ``[T,K] -> [N,1]`` view
is physically non-contiguous and no output stick over the N=256 row dim is
representable.

This probe isolates JUST that reshape, with a trivial pointwise (no gathers, no
bmm), and contrasts three ways of producing the ``[N,1]`` row-weight operand
for a ``[N,H] * [N,1]`` broadcast multiply:

    reshape  : row_w = w.reshape(N, 1)                 # the region's current path
    contig   : row_w = w.reshape(N, 1).contiguous()    # force a real [N,1] buffer
    flat1d   : row_w = w.reshape(N).contiguous()[:, None]  # rebuild from flat 1-D

RESULT (recorded 2026-08-04): the trigger is the ``[T,K] <-> N`` reshape ACROSS
the sub-64 (K=4) partial stick, and it is NOT fixable adapter-side. The full
table:

    reshape  : LAYOUT-ABORT  (w.reshape(N,1))
    contig   : LAYOUT-ABORT  (.reshape(N,1).contiguous())
    flat1d   : LAYOUT-ABORT  (.reshape(N).contiguous()[:,None])
    compute  : LAYOUT-ABORT  ((w.reshape(N)*1.0).reshape(N,1)) -- abort MOVES
                             onto the flatten op: "buf0 aten.mul: no supported
                             output layout ... output size=[256]"
    addzero  : LAYOUT-ABORT  ((w.reshape(N)+0.0)[:,None]) -- same as compute
    tk_space : OTHER-ABORT   (multiply in [T,K,H] space instead of flattening)
                             -- "arg0_1 STL 0 --> Out STL 1": the [N,H]->[T,K,H]
                             reshape of the ACTIVATION hits the same partial
                             stick from the other side
    native   : OK            (a device-native rand(N,1), never [T,K]-derived)

CONCLUSION: NO adapter rewrite that crosses the [T,K] <-> N boundary compiles;
only a [N,...]-native operand does. Every materialization strategy
(.contiguous(), *1.0, +0.0) just relocates the abort onto the flatten op,
because the copy's INPUT READ of the [64,4]-partial-stick buffer with a flat
[256] index is itself the unrepresentable cross-stick expression
(Mod(d0,4)+floor(d0/4), not Mod(var,64)). This is a torch-spyre backend
limitation: a reshape across a sub-stick dim needs a device RESTICKIFY (HBM
round-trip) to re-lay the buffer N-outermost before any flat consumer. It
CANNOT be worked around in hf_gemma4_moe.py under the "no CPU fallback; move
router weights to spyre; change stickification not compute location"
constraint -- changing the stickification is exactly the backend fix required.

DISTINCT from the indirect-gather 4x4 xfail (Approach-B ask #2): no gather, no
bmm here -- it is a reshape-across-partial-stick bug. Two backend asks, not one.

Run (Spyre host, NO model load, NO real weights):
    PYTHONPATH=<hf-adapters> python3 -u repros/gemma4_moe/rowweight_reshape_probe.py
"""
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device + backend
from torch_spyre._inductor.propagate_hints import spyre_hint

T = 64  # tokens
K = 4  # top-K (Spyre topk cap)
N = T * K  # 256 rows
H = 2816  # hidden_size
TILE = 32  # row tile (matches the region's spyre_hint)

_LAYOUT_MSG = "no supported output layout found"


def _classify(name, fn, args):
    """Compile+run fn for spyre; classify OK / LAYOUT-ABORT / OTHER-ABORT."""
    try:
        out = torch.compile(fn, dynamic=False)(*args)
        out.cpu()
        print(f"  {name:<8}: OK  (out {tuple(out.shape)})")
        return "OK"
    except Exception as e:  # noqa: BLE001 - probe: classify, don't crash
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
    print(f"shapes: T={T} K={K} N={N} H={H} TILE={TILE}")

    # w: the [T,K] router weights (post softmax/topk/normalize). The physical
    # provenance that matters is the [T,K] shape with K < stick(64), so plain
    # randn([T,K]) on device reproduces the partial-stick layout.
    w_tk = torch.randn(T, K, dtype=torch.float16).to("spyre")
    x_nh = torch.randn(N, H, dtype=torch.float16).to("spyre")

    # PATH reshape: exactly what the region does today.
    def path_reshape(w, x):
        with spyre_hint(tiles={"row": TILE}):
            row_w = w.reshape(N, 1)  # view of [T,K]
            return x * row_w

    # PATH contig: force row_w into a materialized [N,1] buffer.
    def path_contig(w, x):
        with spyre_hint(tiles={"row": TILE}):
            row_w = w.reshape(N, 1).contiguous()
            return x * row_w

    # PATH flat1d: rebuild the operand from a contiguous flat [N] then unsqueeze.
    def path_flat1d(w, x):
        with spyre_hint(tiles={"row": TILE}):
            row_w = w.reshape(N).contiguous()[:, None]
            return x * row_w

    # PATH native: the [N,1] operand NEVER passed through a [T,K] shape -- it is
    # a device-native [N,1] tensor. Contrast against the [T,K]-derived paths to
    # decide whether the trigger is the [T,K] partial-stick provenance or the
    # [N,H]*[N,1] broadcast-under-hint itself. (nobmm/noidx in the sibling repro
    # already compile with a native [N,1]; this repeats it in-file for one
    # apples-to-apples table.)
    w_n1_native = torch.randn(N, 1, dtype=torch.float16).to("spyre")

    def path_native(w_n1, x):
        with spyre_hint(tiles={"row": TILE}):
            return x * w_n1

    # PATH compute: force row_w to be its OWN ComputedBuffer by putting a real
    # pointwise (mul by 1.0) on the flattened operand, so Inductor cannot fold
    # it back into a view of the [T,K] buffer. This is the debugger's "preferred
    # bounded fix" in its strongest form -- a materialized [N,1] producer.
    def path_compute(w, x):
        with spyre_hint(tiles={"row": TILE}):
            row_w = (w.reshape(N) * 1.0).reshape(N, 1)
            return x * row_w

    # PATH addzero: alternate forced-materialization (add 0.0 after flatten).
    def path_addzero(w, x):
        with spyre_hint(tiles={"row": TILE}):
            row_w = (w.reshape(N) + 0.0)[:, None]
            return x * row_w

    # PATH tk_space: DON'T flatten row_w to [N,1] at all. Reshape the [N,H]
    # expert output BACK to [T,K,H] and multiply by w[T,K,1] there, keeping K a
    # logical dim (never flattened into the stick). If this compiles, it is a
    # true adapter-side fix that avoids the unrepresentable [T,K]->[N] flatten.
    # (x here stands in for the [N,H] expert output row_out.)
    def path_tk_space(w, x):
        with spyre_hint(tiles={"row": TILE}):
            x_tkh = x.reshape(T, K, H)  # [T,K,H]
            out = x_tkh * w[:, :, None]  # [T,K,H] * [T,K,1]
            return out.reshape(N, H)  # back to [N,H]

    results = {}
    results["reshape"] = _classify("reshape", path_reshape, (w_tk, x_nh))
    results["contig"] = _classify("contig", path_contig, (w_tk, x_nh))
    results["flat1d"] = _classify("flat1d", path_flat1d, (w_tk, x_nh))
    results["compute"] = _classify("compute", path_compute, (w_tk, x_nh))
    results["addzero"] = _classify("addzero", path_addzero, (w_tk, x_nh))
    results["tk_space"] = _classify("tk_space", path_tk_space, (w_tk, x_nh))
    results["native"] = _classify("native", path_native, (w_n1_native, x_nh))

    print("\nsummary:")
    for k, v in results.items():
        print(f"  {k:<8}: {v}")
    # Interpretation:
    #   native OK + all [T,K]-derived ABORT  -> the [T,K] partial-stick
    #     provenance is the trigger; the fix must RESTICKIFY into a real
    #     N-outermost buffer (a logical .reshape/.contiguous is not enough).
    #   native also ABORT -> the [N,H]*[N,1] broadcast under the row hint is
    #     itself unsupported, independent of provenance (a broader backend ask).
    # Exit 0 only if the native path compiles (isolates which of the two).
    raise SystemExit(0 if results["native"] == "OK" else 1)


if __name__ == "__main__":
    main()
