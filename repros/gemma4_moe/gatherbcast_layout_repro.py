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

"""Repro: the Gemma-4 MoE loop-on-topk region aborts in the layout pass.

Reproduces, with NO model load, the compile abort that the on-card gate
(``repros/gemma4_moe/gateA_loop_on_topk.py``) hits when it compiles the
Gemma-4 26B-A4B MoE loop-on-topk region
(``hf_adapters/hf_gemma4_moe.py`` ``_compiled_moe_loop_region``). The region
runs the whole router + expert FFN per routed row on-device, then scales each
row by its router weight:

    probs        = softmax(router(x_router))       # [T,E]
    w, idx       = topk(probs, K)                  # [T,K],[T,K]  (int64 post-fix)
    row_w        = (w / w.sum(-1, keepdim=True) * per_expert_scale[idx])
                     .reshape(N, 1)                # [N,1]  <-- [T,K] -> [N,1] view
    gathered     = x_expert[token_of_row]          # [N,H]    indirect gather
    W_gu         = gate_up_dev[expert_of_row]      # [N,H,2M] indirect gather
    W_dn         = down_dev[expert_of_row]         # [N,M,H]  indirect gather
    gu           = bmm(gathered.unsqueeze(1), W_gu)
    g, u         = gu.chunk(2, -1); act = gelu(g,'tanh') * u
    row_out      = bmm(act, W_dn).squeeze(1)       # [N,H]
    row_out      = row_out * row_w                 # [N,H] * [N,1]  buf1 (ABORTS)

On device (real shapes N=T*K=256, H=2816, 2M=1408, M=704) the compile aborts
in the torch-spyre layout pass, NOT at runtime and NOT on the topk index
(the topk index dtype bug is already fixed upstream):

    InductorError: Unsupported: Spyre backend does not support:
      Multi-arg pointwise (buf1): no supported output layout found
      with size=[256, 2816] and coordinates=[d0, d1]
    propagate_layouts.py:1099  _multi_arg_pointwise_layouts

buf1 is the last op, ``row_out * row_w`` -- a broadcasting multiply
``f16[256,2816] * f16[256,1]``. WHAT MAKES IT UNSTICKABLE is the SECOND
operand's layout provenance, not the gather-derived first operand:
``row_w [256,1]`` is a ``.reshape(N,1)`` VIEW of the router weights
``[T=64, K=4]``. A ``[64,4]`` fp16 tensor lays its K=4 dim out as a partial
stick padded to 64 (4 real of 64), so the ``[T,K] -> [N,1]`` view is
physically non-contiguous: flat row ``d0`` of ``row_w`` maps to
``router_w[d0//4, d0%4]``, scattered one-per-padded-stick. A row-dim (d0)
output stick over that view is therefore unrepresentable, and the size-1 H
dim gives no alternative -- so ``_multi_arg_pointwise_layouts`` finds no
candidate and raises. (See the standalone-fragment stages below, which
confirm the plain gather+bcast path stickifies FINE -- it is specifically
the ``[T,K]`` router-weight reshape that defeats it.)

The sibling probe ``repros/gemma4_moe/rowweight_reshape_probe.py`` isolates
JUST this ``[T,K]->[N,1]`` reshape (no gathers, no bmm) and confirms with a
7-way table that NO adapter-side materialization avoids it -- ``.contiguous()``,
``* 1.0``, ``+ 0.0`` and multiplying in ``[T,K,H]`` space all relocate the same
abort; only a device-native ``[N,1]`` operand compiles. The fix is a backend
restickify at the reshape (see backend-asks doc ask #5).

This script STAGES the abort so the backend team gets both the faithful
failing case and the smallest fragment that isolates the trigger:

    STAGE region : the REAL _compiled_moe_loop_region with gate-faithful
                   input layouts (router + expert stacks built exactly as
                   prepare_for_spyre does). Reproduces the gate's abort
                   verbatim -- this is the case to fix.
    STAGE full   : a HAND-BUILT region body (gather -> bmm -> gelu*u -> bmm
                   -> * row_w) with FABRICATED row_w = rand(N,1). Because its
                   row_w has no [T,K] provenance, it does NOT reproduce the
                   layout abort -- it fails earlier/elsewhere, which is itself
                   the evidence that fabricated [N,1] weights are not the
                   trigger. Kept as a contrast, not as the reproducer.
    STAGE nobmm  : gather x_expert -> * fabricated row_w.  Compiles OK.
    STAGE noidx  : plain [N,H] input -> * fabricated row_w.  Compiles OK.

For each stage it prints one of:
    OK ............ compiled + ran (layout pass found an output layout)
    LAYOUT-ABORT .. Unsupported: Multi-arg pointwise ... no supported output
                    layout  (the failure under investigation)
    OTHER-ABORT ... some other Unsupported / InductorError (printed verbatim)

Exit status is 0 iff the faithful ``region`` stage compiles.

Run (on a Spyre host, real weights NOT required):
    PYTHONPATH=<hf-adapters> python3 -u \\
      repros/gemma4_moe/gatherbcast_layout_repro.py
"""
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device + backend
from torch_spyre._inductor.propagate_hints import spyre_hint

# The REAL adapter region under test — the faithful reproducer calls this
# directly with gate-faithful input layouts (see stage_region below).
from hf_adapters.hf_gemma4_moe import _compiled_moe_loop_region

# Real google/gemma-4-26B-A4B-it MoE shapes (see topk_router_shapes.py).
T = 64  # tokens fed at once (the gate's token count)
K = 4  # bring-up top-K (Spyre topk caps at k<=4)
N = T * K  # 256 routed (token, expert) rows
H = 2816  # hidden_size
M = 704  # moe_intermediate_size
TWO_M = 2 * M  # fused gate_up inner dim
E = 128  # num_experts
TILE = 32  # row tile (matches _MOE_TILE / the region's spyre_hint)

_LAYOUT_MSG = "no supported output layout found"


def _run(name, fn, args):
    """Compile fn for spyre, run it, and classify the outcome."""
    try:
        compiled = torch.compile(fn, dynamic=False)
        out = compiled(*args)
        # The region returns (row_out, token_of_row); the reductions return one
        # tensor. Move the first output to host to force execution to complete.
        primary = out[0] if isinstance(out, tuple) else out
        primary.cpu()
        print(f"  {name:<7}: OK  (out {tuple(primary.shape)})")
        return "OK"
    except Exception as e:  # noqa: BLE001 - repro: classify, don't crash
        # The top-level InductorError.__str__ is a generic "set TORCHDYNAMO_
        # VERBOSE=1" hint; the real layout reason is on the chained cause. Walk
        # the __cause__/__context__ chain and concatenate every message so the
        # classifier sees the nested "no supported output layout" text.
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
            print(f"  {name:<7}: LAYOUT-ABORT  {reason}")
            return "LAYOUT-ABORT"
        print(f"  {name:<7}: OTHER-ABORT   {msg.splitlines()[-1]}")
        return "OTHER-ABORT"


def _dev(t):
    return t.to("spyre")


def main():
    torch.manual_seed(0)
    print(f"shapes: N={N} H={H} 2M={TWO_M} M={M} E={E} TILE={TILE}")

    # --- Inputs, laid out EXACTLY as the on-card gate / prepare_for_spyre does.
    # Router: proj.weight [E,H], scale [H], per_expert_scale [E]; all device-
    # resident (the whole router runs in the region). scalar_root_size is a
    # Python float. x_router / x_expert are [T,H] fp16 device residuals.
    x_router = _dev(torch.randn(T, H, dtype=torch.float16))
    x_expert = _dev(torch.randn(T, H, dtype=torch.float16))
    router_proj_w = _dev(torch.randn(E, H, dtype=torch.float16))
    router_scale = _dev(torch.randn(H, dtype=torch.float16))
    root_size = float(H**-0.5)
    per_expert_scale = _dev(torch.randn(E, dtype=torch.float16))
    token_ids = _dev(torch.arange(T, dtype=torch.int32))
    eps = 1e-6

    # Expert stacks built with the gate's provenance: the stock params are
    # [E,2M,H] / [E,H,M]; prepare_for_spyre does transpose(1,2).contiguous()
    # -> [E,H,2M] / [E,M,H], then .to("spyre") (E outermost, no restickify).
    gate_up_stock = torch.randn(E, TWO_M, H, dtype=torch.float16)
    down_stock = torch.randn(E, H, M, dtype=torch.float16)
    gate_up_dev = _dev(gate_up_stock.transpose(1, 2).contiguous())  # [E,H,2M]
    down_dev = _dev(down_stock.transpose(1, 2).contiguous())  # [E,M,H]

    # STAGE region: the REAL adapter region, called with gate-faithful inputs.
    # This is the faithful reproducer for the backend team — the exact op graph
    # and layouts the on-card gate compiles. Router runs on-device, topk emits
    # int64 indices (post-fix), three indirect gathers under the row hint, two
    # bmms, gelu-tanh SwiGLU, then the [N,H]*[N,1] row-weight multiply.
    def stage_region():
        return _compiled_moe_loop_region(
            x_router,
            x_expert,
            router_proj_w,
            router_scale,
            root_size,
            per_expert_scale,
            gate_up_dev,
            down_dev,
            token_ids,
            K,
            TILE,
            eps,
        )

    # --- Reduction stages: hand-built fragments to localize the trigger. They
    # use precomputed index tensors (the region's reshaped views) so no router
    # or topk runs — isolating just the gather/bmm/broadcast layout question.
    token_of_row = _dev(torch.arange(T, dtype=torch.int32).repeat_interleave(K))
    expert_of_row = _dev(torch.randint(0, E, (N,), dtype=torch.int64))
    row_w = _dev(torch.rand(N, 1, dtype=torch.float16))
    nh_input = _dev(torch.randn(N, H, dtype=torch.float16))

    # STAGE full: the region BODY (gathers + bmms + gelu + broadcast), no router.
    def stage_full(x_e, gu_w, dn_w, tok, exp, rw):
        with spyre_hint(tiles={"row": TILE}):
            gathered = x_e[tok]  # [N,H]
            W_gu = gu_w[exp]  # [N,H,2M]
            W_dn = dn_w[exp]  # [N,M,H]
            gu = torch.bmm(gathered.unsqueeze(1), W_gu)  # [N,1,2M]
            g, u = gu.chunk(2, dim=-1)  # [N,1,M]
            act = torch.nn.functional.gelu(g, approximate="tanh") * u
            row_out = torch.bmm(act, W_dn).squeeze(1)  # [N,H]
            return row_out * rw  # [N,H]*[N,1]  <-- the reported buf1

    # STAGE nobmm: gather then broadcast-multiply, no bmm chain in between.
    # Tests whether the gather-derived [N,H] layout alone defeats the mul.
    def stage_nobmm(x_e, tok, rw):
        with spyre_hint(tiles={"row": TILE}):
            gathered = x_e[tok]  # [N,H] gather-derived layout
            return gathered * rw  # [N,H]*[N,1]

    # STAGE noidx: plain [N,H] input, no gather at all. Tests whether the
    # [N,H]*[N,1] broadcast is unsupported even on a STANDARD input layout.
    def stage_noidx(x_nh, rw):
        with spyre_hint(tiles={"row": TILE}):
            return x_nh * rw  # [N,H]*[N,1]

    results = {}
    results["region"] = _run("region", stage_region, ())
    results["full"] = _run(
        "full",
        stage_full,
        (x_expert, gate_up_dev, down_dev, token_of_row, expert_of_row, row_w),
    )
    results["nobmm"] = _run("nobmm", stage_nobmm, (x_expert, token_of_row, row_w))
    results["noidx"] = _run("noidx", stage_noidx, (nh_input, row_w))

    print("\nsummary:")
    for k, v in results.items():
        print(f"  {k:<7}: {v}")
    # Exit non-zero if the faithful region stage aborts (the gate's blocker).
    raise SystemExit(0 if results["region"] == "OK" else 1)


if __name__ == "__main__":
    main()
