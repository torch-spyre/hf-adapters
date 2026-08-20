# Gemma 4 MoE — Loop-on-Topk Device Path + Backend Grouping Collaboration

**Date:** 2026-08-01
**Model:** `google/gemma-4-26B-A4B-it` (MoE: 128 experts, top-8 trained, `moe_intermediate_size=704`, `hidden_size=2816`, 30 all-MoE layers parallel to a dense MLP).
**Adapter:** `hf_adapters/hf_gemma4_moe.py` (already ships a working device/host-split path; this design replaces the device MoE-FFN formulation).

## Goal

Two deliverables:

- **Approach A (implement now):** run the whole MoE FFN on-device *except the scatter-combine*, looping **directly over the `[T,K]` topk results** (no CPU grouping, no argsort). The expert-weight select becomes an **on-device `index_select`** from an **HBM-resident** expert stack, tiled by the `row` axis via `spyre_hint` so only a scratchpad window is materialized per tile. Host does only the `index_add` scatter-combine (scatter does not yet lower on device).
- **Approach B (specify only, backend collaboration):** turn A's per-row weight fetches into **once-per-expert** fetches by grouping routed rows into contiguous per-expert segments and driving a **fixed-size loop over a single static device program** that gathers one expert weight slab + its group of activations. Grouping runs first on **host CPU** (B-Stage 1), then migrates to the **RISC-V CPU inside Spyre** (B-Stage 2). This section names the exact backend primitives that are missing today.

## Background: why the current path is being replaced

The shipped adapter keeps expert weights **host-resident**, selects the per-row weight on CPU (`gate_up_t[row_expert]`), and moves the small `[N,·]` slices to device. That was forced by two findings:
- `aten::index.Tensor` fancy-index of the expert stack did not lower on-device at the time; and
- keeping all experts resident **and** materializing the per-row `[N,H,2M]` weight tensor simultaneously exhausted the card (~46 GB resident + ~46 GB materialized on a ~103 GB card).

The insight driving this redesign: the OOM was never "experts in HBM" per se — it was **indexing/materializing the whole stack at once**. If the on-device `index_select` is **tiled** so each iteration touches only a scratchpad window of rows (and their experts), the giant per-row tensor never materializes, and the weight select can run **on Spyre** instead of the host.

## Backend facts this design rests on (verified in torch-spyre)

All citations are to `/mnt/devel/inductor_src/torch-spyre/torch_spyre/_inductor/`.

1. **The `spyre_hint` IS the loop — there is no Python `for`.** `spyre_hint(**kwargs)` only tags FX nodes (`propagate_hints.py:85-93`). `hints_to_coarse_tile_groups` (`wsr/coarse_tile_hints.py:269-378`) unions hinted ops into tile groups; `coarse_tile._apply_plan` (`wsr/coarse_tile.py:1034-1051`) **shrinks a single op's own `data.ranges` in place** by the split factor and stamps `loop_info`; `scheduler.py:639-677` wraps that one shrunk `OpSpec` in a `LoopSpec(count=n, body=[op_spec])` (`op_spec.py:278-297`). So you write **single tiled ops** over the full `[N,…]` iteration space, attach the hint, and the backend iterates — no op duplication in the graph.

2. **Tiling binds to an op's OUTPUT ranges.** `plan_coarse_tile_groups` and the auto span-overflow planner resolve every tiled dim against `op_out_coords` / `_loop_var_to_ranges_pos` (`wsr/coarse_tile.py:758-768`). Consequence: you **cannot** tile the *expert* axis of a gather, because `Wstack[expert_ids]`'s output has no expert coordinate (E is consumed by the index). **You tile the `row` (`N = T·K`) axis instead** — that is a legitimate output-range dimension of the gather, both bmms, and the pointwise ops. (This is why Approach A loops on topk rows, not on experts.)

3. **Expert-dim-outermost layout for the gather, with zero restickify.** `enforce_indirect_access_layout.py:68-89` requires the indexed dim at **device position 0** (outermost, stick fixed at the last position). `Wstack` is a **graph input**, so a non-compliant layout **cannot be mutated in place** and forces a full-tensor `spyre.restickify` (a ~46 GB HBM round-trip) — therefore the compliant layout must be produced **by construction** in `prepare_for_spyre`. Row-major `[E,H,2M]` / `[E,M,H]` already put E at position 0 → compliant, no restickify.

4. **Stick dim for the two expert bmms (weight operand sticks on the free/generated dim).** The DF16 matmul stick rule (`propagate_layouts.py:796-799`): Input2 (the weight `y`) sticks on the **generated dim** (present in output, absent from `x`), not the contraction dim.
   - gate/up: `gathered[TILE,1,H] @ W_gu[TILE,H,2M] → [TILE,1,2M]` — contraction `H`, generated `2M` ⇒ **`W_gu` sticks on `2M`**.
   - down: `act[TILE,1,M] @ W_dn[TILE,M,H] → [TILE,1,H]` — contraction `M`, generated `H` ⇒ **`W_dn` sticks on `H`**.
   Row-major `[E,H,2M]` (stick=`2M`) and `[E,M,H]` (stick=`H`) satisfy this **and** point-3's E-outermost **simultaneously** — E at position 0, generated dim on the stick, contraction dim in the middle. No transpose, no restickify.

5. **`per_tile_fixed` (loop-invariant load) exists and is not reduction-only.** `insert_restickify.py:281-345` marks an op's output loop-invariant (loaded once per tile) when its tiled dims are empty; general mechanism, applies to a matmul weight operand whose output does not vary across the tiled loop. Relevant to Approach B (one weight slab per expert-segment tile).

6. **Known gaps (the assumptions under test / the backend asks).**
   - `expert_w[expert_ids]` gather reaches a real op-spec + SDSC bundle and runs on hardware, but the suite defaults to `pytest.xfail` on numeric divergence for on-device indirect gather (`tests/inductor/indirect_access_common.py:413-434`); the literal MoE case `test_moe` (`tests/inductor/test_indirect_access_gather.py:447-465`) is **skipped** for output-span overflow (work-division tries to split the gather's *output* B/S span, not the expert/index axis, and gives up). So on-device gather **correctness at our shapes is an assumption** — Approach A's on-card gate is the oracle.
   - On-device `topk(k>4)` historically **SIGABRT'd** in dxp_standalone and `argsort` had no path (ledger Task 2). Approach A moves `topk(k=4)` on-device and **assumes it now lowers**; the gate is the oracle. K stays pinned to 4 until top-8 restore (below).
   - `scatter` / `index_add` do **not** lower on device (confirmed) → the combine stays on host.

## Approach A — loop-on-topk device region (implement now)

### Device region (single compiled region; single tiled ops, no Python loop)

Host provides `residual_flat [T,H]`, `token_ids [T]`. Everything below is one compiled region:

```
x_router = router_norm(residual_flat)            # [T,H]
logits   = x_router @ router_proj_T              # [T,E]
probs    = softmax(logits, -1)
tw, te   = topk(probs, K=4)                       # [T,K],[T,K]   ASSUMED to lower on-device
tw       = renorm(tw) * per_expert_scale[te]      # [T,K]
expert_of_row = te.reshape(N)                     # [N]  (N = T*K)
token_of_row  = token_ids.reshape(N)              # [N]

with spyre_hint(tiles={"row": TILE}):             # the hint IS the loop over ceil(N/TILE) tiles
    gathered = x_expert[token_of_row]             # [N,H]    on-device activation gather
    W_gu     = gate_up_dev[expert_of_row]         # [N,H,2M] on-device index_select, E-outermost
    W_dn     = down_dev[expert_of_row]            # [N,M,H]
    gu       = bmm(gathered[:,None,:], W_gu)      # [N,1,2M]  3D throughout (no squeeze mid-region)
    g, u     = gu.chunk(2, -1)
    act      = gelu(g, approximate="tanh") * u    # [N,1,M]
    out      = bmm(act, W_dn).squeeze(1)          # [N,H]
    out      = out * tw.reshape(N, 1)             # per-row expert weight (on device)
```

Host, after the region:
```
combined = index_add(zeros[T,H], token_of_row, out)   # scatter-combine (host; scatter not on device)
y        = combined + dense_mlp_out                    # parallel dense MLP add
```

### Invariants (each tied to a backend fact)

- **`x_expert = pre_ff_ln_2(residual_flat)`** is the expert input; **`x_router = router_norm(residual_flat)`** is the router input — threaded **separately** from the raw residual (this is the Task 6 correctness fix; router must NOT be double-normalized).
- **Experts HBM-resident on device**, laid out `gate_up_dev [E,H,2M]` (stick=`2M`), `down_dev [E,M,H]` (stick=`H`), **E outermost** — produced by construction in `prepare_for_spyre`; zero restickify (facts 3–4). This **reverses** the prior host-resident decision — the whole point is that the `index_select` runs on Spyre.
- **Tiling is on `row` (the N output axis)**, not the expert axis (fact 2). The per-tile scratchpad-windowing of experts is the efficiency **assumption under test**.
- **3D `[·,1,·]` through both bmms**, squeeze only at the end (Task 2 shape rule — avoids the layout-prop abort).
- **No padding to a constant tile count** — `N = T·K` is static for a fixed prompt shape; trip count is `ceil(N/TILE)`.
- **`TILE`** is a module constant (bring-up default e.g. 32–64 rows/tile), a tuning knob.
- **K pinned to 4** for bring-up (top-8 restore below).
- **Host does only `index_add`** + the dense add.

### Validation oracle

`repros/gemma4_moe/gateA_loop_on_topk.py`: single-layer device fp16 vs fp32 CPU reference, K=4 apples-to-apples, criterion **mean_rel < 0.02, max_rel < 0.5** (the project's fp16-vs-fp32 relative-error criterion — NOT device-fp16-vs-CPU-fp16 at tight atol/rtol). This one gate exercises all three assumptions: on-device `topk(k=4)`, on-device `row`-tiled windowed gather, and the E-outermost + stick layout. **If it aborts or diverges, STOP and report the exact failure** (do not pre-engineer a fallback — per the explicit "assume it works; tell me if the device test fails" direction).

## Approach B — backend-collaboration grouping (specify only)

**Not implemented now.** B-Stage-1 host grouping is deliberately *not* landed as scaffolding, because the per-segment device grouped program cannot be tiled until backend primitive #1 (below) exists — building the host side now would only reproduce the existing non-working `4B` path. B is landed once the primitive lands.

### B-Stage 1 — grouping on host CPU

Reuses the existing (working) `_moe_permute` / `_group_offsets`:
- `argsort(expert_of_row) → sort_perm`; `gathered_sorted = gathered[sort_perm]`; `row_expert_sorted` non-decreasing.
- `group_off = cumsum(bincount(expert_of_row, E)) → [E+1]` segment boundaries.
- **Fixed-tile contract:** trip count must be a compile-time constant. Since per-expert segment sizes are data-dependent, Stage 1 defines a **capacity/padding scheme** — pad each expert's segment up to a multiple of `TILE` (segments TILE-aligned) so the loop is `N_TILES = N_pad/TILE` static iterations and **each tile belongs to exactly one expert**. The offset table (`group_off` + per-tile `tile_expert [N_TILES]`) is the host→device side-channel.

Device static program (single program, hint-looped over `N_TILES`):
```
with spyre_hint(tiles={"row": TILE}):
    e    = tile_expert[tile]        # ONE expert id for the whole tile
    W_gu = gate_up_dev[e]           # ONE slab fetch [H,2M] — per_tile_fixed: loaded once/tile
    seg  = gathered_sorted[rows]    # [TILE,H] one expert's group
    ... bmm / gelu-tanh SwiGLU / bmm ...
```
Win vs. A: **one weight slab per tile** (expert constant within a tile) + `per_tile_fixed` marks it loop-invariant (loaded once, fact 5).

### B-Stage 2 — grouping on the RISC-V CPU inside Spyre

Move Stage-1 grouping (argsort + bincount + cumsum + capacity-pad + offset-table build) onto the **in-Spyre RISC-V CPU**, so topk results never round-trip to the x86 host for grouping. The **device static program is byte-identical to Stage 1** — only the *producer* of `group_off` / `tile_expert` / `sort_perm` moves. Defines a **RISC-V ↔ device-program ABI**: the RISC-V code writes the offset table + permutation into a known HBM/scratchpad region the static program reads, with a defined sync/fence contract.

### Named backend deliverables (the collaboration asks)

1. **Per-segment operand-select tiling primitive.** A way to tile the loop by **per-expert segment** so one static program iterates `N_TILES` times, each binding one `expert_id → one weight slab`. Today `tiles={...}` binds only to an op's **output** ranges (fact 2); there is no hint to make a **per-tile scalar `tile_expert[tile]` select the weight operand** from `group_off`. This is the core new primitive — either a backend **grouped-GEMM op** or a `group_off`-driven **per-tile operand-select hint**. (This is the Task 9 follow-up, made precise.)
2. **Windowed HBM→scratchpad gather correctness.** Harden the indirect-gather execution path: `expert_w[expert_ids]` reaches SDSC but is `xfail` on divergence and `test_moe` is skipped for output-span overflow (fact 6). **This is also Approach A's dependency** — surfaced by A's gate; if A's gate fails here, this ask is what unblocks it.
3. **`per_tile_fixed` for the weight operand.** Confirm/extend that the loop-invariant-load flag fires for the expert weight slab within a segment tile (mechanism exists, fact 5).
4. **RISC-V grouping ABI (Stage 2).** Memory region, layout, and synchronization/fence contract for RISC-V-produced `group_off` / `tile_expert` / `sort_perm` → static device program. (Ties to the Spyre correction-path/host-compute ordering constraints — HostCompute runs inline, H2D→Compute auto-barriered; a RISC-V→device handoff needs an explicit fence design.)

## Validation & top-8 restore (both approaches)

- Each stage validated by the **single-layer fp16-vs-fp32 rel-err gate** (mean_rel<0.02 / max_rel<0.5), then the **e2e token-compare** (`tests/spyre/test_e2e_token_compare_spyre.py -k 26B-A4B`, non-blocking xfail during bring-up).
- **Top-8 restore criterion:** K=4 is pinned *because* on-device `topk(k>4)` SIGABRT'd and grouping ops didn't lower (ledger Task 2). Once routing runs on-device with a working `topk` (Approach A) — or once grouping is RISC-V-side where k is unconstrained (B-Stage 2) — lift K from 4 → the trained **top-8** and re-validate. **Gate:** token-compare top-1 agreement must recover with K=8 + a correct routing/grouping path; that recovery is the signal that the K=4 pin (not an adapter bug) was the cause of the current 0/5 divergence.

## Deliverables

- **This spec:** `docs/superpowers/specs/2026-08-01-gemma4-moe-loop-on-topk-design.md`.
- **Implementation plan:** `docs/superpowers/plans/2026-08-01-gemma4-moe-loop-on-topk.md` — Approach A as concrete adapter + gate build tasks; Approach B as spec/backend-facing tasks (write the backend-asks doc section, file the four asks). No B-Stage-1 code until backend primitive #1 lands.

## Out of scope

- Implementing B-Stage-1 host grouping or B-Stage-2 RISC-V grouping code (specified only).
- Reconciling the pre-existing ARCHITECTURE.md / README adapter-count discrepancy.
- The dense `hf_gemma4.py` path (untouched).
