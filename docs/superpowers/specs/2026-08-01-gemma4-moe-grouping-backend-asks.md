# Gemma 4 MoE — Backend-Collaboration Asks for Per-Expert Grouping (Approach B)

**Date:** 2026-08-01
**Status:** Specify only — nothing in this document is implemented. No host-grouping
code and no RISC-V code has been written against this spec.
**Audience:** torch-spyre / deeptools backend engineers. This document is
self-contained — it does not assume you have read the design spec it is
extracted from.
**Source spec:** `docs/superpowers/specs/2026-08-01-gemma4-moe-loop-on-topk-design.md`
(section "Approach B — backend-collaboration grouping (specify only)" and
"Validation & top-8 restore"). This document restates that content
verbatim-faithfully as a standalone reference; it does not add new facts or
citations beyond what that spec states.
**Implementation plan:** `docs/superpowers/plans/2026-08-01-gemma4-moe-loop-on-topk.md`.

## Context in one paragraph

Gemma 4's MoE FFN (`google/gemma-4-26B-A4B-it`: 128 experts, top-8 trained,
`moe_intermediate_size=704`, `hidden_size=2816`) is being ported to run on
Spyre. A separate, already-in-progress path ("Approach A") runs the whole
MoE FFN on-device except the scatter-combine, by looping directly over the
`[T,K]` topk results with an on-device `index_select` from an HBM-resident
expert stack — no CPU grouping, no argsort. Approach A does one weight fetch
**per row**. Approach B, described here, is the follow-on: turn Approach A's
per-row weight fetches into **once-per-expert** fetches, by grouping routed
rows into contiguous per-expert segments and driving a fixed-size loop over a
single static device program that gathers one expert weight slab plus its
group of activations per iteration.

Approach B is **not implemented in the current plan** because the per-segment
device grouped program cannot be tiled until backend primitive #1 (below)
exists. Building the host-grouping side now would only reproduce the
project's existing, non-working `4B` path. **B is landed once the primitive
lands** — that is the purpose of this document: to hand the backend team a
precise, actionable spec of what's needed, so grouping can be picked up as
soon as the primitive is available.

## B-Stage 1 — grouping on host CPU

### What "grouping" means, from scratch

The router has already run. For each of the `T` tokens it picked `K` experts,
so there are `N = T·K` **(token, expert) pairs** to compute. Three flat arrays
describe them, all length `N`, in the same order:

- `token_of_row[i]` — which token the `i`-th pair belongs to (values in `0..T-1`,
  each appearing exactly `K` times).
- `expert_of_row[i]` — which expert the `i`-th pair routes to (values in
  `0..E-1`, `E = 128`; arbitrary, data-dependent, in no particular order).
- `row_weight[i]` — the scalar router weight for the `i`-th pair (post
  softmax → top-`K` → renormalize → `per_expert_scale`). It is the flattened
  `[T,K]` router-weight tensor `w`, i.e. `row_weight = w.reshape(N)`. Every
  expert output row is scaled by its `row_weight` before the scatter-combine —
  this is the multiply that blocker #5 is about. Carrying it here, alongside
  the other two arrays, is what makes #5 visible in the spec body.

  **The `w.reshape(N)` on this line is blocker #5's exact trigger** — a
  `[T=64, K=4]→[N=256]` reshape across the partial (`K=4 < 64`) stick — no
  matter where the reshape executes. It is the same unrepresentable cross-stick
  index whether it runs in Approach A's device region (`out * tw.reshape(N,1)`),
  in a device-side Stage-1 flatten, or on the RISC-V CPU writing a device
  buffer (Stage 2). Grouping does **not** escape #5 by "flattening earlier": the
  flatten *is* the reshape #5 blocks. #5 must clear for the router weight to
  reach the device in any form.

Approach A computes these `N` pairs in their natural (per-token) order, and so
must fetch a fresh expert weight slab for **every row** — even when two adjacent
rows happen to use the same expert. Grouping removes that waste by physically
**reordering the rows so that all rows routed to the same expert sit next to
each other**. Once the rows are expert-contiguous, a tile of consecutive rows is
guaranteed to share one expert, so the program fetches that expert's weight slab
**once per tile** instead of once per row.

Grouping is three plain array operations — no model knowledge, just sorting a
list of integers and recording where the runs begin and end:

1. **Sort the pairs by expert.** Compute the permutation that would sort
   `expert_of_row` into non-decreasing order — i.e. for each output position,
   which input row lands there. Call it `sort_perm [N]`. (In PyTorch this is
   `sort_perm = argsort(expert_of_row)`.) Applying `sort_perm` to the row arrays
   gives the reordered views:
   - `expert_sorted = expert_of_row[sort_perm]` — now non-decreasing, e.g.
     `[0,0,0,2,2,5,5,5,5,…]`: all of expert 0's rows, then all of expert 2's,
     and so on. Experts that got no rows simply don't appear.
   - `gathered_sorted = activations[sort_perm]` — the token activations
     reordered to match, so row `i` of `gathered_sorted` is the input for the
     expert named in `expert_sorted[i]`.
   - `token_of_row_sorted = token_of_row[sort_perm]` — kept so the final result
     can be scattered back to the right token.
   - `row_weight_sorted = row_weight[sort_perm]` — the router weights reordered
     the same way, so row `i` of `row_weight_sorted` is the scale for the output
     row produced from `gathered_sorted[i]`. Applied at the very end (below).

2. **Count how many rows each expert got.** Tally the sorted expert list into a
   length-`E` histogram: `counts[e]` = number of rows routed to expert `e`
   (`counts = bincount(expert_sorted, minlength=E)`). With top-8 over 128
   experts and a typical prompt, most counts are small and some are zero.

3. **Turn counts into segment boundaries.** Take the running (prefix) sum of the
   counts, prepended with a leading `0`, giving `group_off [E+1]`
   (`group_off[0]=0`; `group_off[1:] = cumsum(counts)`). Then expert `e` owns
   exactly the contiguous row range `group_off[e] : group_off[e+1]` of the
   sorted arrays — that half-open interval is expert `e`'s **segment**. An
   expert with `counts[e]==0` has `group_off[e]==group_off[e+1]` (an empty
   segment).

Worked micro-example (`T=3`, `K=2`, so `N=6`, and suppose `E=6`):

```
expert_of_row      = [2, 0, 2, 5, 0, 2]      # token0→{2,0}, token1→{2,5}, token2→{0,2}
sort_perm          = [1, 4, 0, 2, 5, 3]      # positions of experts 0,0,2,2,2,5
expert_sorted      = [0, 0, 2, 2, 2, 5]      # non-decreasing after the sort
counts (E=6)       = [2, 0, 3, 0, 0, 1]      # expert0:2 rows, expert2:3, expert5:1
group_off (E+1=7)  = [0, 2, 2, 5, 5, 5, 6]   # expert2 owns 2:5, expert5 owns 5:6
```

So expert 2's segment is `group_off[2]:group_off[3] = 2:5` — rows 2, 3, 4 of the
sorted block, all of which need expert 2's weights fetched exactly once.

(The adapter already ships these three steps as `_moe_permute` (step 1) and
`_group_offsets` (steps 2–3); the algorithm above is the whole of what they do —
you do not need to read that code to implement or review this spec.)

### From variable-size segments to a fixed-trip static loop

#### The problem this step solves

After grouping (previous section) the rows are expert-contiguous and expert
`e` owns the half-open row range `group_off[e] : group_off[e+1]`. But those
segment sizes are **data-dependent** — a different prompt routes a different
number of rows to each expert, so `counts` (and therefore every segment
length) changes on every forward pass. A Spyre device loop can have a loop 
count determined at runtime, but the program it executes must be static (aka
do the same amount of work every single time). This step turns the ragged,
runtime-sized segments into a **fixed number of equal-size tiles**, each
belonging to exactly one expert, so the same static program can be issued a
variable number of times.

#### The capacity/padding scheme, step by step

The idea is to **round every expert's segment length up to a whole number of
`TILE`-row tiles**, inserting inert padding rows to fill the last partial tile
of each expert. Two invariants must hold afterwards:

1. **`TILE`-alignment** — every expert's (padded) segment starts and ends on a
   `TILE` boundary, so a tile never spans the seam between two experts.
2. **Single-expert tiles** — because of (1), every one of the `N_TILES` tiles
   contains rows from exactly one expert, so a single `tile_expert[tile]`
   scalar names the weight slab for the whole tile.

Concretely, from the `counts [E]` and `group_off [E+1]` the previous section
produced, build the padded layout with plain integer/tensor ops (no model
knowledge — this is arithmetic on the histogram):

```python
# INPUT (from the grouping section):
#   sort_perm     [N]    argsort(expert_of_row)           — the by-expert order
#   expert_sorted [N]    expert_of_row[sort_perm]         — non-decreasing
#   counts        [E]    bincount(expert_sorted, E)       — rows per expert
#   gathered_sorted [N,H], token_of_row_sorted [N]        — reordered activations
#   row_weight_sorted [N] row_weight[sort_perm]           — reordered router weights
TILE = 32                                     # rows per tile (module constant)

# 1. Tiles each expert needs = ceil(counts / TILE). Experts with 0 rows need 0.
tiles_per_expert = (counts + TILE - 1) // TILE          # [E]  int
padded_counts    = tiles_per_expert * TILE              # [E]  each a TILE multiple

# 2. Padded segment boundaries (same shape/role as group_off, TILE-aligned).
pad_off = torch.zeros(E + 1, dtype=torch.long)          # [E+1]
pad_off[1:] = torch.cumsum(padded_counts, 0)            # expert e -> pad_off[e]:pad_off[e+1]
N_pad   = int(pad_off[-1].item())                       # total padded rows
N_TILES = N_pad // TILE                                 # runtime loop count

# 3. tile_expert[t] = the one expert that owns padded tile t.
#    repeat_interleave expands "expert e owns tiles_per_expert[e] tiles" into a
#    flat per-tile expert id, e.g. tiles_per_expert=[1,0,2,0,0,1] -> [0,2,2,5].
experts     = torch.arange(E)
tile_expert = experts.repeat_interleave(tiles_per_expert)   # [N_TILES]

# 4. Scatter the real (unpadded) rows into their padded slots; the gap between
#    pad_off[e]+counts[e] and pad_off[e+1] stays as padding rows.
#    dst_pos[i] = padded destination of the i-th sorted row: within expert e,
#    row j (0-based inside the segment) lands at pad_off[e]+j.
seg_start_of_row = pad_off[:-1].repeat_interleave(counts)   # [N] padded seg start per row
intra_base       = group_off[:-1].repeat_interleave(counts)  # [N] unpadded seg start per row
intra            = torch.arange(N) - intra_base             # [N] j within segment
dst_pos          = seg_start_of_row + intra                 # [N] -> padded index

gathered_pad = gathered_sorted.new_zeros(N_pad, H)          # padding rows = 0
gathered_pad[dst_pos] = gathered_sorted                     # real rows into their slots
# padding rows carry the SINK token id T (one past the last real token) so the
# on-device per-tile index_add routes them to a throwaway sink row out[T] that
# is sliced off after the loop — no data-dependent mask in the static program.
token_of_row_pad = token_of_row_sorted.new_full((N_pad,), T)
token_of_row_pad[dst_pos] = token_of_row_sorted
# the router weights ride along into the same padded slots; padding rows get a
# weight of 0, so even the sink-row accumulation for them is zero.
row_weight_pad = row_weight_sorted.new_zeros(N_pad)         # [N_pad] padding = 0
row_weight_pad[dst_pos] = row_weight_sorted
```

Worked micro-example continuing the grouping section's `T=3, K=2, E=6` case
(`counts = [2,0,3,0,0,1]`), with `TILE = 2`:

```
counts            = [2, 0, 3, 0, 0, 1]
tiles_per_expert  = [1, 0, 2, 0, 0, 1]      # ceil([2,0,3,0,0,1] / 2)
padded_counts     = [2, 0, 4, 0, 0, 2]      # * TILE
pad_off (E+1=7)   = [0, 2, 2, 6, 6, 6, 8]   # cumsum, leading 0; N_pad = 8
N_TILES           = 4                        # 8 / 2
tile_expert       = [0, 2, 2, 5]            # expert 2 spans two tiles (3 rows -> 2 tiles, 1 pad)
# padded row block (8 rows): [e0 e0 | e2 e2 | e2 PAD | e5 PAD]
#   tile 0 -> expert 0 (2 real)   tile 2 -> expert 2 (1 real + 1 pad)
#   tile 1 -> expert 2 (2 real)   tile 3 -> expert 5 (1 real + 1 pad)
```

#### The host→device side-channel

Two small integer tables travel from host to device alongside the padded,
reordered activations `gathered_pad [N_pad,H]` and the per-row router weights
`row_weight_pad [N_pad]`:

- `pad_off [E+1]` — the `TILE`-aligned segment boundaries (the padded analogue
  of `group_off`).
- `tile_expert [N_TILES]` — for each padded tile, the single expert id it
  belongs to.

`row_weight_pad` is not an integer side-channel table — it is a per-row column
that rides with `gathered_pad` (one scalar per padded row) and is consumed by
the device program's final row-scale multiply (below).

The device program reads `tile_expert[tile]` to know which weight slab to
fetch for the current tile. Padding rows (zeroed activation, sink token id `T`,
zero router weight) flow through the FFN harmlessly and are **routed to the
sink row `out[T]` by the on-device scatter-combine**, which is sliced off after
the loop — so they contribute nothing to any real token's output.

Device static program (single program, hint-looped over the runtime
`N_TILES` — the trip count is now runtime dependent):

```python
out = gathered_pad.new_zeros(T + 1, H)       # [T+1,H] accumulator; row T is the padding sink
with spyre_hint(tiles={"row": TILE}):        # exactly N_TILES iterations
    e    = tile_expert[tile]                 # ONE expert id for the whole tile
    W_gu = gate_up_dev[e]                     # ONE slab [H,2M] — per_tile_fixed: once/tile
    W_dn = down_dev[e]                         # ONE slab [M,H]  — per_tile_fixed
    seg  = gathered_pad[rows]                 # [TILE,H] this tile's rows (one expert + any pad)
    w_r  = row_weight_pad[rows]               # [TILE] this tile's router weights
    dst  = token_of_row_pad[rows]             # [TILE] destination token per row (T for padding)
    gu   = torch.bmm(seg.unsqueeze(1), W_gu)  # [TILE,1,2M]
    g, u = gu.chunk(2, dim=-1)                # [TILE,1,M]
    act  = F.gelu(g, approximate="tanh") * u  # gelu-tanh SwiGLU
    seg_out = torch.bmm(act, W_dn).squeeze(1) # [TILE,H]
    seg_out = seg_out * w_r.reshape(TILE, 1)  # per-row router-weight scale (blocker #5)
    out.index_add_(0, dst, seg_out)           # ON-DEVICE scatter-combine, per tile
out = out[:T]                                 # drop the padding sink row -> [T,H]
```

**The scatter-combine is wired through the tiles and runs on-device.** Each
tile's `index_add_` accumulates its `TILE` output rows straight onto the shared
`out` accumulator inside the same hint-looped region — no separate post-loop
scatter, no host round-trip. Two consequences of doing the combine per tile:

- **Padding rows are routed to a sink row, not masked.** Instead of dropping
  sentinel rows before a bulk scatter, padding rows carry destination token id
  `T` (not `-1`), so their contribution lands in `out[T]`, a throwaway `[1,H]`
  sink that is sliced off after the loop. Set `token_of_row_pad`'s padding fill
  to `T` (in the Stage-1 padding block) rather than `-1` for this path. Their
  activations are already zero, so `seg_out` for those rows is zero and the sink
  stays inert — the sink exists only so the on-device `index_add_` never needs a
  data-dependent mask (masking would reintroduce a variable-length gather the
  static tile program cannot express).
- **`out` is the one loop-carried accumulator.** It is written by every tile
  (an `index_add_` reduction into a fixed `[T+1,H]` buffer), so it is *not*
  `per_tile_fixed`; the expert slabs `W_gu`/`W_dn` still are (one fetch per
  tile). This is the ask that #2 (windowed indirect-gather / scatter
  correctness) must confirm lowers at these shapes; the design assumes it does
  and the on-card gate is the oracle.

Win vs. Approach A: **one weight slab per tile** (the expert is constant within
a tile) plus `per_tile_fixed` marks it loop-invariant so the slab is loaded
**once per tile** (see deliverable #3 below). Approach A fetches a slab per
**row**; grouping fetches a slab per **`TILE` rows**. The cost is the padding
rows — with top-8 over 128 experts and short prompts many segments are far
smaller than `TILE`, so the padding overhead (and the right `TILE`) is the key
tuning trade-off this scheme exposes.

## B-Stage 2 — grouping on the RISC-V CPU inside Spyre

Move Stage-1 grouping (argsort + bincount + cumsum + capacity-pad +
offset-table build) onto the **in-Spyre RISC-V CPU**, so topk results never
round-trip to the x86 host for grouping. The **device static program is
byte-identical to Stage 1** — only the *producer* of `group_off` /
`tile_expert` / `sort_perm` moves. This defines a **RISC-V ↔ device-program
ABI**: the RISC-V code writes the offset table plus permutation into a known
HBM/scratchpad region that the static program reads, with a defined
sync/fence contract.

## The five named backend deliverables (the collaboration asks)

1. **Per-segment operand-select tiling primitive.** A way to tile the loop by
   **per-expert segment** so one static program iterates `N_TILES` times,
   each binding one `expert_id → one weight slab`. Today `tiles={...}` binds
   only to an op's **output** ranges (`wsr/coarse_tile.py:758-768`,
   `wsr/coarse_tile_hints.py`); there is no hint to make a **per-tile scalar
   `tile_expert[tile]` select the weight operand** from `group_off`. This is
   the core new primitive — either a backend **grouped-GEMM op** or a
   `group_off`-driven **per-tile operand-select hint**.
2. **Windowed HBM→scratchpad indirect-gather + per-tile scatter-reduce
   correctness.** Harden the indirect-gather execution path: `expert_w[expert_ids]`
   reaches SDSC but is `xfail` on divergence by default (indirect gather defaults
   to xfail on numeric divergence — `tests/inductor/indirect_access_common.py:413-434`)
   and the literal MoE case is skipped for output-span overflow
   (`test_moe` skipped, `tests/inductor/test_indirect_access_gather.py:447-465`).
   **This is also Approach A's dependency** — it is the on-device gather
   correctness Approach A's gate must eventually clear once the reshape ask
   (#5 below) unblocks compilation. **This ask now also covers the on-device
   per-tile `index_add_` scatter-combine** that both approaches wire through the
   tiled loop (the B-Stage-1 device program above; Approach A's per-region
   `index_add`): a scatter-reduce into a fixed `[T+1,H]` accumulator, one tile's
   `TILE` rows at a time, with padding rows routed to the sink row `out[T]`. The
   spec **assumes `index_add`/`scatter-reduce` lowers on device** at these
   shapes; confirm it does (or land it), since neither the per-tile combine nor
   the sink-row scheme has an off-device fallback in this design.
3. **`per_tile_fixed` for the weight operand.** Confirm/extend that the
   loop-invariant-load flag fires for the expert weight slab within a
   segment tile (mechanism exists: `insert_restickify.py:281-345`).
4. **RISC-V grouping ABI (Stage 2).** Memory region, layout, and
   synchronization/fence contract for the RISC-V-produced grouping outputs
   handed to the static device program — `pad_off` / `tile_expert` / `sort_perm`
   plus the per-row padded columns the device program reads (`gathered_pad`,
   `row_weight_pad`, and `token_of_row_pad` with its sink id `T`). Ties to
   the Spyre correction-path/host-compute ordering constraints — HostCompute
   runs inline, H2D→Compute is auto-barriered; a RISC-V→device handoff needs
   an explicit fence design.
5. **Restickify on a reshape across a sub-stick dim.** *(New — this is the
   FIRST blocker Approach A's on-card gate actually hit, ahead of #2.)* A
   reshape that flattens a small trailing dim into the row/outer dim aborts
   the layout pass when that trailing dim is a **partial stick** (smaller than
   one 64-element stick). Concretely, the router weights are `[T=64, K=4]`;
   `K=4` occupies a stick padded to 64, so `w.reshape(N=256, 1)` — the
   `row_weight = w.reshape(N)` flatten that feeds the per-row weight multiply in
   the B-Stage-1 body above (and, identically, Approach A's
   `out * tw.reshape(N,1)`) — makes the layout pass read the
   `[T,K]` buffer with a flat `[256]` index. It blocks **both approaches**: the
   flatten *is* this reshape, whether it runs in A's device region or ahead of
   B's grouping, so grouping does not route around it. That read is the
   cross-stick
   expression `Mod(d0,4) + floor(d0/4)` (mod **4**, not mod 64), which
   `is_stick_expr_offset_free` rejects, so `_multi_arg_pointwise_layouts` finds
   no supported output layout and raises:

   ```
   InductorError: Unsupported: Spyre backend does not support:
     Multi-arg pointwise (buf1): no supported output layout found
     with size=[256, 2816] and coordinates=[d0, d1]
   propagate_layouts.py:1099  (load-bearing gate at :985)
   ```

   **Not fixable adapter-side.** Every materialization strategy
   (`.contiguous()`, `* 1.0`, `+ 0.0`, or multiplying in `[T,K,H]` space
   instead of flattening) merely relocates the same abort — because the copy's
   *input read* of the `[T,K]` partial-stick buffer with a flat index is itself
   the unrepresentable expression. Only a device-native `[N,·]` operand (one
   that never passed through the `[T,K]` shape) compiles. **The fix is a
   backend restickify** (HBM round-trip) inserted at the `[T,K]→[N]` reshape,
   re-laying the partial-stick buffer `N`-outermost before any flat consumer
   reads it. The comment at `propagate_layouts.py:987-989` already anticipates
   this ("a restickify will be needed to move the … coordinate away from the
   stick") but the code only inserts it for the index-symbol case, not the
   partial-stick-reshape case; `AllSameNode` cannot rescue it today (it asserts
   `out_layouts` non-empty at `optimize_restickify.py:183` and never synthesizes
   a reconciling geometry, and the abort at `:1098` fires before it is even
   constructed). **Distinct bug class from #2** — no gather, no bmm; they share
   only the raise site (`:1099`). Two separate asks.

   Reproducers (no model load, no real weights, run on any Spyre host):
   `repros/gemma4_moe/gatherbcast_layout_repro.py` (the faithful full-region
   case — its `region` stage reproduces the gate's abort verbatim) and
   `repros/gemma4_moe/rowweight_reshape_probe.py` (a 7-way isolation that pins
   the trigger to the `[T,K]↔N` reshape and rules out every adapter-side
   workaround).

## Validation & top-8 restore (both approaches)

- Each stage is validated by the **single-layer fp16-vs-fp32 rel-err gate**
  (mean_rel < 0.02 / max_rel < 0.5), then the **e2e token-compare**
  (`tests/spyre/test_e2e_token_compare_spyre.py -k 26B-A4B`, non-blocking
  xfail during bring-up).
- **Top-8 restore criterion:** K is pinned to 4 *because* on-device
  `topk(k>4)` SIGABRT'd and grouping ops didn't lower (ledger Task 2). Once
  routing runs on-device with a working `topk` (Approach A) — or once
  grouping is RISC-V-side where `k` is unconstrained (B-Stage 2) — lift `K`
  from 4 to the trained **top-8** and re-validate. **Gate:** token-compare
  top-1 agreement must recover with K=8 plus a correct routing/grouping
  path; that recovery is the signal that the K=4 pin (not an adapter bug)
  was the cause of the current 0/5 divergence.
