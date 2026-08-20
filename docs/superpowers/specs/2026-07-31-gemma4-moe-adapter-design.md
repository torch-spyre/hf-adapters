# Gemma 4 MoE Adapter for Spyre — Design

**Status:** Draft for review
**Target model:** `google/gemma-4-26B-A4B-it` (`model_type=gemma4`, `enable_moe_block=True`)
**Adapter approach:** Extend the existing dense `hf_gemma4.py` with a sparse-MoE FFN branch, using a
lossless permute + grouped-GEMM formulation (DeepSeek/DeepGEMM "contiguous layout" style), with the
expert compute tiled across Spyre cores via `spyre_hint`.

---

## 1. Background & scope

### 1.1 What the model is

`gemma-4-26B-A4B-it` is the MoE member of the Gemma 4 family: ~26B total parameters, ~4B active per
token. It reuses the *exact same* `gemma4` modeling code as the dense 12B/31B variants; MoE is gated
behind a single config flag. The attention stack, norms, RoPE, embedding scaling, and logit softcap
are therefore **identical to what `hf_gemma4.py` already implements and validates**. This design adds
only the FFN (feed-forward) portion.

Verified config (from the hub `config.json`):

| field | value | note |
|---|---|---|
| `enable_moe_block` | `True` | master switch |
| `num_experts` | `128` | |
| `top_k_experts` | `8` | field name is `top_k_experts`, not `num_experts_per_tok` |
| `moe_intermediate_size` | `704` | per-expert FFN hidden |
| `num_hidden_layers` | `30` | **all layers are MoE** (no dense/MoE interleave) |
| `hidden_size` | `2816` | |
| `intermediate_size` (dense MLP) | `2112` | the parallel dense branch |
| `num_attention_heads` / `num_key_value_heads` | `16` / `8` | GQA |
| `head_dim` / `global_head_dim` | `256` / `512` | per-layer-type head dims |
| `layer_types` | 5 `sliding`:1 `full` | full at 5,11,17,23,29 |
| `attention_k_eq_v` | `True` | global layers alias V from K |
| `final_logit_softcapping` | `30.0` | |
| `hidden_size_per_layer_input` / `num_kv_shared_layers` | `0` / `0` | no PLE, no KV-sharing |

Weight footprint: experts hold ~22.8B params (~45.7 GB bf16) — the experts *are* the model. This
matters for on-device residency and motivates the sparse (not dense-all-experts) formulation.

### 1.2 The distinctive structural quirk — parallel dense + MoE

Unlike a typical MoE that *replaces* the FFN, Gemma 4 runs the dense MLP **and** the MoE block in
parallel off the same pre-MLP residual and **sums** them. The dense MLP is effectively an always-on
shared expert. Reference structure (`transformers` `Gemma4TextDecoderLayer.forward`):

```
residual = h
h_ln     = pre_feedforward_layernorm(h)         # dense branch input
h_dense  = post_feedforward_layernorm_1( mlp(h_ln) )
h_moe    = post_feedforward_layernorm_2(
               moe( pre_feedforward_layernorm_2(residual) )   # NB: reads residual, not h_ln
           )
h        = post_feedforward_layernorm( h_dense + h_moe )
h        = residual + h
h        = h * layer_scalar
```

Key subtleties the adapter must preserve:
- The MoE branch reads the **pre-MLP residual**, and it has its **own** pre/post RMSNorms
  (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`) that
  only exist in MoE mode.
- Router: `softmax` over all 128 experts → top-8 → **renormalize** the 8 weights to sum to 1 → multiply
  by a learned `per_expert_scale[expert_id]`.
- Experts are stored as **3D packed tensors**: `gate_up_proj [128, 2*704, 2816]` (gate and up fused,
  split by `chunk(2,-1)`), `down_proj [128, 2816, 704]`. Activation is `gelu_pytorch_tanh` (**not**
  silu).

### 1.3 What is explicitly out of scope

- PLE / E2B-E4B variants (`hidden_size_per_layer_input>0`) — asserted absent.
- Cross-layer KV sharing (`num_kv_shared_layers>0`) — asserted absent.
- Multimodal (vision) input — this adapter is the **text** causal-LM. (The dense VLM path lives in
  `hf_gemma4_mm.py`; a MoE-VLM is a later, separate effort.)
- Training-only concerns (router aux loss, jitter) — inference ignores them.

---

## 2. Why not the upstream forward

The `transformers` reference routes with `torch.nonzero` + a Python loop over "experts that were hit"
+ `index_add_`. That is data-dependent in both shape and control flow and does not fit Spyre's static
compile model. We do **not** port it.

Instead we use the standard efficient formulation from the MoE-kernel literature (Megablocks / vLLM
fused-MoE / DeepSeek DeepGEMM): **route → permute tokens into expert-contiguous order → one grouped
GEMM → unpermute → weighted combine.** The variant we target is the **lossless contiguous-layout
grouped GEMM** (DeepSeek/DeepGEMM `m_grouped_gemm_contiguous`), which needs **no worst-case per-expert
capacity padding** and drops no tokens.

### 2.1 Empirical op-support result (gate outcome, 2026-07-31) — device/host split

The §6 gates were run on the card *before* writing adapter code. The design originally assumed (per the
discussion) that `topk` at k=8, `argsort`, `gather`, and scatter (`index_add`) all compiled on the
current backend. **That assumption did not hold.** Isolated op-by-op on the card:

| op | on-device status |
|---|---|
| `bmm` `[T*K,1,H]×[T*K,H,F]` and `[E,T,H]×[E,H,F]` | ✅ works, numerically correct (gate 1) |
| `x[token_of_row]` plain gather | ✅ works |
| `topk(k=4)` (with softmax) | ❌ `dxp_standalone` SIGABRT |
| `topk(k>4)` | ❌ `Unsupported` (hard `k>4` guard in `spyre_topk`) |
| `argsort` / `sort` | ❌ `aten::sort.values_stable` not implemented for spyre |
| `(arange//K)[perm]` index arithmetic | ❌ multi-arg pointwise, no supported layout on the 1-D index |
| `index_add` (scatter combine) | ❌ `KeyError: indirect0` |

**Decision (per user):** keep the permute + grouped-GEMM formulation, but run the unsupported routing
ops (`topk`, `argsort`, the index arithmetic, and `index_add`) as **CPU/host fallbacks**, while the
matmul-heavy work (router `Linear`, the `[T*K,H]` gather, and the expert grouped GEMM) stays on-device.
The routing tensors are small — `[T,E]` logits, `[T,K]` weights/ids, `[T*K]` permutations — so the host
round-trip is cheap relative to the expert compute. This composes cleanly: separate `torch.compile`
regions for the device pieces with plain host PyTorch between them (verified on the card — device router
→ host route → device gather+GEMM → host combine round-trips successfully).

**Top-k value:** the model config specifies `top_k_experts=8`, but `topk` on the current backend is
capped at (and in fact crashes at) k>4. For bring-up we therefore run with **K=4** (a tracked deviation
from the real model) and revisit K=8 once torch-spyre lifts the cap / fixes the `topk` crash. Because
`topk` runs on the **host** in this formulation, K is not actually constrained by the backend guard;
K=4 is a bring-up choice to keep parity with what a future all-on-device version could support, and to
bound the host↔device traffic. The router weight renormalization and `per_expert_scale` still apply over
the K selected experts regardless of K's value.

*Known-issue note:* a `corrupted double-linked list` SIGABRT can fire on **process teardown** after a
successful device run; per the user this is a known torch-spyre lifetime issue unrelated to compute
correctness and is ignored for this work.

---

## 3. The MoE forward — target formulation (lossless grouped GEMM)

Let `T` = number of tokens in the block (batch·seqlen, flattened), `E=128`, `K=4` (bring-up value;
the model config wants `top_k_experts=8` — see §2.1), `H=2816`, `M=704`. The MoE branch input is
`x = pre_feedforward_layernorm_2(residual)`, shape `[T, H]`.

**Device/host partition (§2.1):** each step below is annotated `[device]` or `[host]`. The compiled
regions are the `[device]` pieces; the `[host]` pieces are plain PyTorch on the small routing tensors,
run between the compiled regions.

### 3.1 Routing

```
logits   = x @ W_router.T   # [T, E]  [device] — router is a plain Linear (matmul, supported)
probs    = softmax(logits, dim=-1)             # [T, E]   [host]  (moved to cpu after the router)
w, idx   = topk(probs, K, dim=-1)              # [T, K], [T, K]   [host]  (topk unsupported on device)
w        = w / w.sum(-1, keepdim=True)         # renormalize (Gemma4 does this)   [host]
w        = w * per_expert_scale[idx]           # learned per-expert scale   [host]
```

`per_expert_scale` is a host tensor for this indexing; `W_router` stays a device weight for the matmul.

### 3.2 Permutation into expert-contiguous order

Flatten the `T*K` (token, expert) assignments and sort by expert id. The sort/index bookkeeping is
`[host]` (argsort + index arithmetic are unsupported on device); only the `gathered = x[token_of_row]`
gather is `[device]` (gather is the one indirect-access op that works on the current backend). This
produces:
- `sort_perm`  : `[T*K]` — permutation ordering pairs by expert id (`argsort(idx.flatten())`) `[host]`.
- `token_of_row = (arange(T*K) // K)[sort_perm]` : source token per sorted row `[host]`.
- `gathered   = x[token_of_row]`                : `[T*K, H]` — tokens gathered into expert-contiguous order `[device]` (x is a device tensor; `token_of_row` is moved to device as int32).
- `counts     = bincount(idx.flatten(), E)`     : `[E]` — tokens per expert `[host]`.
- `group_off  = cumsum(counts)`                 : `[E]` — segment boundaries (prefix sum) `[host]`.

Total rows are exactly `T*K` — **fixed and static**. Only the *segment boundaries* (`group_off`) are
data-dependent; the buffer size is not. This is precisely the property that makes the DeepGEMM
contiguous layout lossless without capacity padding.

### 3.3 Grouped GEMM (the expert compute)

Each contiguous segment `[group_off[e-1] : group_off[e]]` of `gathered` is multiplied by expert `e`'s
weights:

```
gate_up = grouped_gemm(gathered, gate_up_proj, group_off)   # [T*K, 2M]
g, u    = gate_up.chunk(2, dim=-1)
act     = gelu_tanh(g) * u                                  # [T*K, M]
out     = grouped_gemm(act, down_proj, group_off)           # [T*K, H]
```

`grouped_gemm(A, Wstack, offsets)` computes, for each expert segment, `A_seg @ Wstack[e].T`. On Spyre
this is expressed as a batched/segmented matmul; §4 details the two candidate lowerings.

### 3.4 Unpermute + weighted combine

```
out            = out * w.flatten()[sort_perm, None]         # apply router weight per row
scattered      = scatter_add over token_of_row into [T, H]  # sum the K contributions per token
moe_out        = scattered                                  # [T, H]
```

Then the layer combines `post_ff_ln_2(moe_out)` with the dense branch as in §1.2.

### 3.5 Device-layout constraint on the indirectly-addressed tensors

Spyre imposes a hard layout requirement on every tensor that is read or written through an indirect
(gather/scatter) access: **the indirectly-addressed dimension must be the outermost dimension of that
tensor's on-device (stickified) layout.** In the three-pass restickify pipeline this is enforced by the
`enforce_indirect_access_layout` pass (`torch_spyre/_inductor/enforce_indirect_access_layout.py`), which
runs after `insert_restickify`, checks each indirect-access op's value-tensor `dim_order`, and — if the
indexed dimension is not outermost — either rewrites the producer's output layout in place (when the
producer is a non-output `ComputedBuffer`) or inserts a `spyre.restickify` copy into the required
layout.

For this design that means the row (token) dimension being gathered/scattered must be outermost, and
the hidden dimension stickified innermost, on every indirect participant:

- `x` before the §3.2 token gather — the `[T, H]` layout must have `T` (the indexed dim) outermost.
- `gathered` `[T*K, H]` produced by the gather — `T*K` outermost.
- the §3.4 scatter target `[T, H]` and its `out` source `[T*K, H]` — the scattered row dim outermost.
- in Option 4A, the per-row weight gather `gate_up_proj[row_expert]` — the expert dim indexed there
  must be outermost in `gate_up_proj`'s device layout.

The adapter should lay these buffers out so the pass finds the constraint already satisfied (row/expert
dim outermost, `H`/`M` innermost as the stick dimension) rather than relying on inserted restickify
copies — an unplanned restickify of a `[T*K, H]` tensor is a full HBM round-trip on the hot path.
Prototype gate §6.3 must confirm the committed layouts match this and that no surprise restickify is
inserted (check the compile artifacts / restickify insertions).

---

## 4. The core open question: expressing the grouped GEMM on Spyre

This is the piece that **stays on-device** under the §2.1 split — the routing/permutation indices
(`row_expert`, `sort_perm`, `token_of_row`) arrive from the host as plain `int32` tensors, and the
device computes the actual expert matmuls. The segment boundaries `group_off` are host-computed values,
but Spyre still wants static tiling on the device side. Two candidate lowerings, to be chosen during
prototyping (§6):

**Option 4A — gather the per-row weight, then row-wise GEMM.**
The host supplies a per-row expert id `row_expert` (`= idx.flatten()[sort_perm]`, computed on the host);
the device does `W_row = gate_up_proj[row_expert]` (a plain device gather — the only indirect-access op
verified working on-card, §2.1) giving `[T*K, 2M, H]`, and a batched `[T*K,1,H] × [T*K,H,2M]` matmul
(verified correct on-card, §6 gate 1). Simple to express; heavy on weight gather bandwidth (materializes
a weight per row). This is the **bring-up path** — correct grouped result on-device with only the one
device indirect-access primitive that works today.

**Option 4B — tiled contiguous grouped GEMM via `spyre_hint` (the DeepGEMM analogue).**
Keep `gathered` contiguous and tile the M (row) dimension into fixed tiles; each tile reads a single
expert's weight slab selected by host-supplied `group_off`. Express with `spyre_hint(tiles={...})` /
`named_dims=[...]` so the compiler schedules one expert-weight load per tile across the 32 cores —
structurally the same as a Triton grouped-GEMM `group_id` loop. This is the performant target: expert
weights are loaded once per tile, not once per row. It depends on hints propagating through the
gather/segment structure (an area with known fragility — see risks).

Recommendation: implement **4A first** to get a correct grouped result on-device, then move the hot
path to **4B**. Both share the host-side routing (§3.1) and permutation-index computation (§3.2) and the
host-side combine (§3.4); only §3.3's device kernel differs.

---

## 5. Adapter code structure

New file: `hf_adapters/hf_gemma4_moe.py`. It **reuses** the dense attention machinery from
`hf_gemma4.py` and only swaps the FFN branch.

- **Reuse from `hf_gemma4.py`:** `_patch_gemma4_rmsnorm`, `_gemma4_backbone`, `_build_layer_masks`,
  per-layer-type RoPE setup, KV-shape computation, `pad_lm_head`, the attention half of the block, the
  final-norm + logit-softcap in `_run_forward`. Factor the attention half of `block_forward` out of
  `hf_gemma4.py` into a shared helper so both dense and MoE blocks call it (targeted refactor, no
  behavior change to the dense adapter).
- **New in `hf_gemma4_moe.py`:**
  - `_moe_ffn(x, router, gate_up_proj, down_proj, per_expert_scale, K, E)` — the §3 forward.
  - `_make_compiled_moe_block(layer)` — attention (shared helper) + parallel dense MLP + `_moe_ffn`,
    combined per §1.2, `torch.compile(dynamic=False)`.
  - `prepare_for_spyre(model)` — assert PLE/KV-share absent; assert `enable_moe_block=True`; patch
    RMSNorm; build per-type RoPE; record KV shapes; `pad_lm_head`; stack expert weights into the 3D
    packed tensors as registered buffers/parameters (so `.to("spyre")` moves them); compile blocks.
- **`layer_scalar`** stays a tensor argument (not a captured float) — same recompile-avoidance rationale
  documented in `hf_gemma4.py`.
- **Registry:** add `gemma4_moe` → `google/gemma-4-26B-A4B-it` → `hf_gemma4_moe.py` in
  `tests/model_registry.py`; add to `ARCHITECTURE.md` coverage tables + README badge on completion.

---

## 6. Prototype gates (de-risk before full adapter)

Build these tiny repros *first*, on-device, in order. Each gate must pass before proceeding. Under the
§2.1 device/host split, the gates de-risk the **device** pieces (matmul, gather, weighted combine) and
that the **host↔device composition** runs; the routing/permute *math* (softmax/topk/argsort/index-add)
runs in plain host PyTorch and is correct by construction.

1. **Expert-dim batched matmul compiles & is correct.** ✅ **DONE** (`gate1_grouped_gemm.py`). Both the
   row-batched `[T*K,1,H] × [T*K,H,F]` (Option 4A) and dense expert-batched `[E,T,H] × [E,H,F]`
   geometries compile with no `out_reuse_dim.size()==1` abort and match fp32 ground truth at
   mean-rel ~0.7%. The abort that fires in the attention output-projection path does **not** fire here.
2. **Device/host split composes for the full route→permute→GEMM→combine round-trip.**
   (`gate2_route_permute.py`, K=4.) A device `torch.compile` region for the router `Linear`; host
   `softmax`/`topk(K)`/renormalize + `argsort` + `token_of_row` index arithmetic; a device `torch.compile`
   region for the `x[token_of_row]` gather + expert GEMM; host weighted `index_add` combine. Verify the
   three host↔device hand-offs run without error and the recombined output matches a pure-CPU MoE
   reference within fp16 tolerance. This gate exists because the routing ops (`topk`, `argsort`,
   index-arithmetic, `index_add`) do **not** lower on the current backend (§2.1) — the gate proves the
   split is a working substitute.
3. **Device indirect-access layout is satisfied by construction.** For the device gather region of
   gate 2, confirm the gathered/scattered token buffers get the row dim committed **outermost** (§3.5)
   with no surprise `spyre.restickify` copy inserted on the hot path. Inspect the compile artifacts to
   confirm the layout is satisfied by construction, not by an inserted copy.
4. **End-to-end single MoE layer** vs a CPU/HF reference on real weights (bit-close, fp16 tol), using
   the device/host split for the FFN.
5. **Full 30-layer forward** vs HF reference (token-compare), then generation smoke.

If gate 2 or 3 regresses the assumptions from §2/§2.1, stop and revisit the formulation (e.g. move more
of the FFN to the host, or fall back to a dense-masked expert compute for the affected step) rather than
shipping wrong output.

---

## 7. Numerical validation plan

- Run the HF reference forward **before** `prepare_for_spyre` (RMSNorm patch is global — same ordering
  rule as every other adapter).
- Per-layer MoE compare on real 26B-A4B weights (§6 gate 4): assert max-abs / relative error within the
  fp16 tolerances used by the other Gemma 4 adapters.
- End-to-end: `tests/spyre/test_e2e_token_compare_spyre.py -k gemma4_moe`.
- Watch the parallel-branch sum and the router renormalize + `per_expert_scale` — these are the spots
  most likely to diverge from HF if mis-ordered.

---

## 8. Definition of done

- [ ] `hf_gemma4_moe.py` adapter (attention shared with `hf_gemma4.py`, new MoE FFN branch).
- [ ] Prototype gates §6.1–6.3 pass on-device.
- [ ] Registry entry `gemma4_moe`.
- [ ] Compiles + runs e2e on Spyre with no crash/NaN; token-compare within tolerance vs HF.
- [ ] Grouped GEMM on the performant path (Option 4B) — or 4A shipped with a tracked follow-up to 4B.
- [ ] `ARCHITECTURE.md` coverage tables + README badge updated.

---

## 9. Open items for prototyping (not blockers to the plan)

- **4A vs 4B choice** — decided by gate 1 (done) + perf on-device.
- **`per_expert_scale` exact application point** — confirm against `transformers` source at
  implementation time (multiply into `w` before or after renormalize). Applied host-side under §2.1.
- **`K` back to 8, routing back on-device** — the current split runs routing on the host and uses `K=4`
  only as a host↔device-traffic bound (§2.1); `K` is not backend-constrained while topk is host-side.
  When torch-spyre lifts the topk/argsort/index_add limits, revisit moving routing on-device and raising
  `K` to the model's 8. Tracked as a follow-up, not a blocker to a correct adapter.
- **Grouped-GEMM tile size / stick alignment** for 4B — DeepGEMM aligns each group's M to a small tile
  (e.g. 128); map that to Spyre's 64-element stick.
- **Host↔device round-trip cost** — the split adds three transfers per MoE layer (logits down, indices
  up, expert-out down). Measure in §6 gate 4/5; if it dominates, that motivates the on-device routing
  follow-up above.
```
