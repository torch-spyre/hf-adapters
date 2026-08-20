# Gemma 4 MoE Loop-on-Topk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gemma 4 MoE device-FFN formulation so the whole MoE FFN runs on-device *except the scatter-combine*, looping directly over the `[T,K]` topk results with an on-device `index_select` from an HBM-resident expert stack; and produce a backend-collaboration spec for the per-expert grouping follow-on (Approach B).

**Architecture:** The router (norm → proj → softmax → topk(K=4) → renorm → per_expert_scale), the activation gather, the on-device expert-weight `index_select`, and the two expert bmms all run inside one compiled region; a single `spyre_hint(tiles={"row": TILE})` tiles the `N = T·K` row axis so the backend loops over `ceil(N/TILE)` tiles (no Python `for`). Expert weights sit resident in Spyre HBM laid out `[E,H,2M]`/`[E,M,H]` row-major (E outermost, generated dim on the stick) so both the indirect gather and the DF16 matmul lower with zero restickify. Host does only the `index_add` scatter-combine and the parallel dense-MLP add.

**Tech Stack:** PyTorch + `torch.compile(dynamic=False)` on the torch_spyre device backend; HuggingFace `transformers` Gemma4; the `spyre_hint` API in `torch_spyre/_inductor/propagate_hints.py`.

## Global Constraints

- **Design source of truth:** `docs/superpowers/specs/2026-08-01-gemma4-moe-loop-on-topk-design.md`. Every task's requirements implicitly include it.
- **Approach A is implemented; Approach B is specified only** — no B host-grouping or RISC-V code is written (it is blocked on backend primitive #1). B's sole deliverable in this plan is a backend-asks doc.
- **"Assume it works":** the on-device `topk(k=4)`, the on-device windowed indirect gather of expert weights, and the E-outermost+stick layout are assumptions. The single-layer on-card gate is the pass/fail oracle. **If the gate aborts or diverges, STOP and report the exact failure — do NOT engineer a fallback.**
- **fp16 numeric criterion:** device fp16 vs **fp32** CPU ground truth, relative error **mean_rel < 0.02, max_rel < 0.5**. NOT device-fp16-vs-CPU-fp16 at tight atol/rtol (a known false-blocker on this project).
- **K pinned to 4** for bring-up (`_MOE_BRINGUP_K = 4`). Top-8 restore is a separate, later gate.
- **Device shape rule 1:** expert tensors stay **3D `[·,1,·]`** through both bmms; squeeze only at the very end.
- **Device shape rule 2 / layout:** expert weights pre-transposed and **E-outermost**: `gate_up_dev [E,H,2M]` (stick = `2M`), `down_dev [E,M,H]` (stick = `H`), produced by construction in `prepare_for_spyre` (never fixed up post-hoc — `Wstack` is a graph input, so a non-compliant layout forces a full-tensor restickify).
- **Scatter/`index_add` stays on host** (does not lower on device).
- **Router double-normalization rule (Task 6 fix, preserved):** router reads a scale-free norm of the **raw** residual; experts read a **separate** `pre_feedforward_layernorm_2` norm of the same residual. Never feed the pre_ff_ln_2 output to the router.
- **Every source file** carries the Apache 2.0 header already present in the repo. **Line length ≤ 88 chars.**
- **The existing shipped path** (`_moe_ffn_split`, host-resident experts, per-row select) is retained behind a flag until the loop path passes its on-card gate; do not delete it in the same task that adds the new path.

- **VERIFIED router attribute surface + math (preflight-checked against transformers 5.12.1 `modeling_gemma4.py`, both a probe and an independent skeptic agreeing — this CORRECTS the design spec's shorthand).** The stock `Gemma4TextRouter` on a decoder layer:
  - **`layer.router.norm` is `Gemma4RMSNorm(H, with_scale=False)` — it has NO `.weight` attribute.** Do NOT read `layer.router.norm.weight` (AttributeError). The norm is pure scale-free RMSNorm with **eps INSIDE the sqrt** (Gemma variant): `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` in fp32, `eps = config.rms_norm_eps`. **No `+1` gain.**
  - **`layer.router.scale` is an `nn.Parameter` of shape `[H]` (a per-channel vector), NOT a scalar** — applied elementwise AFTER the norm.
  - **`layer.router.scalar_root_size` is a Python float** `= hidden_size ** -0.5`.
  - **Exact router order:** `normed = rmsnorm_scalefree(x); normed = normed * scale[H] * scalar_root_size; logits = proj(normed); probs = softmax(logits,-1); w,idx = topk(probs, K); w = w / w.sum(-1,keepdim=True); w = w * per_expert_scale[idx]`.
  - `layer.router.proj.weight` is `[E,H]`; `layer.router.per_expert_scale` is `[E]`. (These match the spec.)
  - `layer.router.forward` returns a **3-tuple** `(router_probabilities, top_k_weights, top_k_index)` — a 2-value unpack breaks. (We inline the math, so we don't call it, but any code that does must unpack 3.)
- **VERIFIED experts layout:** `layer.experts.gate_up_proj` is **fused** `[E,2M,H]` (chunk into `(gate,up)` AFTER the linear, `chunk(2,dim=-1)`); `layer.experts.down_proj` is `[E,H,M]`. Our pre-transpose (`prepare_for_spyre`) turns these into `[E,H,2M]`/`[E,M,H]`; the fused-then-chunk order is preserved by chunking the `2M` output of the gate/up bmm.
- **VERIFIED on-device-gather guardrails (preflight UNCERTAIN — the very assumptions the on-card gate exists to settle):**
  - The on-device fancy-index `Wstack[ids]` gather is **compiled-only** (eager `aten::index.Tensor` raises `NotImplementedError`) — it MUST sit inside the compiled region, never run eager on device tensors. (It already does, inside the `spyre_hint` region.)
  - **Never a single-row (P=1) index** — `test_advanced_indexing_single_row` SIGABRTs in dxp_standalone (`layoutDimOrder_.empty()`). The row tile must be P>1 (`_MOE_TILE ≥ 2`; default 32 is fine). The gate at K=4 with T≥16 gives N=T·K≥64 rows, tiled ≥2 per tile — compliant.
  - On-device gather **correctness** at our shapes is NOT proven by the torch-spyre suite (every indirect-gather e2e test is `expect_close=None` → `xfail` on divergence; `test_moe` is skipped for output-span overflow). This is the "assume it works" assumption; **the on-card gate (Task 4) is the sole oracle. If it diverges or aborts, STOP and report — do not add a host fallback.**
- **VERIFIED residency mechanism:** a plain (non-Parameter, non-buffer) attribute holding a `.to("spyre")` tensor IS left device-resident by `hf_common._move_to_spyre_with_layout` (it sweeps only `named_parameters()`/`named_buffers()`), and a device tensor is a valid `bmm` operand. So `layer._spyre_gate_up_dev = t.to("spyre")` is sound plumbing. (Capacity caveat: all 128×30 experts resident ≈46 GB fp16 — the single-layer gate touches one layer, so it fits; full-model residency is a separate scaling concern, out of scope for the gate.)
- **VERIFIED layout hazard:** the real restickify risk is an **in-graph `.transpose`** of a weight (forces a huge-offset restickify → `L3_ADDEARIMM` "immediate out of boundary" abort), NOT the E-outermost property per se. Mitigation: supply weights **pre-transposed** by construction (`prepare_for_spyre`), never transpose in-graph. E-outermost + stick-on-generated-dim are compatible on the row-major layout; "zero restickify" is expected but not fully proven — the gate's compile artifacts confirm it.

---

### Task 1: CPU reference for the loop-on-topk FFN math

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` (add `_moe_ffn_loop_ref` near the other CPU helpers, after `_moe_ffn` ~line 280)
- Test: `tests/test_gemma4_moe_ffn.py`

**Interfaces:**
- Consumes: existing `_moe_route(x, W_router, per_expert_scale, K) -> (w[T,K], idx[T,K])` (line 90).
- Produces: `_moe_ffn_loop_ref(x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=None) -> [T,H]` — a pure-CPU fp32 reference that computes the MoE FFN in the **loop-on-topk row order** (NO expert sorting/grouping), so it is the numeric oracle for the new device region. `gate_up_t` is `[E,H,2M]`, `down_t` is `[E,M,H]` (pre-transposed, matching the device layout). `x_router` is the **already-preprocessed router input** (scale-free RMSNorm → `*scale[H]*root_size`); when `None` it defaults to `x` (so the plain `test_moe_ffn_loop_ref_matches_dense_reference` test routes on raw `x`). The expert FFN always reads `x`, never `x_router` — routing and expert input are threaded separately (the double-normalization rule).

This reference deliberately mirrors the device dataflow (per-row weight select in topk order, no argsort) so Task 2's region and Task 4's gate compare against identical math. The `x_router` seam exists so Task 2's region test and Task 4's gate can feed the reference the SAME router-preprocessed input the device region computes, making router math cancel out of the comparison.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gemma4_moe_ffn.py`:

```python
def test_moe_ffn_loop_ref_matches_dense_reference():
    """Loop-on-topk FFN (no grouping) equals the dense per-token top-K sum."""
    torch.manual_seed(0)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    # Pre-transposed expert weights, matching the device layout.
    gate_up_t = torch.randn(E, H, 2 * M)  # [E,H,2M]
    down_t = torch.randn(E, M, H)          # [E,M,H]
    per_expert_scale = torch.rand(E) + 0.5

    got = _moe_ffn_loop_ref(x, W_router, gate_up_t, down_t, per_expert_scale, K)

    # Independent dense reference: route, then for each token sum its K experts.
    w, idx = _moe_route(x, W_router, per_expert_scale, K)  # [T,K],[T,K]
    ref = torch.zeros(T, H)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            gu = x[t] @ gate_up_t[e]            # [2M]
            g, u = gu.chunk(2, dim=-1)          # [M],[M]
            act = torch.nn.functional.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * (act @ down_t[e])  # [H]
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_moe_ffn_loop_ref_matches_dense_reference -v`
Expected: FAIL with `NameError`/`AttributeError` — `_moe_ffn_loop_ref` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `hf_adapters/hf_gemma4_moe.py`:

```python
def _moe_ffn_loop_ref(
    x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=None
):
    """CPU fp32 reference for the loop-on-topk MoE FFN (no expert grouping).

    Mirrors the on-device dataflow: route to top-K, flatten the [T,K] result to
    N=T*K rows in topk order (NO argsort), per-row select the expert weight,
    two bmms with a gelu-tanh SwiGLU between, weight by the router weight, and
    scatter-add back to [T,H]. The numeric oracle for _compiled_moe_loop_region.

    Args:
        x: Expert-FFN input token embeddings [T, H] (fed to the experts).
        W_router: Router projection weights [E, H].
        gate_up_t: Pre-transposed gate_up per expert [E, H, 2M].
        down_t: Pre-transposed down per expert [E, M, H].
        per_expert_scale: Per-expert scale [E].
        K: Top-K experts per token.
        x_router: Router input [T, H] (already RMSNorm/scale-preprocessed). When
            None, defaults to x. Routing uses x_router; the experts always read
            x -- router and expert inputs are threaded separately (the double-
            normalization rule from the design spec).

    Returns:
        out: MoE FFN output [T, H].
    """
    T, H = x.shape
    if x_router is None:
        x_router = x
    w, idx = _moe_route(x_router, W_router, per_expert_scale, K)  # [T,K],[T,K]
    expert_of_row = idx.reshape(-1)  # [N] N=T*K, topk order (NOT sorted)
    token_of_row = (torch.arange(T * K, device=x.device) // K)  # [N]
    row_w = w.reshape(-1, 1)  # [N,1]

    gathered = x[token_of_row]  # [N,H]
    gu = torch.bmm(
        gathered.unsqueeze(1), gate_up_t[expert_of_row]
    )  # [N,1,2M]
    g, u = gu.chunk(2, dim=-1)  # [N,1,M]
    act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
    row_out = torch.bmm(act, down_t[expert_of_row]).squeeze(1)  # [N,H]
    row_out = row_out * row_w  # [N,1] broadcast

    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out = out.index_add(0, token_of_row.long(), row_out)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_moe_ffn_loop_ref_matches_dense_reference -v`
Expected: PASS.

- [ ] **Step 5: Run the full CPU FFN test module to confirm no regression**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py -q`
Expected: all prior tests + the new one PASS.

- [ ] **Step 6: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py tests/test_gemma4_moe_ffn.py
git commit -m "feat: gemma4 MoE loop-on-topk CPU reference (_moe_ffn_loop_ref)"
```

---

### Task 2: Compiled loop-on-topk device region

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` (add `_MOE_TILE` constant near `_MOE_GEMM_4B` ~line 87; add `_compiled_moe_loop_region` after `_compiled_moe_device_region` ~line 306)
- Test: `tests/test_gemma4_moe_ffn.py`

**Interfaces:**
- Consumes: nothing from other tasks (self-contained device fn).
- Produces: `_compiled_moe_loop_region(x_router, x_expert, router_proj_w, router_scale, router_scalar_root_size, per_expert_scale, gate_up_dev, down_dev, token_ids, K, tile, eps) -> (row_out[N,H], token_of_row[N])` — the compiled region computing router→topk→gather→index_select→bmm→gelu→bmm→row-weight, tiled on the row axis. Returns per-row weighted expert outputs `[N,H]` **and** `token_of_row[N]` so the host can scatter-combine.
  - **NOTE (preflight-corrected):** there is NO `router_norm_weight` parameter — the stock `Gemma4TextRouter.norm` is `with_scale=False` and has no `.weight`. The router norm is scale-free RMSNorm (eps INSIDE sqrt); the learnable gain is `router_scale`, an `[H]` vector applied AFTER the norm; `router_scalar_root_size` is a float. `eps` is passed in (`config.rms_norm_eps`) rather than hardcoded.

This region is written as **single tiled ops under one `spyre_hint(tiles={"row": tile})`** — there is NO Python `for` loop over tiles (the hint generates the loop; `coarse_tile._apply_plan` + `scheduler.py:677 LoopSpec`). The router math is inlined (not via `router` module) so the whole region is one compilable function of plain tensors. **The `spyre_hint` CPU-eager no-op behavior is preflight-verified** (`is_compiling()` is False in eager → the Dynamo path is skipped → ordinary aten ops run untouched), so Step 1's CPU-eager equivalence test is valid.

- [ ] **Step 1: Write the failing test (CPU-eager equivalence to the reference)**

The region is a plain function of tensors; on CPU-eager it must equal `_moe_ffn_loop_ref` post-combine. Add to `tests/test_gemma4_moe_ffn.py`:

```python
def test_compiled_moe_loop_region_matches_reference_eager():
    """The device-region fn, run eager on CPU, matches _moe_ffn_loop_ref."""
    torch.manual_seed(1)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    gate_up_t = torch.randn(E, H, 2 * M)
    down_t = torch.randn(E, M, H)
    per_expert_scale = torch.rand(E) + 0.5
    token_ids = torch.arange(T)

    # The region and the reference share the SAME router math. To make them
    # agree, drive the region with scale=ones(H), root_size=1.0, eps=1e-6, and
    # give _moe_ffn_loop_ref the SAME router preprocessing via its x_router arg
    # (the reference applies F.linear to x_router, so pass the region's normed
    # input). Both then route identically; only the expert FFN math is compared.
    row_out, token_of_row = _compiled_moe_loop_region(
        x, x, W_router,
        torch.ones(H), 1.0,
        per_expert_scale, gate_up_t, down_t, token_ids, K, tile=4, eps=1e-6,
    )
    combined = torch.zeros(T, H).index_add(0, token_of_row.long(), row_out)

    var = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + 1e-6)  # == region's scale-free RMSNorm
    ref = _moe_ffn_loop_ref(
        x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=x_normed
    )
    assert torch.allclose(combined, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_compiled_moe_loop_region_matches_reference_eager -v`
Expected: FAIL — `_compiled_moe_loop_region` not defined.

- [ ] **Step 3: Write minimal implementation**

Add the constant near `_MOE_GEMM_4B`:

```python
# Row-tile size for the loop-on-topk device region (spec Approach A). The
# spyre_hint(tiles={"row": _MOE_TILE}) tiles the N=T*K row axis so the backend
# loops over ceil(N/_MOE_TILE) tiles; a tuning knob (scratchpad window size).
_MOE_TILE = 32
```

Add the region:

```python
def _compiled_moe_loop_region(
    x_router,
    x_expert,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    gate_up_dev,
    down_dev,
    token_ids,
    K,
    tile,
    eps,
):
    """Whole MoE FFN on-device except the scatter-combine (spec Approach A).

    Router (inlined SCALE-FREE RMSNorm on the RAW residual x_router with eps
    INSIDE the sqrt, then * router_scale[H] * router_scalar_root_size, then
    proj) -> softmax -> topk(K) -> renorm -> per_expert_scale; then, under a
    single spyre_hint(tiles={"row": tile}) that tiles the N=T*K row axis (the
    hint IS the loop -- no Python for), gather the expert-input rows from
    x_expert, index_select the per-row expert weights from the HBM-resident
    E-outermost stacks, two bmms (3D [*,1,*] throughout) with a gelu-tanh
    SwiGLU, and weight by the router weight.

    Preflight-corrected router surface: the stock router.norm has NO .weight
    (Gemma4RMSNorm with_scale=False); the learnable gain is router_scale, an
    [H] vector applied AFTER the scale-free norm. router_scalar_root_size is a
    Python float (hidden_size ** -0.5). eps is config.rms_norm_eps.

    gate_up_dev: [E,H,2M] (stick=2M), down_dev: [E,M,H] (stick=H), E outermost.
    tile must be >= 2 (single-row P=1 gather SIGABRTs in dxp_standalone).
    Returns (row_out[N,H], token_of_row[N]) for the host index_add combine.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape
    N = T * K

    # --- router (all device-lowerable): SCALE-FREE RMSNorm (eps inside sqrt)
    # on the RAW residual, then the [H] scale vector and the root-size scalar.
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free (no gain in norm)
    normed = normed * router_scale * router_scalar_root_size  # scale is [H]
    logits = F.linear(normed, router_proj_w)  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K],[T,K]  ASSUMED to lower
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[idx]  # [T,K]

    expert_of_row = idx.reshape(N)  # [N] topk order (no sort)
    token_of_row = (token_ids.reshape(T, 1).expand(T, K)).reshape(N)  # [N]
    row_w = w.reshape(N, 1)  # [N,1]

    with spyre_hint(tiles={"row": tile}):
        gathered = x_expert[token_of_row]  # [N,H]
        W_gu = gate_up_dev[expert_of_row]  # [N,H,2M] on-device index_select
        W_dn = down_dev[expert_of_row]  # [N,M,H]
        gu = torch.bmm(gathered.unsqueeze(1), W_gu)  # [N,1,2M] 3D throughout
        g, u = gu.chunk(2, dim=-1)  # [N,1,M]
        act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
        row_out = torch.bmm(act, W_dn).squeeze(1)  # [N,H]
        row_out = row_out * row_w  # [N,1] broadcast
    return row_out, token_of_row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_compiled_moe_loop_region_matches_reference_eager -v`
Expected: PASS (the `spyre_hint` import resolves; on CPU-eager it is a no-op context manager over plain tensor ops).

- [ ] **Step 5: Line-length + full-module check**

Run: `awk 'length>88{print NR": "length}' hf_adapters/hf_gemma4_moe.py` (expect no output) and `python3 -m pytest tests/test_gemma4_moe_ffn.py -q` (expect all PASS).

- [ ] **Step 6: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py tests/test_gemma4_moe_ffn.py
git commit -m "feat: gemma4 MoE loop-on-topk compiled device region (row-tiled)"
```

---

### Task 3: HBM-resident E-outermost expert layout + loop-path orchestrator + flag

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` — add `_MOE_LOOP_ON_TOPK` flag (near `_MOE_TILE`); add `_moe_ffn_loop` orchestrator (after `_moe_ffn_split` ~line 413); branch `_make_moe_block.block_forward` on the flag; branch `prepare_for_spyre` expert layout on the flag.
- Test: `tests/test_gemma4_moe_ffn.py`

**Interfaces:**
- Consumes: `_compiled_moe_loop_region(...)` (Task 2); existing `_make_moe_block` (line 416), `prepare_for_spyre` (line 559), the `layer.router` submodule attributes (**preflight-corrected surface** — `router.proj.weight` `[E,H]`, `router.scale` `[H]` nn.Parameter, `router.scalar_root_size` float, `router.per_expert_scale` `[E]`, and `config.rms_norm_eps` for the norm eps). **There is NO `router.norm.weight`** (`Gemma4RMSNorm` is `with_scale=False`); do not read it.
- Produces: `_moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_up_dev, down_dev, K, tile, eps) -> [T,H]` — host orchestrator that calls the compiled loop region (passing the router's unpacked tensors + device-resident expert stacks + `eps`) and does the host `index_add` combine. Sets, when `_MOE_LOOP_ON_TOPK` is True, `layer._spyre_gate_up_dev`/`layer._spyre_down_dev` as **device-resident** pre-transposed stacks.

**Note:** the flag defaults **False** so the shipped host-split path is unchanged until the on-card gate (Task 4) validates the loop path. This task adds the loop path in parallel; it does not remove `_moe_ffn_split`.

- [ ] **Step 1: Write the failing test (orchestrator equals reference on CPU)**

Add to `tests/test_gemma4_moe_ffn.py`:

```python
def test_moe_ffn_loop_orchestrator_matches_reference():
    """_moe_ffn_loop (host combine + region) equals _moe_ffn_loop_ref on CPU."""
    import types
    torch.manual_seed(2)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    gate_up_t = torch.randn(E, H, 2 * M)
    down_t = torch.randn(E, M, H)
    per_expert_scale = torch.rand(E) + 0.5

    # Minimal router stub with the PREFLIGHT-CORRECTED Gemma4TextRouter surface:
    # norm has NO .weight (scale-free); scale is an [H] vector Parameter;
    # scalar_root_size is a float. Neutral values (scale=ones, root_size=1.0)
    # so the orchestrator's router preprocessing reduces to a plain scale-free
    # RMSNorm, matched below via _moe_ffn_loop_ref's x_router seam.
    router = types.SimpleNamespace(
        scale=torch.ones(H),
        scalar_root_size=1.0,
        proj=types.SimpleNamespace(weight=W_router),
        per_expert_scale=per_expert_scale,
    )

    def compiled_loop(*args):
        return _compiled_moe_loop_region(*args)

    got = _moe_ffn_loop(
        x, x, router, compiled_loop, gate_up_t, down_t, K, tile=4, eps=1e-6
    )
    # Feed the reference the SAME router-preprocessed input the region computes
    # (scale-free RMSNorm; scale=ones and root_size=1.0 add nothing) so routing
    # is identical and only the expert FFN math is under test.
    var = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + 1e-6)
    ref = _moe_ffn_loop_ref(
        x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=x_normed
    )
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_moe_ffn_loop_orchestrator_matches_reference -v`
Expected: FAIL — `_moe_ffn_loop` not defined.

- [ ] **Step 3: Write the flag, the orchestrator, and the block/prepare branches**

Add near `_MOE_TILE`:

```python
# Device-FFN formulation selector (spec Approach A).
#   False (default) -> shipped host-split path (_moe_ffn_split): experts
#       host-resident, per-row weight select on CPU, [N,.] slices to device.
#   True            -> loop-on-topk path (_moe_ffn_loop): experts HBM-resident
#       on device, on-device index_select under a row-tiled spyre_hint.
# Flip to True only after gateA_loop_on_topk.py passes on-card.
_MOE_LOOP_ON_TOPK = False
```

Add the orchestrator after `_moe_ffn_split`:

```python
def _moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_up_dev,
                  down_dev, K, tile, eps):
    """Loop-on-topk MoE FFN orchestrator (spec Approach A).

    Unpacks the router's tensors, calls the compiled loop region (router +
    gather + on-device expert-weight index_select + bmms, row-tiled), then does
    the host index_add scatter-combine (scatter does not lower on device). The
    expert stacks are DEVICE-resident here (unlike _moe_ffn_split's host-
    resident stacks) -- the whole point of Approach A is the on-device select.

    Router surface (preflight-corrected): router.norm has NO .weight; the region
    applies a scale-free RMSNorm (eps INSIDE sqrt) then the [H] router.scale
    vector and the router.scalar_root_size float. Pass eps=config.rms_norm_eps.

    Returns the combined [T,H] MoE output on x_expert's device.
    """
    T, H = x_expert.shape
    token_ids = torch.arange(T, device=x_expert.device, dtype=torch.int32)
    row_out, token_of_row = compiled_loop(
        x_router,
        x_expert,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        gate_up_dev,
        down_dev,
        token_ids,
        K,
        tile,
        eps,
    )
    row_out = row_out.cpu().float()
    token_of_row = token_of_row.cpu().long()
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row, row_out)
    return out.to(dtype=x_expert.dtype, device=x_expert.device)
```

In `_make_moe_block`, after the existing `compiled_expert = torch.compile(...)` line (~481), add a compiled loop region:

```python
    compiled_loop = torch.compile(_compiled_moe_loop_region, dynamic=False)
```

The loop region needs the RMSNorm eps (`config.rms_norm_eps`). Capture it from the `pre_ff_ln_2` RMSNorm instance already bound in the factory (it is a `Gemma4RMSNorm`, which stores its epsilon on `.eps == config.rms_norm_eps` — NOT `.variance_epsilon`, which is the generic HF-patched name; the same eps the scale-free router norm uses). Add near the other captured norms (~467):

```python
    moe_rms_eps = pre_ff_ln_2.eps  # Gemma4RMSNorm uses .eps (== config.rms_norm_eps)
```

and in `block_forward`, replace the single `moe_out = _moe_ffn_split(...)` call (~537-546) with a flag branch:

```python
        if _MOE_LOOP_ON_TOPK:
            moe_out = _moe_ffn_loop(
                flat,
                x_moe,
                router,
                compiled_loop,
                layer._spyre_gate_up_dev,
                layer._spyre_down_dev,
                K,
                _MOE_TILE,
                moe_rms_eps,
            )  # [T,H]
        else:
            moe_out = _moe_ffn_split(
                flat,
                x_moe,
                router,
                compiled_gather,
                compiled_expert,
                layer._spyre_gate_up_t,
                layer._spyre_down_t,
                K,
            )  # [T,H]
```

In `prepare_for_spyre`, branch the expert-layout loop (the `for layer in backbone.layers:` block ~618-627). When `_MOE_LOOP_ON_TOPK`, keep the pre-transposed E-outermost stacks **on the device** (do NOT `.cpu()`), stored under the `_dev` names so the spyre move sweep still leaves them — they are plain attributes, so move them explicitly:

```python
    for layer in backbone.layers:
        experts = layer.experts
        # gate_up_proj: [E,2M,H] -> [E,H,2M]; down_proj: [E,H,M] -> [E,M,H].
        gate_up_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()
        down_t = experts.down_proj.data.transpose(1, 2).contiguous()
        del experts.gate_up_proj
        del experts.down_proj
        if _MOE_LOOP_ON_TOPK:
            # Approach A: experts HBM-RESIDENT on device, E outermost. Row-major
            # [E,H,2M]/[E,M,H] is E-outermost (enforce_indirect_access: indexed
            # dim at device position 0) AND stick-correct for the bmm weight
            # operand (2M / H is the generated dim on the stick) -> zero
            # restickify. Move explicitly (plain attrs are not in the buffer
            # sweep).
            layer._spyre_gate_up_dev = gate_up_t.to("spyre")  # [E,H,2M]
            layer._spyre_down_dev = down_t.to("spyre")  # [E,M,H]
        else:
            layer._spyre_gate_up_t = gate_up_t.cpu()  # [E,H,2M] host-resident
            layer._spyre_down_t = down_t.cpu()  # [E,M,H] host-resident
        layer._spyre_moe_k = _MOE_BRINGUP_K
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py::test_moe_ffn_loop_orchestrator_matches_reference -v`
Expected: PASS.

- [ ] **Step 5: Confirm the default path is byte-identical (flag OFF regression)**

Run: `python3 -m pytest tests/test_gemma4_moe_ffn.py tests/test_gemma4_moe_dispatch.py -q`
Expected: all PASS. The `_MOE_LOOP_ON_TOPK=False` default means `block_forward`/`prepare_for_spyre` take the existing branch — no behavior change to the shipped path.

- [ ] **Step 6: Line-length check + commit**

Run: `awk 'length>88{print NR": "length}' hf_adapters/hf_gemma4_moe.py` (expect no output).

```bash
git add hf_adapters/hf_gemma4_moe.py tests/test_gemma4_moe_ffn.py
git commit -m "feat: gemma4 MoE loop-on-topk orchestrator + HBM-resident experts (flag OFF)"
```

---

### Task 4: On-card single-layer gate for the loop-on-topk path (the oracle)

**Files:**
- Create: `repros/gemma4_moe/gateA_loop_on_topk.py`

**Interfaces:**
- Consumes: `_MOE_LOOP_ON_TOPK`, `prepare_for_spyre`, `_make_moe_block`, `_moe_ffn_loop_ref` from `hf_adapters/hf_gemma4_moe.py`; the real `google/gemma-4-26B-A4B-it` weights.
- Produces: a standalone on-card gate script; prints `mean_rel`/`max_rel` and PASS/FAIL against the fp16 criterion.

This task is the **assumption oracle** (Global Constraints). It runs ON-CARD. Per the ledger's host-env memory: run locally on this host (Spyre card directly reachable), `HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1`, gated model needs the cache present.

- [ ] **Step 1: Write the gate script**

Create `repros/gemma4_moe/gateA_loop_on_topk.py` (Apache header + this body). It forces `_MOE_LOOP_ON_TOPK=True`, builds layer-0's MoE block on-card at K=4, and compares device fp16 output to a fp32 CPU `_moe_ffn_loop_ref` at the same K:

```python
# <Apache 2.0 header — copy verbatim from hf_gemma4_moe.py lines 1-13>
"""On-card single-layer gate for the Gemma 4 MoE loop-on-topk path (Approach A).

Compares the compiled loop-on-topk MoE FFN (device fp16, experts HBM-resident,
on-device index_select, row-tiled spyre_hint) for decoder layer 0 against a
pure-CPU fp32 reference on the real google/gemma-4-26B-A4B-it weights. This is
the pass/fail oracle for the three Approach-A assumptions: on-device topk(k=4),
on-device windowed indirect gather, and the E-outermost + stick layout.

Criterion: mean_rel < 0.02, max_rel < 0.5 (device fp16 vs fp32 CPU truth).
If it aborts or diverges, STOP and report the exact failure (do not fall back).

Run:
  HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \\
    python3 -u repros/gemma4_moe/gateA_loop_on_topk.py
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import hf_adapters.hf_gemma4_moe as moe
from hf_adapters.hf_gemma4 import _gemma4_backbone
from hf_adapters.hf_gemma4_moe import _moe_ffn_loop_ref

MODEL = "google/gemma-4-26B-A4B-it"
K = 4


def main():
    moe._MOE_BRINGUP_K = K  # apples-to-apples with the reference
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    # gemma-4-26B loads as multimodal Gemma4ForConditionalGeneration; the decoder
    # stack lives at the text backbone (Gemma4TextModel), NOT model.model.layers.
    layer = _gemma4_backbone(model).layers[0]

    # fp32 CPU ground truth on a fixed random input, BEFORE prepare (the RMSNorm
    # patch is global; capture the reference first). K=4, N=T*K=256 (>1, tiled).
    torch.manual_seed(0)
    T = 64  # small token count; N = T*K = 256 rows
    H = cfg.text_config.hidden_size
    eps = cfg.text_config.rms_norm_eps
    x = torch.randn(T, H, dtype=torch.float32)

    # Router preprocessing must match the device region EXACTLY: scale-free
    # RMSNorm (eps inside sqrt) -> * scale[H] -> * scalar_root_size, THEN proj.
    # Feed that preprocessed input to the reference via its x_router seam so the
    # gate compares the expert FFN + on-device machinery, not router-math skew.
    W_router = layer.router.proj.weight.data.float()
    router_scale = layer.router.scale.data.float()  # [H] vector, NOT scalar
    root_size = float(layer.router.scalar_root_size)  # hidden_size ** -0.5
    var = x.pow(2).mean(-1, keepdim=True)
    x_router = x * torch.rsqrt(var + eps)  # scale-free RMSNorm
    x_router = x_router * router_scale * root_size
    # gate_up_proj is FUSED [E,2M,H]; transpose to [E,H,2M] (chunk stays post-bmm).
    gate_up_t = layer.experts.gate_up_proj.data.transpose(1, 2).contiguous().float()
    down_t = layer.experts.down_proj.data.transpose(1, 2).contiguous().float()
    per_expert_scale = layer.router.per_expert_scale.data.float()
    ref = _moe_ffn_loop_ref(
        x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=x_router
    )

    # Device path: force the loop-on-topk formulation, prepare, run the region.
    moe._MOE_LOOP_ON_TOPK = True
    moe.prepare_for_spyre(model)
    router = layer.router
    compiled_loop = torch.compile(moe._compiled_moe_loop_region, dynamic=False)
    x_dev = x.to(torch.float16).to("spyre")
    got = moe._moe_ffn_loop(
        x_dev, x_dev, router, compiled_loop,
        layer._spyre_gate_up_dev, layer._spyre_down_dev, K, moe._MOE_TILE, eps,
    )
    got = got.cpu().float()

    diff = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    mean_rel = (diff / denom).mean().item()
    max_rel = (diff / denom).max().item()
    print(f"ref fp32 shape={tuple(ref.shape)} min={ref.min():.4f} max={ref.max():.4f}")
    print(f"got fp16->fp32 min={got.min():.4f} max={got.max():.4f}")
    print(f"mean_rel={mean_rel*100:.4f}%  max_rel={max_rel:.4f}")
    ok = mean_rel < 0.02 and max_rel < 0.5
    print("PASS" if ok else "FAIL", "gemma4 MoE loop-on-topk layer-0 parity")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate on-card**

Run:
```bash
HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
  python3 -u repros/gemma4_moe/gateA_loop_on_topk.py
```
Expected (assumptions hold): `PASS gemma4 MoE loop-on-topk layer-0 parity`, `mean_rel` well under 2%.

**If it aborts (dxp SIGABRT / lowering error / restickify) or FAILs the criterion:** STOP. Capture the exact error and which assumption it implicates (on-device topk; indirect gather correctness/span; E-outermost layout / restickify), record it, and **report to the human partner** — this is the "assume it works; tell me if the device test fails" contract. Do not add a host-fallback branch.

- [ ] **Step 3: Commit the gate (regardless of PASS/FAIL — it is the durable oracle)**

```bash
git add repros/gemma4_moe/gateA_loop_on_topk.py
git commit -m "test: on-card single-layer gate for gemma4 MoE loop-on-topk (Approach A)"
```

- [ ] **Step 4: If PASS — flip the default flag on and re-confirm**

Only if Step 2 PASSed: set `_MOE_LOOP_ON_TOPK = True` in `hf_adapters/hf_gemma4_moe.py`, re-run the CPU suite (`python3 -m pytest tests/test_gemma4_moe_ffn.py tests/test_gemma4_moe_dispatch.py -q`, expect PASS), and commit:

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -m "feat: enable gemma4 MoE loop-on-topk device path by default (gate passed)"
```

If Step 2 did not pass, leave the flag OFF and stop here with the failure report.

---

### Task 5: Approach B backend-collaboration spec (specify only)

**Files:**
- Create: `docs/superpowers/specs/2026-08-01-gemma4-moe-grouping-backend-asks.md`

**Interfaces:** none (documentation deliverable).

Approach B is **not implemented** (blocked on backend primitive #1). This task extracts the Approach-B content of the design doc into a standalone, backend-team-facing asks document, so the torch-spyre/deeptools team has a single reference. It restates the four named deliverables with their code citations and the fixed-tile contract + RISC-V ABI.

- [ ] **Step 1: Write the backend-asks doc**

Create `docs/superpowers/specs/2026-08-01-gemma4-moe-grouping-backend-asks.md` containing, verbatim-faithful to the design doc's Approach-B section:
  - The B-Stage-1 (host CPU) grouping algorithm + the **fixed-tile contract** (TILE-aligned per-expert segments; `group_off = cumsum(bincount(expert_of_row, E))`; per-tile `tile_expert[N_TILES]`).
  - The B-Stage-2 RISC-V migration (byte-identical device program; only the producer of `group_off`/`tile_expert`/`sort_perm` moves).
  - The **four named backend deliverables**, each with its torch-spyre citation:
    1. Per-segment operand-select tiling primitive (grouped-GEMM op OR `group_off`-driven per-tile operand-select hint) — today `tiles={...}` binds only to output ranges (`wsr/coarse_tile.py:758-768`, `wsr/coarse_tile_hints.py`).
    2. Windowed HBM→scratchpad indirect-gather correctness (`test_moe` skipped for output-span overflow, `tests/inductor/test_indirect_access_gather.py:447-465`; indirect gather defaults to xfail on divergence, `indirect_access_common.py:413-434`) — also Approach A's dependency.
    3. `per_tile_fixed` for the weight operand (`insert_restickify.py:281-345`).
    4. RISC-V grouping ABI (memory region/layout/fence contract).
  - The validation + top-8 restore criteria.
  - A pointer back to the design doc and this plan.

- [ ] **Step 2: Verify the doc references resolve**

Run: `python3 -c "import pathlib; p=pathlib.Path('docs/superpowers/specs/2026-08-01-gemma4-moe-grouping-backend-asks.md'); assert p.exists() and p.stat().st_size > 1500; print('ok', p.stat().st_size)"`
Expected: `ok <bytes>`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-01-gemma4-moe-grouping-backend-asks.md
git commit -m "docs: gemma4 MoE grouping backend-collaboration asks (Approach B, specify only)"
```

---

### Task 6: Documentation — record the loop-on-topk path status

**Files:**
- Modify: `ARCHITECTURE.md` (Gemma 4 MoE footnote — update to reflect the loop-on-topk path status)
- Modify: `hf_adapters/hf_gemma4_moe.py` (module docstring — describe the two selectable device formulations)

**Interfaces:** none.

- [ ] **Step 1: Update the module docstring**

In `hf_adapters/hf_gemma4_moe.py` (docstring lines ~15-52), add a short paragraph documenting the two device formulations selectable via `_MOE_LOOP_ON_TOPK` (host-split default vs. loop-on-topk), and that the loop path keeps experts HBM-resident with E-outermost/stick layout. Keep the existing shape-rule text.

- [ ] **Step 2: Update the ARCHITECTURE.md footnote**

In `ARCHITECTURE.md`, extend the existing Gemma 4 MoE footnote to note the loop-on-topk device path and its gate (`gateA_loop_on_topk.py`), with its on-card status (PASS/FAIL per Task 4) stated honestly. Do NOT change the verified-checkpoint count. Do NOT mark the model verified unless the e2e token-compare passed.

- [ ] **Step 3: Confirm counts unchanged + commit**

Run: `grep -n "adapters-\|Supported Models\|Verified Checkpoints" README.md ARCHITECTURE.md | head` and confirm no count changed (this task documents status only).

```bash
git add ARCHITECTURE.md hf_adapters/hf_gemma4_moe.py
git commit -m "docs: document gemma4 MoE loop-on-topk device path + gate status"
```

---

## Self-Review Notes

- **Spec coverage:** Approach A device region → Task 2; router+topk on-device → Task 2 (inlined router math); E-outermost/stick HBM-resident layout → Task 3 `prepare_for_spyre` branch; host index_add combine → Task 3 orchestrator; single-layer fp16-vs-fp32 gate (the oracle) → Task 4; K=4 pin → Tasks 1-4 all force K=4; Approach B specify-only + four backend asks + fixed-tile contract + RISC-V ABI + top-8 restore → Task 5; docs/status → Task 6. `_MOE_TILE` knob → Task 2. Retain shipped path behind flag → Task 3 (flag defaults OFF; Task 4 flips it only on PASS).
- **"Assume it works" contract:** encoded in Global Constraints + Task 4 Step 2 (STOP + report, no fallback).
- **Type consistency:** `_moe_ffn_loop_ref(x, W_router, gate_up_t, down_t, per_expert_scale, K)` (Task 1) is the oracle for both `_compiled_moe_loop_region` (Task 2) and `_moe_ffn_loop` (Task 3); the region returns `(row_out[N,H], token_of_row[N])` consumed by the orchestrator's `index_add`; `gate_up_dev [E,H,2M]` / `down_dev [E,M,H]` names/shapes consistent across Tasks 2/3/4; the loop-path stacks are `layer._spyre_gate_up_dev`/`_spyre_down_dev` (device) vs. the shipped `_spyre_gate_up_t`/`_spyre_down_t` (host).
- **Reboot-safety:** each task commits locally (this host may reboot; local commits are the durable record — consistent with prior session).
```
