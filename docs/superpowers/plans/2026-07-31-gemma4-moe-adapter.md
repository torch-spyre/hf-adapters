# Gemma 4 MoE Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Spyre adapter for `google/gemma-4-26B-A4B-it` (the MoE Gemma 4 variant) that compiles and runs end-to-end with token output matching the HuggingFace reference.

**Architecture:** Reuse the dense `hf_gemma4.py` attention stack unchanged and add a sparse-MoE FFN branch that runs in parallel with the dense MLP (summed). The MoE forward uses a lossless permute + grouped-GEMM (DeepSeek/DeepGEMM contiguous-layout) formulation — route → sort tokens by expert → single grouped GEMM → unpermute → weighted combine — with the expert compute tiled across Spyre cores via `spyre_hint`. No worst-case capacity padding, no token dropping.

**Device/host split (empirically forced — see spec §2.1):** on the current torch-spyre backend the routing/permute ops (`topk`, `argsort`/`sort`, 1-D index arithmetic, `index_add`/`scatter_add`) do **not** lower — only plain gather and `bmm` do. So the MoE forward is partitioned: the **device** (`torch.compile`) does the router `Linear`, the `x[token_of_row]` token gather, and the expert grouped GEMM; the **host** (plain eager PyTorch on small routing tensors) does softmax/topk/renormalize/`per_expert_scale`, `argsort` + `token_of_row` index arithmetic, and the weighted `index_add` combine. The three host↔device hand-offs are `.cpu()`/`.to("spyre")` on `[T,E]`/`[T,K]`/`[T*K]`/`[T*K,H]` tensors. `top_k_experts` is pinned to **K=4** for bring-up (host-side, so not backend-constrained — a host↔device-traffic bound / parity target with a future on-device version); the model wants 8, revisited when torch-spyre lifts the topk/argsort/index_add limits. This is a bring-up formulation; moving routing on-device + raising K is a tracked follow-up, not a blocker to a correct adapter.

**Tech Stack:** Python, PyTorch, `torch.compile(dynamic=False)`, torch-spyre inductor backend, HuggingFace `transformers` `gemma4` modeling code. Tests via pytest against the model registry.

## Global Constraints

- **License header:** every new source file carries the 14-line Apache 2.0 Python header (copy from any `hf_adapters/*.py`).
- **Line length:** 88 chars; code must pass the repo's pre-commit (ruff etc.). Run `.pre-commit-config.yaml` hooks before finishing a task.
- **Spyre dtype:** float16 default; the registry entry may request bfloat16. Spyre does not support float32; zero-length tensors must be created with `device=`, never `.to("spyre")`.
- **Test ordering:** always run the HF reference forward BEFORE `prepare_for_spyre` — the RMSNorm patch is global.
- **Recompile rule:** per-layer scalars/values that differ across layers must be passed as **tensor arguments** to the compiled block, never captured as Python floats (see the `layer_scalar` note in `hf_gemma4.py`).
- **head_dim:** both `head_dim` (256) and `global_head_dim` (512) already satisfy `head_dim/2 >= 64`; no head padding needed.
- **On-device gates:** Tasks 1–2 run on the Spyre card directly reachable on this host (no pod indirection). They are compile/numeric gates, not unit tests. Do not proceed past a failed gate — revisit the formulation instead of shipping wrong output.
- **Verified op support (spec §2.1):** on-device, only plain gather (`x[idx]`) and `bmm` lower and are numerically correct. `topk` (SIGABRTs even at k=4), `argsort`/`sort` (`aten::sort.values_stable` not implemented), 1-D index arithmetic on the permutation, and `index_add`/`scatter_add` do **not** lower. Therefore all routing/permute/combine math runs **host-side** in eager PyTorch (Tasks 4/5 functions are host code); only the router matmul, token gather, and expert GEMM run on-device. Do not put `topk`/`argsort`/`index_add` inside a `torch.compile` region targeting spyre.
- **Indirect-access layout:** every tensor read via the on-device gather must have its indexed (row/expert) dimension **outermost** in the on-device layout, feature dim innermost as the stick dim (enforced by `enforce_indirect_access_layout`; an inserted `spyre.restickify` on a `[T*K, H]` tensor is a full HBM round-trip). Lay buffers out to satisfy this by construction.
- **`top_k_experts` = 4 (bring-up):** the router uses K=4 (host-side); the model config wants 8. Revisit when torch-spyre lifts the topk/argsort/index_add limits (spec §2.1, §9).
- **Teardown crash:** a `corrupted double-linked list` SIGABRT on process exit after a successful compute is a known torch-spyre lifetime issue — ignore it; it does not indicate a compute failure.

**Reference spec:** `docs/superpowers/specs/2026-07-31-gemma4-moe-adapter-design.md` (read §1.2, §3, §3.5, §4 before starting).

---

**Note on code snippets:** the Apache license header is elided from the snippets below to keep them short. Per Global Constraints, every *new* file (`hf_gemma4_moe.py`, all `repros/` and `tests/` files) MUST begin with the 14-line header copied from an existing `hf_adapters/*.py`.

## File Structure

- **Create** `repros/gemma4_moe/` — throwaway on-device prototype scripts for Tasks 1–2 (not shipped; kept in the branch for reproducibility).
- **Modify** `hf_adapters/hf_gemma4.py` — extract the attention half of `block_forward` into a shared helper `_gemma4_attention(...)` (behavior-preserving refactor) so both the dense and MoE blocks call it.
- **Create** `hf_adapters/hf_gemma4_moe.py` — the MoE adapter: router, permute, grouped GEMM, unpermute/combine, parallel dense+MoE block, `prepare_for_spyre`.
- **Modify** `hf_adapters/auto_spyre_model.py` — branch `Gemma4Config`/`Gemma4TextConfig` dispatch on `enable_moe_block` to route MoE checkpoints to `hf_gemma4_moe`.
- **Modify** `tests/model_registry.py` — add the `gemma4_moe` entry (+ non-blocking xfail while bringing up).
- **Create** `tests/test_gemma4_moe_ffn.py` — CPU unit tests for the MoE FFN vs a plain reference (routing, permute round-trip, full FFN parity).
- **Modify** `ARCHITECTURE.md` + `README.md` — coverage tables + badge (final task).

---

### Task 1: On-device gate — expert-dim grouped/batched matmul compiles and is correct

**Files:**
- Create: `repros/gemma4_moe/gate1_grouped_gemm.py`

**Interfaces:**
- Produces: confirmation that a `[T*K, 1, H] × [T*K, H, F]` row-batched matmul (Option 4A shape) and a `[E, T, H] × [E, H, F]` expert-batched matmul (dense reference shape) both compile on Spyre and match CPU, without the `out_reuse_dim.size()==1` abort.

- [ ] **Step 1: Write the compile+compare repro**

```python
# repros/gemma4_moe/gate1_grouped_gemm.py — 14-line Apache header omitted here for brevity
import torch

H, F, E, T, K = 2816, 704, 8, 32, 8  # small E/T to iterate fast; F=moe_intermediate_size

def row_batched(a, w):           # 4A geometry: one weight per row
    return torch.bmm(a, w)       # [N,1,H] x [N,H,F] -> [N,1,F]

def expert_batched(a, w):        # dense geometry: all experts
    return torch.bmm(a, w)       # [E,T,H] x [E,H,F] -> [E,T,F]

def check(fn, a_shape, w_shape):
    a = torch.randn(*a_shape, dtype=torch.float16)
    w = torch.randn(*w_shape, dtype=torch.float16)
    ref = fn(a, w)
    cfn = torch.compile(fn, dynamic=False)
    got = cfn(a.to("spyre"), w.to("spyre")).cpu()
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
    print(f"OK {a_shape} x {w_shape}")

if __name__ == "__main__":
    check(row_batched, (T * K, 1, H), (T * K, H, F))
    check(expert_batched, (E, T, H), (E, H, F))
```

- [ ] **Step 2: Run on the Spyre pod**

Run: `python3 repros/gemma4_moe/gate1_grouped_gemm.py`
Expected: both print `OK ...`. If it aborts with `out_reuse_dim.size()==1` or a layout error, STOP: capture the failing kernel and revisit §4 (try `spyre_hint(tiles={...})` on the expert/row dim, or fall back to the dense-masked expert compute for this step). Record the outcome in the repro file's docstring.

- [ ] **Step 3: Commit the gate result**

```bash
git add repros/gemma4_moe/gate1_grouped_gemm.py
git commit -m "test: gate 1 — expert-dim grouped/batched matmul on Spyre"
```

---

### Task 2: On-device gate — device/host-split route → permute → GEMM → combine round-trip

**Files:**
- Create: `repros/gemma4_moe/gate2_route_permute.py`

**Interfaces:**
- Consumes: nothing.
- Produces: confirmation that the §2.1 device/host split composes end-to-end (K=4): a device `torch.compile` region for the router `Linear`; host `softmax`/`topk`/renormalize + `argsort` + `token_of_row`; a device `torch.compile` region for the `x[token_of_row]` gather + expert GEMM; host weighted `index_add` combine — and that the recombined output matches a pure-CPU MoE reference within fp16 tolerance, with the device gather's `[T*K, H]` buffer committing row-dim-outermost (no surprise `spyre.restickify`).

**Why this shape:** the routing ops (`topk`, `argsort`, index-arith, `index_add`) do **not** lower on the current backend (Global Constraints / spec §2.1). This gate proves the host/device split is a working substitute: the unsupported ops run in eager CPU on the small routing tensors, and only the matmul + gather cross to the device.

- [ ] **Step 1: Write the split round-trip repro**

```python
# repros/gemma4_moe/gate2_route_permute.py — Apache header omitted here
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch_spyre
torch_spyre._autoload()
import torch
import torch.nn.functional as F

H, E, T, K, M = 2816, 128, 64, 4, 704  # K=4 bring-up; M=moe_intermediate_size

# --- device regions (compiled, spyre) ---
def device_router(x, W_router):          # [T,H] x [E,H] -> [T,E]
    return F.linear(x, W_router)

def device_expert(gathered, token_of_row, gate_up_row, down_row):
    # gathered token rows + per-row expert weights already selected on host side
    g_u = torch.bmm(gathered.unsqueeze(1), gate_up_row.transpose(1, 2)).squeeze(1)  # [N,2M]
    g, u = g_u.chunk(2, dim=-1)
    act = F.gelu(g, approximate="tanh") * u                                         # [N,M]
    return torch.bmm(act.unsqueeze(1), down_row.transpose(1, 2)).squeeze(1)         # [N,H]

def device_gather(x, token_of_row):      # the indirect-access op under test
    return x[token_of_row]               # [T*K,H]

# --- pure-CPU reference (dense: compute all experts, select top-K) ---
def ref_moe(x, W_router, gate_up, down, scale, K):
    probs = torch.softmax(F.linear(x, W_router).float(), dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx]
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(K):
            e = idx[t, k].item()
            g, u = F.linear(x[t].float(), gate_up[e].float()).chunk(2, dim=-1)
            h = F.linear(F.gelu(g, approximate="tanh") * u, down[e].float())
            out[t] += w[t, k] * h
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(T, H, dtype=torch.float16)
    W_router = torch.randn(E, H, dtype=torch.float16)
    gate_up = torch.randn(E, 2 * M, H, dtype=torch.float16) * 0.02
    down = torch.randn(E, H, M, dtype=torch.float16) * 0.02
    scale = (torch.rand(E) + 0.5).half()

    ref = ref_moe(x, W_router, gate_up, down, scale, K)

    crouter = torch.compile(device_router, dynamic=False)
    cexpert = torch.compile(device_expert, dynamic=False)
    cgather = torch.compile(device_gather, dynamic=False)

    # 1) device: router logits
    logits = crouter(x.to("spyre"), W_router.to("spyre")).cpu().float()   # -> host
    # 2) host: softmax/topk/renorm/scale/argsort/token_of_row (unsupported on device)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx].float()
    flat_expert = idx.reshape(-1)                                          # [T*K]
    sort_perm = torch.argsort(flat_expert)
    row_expert = flat_expert[sort_perm]
    token_of_row = (torch.arange(T * K) // K)[sort_perm].to(torch.int32)
    # 3) device: gather token rows
    gathered = cgather(x.to("spyre"), token_of_row.to("spyre")).cpu()      # [T*K,H]
    # host: select per-row expert weights (gather on expert dim — done on host here
    # to keep the gate focused on the token gather + GEMM; Task 5 does it on device)
    gate_up_row = gate_up[row_expert]                                      # [N,2M,H]
    down_row = down[row_expert]                                            # [N,H,M]
    # 4) device: expert GEMM
    expert_out = cexpert(gathered.to("spyre"), token_of_row.to("spyre"),
                         gate_up_row.to("spyre"), down_row.to("spyre")).cpu().float()
    # 5) host: weighted index_add combine
    row_w = w.reshape(-1)[sort_perm].unsqueeze(-1).float()
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row.long(), expert_out * row_w)

    denom = ref.abs().clamp_min(1.0)
    rel = (out - ref).abs() / denom
    print(f"mean_rel={rel.mean():.4%} max_rel={rel.max():.4f}")
    assert rel.mean() < 0.02 and rel.max() < 0.5, "split MoE round-trip diverged"
    print("OK device/host-split route/permute/GEMM/combine round-trip")
```

- [ ] **Step 2: Run on the card with compile artifacts enabled**

Run: `TORCH_COMPILE_DEBUG=1 python3 repros/gemma4_moe/gate2_route_permute.py`
Expected: prints `mean_rel=...` then `OK ...`. A `corrupted double-linked list` SIGABRT *after* the `OK` line is the known teardown issue — ignore it (Global Constraints). Then inspect the dumped artifacts / restickify insertions to confirm the `device_gather` `[T*K, H]` buffer has the row dim outermost with **no** inserted `spyre.restickify` on the hot path (§3.5). Record findings in the docstring. If any device region fails to lower or the round-trip diverges, STOP and escalate.

- [ ] **Step 3: Commit**

```bash
git add repros/gemma4_moe/gate2_route_permute.py
git commit -m "test: gate 2 — device/host-split route/permute/GEMM/combine on Spyre"
```

---

### Task 3: Refactor — extract shared Gemma 4 attention helper

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (the attention region of `_make_compiled_block`'s `block_forward`, lines ~203–253)
- Test: existing `tests/spyre/test_e2e_token_compare_spyre.py -k gemma4` (regression — must still pass unchanged)

**Interfaces:**
- Produces: `_gemma4_attention(hidden_states, *, input_ln, post_attn_ln, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, v_norm, scaling, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, selected_freqs, attn_mask, key_cache, value_cache, is_filling, token_index, cache_position) -> (h, key_cache, value_cache)` where `h = residual + post_attn_ln(o_proj(attn_out))`. This is the attention half of the current dense block up to and including the sandwich post-attention residual add.

- [ ] **Step 1: Extract the helper (pure move, no behavior change)**

Cut the attention body from `block_forward` in `_make_compiled_block` (the region from `residual = hidden_states` through `h = residual + attn_out` after `post_attn_ln`) into a module-level `_gemma4_attention(...)` that takes the captured modules as keyword args. Have the dense `block_forward` call it, then continue with its existing FFN sandwich (`pre_ff_ln → mlp → post_ff_ln → residual add → * layer_scalar`).

- [ ] **Step 2: Run the dense regression on CPU-side sanity + module test**

Run: `bash tests/run_oot_module_configs.sh tests/configs/module_tests/ -v -k gemma4` (if a gemma4 module config exists) or the smoke import: `python3 -c "import hf_adapters.hf_gemma4"`
Expected: import clean; no signature errors.

- [ ] **Step 3: Run the dense Spyre token-compare to prove no regression**

Run (pod): `pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k gemma4_google`
Expected: same PASS/xfail status as before the refactor (the dense adapter output is unchanged).

- [ ] **Step 4: Commit**

```bash
git add hf_adapters/hf_gemma4.py
git commit -m "refactor: extract _gemma4_attention shared helper (no behavior change)"
```

---

### Task 4: MoE router + permute (CPU unit test first)

**Files:**
- Create: `hf_adapters/hf_gemma4_moe.py` (router + permute functions)
- Test: `tests/test_gemma4_moe_ffn.py`

**Interfaces:**
- Consumes: nothing from prior tasks (pure functions).
- Produces:
  - `_moe_route(x, W_router, per_expert_scale, K) -> (w, idx)` : `x [T,H]`, returns router weights `w [T,K]` (softmax→topk→renormalize→×per_expert_scale[idx]) and expert ids `idx [T,K]`.
  - `_moe_permute(x, idx, K) -> (gathered, token_of_row, row_expert, sort_perm)` : sorts the `T*K` pairs by expert, returns `gathered [T*K,H]`, `token_of_row [T*K]`, `row_expert [T*K]`, `sort_perm [T*K]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gemma4_moe_ffn.py — Apache header omitted here
import torch
from hf_adapters.hf_gemma4_moe import _moe_route, _moe_permute

def test_route_shapes_and_renorm():
    T, H, E, K = 4, 16, 8, 2
    x = torch.randn(T, H)
    W = torch.randn(E, H)
    scale = torch.ones(E)
    w, idx = _moe_route(x, W, scale, K)
    assert w.shape == (T, K) and idx.shape == (T, K)
    # with per_expert_scale == 1, weights renormalize to sum 1 per token
    torch.testing.assert_close(w.sum(-1), torch.ones(T), atol=1e-5, rtol=1e-5)

def test_permute_roundtrip():
    T, H, E, K = 4, 16, 8, 2
    x = torch.randn(T, H)
    idx = torch.tensor([[0, 3], [3, 1], [7, 0], [2, 2]])
    gathered, token_of_row, row_expert, sort_perm = _moe_permute(x, idx, K)
    assert gathered.shape == (T * K, H)
    # rows are sorted by expert id
    assert torch.equal(row_expert, torch.sort(idx.reshape(-1)).values)
    # gathered row r is the source token for that pair
    torch.testing.assert_close(gathered, x[token_of_row])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_gemma4_moe_ffn.py -v`
Expected: FAIL (`ImportError` / functions not defined).

- [ ] **Step 3: Implement `_moe_route` and `_moe_permute`**

```python
import torch
import torch.nn.functional as F

def _moe_route(x, W_router, per_expert_scale, K):
    logits = F.linear(x, W_router)                     # [T,E]
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)              # [T,K],[T,K]
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[idx]
    return w, idx

def _moe_permute(x, idx, K):
    flat_expert = idx.reshape(-1)                      # [T*K]
    sort_perm = torch.argsort(flat_expert)             # [T*K]
    row_expert = flat_expert[sort_perm]                # [T*K] expert id per sorted row
    token_of_row = (torch.arange(idx.shape[0] * K, device=x.device) // K)[sort_perm]
    gathered = x[token_of_row]                         # [T*K,H]
    return gathered, token_of_row, row_expert, sort_perm
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_gemma4_moe_ffn.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py tests/test_gemma4_moe_ffn.py
git commit -m "feat: gemma4 MoE router + expert permutation (CPU-tested)"
```

---

### Task 5: MoE grouped GEMM + unpermute/combine — full FFN parity (CPU)

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`
- Test: `tests/test_gemma4_moe_ffn.py`

**Interfaces:**
- Consumes: `_moe_route`, `_moe_permute` from Task 4.
- Produces:
  - `_grouped_gemm(gathered, Wstack, row_expert) -> out` : for each row `r`, `gathered[r] @ Wstack[row_expert[r]].T`. Option 4A implementation (gather the per-row weight, row-batched bmm). `Wstack [E, out, in]`, `gathered [N,in]`, returns `[N,out]`. This is the piece that runs **on-device** in Task 6 (gather + bmm are the two ops that lower); on CPU here it is plain eager.
  - `_moe_ffn(x, W_router, gate_up_proj, down_proj, per_expert_scale, K) -> [T,H]` : the full §3 forward (route → permute → grouped gate_up → gelu_tanh SwiGLU → grouped down → weight by `w` → scatter_add combine). Pure CPU/eager reference here; **Task 6 splits it** so route/permute/combine stay host-eager and the gather+GEMM run in a compiled device region. Keep it as a single readable eager function so it doubles as the host reference for the §6 gate-4 parity.

- [ ] **Step 1: Write failing parity test vs a plain reference**

```python
# add to tests/test_gemma4_moe_ffn.py
import torch.nn.functional as F
from hf_adapters.hf_gemma4_moe import _moe_ffn

def _ref_moe(x, W_router, gate_up, down, scale, K):
    # dense reference: compute all experts, select top-K, weighted sum
    T, H = x.shape; E = W_router.shape[0]
    probs = torch.softmax(F.linear(x, W_router), dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx]
    out = torch.zeros_like(x)
    for t in range(T):
        for k in range(K):
            e = idx[t, k].item()
            g, u = F.linear(x[t], gate_up[e]).chunk(2, dim=-1)
            h = F.linear(F.gelu(g, approximate="tanh") * u, down[e])
            out[t] += w[t, k] * h
    return out

def test_moe_ffn_matches_reference():
    T, H, E, K, M = 4, 16, 8, 2, 5
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    gate_up = torch.randn(E, 2 * M, H)
    down = torch.randn(E, H, M)
    scale = torch.rand(E) + 0.5
    ref = _ref_moe(x, W_router, gate_up, down, scale, K)
    got = _moe_ffn(x, W_router, gate_up, down, scale, K)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_gemma4_moe_ffn.py::test_moe_ffn_matches_reference -v`
Expected: FAIL (`_moe_ffn` not defined).

- [ ] **Step 3: Implement `_grouped_gemm` (4A) and `_moe_ffn`**

```python
def _grouped_gemm(gathered, Wstack, row_expert):
    # Option 4A: gather the per-row weight, then row-batched bmm.
    # W_row: [N, out, in]; gathered: [N, in]
    W_row = Wstack[row_expert]                              # index_select on expert dim (outermost)
    out = torch.bmm(gathered.unsqueeze(1), W_row.transpose(1, 2))  # [N,1,out]
    return out.squeeze(1)                                   # [N,out]

def _moe_ffn(x, W_router, gate_up_proj, down_proj, per_expert_scale, K):
    T, H = x.shape
    w, idx = _moe_route(x, W_router, per_expert_scale, K)
    gathered, token_of_row, row_expert, sort_perm = _moe_permute(x, idx, K)
    gate_up = _grouped_gemm(gathered, gate_up_proj, row_expert)  # [N,2M]
    g, u = gate_up.chunk(2, dim=-1)
    act = F.gelu(g, approximate="tanh") * u                      # [N,M]
    expert_out = _grouped_gemm(act, down_proj, row_expert)       # [N,H]
    expert_out = expert_out * w.reshape(-1)[sort_perm].unsqueeze(-1)
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out = out.index_add(0, token_of_row, expert_out)            # scatter_add combine
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_gemma4_moe_ffn.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py tests/test_gemma4_moe_ffn.py
git commit -m "feat: gemma4 MoE grouped GEMM + combine, parity vs reference (4A)"
```

---

### Task 6: Compiled MoE decoder block + `prepare_for_spyre`

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`
- Test: import + structural sanity (`python3 -c "import hf_adapters.hf_gemma4_moe"`)

**Interfaces:**
- Consumes: `_gemma4_attention` (Task 3), `_moe_route`/`_moe_permute` (Task 4), `_grouped_gemm` (Task 5), and from `hf_gemma4.py`: `_patch_gemma4_rmsnorm`, `_gemma4_backbone`, `_build_layer_masks`, RoPE/KV-shape setup.
- Produces:
  - `_compiled_moe_device(gathered, gate_up_row, down_row) -> [N,H]` — the **device** portion of the FFN: the two grouped GEMMs + gelu_tanh SwiGLU (from `_grouped_gemm`), wrapped in `torch.compile(dynamic=False)`. `gathered [N,H]`, `gate_up_row [N,2M,H]`, `down_row [N,H,M]`. (The token gather `x[token_of_row]` and per-row weight gather `Wstack[row_expert]` may live in this region too, or be done as separate compiled gathers — implementer's choice, but they must be on-device gather ops, not host indexing, on the hot path.)
  - `_moe_ffn_split(x_dev, W_router_dev, gate_up_dev, down_dev, per_expert_scale_host, K)` — host orchestrator implementing the §2.1 split: device router `Linear` → `.cpu()` → host softmax/`_moe_route`/`_moe_permute` (topk/argsort/index-arith) → move `token_of_row`/`row_expert` to device → `_compiled_moe_device` → `.cpu()` → host weighted `index_add` combine → return `[T,H]` (on the device, moved back for the block sum). Router weights, expert weights stay resident on-device across calls; only the small routing tensors and the `[T*K,H]` gathered/expert-out buffers cross the boundary.
  - `_make_moe_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v)` → a block callable (NOT a single `torch.compile` of the whole block — the FFN's routing is host-side). It runs the **compiled** attention (shared helper) and **compiled** dense MLP, calls `_moe_ffn_split` for the sparse branch, and combines per §1.2. Matches the dense block's call signature plus the MoE weights.
  - `prepare_for_spyre(model)` — asserts (no PLE, no KV-share, `enable_moe_block=True`, `top_k_experts` coerced/asserted to K=4 for bring-up); patches RMSNorm; builds per-type RoPE + KV shapes (reuse `hf_gemma4.prepare_text_decoder_for_spyre` logic); `pad_lm_head`; registers the 3D packed expert weights + router + `per_expert_scale` as buffers and moves them to `spyre` so they stay resident; lays expert weights out expert-dim-outermost (§3.5); compiles the attention/dense-MLP/device-FFN regions.
  - `_run_forward(...)` / `_run_backbone_forward(...)` — same signatures as `hf_gemma4._run_forward` (delegate to the shared `_run_blocks_over_embeds` machinery, swapping in the MoE blocks), plus the final logit softcap.

- [ ] **Step 1: Implement the block, the split FFN, `prepare_for_spyre`, and `_run_forward`**

Model the norm wiring on the spec §1.2 exactly: `h_dense = post_ff_ln_1(dense_mlp(pre_ff_ln(h)))`, `h_moe = post_ff_ln_2(_moe_ffn_split(pre_ff_ln_2(residual), ...))`, `h = post_ff_ln(h_dense + h_moe)`, `h = residual + h`, `h = h * layer_scalar`. Pass `layer_scalar` as a tensor arg (Global Constraints). Reuse `hf_gemma4.prepare_text_decoder_for_spyre` for the attention-side prep (RoPE, KV shapes, RMSNorm patch, pad_lm_head) and add the expert-weight registration on top. Flatten `[B,S,H]`→`[T,H]` for the FFN and reshape back inside the block. Do **not** place `topk`/`argsort`/`index_add` inside any `torch.compile(...)` targeting spyre (Global Constraints, spec §2.1) — those stay in eager host code inside `_moe_ffn_split`. The per-token norms (`pre_ff_ln_2`, `post_ff_ln_2`) apply on-device on the `[T,H]` tensor before/after the split, so the host only ever sees the small routing tensors plus the `[T*K,H]` gathered/expert-out buffers.

- [ ] **Step 2: Import + config-shape sanity (CPU, no weights)**

Run: `python3 -c "import hf_adapters.hf_gemma4_moe as m; print([n for n in dir(m) if not n.startswith('__')])"`
Expected: prints the public names; no import/syntax error.

- [ ] **Step 3: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -m "feat: gemma4 MoE compiled block + prepare_for_spyre + _run_forward"
```

---

### Task 7: Adapter dispatch + registry entry

**Files:**
- Modify: `hf_adapters/auto_spyre_model.py:194-219` (`resolve_adapter_module`) and the import block (~line 85) + `CONFIG_TO_ADAPTER_MODULE_MAPPING`
- Modify: `tests/model_registry.py` (add `gemma4_moe` + non-blocking xfail)
- Test: `tests/test_*` that exercises `resolve_adapter_module` (or a new focused unit test)

**Interfaces:**
- Consumes: `hf_gemma4_moe` module (Task 6).
- Produces: `resolve_adapter_module("google/gemma-4-26B-A4B-it")` returns `hf_gemma4_moe`; dense gemma4 paths still return `hf_gemma4`.

- [ ] **Step 1: Write a failing dispatch test**

```python
# tests/test_gemma4_moe_dispatch.py — Apache header omitted here
from types import SimpleNamespace
from hf_adapters import auto_spyre_model as asm

def test_moe_config_routes_to_moe_adapter(monkeypatch):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    cfg = Gemma4TextConfig(enable_moe_block=True, num_experts=8, top_k_experts=4,
                           moe_intermediate_size=8)
    cfg.architectures = ["Gemma4ForConditionalGeneration"]
    monkeypatch.setattr(asm.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: cfg))
    monkeypatch.setattr(asm, "assert_spyre_dimensions", lambda *a, **k: None)
    assert asm.resolve_adapter_module("dummy").__name__.endswith("hf_gemma4_moe")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_gemma4_moe_dispatch.py -v`
Expected: FAIL (routes to `hf_gemma4`, not `hf_gemma4_moe`).

- [ ] **Step 3: Add the `enable_moe_block` branch to `resolve_adapter_module`**

After the arch-name loop and before the config-class lookup, add:

```python
    # Gemma 4 shares one config class across dense and MoE checkpoints; the MoE
    # variant (enable_moe_block=True) needs the dedicated hf_gemma4_moe adapter.
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config, Gemma4TextConfig,
    )
    if isinstance(model_config, (Gemma4Config, Gemma4TextConfig)) or hasattr(
        model_config, "text_config"
    ):
        text_cfg = getattr(model_config, "text_config", model_config)
        if getattr(text_cfg, "enable_moe_block", False):
            assert_spyre_dimensions(model_config, model_name=str(model_name_or_path))
            from hf_adapters import hf_gemma4_moe
            return hf_gemma4_moe
```

(Import `hf_gemma4_moe` in the top import block alongside `hf_gemma4`.)

- [ ] **Step 4: Add the registry entry**

In `tests/model_registry.py`, after the `hf_gemma4` block:

```python
    # hf_gemma4_moe.py
    "gemma4_moe": {
        "name": "Gemma 4 26B-A4B (MoE)",
        "path": "google/gemma-4-26B-A4B-it",
        "adapter": "hf_gemma4_moe.py",
        "size": "26b",
    },
```

Add `"gemma4_moe"` to `NON_BLOCKING_CAUSAL_MODELS`'s path tuple while bringing up.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_gemma4_moe_dispatch.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hf_adapters/auto_spyre_model.py tests/model_registry.py tests/test_gemma4_moe_dispatch.py
git commit -m "feat: route gemma4 enable_moe_block checkpoints to hf_gemma4_moe + registry"
```

---

### Task 8: End-to-end on Spyre — single-layer numeric, then full model

**Files:**
- Create: `repros/gemma4_moe/gate4_single_layer.py` (per-layer compare on real weights)
- Test: `tests/spyre/test_e2e_token_compare_spyre.py -k gemma4_moe`, `tests/spyre/test_e2e_smoke_spyre.py -k gemma4_moe`

**Interfaces:**
- Consumes: the full adapter (Tasks 3–7).
- Produces: a green (or expected-xfail-cleared) token-compare for `gemma4_moe`.

- [ ] **Step 1: Single MoE layer parity on real weights (pod)**

Write `gate4_single_layer.py`: load layer 0 of the real 26B-A4B (weights at `/mnt/models/hf_cache/hub` per the test-host memory; use `HF_HUB_OFFLINE`), run the reference layer forward on CPU (before `prepare_for_spyre`), then the split MoE block (compiled attention/dense-MLP/device-FFN + host routing per Task 6) on Spyre, assert max-abs error within fp16 tol (match the tolerance used in `docs/spyre-numerical-findings.md`). Note the config's `top_k_experts` is coerced to K=4 for bring-up, so the CPU reference must also use K=4 for an apples-to-apples compare (the divergence from the true-8 model is expected and tracked, spec §9).

Run (pod): `python3 repros/gemma4_moe/gate4_single_layer.py`
Expected: prints PASS within tolerance. If the parallel-branch sum or the router renorm/`per_expert_scale` diverge, fix ordering per spec §1.2 / §7 before proceeding.

- [ ] **Step 2: Full-model token compare (pod)**

Run: `pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k gemma4_moe`
Expected: logits/token match within tolerance (entry is non-blocking xfail during bring-up, so a failure is visible but non-gating — read the report, don't assume green).

- [ ] **Step 3: Smoke + load**

Run: `pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k gemma4_moe` and `pytest -s -vvv tests/spyre/test_load_spyre.py -k gemma4_moe`
Expected: no crash/NaN.

- [ ] **Step 4: Commit the gate script + any fixes**

```bash
git add repros/gemma4_moe/gate4_single_layer.py hf_adapters/hf_gemma4_moe.py
git commit -m "test: gemma4 MoE single-layer + e2e numeric gates on Spyre"
```

---

### Task 9: (Perf follow-up) Move the grouped GEMM to the hinted contiguous path (Option 4B)

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` (`_grouped_gemm`)
- Test: `tests/test_gemma4_moe_ffn.py` (unchanged — parity must still hold), plus a perf check on the pod

**Interfaces:**
- Consumes: everything from Tasks 4–8.
- Produces: a `_grouped_gemm` variant that keeps `gathered` contiguous and tiles the row dimension with `spyre_hint(tiles={...})` so each tile loads a single expert's weight slab (via `group_off`) — one weight load per tile instead of per row. Same numeric result as 4A.

- [ ] **Step 1: Implement the hinted contiguous grouped GEMM**

Replace the per-row weight gather with a tiled schedule: compute `group_off = cumsum(bincount(row_expert, E))`, and within a `spyre_hint(tiles={"row": tile})` scope select the expert weight for each tile from `group_off`. Keep the 4A implementation available behind a flag so a compile regression can fall back.

- [ ] **Step 2: Re-run CPU parity (must be unchanged)**

Run: `pytest tests/test_gemma4_moe_ffn.py -v`
Expected: PASS (identical numerics to 4A).

- [ ] **Step 3: Re-run e2e + compare compile artifacts / timing (pod)**

Run: `pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k gemma4_moe`
Expected: still within tolerance; verify (via profiler / artifacts) that expert-weight loads dropped from per-row to per-tile. If 4B fails to compile or regresses numerics, keep 4A and file a follow-up.

- [ ] **Step 4: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -m "perf: gemma4 MoE grouped GEMM via spyre_hint contiguous tiling (4B)"
```

---

### Task 10: Documentation — coverage tables + badge

**Files:**
- Modify: `ARCHITECTURE.md` (Verified Checkpoints + Model Family Coverage tables), `README.md` (badge counts)

**Interfaces:** none.

- [ ] **Step 1: Update the tables**

Add `hf_gemma4_moe.py` / `google/gemma-4-26B-A4B-it` to the coverage tables in `ARCHITECTURE.md` (mark verified only if Task 8 token-compare passed), and bump the README model-count badge.

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md README.md
git commit -m "docs: add Gemma 4 MoE (26B-A4B) to coverage tables + badge"
```

---

## Self-Review Notes

- **Device/host split (spec §2.1):** the routing ops don't lower, so the FFN is partitioned — device does router `Linear` + token gather + expert GEMM, host does softmax/topk/argsort/index-arith/index_add. K pinned to 4 for bring-up. Reflected in Global Constraints, Task 2 (split gate), Task 6 (`_moe_ffn_split`, non-monolithic block), Task 7 (K=4), Task 8 (K=4 reference).
- **Spec coverage:** §1.1 config asserts → Task 6 asserts. §1.2 parallel dense+MoE + norms → Task 6 block. §2 formulation (no upstream loop) → Tasks 4–5. §2.1 device/host split → Global Constraints + Task 2 gate + Task 6 split. §3 routing/permute/grouped-gemm/combine → Tasks 4–5 (host math) + Task 6 (device gather+GEMM). §3.5 layout → Global Constraints + Tasks 2, 6. §4 4A vs 4B → Task 5 (4A) + Task 9 (4B). §6 gates → Tasks 1 (done), 2 (split), 8. §7 numeric plan → Task 8. §8 DoD → Tasks 6–10. §9 open items (K→8, on-device routing, round-trip cost) → tracked follow-ups.
- **Type consistency:** `_moe_route`/`_moe_permute`/`_grouped_gemm`/`_moe_ffn` signatures are defined in Task 4/5 and consumed in Task 6 (`_moe_ffn` becomes the CPU reference; `_moe_ffn_split` is the on-device orchestrator built from the same `_moe_route`/`_moe_permute`/`_grouped_gemm` pieces). `_gemma4_attention` defined in Task 3, consumed in Task 6.
- **Ordering:** Tasks 1–2 are on-device gates first (de-risk). Task 3 refactor is independent and could run anytime before Task 6. Tasks 4–5 are CPU-only TDD (also the host reference). Task 6 assembles the split. Task 7 wires dispatch. Task 8 validates on-device. Tasks 9–10 are follow-ups.
