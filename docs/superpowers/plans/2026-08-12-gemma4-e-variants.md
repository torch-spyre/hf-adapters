# Gemma 4 E-variant (E2B / E4B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Spyre adapter support for the small dense Gemma 4 E-variants (E2B, E4B) by extending `hf_adapters/hf_gemma4.py` in place with Per-Layer Embeddings (PLE) and KV-sharing.

**Architecture:** Both features are config-gated so the 12B/31B path stays byte-identical when the new config fields are zero. PLE is computed once per forward at the model level and injected per layer via a tail block (Linear→GELU→×input→Linear→RMSNorm→residual). KV-sharing is mapped onto the adapter's static per-layer caches by aliasing the producer layer's cache buffers into a second, leaner compiled block used by shared layers.

**Tech Stack:** PyTorch, `torch.compile` (dynamic=False), HuggingFace Transformers 5.12.1, torch-spyre. Tests via pytest (CPU lane + Spyre lane), parametrized off `tests/model_registry.py`.

## Global Constraints

- Copyright header: every new/modified file keeps the existing `# Copyright 2025 The Torch-Spyre Authors.` Apache-2.0 header block (already present in `hf_gemma4.py`).
- No new model classes / no forks — runtime monkey-patch + compiled-block pattern only (repo doctrine, `CLAUDE.md`).
- `head_dim / 2 >= 64` (one Spyre stick) — E-variants have head_dim=256 (÷2=128), already aligned; keep the existing assert.
- Run HF reference forward **before** `prepare_for_spyre()` — the RMSNorm patch mutates the class globally (`CLAUDE.md`, "Test ordering matters"). The CPU accuracy harness already does this (loads a fresh ref model in phase 2).
- Per-layer tensors that vary across layers (`layer_scalar`, `per_layer_input`) are passed as **tensor arguments** read fresh from device buffers, never captured as Python constants — captured varying constants cause per-layer Dynamo recompiles that cross the accumulated-recompile limit (existing `_make_compiled_block` note, `hf_gemma4.py:171-190`).
- MoE stays out of scope: keep `assert not enable_moe_block` in `prepare_text_decoder_for_spyre`.
- Python-only `_inductor`/adapter changes need no torch-spyre rebuild; these are all Python adapter changes.
- Config values (verified, transformers 5.12.1 + cached checkpoints): E2B — 35 layers, hidden 1536, head_dim 256/global 512, 8 q / 1 kv heads, PLE dim 256, num_kv_shared_layers 20, use_double_wide_mlp True, softcap 30.0; E4B — 42 layers, hidden 2560, 8 q / 2 kv, PLE dim 256, num_kv_shared_layers 18, double_wide False. `first_kv_shared_layer_idx` = 15 (E2B) / 24 (E4B). Both have full+sliding producers before the boundary.

---

## Task ordering / registry wrinkle (read before starting)

`CAUSAL_PATHS` in `tests/model_registry.py` selects **one representative model per adapter — the smallest by `size`** (`_select_representative_paths`, `model_registry.py:480-499`). E2B (`size="2b"`) is smaller than the current 12B representative, so **adding E2B makes it the auto-selected `hf_gemma4` representative** for the CPU accuracy harness and the Spyre smoke harness, displacing 12B from those parametrized lists. This is intended (smaller = faster, still exercises the shared code + the new features), but it means:
- After Task 5, the CPU accuracy test and Spyre smoke test will run **E2B** for the `hf_gemma4` slot, not 12B.
- The 12B/31B regression check (Task 6) is therefore run **explicitly by key**, not via the auto-selected list.

---

## Task 1: PLE model-level compute (setup + combined per-layer-inputs)

Add the once-per-forward PLE computation and gate it on config. Producing the combined `[B, S, num_layers, ple_dim]` tensor is independently testable against stock HF's `project_per_layer_inputs`.

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (add `_compute_per_layer_inputs`; extend `prepare_text_decoder_for_spyre`)
- Test: `tests/cpu/test_gemma4_ple_unit.py` (new)

**Interfaces:**
- Consumes: model loaded via `AutoModelForCausalLM` with a `gemma4_text` E-variant config; `text_config`, `_gemma4_backbone` (existing in `hf_gemma4.py`).
- Produces:
  - `_compute_per_layer_inputs(model, inputs_embeds, input_ids) -> torch.Tensor | None`
    returns `[B, S, num_hidden_layers, hidden_size_per_layer_input]` when PLE is present, else `None`.
  - `model._spyre_has_ple: bool` set in `prepare_text_decoder_for_spyre`.

- [ ] **Step 1: Write the failing test**

```python
# tests/cpu/test_gemma4_ple_unit.py
import torch
from transformers import AutoModelForCausalLM
from hf_adapters import hf_gemma4

E2B = "google/gemma-4-E2B-it"


def test_compute_per_layer_inputs_matches_hf():
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    input_ids = torch.tensor([[2, 100, 200, 300, 400]])

    # Stock HF reference (both PLE components combined).
    inputs_embeds = backbone.embed_tokens(input_ids)
    ref_token = backbone.get_per_layer_inputs(input_ids, inputs_embeds)
    ref = backbone.project_per_layer_inputs(inputs_embeds, ref_token)

    got = hf_gemma4._compute_per_layer_inputs(model, inputs_embeds, input_ids)

    assert got is not None
    assert got.shape == ref.shape  # [1, 5, 35, 256]
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_gemma4_ple_unit.py::test_compute_per_layer_inputs_matches_hf -v`
Expected: FAIL with `AttributeError: module 'hf_adapters.hf_gemma4' has no attribute '_compute_per_layer_inputs'`

- [ ] **Step 3: Write minimal implementation**

Add to `hf_adapters/hf_gemma4.py` (near the other helpers):

```python
def _compute_per_layer_inputs(model, inputs_embeds, input_ids):
    """Compute the combined Per-Layer Embeddings tensor, or None if the model
    has no PLE. Mirrors stock Gemma4TextModel.get_per_layer_inputs +
    project_per_layer_inputs (transformers modeling_gemma4).

    Returns [B, S, num_hidden_layers, hidden_size_per_layer_input].
    """
    backbone = _gemma4_backbone(model)
    if not getattr(backbone, "hidden_size_per_layer_input", 0):
        return None

    cfg = text_config(model.config)
    ple_dim = cfg.hidden_size_per_layer_input
    n_layers = cfg.num_hidden_layers

    # Token-identity component (scaled embedding already applies sqrt(ple_dim)).
    token_identity = backbone.embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape, n_layers, ple_dim
    )
    # Context component: project the main embeds, scale, reshape, RMSNorm.
    context = backbone.per_layer_model_projection(inputs_embeds)
    context = context * backbone.per_layer_model_projection_scale
    context = context.reshape(*inputs_embeds.shape[:-1], n_layers, ple_dim)
    context = backbone.per_layer_projection_norm(context)

    return (context + token_identity) * backbone.per_layer_input_scale
```

Then in `prepare_text_decoder_for_spyre`, replace the PLE assert with a flag (leave KV-share/MoE asserts for now — later tasks handle KV-share):

```python
    # (was: assert not hidden_size_per_layer_input ...)
    model._spyre_has_ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_gemma4_ple_unit.py::test_compute_per_layer_inputs_matches_hf -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4.py tests/cpu/test_gemma4_ple_unit.py
git commit -m "feat(gemma4): compute combined per-layer-inputs (PLE) matching HF"
```

---

## Task 2: PLE injection tail inside the compiled block

Add the per-layer PLE residual block to `block_forward` and thread `per_layer_input` as a tensor arg. Gated so non-PLE models (12B/31B) are unaffected.

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (`_make_compiled_block`, its `block_forward`)
- Test: `tests/cpu/test_gemma4_ple_unit.py` (add a case)

**Interfaces:**
- Consumes: a decoder `layer` whose submodules include `per_layer_input_gate` (Linear hidden→ple_dim), `per_layer_projection` (Linear ple_dim→hidden), `post_per_layer_input_norm` (RMSNorm), and `layer.mlp.act_fn`-equivalent gelu-tanh. `_compute_per_layer_inputs` from Task 1.
- Produces: `_make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, has_ple)` — new trailing `has_ple: bool` param; `block_forward` gains a trailing `per_layer_input` tensor arg (may be a zero tensor when `has_ple` is False, but the tail only runs when `has_ple`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/cpu/test_gemma4_ple_unit.py
import torch.nn.functional as F

def test_block_ple_tail_matches_hf_layer():
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    layer = backbone.layers[0]  # non-shared layer (idx 0 < first_shared 15)
    hf_gemma4._patch_gemma4_rmsnorm(type(layer.input_layernorm))

    B, S, H = 1, 4, model.config.hidden_size
    input_ids = torch.tensor([[2, 10, 20, 30]])
    inputs_embeds = backbone.embed_tokens(input_ids)
    ple = hf_gemma4._compute_per_layer_inputs(model, inputs_embeds, input_ids)
    per_layer_input = ple[:, :, 0, :]  # layer 0 slice [B,S,ple_dim]

    # Reference: apply just the PLE tail as stock HF does (post-residual state h).
    h = torch.randn(B, S, H)
    ref = h + layer.post_per_layer_input_norm(
        layer.per_layer_projection(
            F.gelu(layer.per_layer_input_gate(h), approximate="tanh") * per_layer_input
        )
    )

    got = hf_gemma4._ple_tail(layer, h, per_layer_input)
    assert torch.allclose(got, ref, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_gemma4_ple_unit.py::test_block_ple_tail_matches_hf_layer -v`
Expected: FAIL with `AttributeError: ... has no attribute '_ple_tail'`

- [ ] **Step 3: Write minimal implementation**

Add a small helper and call it from `block_forward`. In `hf_adapters/hf_gemma4.py`:

```python
def _ple_tail(layer, h, per_layer_input):
    """Gemma 4 PLE per-layer residual injection (stock modeling_gemma4 tail)."""
    residual = h
    x = layer.per_layer_input_gate(h)
    x = F.gelu(x, approximate="tanh")
    x = x * per_layer_input
    x = layer.per_layer_projection(x)
    x = layer.post_per_layer_input_norm(x)
    return residual + x
```

Change `_make_compiled_block` signature to accept `has_ple` and capture the PLE submodules when present; add `per_layer_input` as the final `block_forward` arg; insert the tail after the post-FF residual and before `* layer_scalar`:

```python
def _make_compiled_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v, has_ple):
    ...
    def block_forward(hidden_states, selected_freqs, attn_mask, key_cache,
                      value_cache, is_filling, token_index, cache_position,
                      layer_scalar, per_layer_input):
        ...
        h = residual + h            # existing post-FF residual add
        if has_ple:
            h = _ple_tail(layer, h, per_layer_input)
        h = h * layer_scalar
        return h, key_cache, value_cache
    return torch.compile(block_forward, dynamic=False)
```

> Note: `_ple_tail` reads `layer.*` submodules via closure — fine, they are on the same device as the rest of the block's captured modules after the Spyre move.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_gemma4_ple_unit.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4.py tests/cpu/test_gemma4_ple_unit.py
git commit -m "feat(gemma4): PLE per-layer residual tail in compiled block"
```

---

## Task 3: KV-share producer map + shared block function

Add the shared-layer compiled block (no k/v path) and compute the producer-index map. This is the correctness core of KV-sharing.

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (add `_shared_producer_map`, `_make_shared_block`)
- Test: `tests/cpu/test_gemma4_kvshare_unit.py` (new)

**Interfaces:**
- Consumes: `text_config` (for `num_hidden_layers`, `num_kv_shared_layers`, `layer_types`); a shared decoder `layer` (has `self_attn.q_proj`, `q_norm`, `o_proj`; no `k_proj`/`v_proj`/`k_norm`/`v_norm`); `apply_rope_matmul`, `_ple_tail` (Task 2).
- Produces:
  - `_shared_producer_map(cfg) -> tuple[int, list[int | None]]` returns `(first_kv_shared_layer_idx, producer_of)` where `producer_of[i]` is the producer layer index for shared layer `i`, else `None` for non-shared layers.
  - `_make_shared_block(layer, num_q_heads, head_dim, has_ple)` → compiled `shared_block_forward(hidden_states, selected_freqs, attn_mask, key_cache, value_cache, layer_scalar, per_layer_input)` returning `hidden_states` only (caches are read-only, not returned).

- [ ] **Step 1: Write the failing test**

```python
# tests/cpu/test_gemma4_kvshare_unit.py
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from hf_adapters import hf_gemma4

E2B = "google/gemma-4-E2B-it"


def test_producer_map_e2b():
    cfg = AutoConfig.from_pretrained(E2B)
    tc = hf_gemma4.text_config(cfg)
    first, producer_of = hf_gemma4._shared_producer_map(tc)
    assert first == tc.num_hidden_layers - tc.num_kv_shared_layers  # 35-20=15
    # Non-shared layers map to None.
    assert all(producer_of[i] is None for i in range(first))
    # Each shared layer maps to a real earlier layer of the SAME type.
    for i in range(first, tc.num_hidden_layers):
        p = producer_of[i]
        assert p is not None and p < first
        assert tc.layer_types[p] == tc.layer_types[i]


def test_shared_block_matches_full_layer_output():
    # A shared block fed the producer's cache must equal a manual attention using
    # that same cache: verifies the lean path drops only k/v recompute, not math.
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    cfg = hf_gemma4.text_config(model.config)
    first, producer_of = hf_gemma4._shared_producer_map(cfg)
    shared_idx = first  # first shared layer
    layer = backbone.layers[shared_idx]
    hf_gemma4._patch_gemma4_rmsnorm(type(layer.input_layernorm))

    assert layer.self_attn.k_proj is None or not hasattr(layer.self_attn, "k_proj")

    block = hf_gemma4._make_shared_block(
        layer, cfg.num_attention_heads, cfg.head_dim, has_ple=False
    )
    # Uncompiled call for a pure-math check (unwrap torch.compile).
    fn = block.__wrapped__ if hasattr(block, "__wrapped__") else block

    B, S, H = 1, 3, cfg.hidden_size
    hd, nkv = cfg.head_dim, cfg.num_key_value_heads
    Lc = 8
    h = torch.randn(B, S, H)
    freqs = torch.randn(1, S, hd // 2, 2)  # matches apply_rope_matmul contract
    mask = torch.zeros(B, 1, S, Lc)
    kcache = torch.randn(B, nkv, Lc, hd)
    vcache = torch.randn(B, nkv, Lc, hd)
    scalar = layer.layer_scalar
    out = fn(h, freqs, mask, kcache, vcache, scalar, torch.zeros(B, S, cfg.hidden_size_per_layer_input))
    assert out.shape == (B, S, H)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_gemma4_kvshare_unit.py -v`
Expected: FAIL with `AttributeError: ... has no attribute '_shared_producer_map'`

- [ ] **Step 3: Write minimal implementation**

Add to `hf_adapters/hf_gemma4.py`:

```python
def _shared_producer_map(cfg):
    """For each layer, the producer layer index whose KV a shared layer reuses.

    Shared layers are the last num_kv_shared_layers layers; each reuses the KV
    of the nearest preceding NON-shared layer of the same layer_type (stock
    Gemma4 store_full_length_kv semantics). Non-shared layers map to None.
    """
    n = cfg.num_hidden_layers
    n_shared = getattr(cfg, "num_kv_shared_layers", 0)
    first = n - n_shared
    layer_types = cfg.layer_types
    producer_of = [None] * n
    if n_shared <= 0:
        return first, producer_of
    # Last non-shared layer index per type (only layers before `first` produce).
    last_by_type = {}
    for i in range(first):
        last_by_type[layer_types[i]] = i
    for i in range(first, n):
        p = last_by_type.get(layer_types[i])
        assert p is not None, (
            f"Gemma 4 KV-share: shared layer {i} (type {layer_types[i]}) has no "
            "same-type producer before the KV-share boundary."
        )
        producer_of[i] = p
    return first, producer_of


def _make_shared_block(layer, num_q_heads, head_dim, has_ple):
    """Compile a KV-sharing decoder layer: Q-only attention against a producer's
    cache. No k/v proj, no k_norm/v_norm, no RoPE-on-K, no cache update.
    """
    attn = layer.self_attn
    q_proj = attn.q_proj
    q_norm = attn.q_norm
    o_proj = attn.o_proj
    scaling = attn.scaling  # 1.0
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    mlp = layer.mlp

    def shared_block_forward(hidden_states, selected_freqs, attn_mask,
                             key_cache, value_cache, layer_scalar, per_layer_input):
        residual = hidden_states
        h = input_ln(hidden_states)
        bsz, seq_len, _ = h.shape
        q = q_proj(h).view(bsz, seq_len, num_q_heads, head_dim)
        q = q_norm(q).transpose(1, 2)
        q = apply_rope_matmul(q, selected_freqs)
        attn_out = F.scaled_dot_product_attention(
            q, key_cache, value_cache, attn_mask=attn_mask,
            dropout_p=0.0, scale=scaling, enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)
        attn_out = post_attn_ln(attn_out)
        h = residual + attn_out

        residual = h
        h = pre_ff_ln(h)
        h = mlp(h)
        h = post_ff_ln(h)
        h = residual + h
        if has_ple:
            h = _ple_tail(layer, h, per_layer_input)
        h = h * layer_scalar
        return h

    return torch.compile(shared_block_forward, dynamic=False)
```

> `torch.compile` returns an `OptimizedModule`/callable; the test unwraps via `.__wrapped__` when present, else calls directly (eager fallback on CPU is fine for the math check).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_gemma4_kvshare_unit.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4.py tests/cpu/test_gemma4_kvshare_unit.py
git commit -m "feat(gemma4): shared-layer block + KV-share producer map"
```

---

## Task 4: Wire prepare + driver for E-variants (integration)

Extend `prepare_text_decoder_for_spyre` to build shared blocks and the cache-index map, and extend `_run_blocks_over_embeds` to select block kind, pass the producer cache, and feed PLE per-layer slices. After this task an E-variant runs end-to-end on CPU.

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (`prepare_text_decoder_for_spyre`, `_run_blocks_over_embeds`, `_run_backbone_forward`)
- Test: `tests/cpu/test_gemma4_kvshare_unit.py` (add an end-to-end forward smoke)

**Interfaces:**
- Consumes: `_compute_per_layer_inputs` (T1), `_ple_tail` (T2), `_shared_producer_map` + `_make_shared_block` (T3), `allocate_kv_caches` (hf_common).
- Produces on `model`:
  - `model._spyre_compiled_blocks: list` — per layer, either a full block (returns 3-tuple) or a shared block (returns 1 tensor).
  - `model._spyre_producer_of: list[int | None]` — producer layer index per layer (None if non-shared).
  - `model._spyre_kv_shapes: list` — one entry **per layer** still (shared layers carry their producer's shape so `allocate_kv_caches` stays uniform-length), but shared layers' own caches are never written.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/cpu/test_gemma4_kvshare_unit.py
def test_e2b_forward_runs_and_is_finite():
    import hf_adapters.hf_gemma4 as g4
    from hf_adapters.hf_common import allocate_kv_caches
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    g4.prepare_for_spyre(model)  # DEVICE is patched to "cpu" by tests/conftest.py
    assert hasattr(model, "_spyre_producer_of")
    assert hasattr(model, "_spyre_has_ple") and model._spyre_has_ple

    input_ids = torch.tensor([[2, 10, 20, 30, 40]])
    S = input_ids.shape[1]
    Lc = S + 2
    kc, vc = allocate_kv_caches(model, 1, Lc, torch.float32, device="cpu")
    pos = torch.arange(S).unsqueeze(0)
    mask = torch.zeros(1, 1, S, Lc)
    for i in range(S):
        mask[:, :, i, i + 1:] = -torch.inf
    with torch.no_grad():
        logits = g4._run_forward(model, input_ids, pos, mask, kc, vc,
                                 is_filling=False, token_index=0, cache_position=0)
    assert logits.shape[1] == S
    assert torch.isfinite(logits[0, -1]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_gemma4_kvshare_unit.py::test_e2b_forward_runs_and_is_finite -v`
Expected: FAIL — `prepare_for_spyre` still asserts `num_kv_shared_layers`, or `_spyre_producer_of` missing.

- [ ] **Step 3: Write minimal implementation**

In `prepare_text_decoder_for_spyre` (`hf_gemma4.py`): remove the `num_kv_shared_layers` assert; keep the `enable_moe_block` assert; build the maps and both block kinds. Replace the block-build section:

```python
    model._spyre_has_ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
    first_shared, producer_of = _shared_producer_map(cfg)
    model._spyre_producer_of = producer_of

    # kv_shapes: keep one entry per layer (uniform-length list for
    # allocate_kv_caches). Shared layers carry their producer's shape; their own
    # buffer is allocated but never written (the driver aliases the producer's).
    kv_shapes = []
    is_kv_eq_v_per_layer = []
    for lt in cfg.layer_types:
        is_global = lt == "full_attention"
        use_kv_eq_v = attention_k_eq_v and is_global
        is_kv_eq_v_per_layer.append(use_kv_eq_v)
        if is_global:
            n_kv = num_global_kv_heads if use_kv_eq_v else num_kv_heads
            hd = global_head_dim
        else:
            n_kv = num_kv_heads
            hd = head_dim
        kv_shapes.append((n_kv, hd, hd))
    model._spyre_kv_shapes = kv_shapes

    pad_lm_head(model)

    blocks = []
    for i, layer in enumerate(backbone.layers):
        if producer_of[i] is None:
            blocks.append(_make_compiled_block(
                layer, num_q_heads, kv_shapes[i][0], kv_shapes[i][1],
                is_kv_eq_v_per_layer[i], model._spyre_has_ple))
        else:
            blocks.append(_make_shared_block(
                layer, num_q_heads, kv_shapes[i][1], model._spyre_has_ple))
    model._spyre_compiled_blocks = blocks
```

In `_run_blocks_over_embeds`, add `per_layer_inputs` handling and block-kind dispatch. Change its signature to accept `per_layer_inputs=None`, and the loop:

```python
    producer_of = model._spyre_producer_of
    zero_ple = None
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        if per_layer_inputs is not None:
            pli = per_layer_inputs[:, :, i, :]
        else:
            # non-PLE model: pass a correctly-shaped zero (tail is gated off, but
            # the arg must exist for the compiled signature).
            if zero_ple is None:
                zero_ple = h.new_zeros(h.shape[0], h.shape[1], 0)
            pli = zero_ple
        p = producer_of[i]
        if p is None:
            h, key_caches[i], value_caches[i] = compiled_block(
                h, freqs[lt], masks[lt], key_caches[i], value_caches[i],
                is_filling, token_index, cache_position,
                backbone_layers[i].layer_scalar, pli,
            )
        else:
            h = compiled_block(
                h, freqs[lt], masks[lt], key_caches[p], value_caches[p],
                backbone_layers[i].layer_scalar, pli,
            )
```

In `_run_backbone_forward`, compute PLE after embedding and pass it through:

```python
    h = backbone.embed_tokens(input_ids)
    per_layer_inputs = _compute_per_layer_inputs(model, h, input_ids)
    return _run_blocks_over_embeds(
        model, h, position_ids, attn_mask, key_caches, value_caches,
        is_filling, token_index, cache_position,
        per_layer_inputs=per_layer_inputs,
    )
```

> The zero-width `[B,S,0]` PLE placeholder for non-PLE models keeps the compiled signature uniform without allocating real data; the tail is gated off by `has_ple=False` so the placeholder is never read. (12B/31B already pass through this path — verify Task 6.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_gemma4_kvshare_unit.py::test_e2b_forward_runs_and_is_finite -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4.py tests/cpu/test_gemma4_kvshare_unit.py
git commit -m "feat(gemma4): wire PLE + KV-share into prepare and block driver"
```

---

## Task 5: Registry entries + CPU token-compare parity

Register E2B/E4B so the CPU accuracy harness and Spyre smoke harness pick them up, and confirm end-to-end greedy parity vs stock HF on CPU (E2B).

**Files:**
- Modify: `tests/model_registry.py` (add `gemma4_e2b`, `gemma4_e4b`)
- Test: reuse `tests/cpu/test_adapter_cpu_accuracy.py` (auto-parametrized)

**Interfaces:**
- Consumes: `CAUSAL_LM_MODELS` dict; `_select_representative_paths`, `NON_BLOCKING_CAUSAL_MODELS` (existing).
- Produces: registry keys `gemma4_e2b`, `gemma4_e4b` mapping to `hf_gemma4.py`.

- [ ] **Step 1: Add registry entries**

In `tests/model_registry.py`, in `CAUSAL_LM_MODELS`, after the `gemma4_31b` block:

```python
    "gemma4_e2b": {
        "name": "Gemma 4 E2B",
        "path": "google/gemma-4-E2B-it",
        "adapter": "hf_gemma4.py",
        "is_gated": True,
        "size": "2b",
    },
    "gemma4_e4b": {
        "name": "Gemma 4 E4B",
        "path": "google/gemma-4-E4B",
        "adapter": "hf_gemma4.py",
        "is_gated": True,
        "size": "4b",
    },
```

Add both keys to the `NON_BLOCKING_CAUSAL_MODELS` tuple (new models start non-blocking in CI until stably green):

```python
        "gemma4_google",
        "gemma4_base",
        "gemma4_e2b",
        "gemma4_e4b",
```

- [ ] **Step 2: Verify E2B is now the representative + run CPU parity**

Run: `python3 -c "import sys; sys.path.insert(0,'tests'); import model_registry as m; print([p for p in m.CAUSAL_PATHS if 'gemma-4' in p])"`
Expected: prints `['google/gemma-4-E2B-it']` (E2B, smallest, is the `hf_gemma4` representative — see the "registry wrinkle" note).

Run: `pytest -s tests/cpu/test_adapter_cpu_accuracy.py -k "E2B" -v`
Expected: PASS — adapter greedy tokens equal stock HF greedy tokens for E2B. (Requires E2B weights, cached locally per project setup; set `HF_HUB_OFFLINE=1` and `HF_HUB_CACHE` per the Spyre-test host env.)

- [ ] **Step 3: Commit**

```bash
git add tests/model_registry.py
git commit -m "test(gemma4): register E2B/E4B; CPU token-compare parity"
```

---

## Task 6: 12B/31B regression check

Confirm the config-gated paths left the dense variants byte-identical. No new code — a verification gate.

**Files:**
- Test: existing `tests/cpu/test_adapter_cpu_accuracy.py`, `tests/cpu/test_generate_cpu.py`

- [ ] **Step 1: Run the dense-variant CPU accuracy explicitly by key**

Run: `pytest -s tests/cpu/test_adapter_cpu_accuracy.py -k "gemma-4-12" -v`
Expected: PASS. (12B is no longer the auto-selected representative — invoke it by path substring so it still runs.)

> If 12B weights aren't resident, download or run against the cached snapshot under `/mnt/models/hf_cache/hub/models--google--gemma-4-12B-it`. If truly unavailable, run the fastest non-PLE gemma4 CPU test present and note the gap in the commit message.

- [ ] **Step 2: Run the full CPU gemma4 suite**

Run: `pytest -s tests/cpu/ -k "gemma4 or gemma-4" -v`
Expected: PASS (E2B parity + any 12B/VLM cases present).

- [ ] **Step 3: Commit (docs/notes only if anything changed)**

No code change expected. If a flake or a needed tweak surfaced, fix it in `hf_gemma4.py` and:

```bash
git add hf_adapters/hf_gemma4.py
git commit -m "fix(gemma4): preserve 12B/31B path under E-variant gating"
```

---

## Task 7: Spyre smoke + docs

Run E2B end-to-end on Spyre and update coverage docs. Requires a Spyre card (per `CLAUDE.md`, run on pod or `aviros-spyre-test`; single-tenant VFIO).

**Files:**
- Test: existing `tests/spyre/test_e2e_smoke_spyre.py`
- Modify: `ARCHITECTURE.md` (coverage tables), `README.md` (badge counts)

- [ ] **Step 1: Spyre smoke run (E2B)**

Run: `pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k "E2B"`
Expected: loads on Spyre, generates non-trivial tokens, no crash / NaN. (E2B is non-blocking xfail per Task 5, so a failure won't gate CI but the outcome is visible — investigate before removing the xfail.)

- [ ] **Step 2: Update ARCHITECTURE.md coverage tables**

Add Gemma 4 E2B and E4B rows to the "Verified Checkpoints" and "Model Family Coverage" tables in `ARCHITECTURE.md` (single source of truth per `CLAUDE.md`). Mark E2B "CPU parity + Spyre smoke", E4B "CPU parity" (or per actual result).

- [ ] **Step 3: Bump README badge counts**

Update the model-count badge(s) in `README.md` to include the two new checkpoints.

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md README.md
git commit -m "docs(gemma4): record E2B/E4B coverage; bump model counts"
```

---

## Self-Review

**Spec coverage:**
- PLE setup + compute → Task 1. PLE per-layer injection → Task 2. ✓
- KV-sharing producer map + shared block → Task 3; wiring/aliasing → Task 4. ✓
- Double-wide MLP → no task needed (auto-handled; both block fns call `mlp()` as-is). ✓
- MoE assert retained → Task 4 (kept in prepare). ✓
- Config-gated, 12B/31B unchanged → Task 4 zero-width placeholder + Task 6 regression. ✓
- Registry + CPU parity → Task 5. Spyre smoke → Task 7. Docs/DoD → Task 7. ✓
- Producer-availability assert → Task 3 (`_shared_producer_map`). ✓

**Placeholder scan:** No TBD/TODO. Task 6 Step 1 has a conditional fallback for missing 12B weights (states exactly what to run and to note the gap) — acceptable, not a placeholder. All code steps carry real code.

**Type consistency:**
- `_compute_per_layer_inputs(model, inputs_embeds, input_ids)` — defined T1, called T4. ✓
- `_ple_tail(layer, h, per_layer_input)` — defined T2, called T3 shared block + T2 full block. ✓
- `_make_compiled_block(..., has_ple)` — new trailing arg T2, called T4. ✓
- `_shared_producer_map(cfg) -> (first, producer_of)` — T3, called T4. ✓
- `_make_shared_block(layer, num_q_heads, head_dim, has_ple)` → block returns **1 tensor**; full block returns **3-tuple**. Driver dispatches on `producer_of[i] is None` (T4) — consistent. ✓
- `model._spyre_producer_of`, `model._spyre_has_ple`, `model._spyre_kv_shapes`, `model._spyre_compiled_blocks` — set T1/T4, read T4. ✓
