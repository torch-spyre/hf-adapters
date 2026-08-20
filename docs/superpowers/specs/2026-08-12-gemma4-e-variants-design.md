# Gemma 4 E-variant (E2B / E4B) Spyre adapter support

**Date:** 2026-08-12
**Status:** Design approved, pending spec review
**Scope:** Add Spyre support for the small dense Gemma 4 E-variants (E2B, E4B) by
extending `hf_adapters/hf_gemma4.py` in place.

## Background

`hf_gemma4.py` today targets the **dense 12B/31B** Gemma 4 text decoder. Its
`prepare_text_decoder_for_spyre` explicitly asserts three features absent so an
unsupported checkpoint fails loudly:

- `hidden_size_per_layer_input` (per-layer embeddings, "PLE")
- `num_kv_shared_layers` (KV-sharing across layers)
- `enable_moe_block` (mixture-of-experts)

Investigation of transformers 5.12.1 `modeling_gemma4.py` and the cached
checkpoints under `/mnt/models/hf_cache/hub` established that the small
**E-variants use only the first two** features. **MoE is NOT an E-variant
feature** — it belongs to the separate `gemma-4-26B-A4B` checkpoint (128 experts,
top-8) and stays out of scope here (it also hits Spyre's known-broken
`topk`/`gather` limits). The MoE assert is retained.

### Verified config values

| | E2B (`google/gemma-4-E2B-it`) | E4B (`google/gemma-4-E4B`) |
|---|---|---|
| model_type | `gemma4_text` | `gemma4_text` |
| num_hidden_layers | 35 | 42 |
| hidden_size | 1536 | 2560 |
| head_dim / global_head_dim | 256 / 512 | 256 / 512 |
| num_attention_heads / kv_heads | 8 / 1 | 8 / 2 |
| hidden_size_per_layer_input (PLE dim) | 256 | 256 |
| vocab_size_per_layer_input | 262144 | 262144 |
| num_kv_shared_layers | 20 | 18 |
| use_double_wide_mlp | True | False |
| enable_moe_block | False | False |
| sliding_window | 512 | 512 |
| final_logit_softcapping | 30.0 | 30.0 |
| hidden_activation | gelu_pytorch_tanh | gelu_pytorch_tanh |
| layer_types | 28 sliding / 7 full | 35 sliding / 7 full |

`head_dim=256` → `head_dim/2 = 128 ≥ 64`: already stick-aligned, no head
padding needed. E2B is cached locally; E4B weights may need download.

## Features and their mechanics (from stock HF)

### Feature 1 — PLE (Per-Layer Embeddings)

A per-token auxiliary signal injected into every decoder layer. Two components,
combined once per forward at the model level, then sliced per layer.

**Model-level setup** (`Gemma4TextModel.__init__`, when `hidden_size_per_layer_input`):
- `embed_tokens_per_layer`: a `Gemma4TextScaledWordEmbedding` of shape
  `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`,
  scale `sqrt(hidden_size_per_layer_input)`.
- `per_layer_model_projection`: `Linear(hidden_size, num_layers * ple_dim, bias=False)`.
- `per_layer_model_projection_scale = hidden_size ** -0.5`.
- `per_layer_projection_norm`: `RMSNorm(ple_dim)`.
- `per_layer_input_scale = 2 ** -0.5`.

**Model-level compute** (once per forward):
1. token-identity = `embed_tokens_per_layer(input_ids)` reshaped to
   `[B, S, num_layers, ple_dim]` (`get_per_layer_inputs`).
2. context = `per_layer_model_projection(inputs_embeds) * projection_scale`,
   reshaped to `[B, S, num_layers, ple_dim]`, then `per_layer_projection_norm`.
3. combined = `(context + token_identity) * per_layer_input_scale`.

Layer *i* receives `combined[:, :, i, :]` as its `per_layer_input`.

**Per-layer injection** (`Gemma4TextDecoderLayer.forward`, tail):
After the post-FF residual add and BEFORE `*= layer_scalar`:
```
residual2 = h
h = per_layer_input_gate(h)          # Linear(hidden -> ple_dim)
h = gelu_tanh(h)
h = h * per_layer_input              # [B,S,ple_dim] elementwise
h = per_layer_projection(h)          # Linear(ple_dim -> hidden)
h = post_per_layer_input_norm(h)     # RMSNorm(hidden)
h = residual2 + h
```
All ops (embedding lookup, linear, gelu-tanh, elementwise mul, RMSNorm) are
Spyre-native.

### Feature 2 — KV-sharing

The last `num_kv_shared_layers` layers reuse the KV of an earlier layer.

- `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`.
  Layers with index `>= first_kv_shared_layer_idx` are **shared** layers.
- Shared layers have **no** `k_proj`, `v_proj`, `k_norm`, `v_norm` (dropped at
  load via `_keys_to_ignore_on_load_unexpected`). They keep `q_proj`, `q_norm`,
  `o_proj`.
- A shared layer reuses the KV of the **last non-shared layer of the same
  `layer_type`** (sliding vs full). In stock HF this is threaded via an
  in-memory `shared_kv_states[layer_type]` dict written by the producer layer
  (`store_full_length_kv`) and read by shared layers.

### Feature 3 — double-wide MLP (E2B only)

`intermediate_size * 2` on kv-shared layers when `use_double_wide_mlp`. **No
adapter work**: the loaded `mlp` submodule already carries the doubled weights;
the compiled block calls `mlp(...)` unchanged.

## Adapter design

Extend `hf_gemma4.py`. All new behavior is **config-gated**: with
`hidden_size_per_layer_input == 0` and `num_kv_shared_layers == 0` (12B/31B) the
code paths are inert and 12B/31B behavior is byte-identical to today.

### Mapping KV-sharing onto static caches (chosen: alias producer buffers)

The adapter uses **per-layer static caches** (`kv_cache_update` driven by
`is_filling` / `token_index` / `cache_position`), not HF's in-memory
`shared_kv_states` dict. KV-sharing maps as:

- Only **non-shared** (producer-bearing) layers allocate a cache. `_spyre_kv_shapes`
  records shapes for those layers.
- For each **shared** layer, precompute its **producer index** = the nearest
  preceding non-shared layer with the same `layer_type`. Store a per-layer
  `producer_cache_index` map on the model.
- At run time the driver passes the **producer layer's** already-updated
  `key_cache` / `value_cache` buffers into the shared block (aliased — no copy,
  no extra buffers). Producer index is always `<` the shared layer index, so the
  producer has already run and its cache is fresh within the step.

### Two compiled block functions (chosen)

- `block_forward` (existing, extended): non-shared layers. Full path:
  input_ln → q/k/v proj+norm → RoPE(q,k) → `kv_cache_update` → SDPA(scale=1.0) →
  o_proj → post_attn_ln (sandwich) → residual → pre_ff_ln → mlp → post_ff_ln →
  residual → **PLE tail (if PLE)** → `* layer_scalar`.
- `shared_block_forward` (new): shared layers. Lean path:
  input_ln → q_proj → q_norm → RoPE(q) → SDPA(q, producer_key_cache,
  producer_value_cache, scale=1.0) → o_proj → post_attn_ln → residual →
  pre_ff_ln → mlp → post_ff_ln → residual → **PLE tail (if PLE)** →
  `* layer_scalar`. No k/v proj/norm, no RoPE-on-K, no `kv_cache_update`.

Each function compiles to one binary reused across all layers of its kind. Both
`layer_scalar` and `per_layer_input` are passed as **tensor arguments** (read
fresh from the device buffers at call time), never captured as constants — same
rationale as the existing `layer_scalar` note (avoids per-layer Dynamo
recompiles that would blow the accumulated-recompile limit).

### `prepare_text_decoder_for_spyre` changes

1. Replace the PLE and KV-share `assert not …` guards with feature wiring; keep
   the `enable_moe_block` assert.
2. No extra norm patching needed for PLE: `per_layer_projection_norm` and
   `post_per_layer_input_norm` are both `Gemma4RMSNorm` (verified in
   transformers 5.12.1, `modeling_gemma4.py:1388,1630`), the same class the
   existing class-level fp16 patch already covers.
3. Compute `first_kv_shared_layer_idx` and the per-shared-layer producer map;
   assert each shared layer found a valid same-type producer.
4. Build `_spyre_kv_shapes` for **producer-bearing (non-shared) layers only**;
   record the per-layer cache-index map (own index if non-shared, producer's
   index if shared).
5. Compile `block_forward` for non-shared layers and `shared_block_forward` for
   shared layers; store both in `_spyre_compiled_blocks` (per-layer, so the
   driver picks the right one by index).

### Backbone driver changes (`_run_backbone_forward` / `_run_blocks_over_embeds`)

1. After the scaled word embedding, if PLE is present compute the combined
   `per_layer_inputs` tensor `[B, S, num_layers, ple_dim]` (steps 1–3 above).
2. In the per-layer loop, pass `per_layer_inputs[:, :, i, :]` (or `None`) as the
   block's `per_layer_input` tensor arg.
3. For each layer, select the compiled block by kind and pass the correct cache
   buffers: own for non-shared, the mapped producer's for shared.
4. The VLM adapter (`hf_gemma4_mm`) reuses `_run_blocks_over_embeds`; its
   behavior is unchanged because E-variants are text-only and the VLM
   checkpoints do not set these config fields. (No VLM E-variant is in scope.)

## Data flow (one decode step)

```
input_ids
  → scaled word embedding  ─┐
  → per_layer_model_projection(embeds)·scale ─┐
  → embed_tokens_per_layer(ids) ──────────────┤ combine·2^-0.5 → per_layer_inputs [B,S,L,ple]
  → per-type RoPE + masks (unchanged)
  → for layer i in 0..L-1:
        non-shared: block_forward(...) updates key/value_cache[i]
        shared:     shared_block_forward(..., producer_key/value_cache) reads only
        (both apply PLE tail with per_layer_inputs[:,:,i,:] when PLE present)
  → final norm → lm_head → final_logit_softcapping (cap=30.0)
```
Producers always precede their shared dependents, so aliased caches are fresh.

## Error handling

- Keep `assert not enable_moe_block` (MoE out of scope).
- Assert every shared layer resolved a same-type producer. Verified true for
  both configs: E2B `first_kv_shared_layer_idx=15`, E4B `=24`, and both full and
  sliding types appear before the boundary in each. The assert guards against
  future/unexpected configs; fail loud otherwise.
- Keep the existing stick-alignment asserts on `head_dim` / `global_head_dim`.

## Testing

Verification target: **CPU token-compare + Spyre smoke**.

1. **Registry** (`tests/model_registry.py`): add
   - `gemma4_e2b` → `google/gemma-4-E2B-it`, adapter `hf_gemma4.py`, size `2b`,
     gated.
   - `gemma4_e4b` → `google/gemma-4-E4B`, adapter `hf_gemma4.py`, size `4b`,
     gated.
   Add both to the smoke/token-compare parametrization groups.
2. **CPU numerical parity**: forward E2B (cached) through stock HF **before**
   `prepare_for_spyre` (the RMSNorm patch mutates the class globally), capture
   reference logits/tokens, then compare the adapter's CPU-path output. This
   exercises PLE + KV-share correctness without a card.
3. **Spyre smoke** (`tests/spyre/test_e2e_smoke_spyre.py -k gemma4_e2b`):
   compile + run end-to-end on Spyre, assert no crash / NaN.
4. E4B follows from the same code; run its CPU parity once weights are present.

## Definition of Done

- [ ] `hf_gemma4.py` extended (PLE + KV-share, config-gated); MoE assert kept.
- [ ] 12B/31B path unchanged (regression check: existing gemma4 tests green).
- [ ] Registry entries `gemma4_e2b`, `gemma4_e4b`.
- [ ] CPU token-compare parity vs stock HF for E2B (and E4B when available).
- [ ] Spyre smoke passes for E2B (compile + run, no crash/NaN).
- [ ] `ARCHITECTURE.md` coverage tables updated; README badge counts bumped.

## Out of scope

- MoE (`gemma-4-26B-A4B`) — separate variant, broken Spyre ops, own effort.
- Any multimodal E-variant.
- E4B on-device token-compare if weights/card time unavailable (CPU parity + E2B
  Spyre smoke is the committed bar).
```
