# Onboarding a New Model

Step-by-step guide for adding a HuggingFace Transformers model to the
Spyre adapter framework.

## Prerequisites

- Python environment with `pip install -e .` (editable install)
- Access to a HuggingFace checkpoint (or path to local weights)
- Familiarity with the model's architecture (attention type, norms,
  any special features)

## Step 1: Check Architecture Constraints

Before writing any code, verify the model is compatible:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("org/model")
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
print(f"head_dim={head_dim}, head_dim/2={head_dim // 2}")
```

### Constraint Checklist

| Check | Condition | Action |
|-------|-----------|--------|
| Stick alignment | `head_dim / 2 < 64` | Use `pad_attention_heads()` to pad to 128 |
| Fused QKV | Single `qkv_proj` weight | Split into separate `q_proj`, `k_proj`, `v_proj` at prepare time |
| Fused MLP | Single `gate_up_proj` weight | Split into separate `gate_proj`, `up_proj` at prepare time |
| Partial RoPE | `partial_rotary_factor < 1.0` | Use `PartialPrecomputedRotaryEmbedding` (see `hf_phi3.py`) |
| Large vocab | vocab_size >= 200K | Chunked LM head (see `hf_phi3.py`) |
| Model multipliers | `embedding_multiplier`, `residual_multiplier`, `logits_scaling` | Preserve in block/forward functions |
| Non-standard norm | Post-norm, LayerNorm (no weight), Q/K norm | Custom block logic (see `hf_olmo.py`, `hf_olmo2.py`, `hf_qwen3.py`) |
| Gated model | Requires HF authentication | Use a non-gated alternative for CPU tests (e.g., TinyLlama for `model_type=llama`) |

## Step 2: Choose Your Starting Point

Most models fall into one of two categories:

### Standard GQA (simple path)

If the model has: separate Q/K/V projections, pre-norm (RMSNorm before
attention/MLP), standard SwiGLU MLP, no multipliers — use the shared
helpers:

```python
from hf_adapters.hf_common import (
    make_standard_gqa_block,
    standard_gqa_forward,
    prepare_standard_gqa,
)
```

Examples: `hf_llama.py`, `hf_qwen2.py`, `hf_mistral.py`

### Custom block (complex path)

If the model has multipliers, unusual norms, fused weights, or other
quirks — write a custom `_make_compiled_block()` and `_run_forward()`.

Examples: `hf_granite.py`, `hf_phi3.py`, `hf_olmo2.py`

## Step 3: Create the Adapter File

Create `hf_adapters/hf_<model>.py` with these four functions:

### `_make_compiled_block(layer)`

A closure over one transformer layer's weights. Returns
`torch.compile(block_forward, dynamic=False)`.

Signature of the inner `block_forward`:

```python
def block_forward(
    hidden_states,      # [B, S, D]
    selected_freqs,     # [B, S, 2, 2, head_dim/2] — precomputed RoPE
    attn_mask,          # [B, 1, S, cache_len] — float16 causal mask
    key_cache,          # [B, num_kv_heads, max_cache_len, head_dim]
    value_cache,        # [B, num_kv_heads, max_cache_len, v_head_dim]
    is_filling,         # bool — True=overwrite mode, False=expand mode
    token_index,        # int — current write position in cache
    cache_position,     # [S] — position indices for cache slice
) -> (hidden_states, key_cache, value_cache)
```

### `_run_forward(model, input_ids, position_ids, attn_mask, key_caches, value_caches, is_filling, token_index, cache_position)`

Full model forward: embedding → RoPE → N compiled blocks → final norm
→ lm_head. Returns logits.

### `prepare_for_spyre(model)`

Patches the model in-place:
1. `prepare_rope_and_heads(model)` — checks head_dim, pads if needed, creates `PrecomputedRotaryEmbedding`
2. Patch RMSNorm class
3. Pad LM head
4. Compile all blocks

Loading and generation are handled by `AutoSpyreModelForCausalLM`
(see Public API in ARCHITECTURE.md) — adapters no longer need
`load_model`/`generate` wrappers.

## Step 4: Register in Tests

Add an entry to the `MODELS` dict in `tests/test_adapter_cpu_accuracy.py`:

```python
"mymodel": {
    "name": "MyModel 1B",
    "path": "org/mymodel-1b-instruct",
    "adapter": "hf_mymodel.py",
},
```

Optional fields:
- `"dtype": "float32"` — use if fp16 overflows on CPU (e.g., large multipliers)
- `"load_fn": True` — use if your adapter has a custom `load_hf_model()` function

## Step 5: Run CPU Accuracy Test

```bash
source .venv/bin/activate
python3 tests/test_adapter_cpu_accuracy.py mymodel
```

This runs prefill + 4 decode steps through both stock HuggingFace and
your adapter, comparing top-1 token selections. All must match.

**Important:** The test runs the HF reference forward *before* calling
`prepare_for_spyre()`, because the RMSNorm patch modifies the class
globally.

## Step 6: Test on Spyre

Once CPU accuracy passes, test on hardware (requires Spyre pod access):

```bash
# Per-layer block comparison (random weights, checks for NaN/crash)
python3 tests/test_block_cpu_vs_spyre.py mymodel

# End-to-end smoke test
python3 tests/test_e2e_smoke_spyre.py mymodel

# Token comparison (CPU vs Spyre, real weights)
python3 tests/test_e2e_token_compare_spyre.py mymodel
```

## Step 7: Update Documentation

1. Add the model to the "Verified Checkpoints" table in `ARCHITECTURE.md`
2. Add the adapter to the "Model Family Coverage" table
3. Add any model-specific features to the "Model-Specific Adaptations" table

## Common Gotchas

- **RMSNorm patch is global** — always run HF reference inference before `prepare_for_spyre()`
- **Zero-length tensors crash Spyre** — create empty caches with `device=` param, never `.to("spyre")` on shape-0 tensor
- **CPU test compile overhead** — use `getattr(block, "_orig_mod", block)` to unwrap `torch.compile` in CPU-only paths
- **Test ordering** — run `test_adapter_cpu_accuracy.py` first; if tokens don't match on CPU, Spyre testing is pointless

---

## Worked Example: Granite 3.3 2B (head_dim padding)

This walks through how Granite 3.3 2B was onboarded. It shares the
same adapter as Granite 3.3 8B (`hf_granite.py`) but required
head_dim padding because `head_dim=64` → `D/2=32`, which is less than
one Spyre stick (64 elements).

### 1. Discovered the constraint

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
print(config.head_dim)  # 64
print(config.head_dim // 2)  # 32 — LESS than 64, fails stick alignment!
```

Without padding, `torch.compile` on Spyre produces:
```
RuntimeError: Could not find a host dimension matching stick expr d4 in [...]
```

### 2. Applied head_dim padding in `prepare_for_spyre()`

The current `hf_granite.py` handles both 8B (no padding) and 2B
(needs padding) via a single call to `prepare_rope_and_heads`:

```python
def prepare_for_spyre(model):
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    prepare_rope_and_heads(model)  # checks head_dim, pads if needed, creates RoPE
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in model.model.layers
    ]
```

Under the hood, `prepare_rope_and_heads` (in `hf_common.py`) does:

```python
def prepare_rope_and_heads(model):
    cfg = model.config
    orig_head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    # Compute minimum stick-aligned head_dim
    stick_aligned_head_dim = (
        (orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)
    ) * (2 * BLOCK_SIZE)

    padded_head_dim = None
    if stick_aligned_head_dim > orig_head_dim:
        padded_head_dim = stick_aligned_head_dim  # 64 → 128
        pad_attention_heads(
            model, model.model.layers, orig_head_dim, padded_head_dim,
            cfg.num_attention_heads, cfg.num_key_value_heads,
        )

    model._spyre_rope = PrecomputedRotaryEmbedding(
        model.model.rotary_emb, padded_head_dim=padded_head_dim,
    )
```

### 3. What `pad_attention_heads()` does

For each layer, it zero-pads the weight matrices:

| Projection | Original shape | Padded shape | Padding strategy |
|-----------|---------------|-------------|-----------------|
| `q_proj` | `[D, num_heads * 64]` | `[D, num_heads * 128]` | Interleaved per RoPE `[2, D/2]` groups |
| `k_proj` | `[D, num_kv_heads * 64]` | `[D, num_kv_heads * 128]` | Interleaved per RoPE groups |
| `v_proj` | `[D, num_kv_heads * 64]` | `[D, num_kv_heads * 128]` | Simple end-padding (no RoPE) |
| `o_proj` | `[num_heads * 64, D]` | `[num_heads * 128, D]` | Simple end-padding |

The interleaved padding for Q/K preserves the `[cos, -sin; sin, cos]`
rotation matrix structure that `apply_rope_matmul` expects.

### 4. Block forward works unchanged

The block function uses `attn.head_dim` (which is now 128 after
padding), so no special-casing is needed in the block itself:

```python
q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
```

### 5. Registered the 2B variant in tests

```python
# tests/test_adapter_cpu_accuracy.py
"granite2b": {
    "name": "Granite 3.3 2B",
    "path": "ibm-granite/granite-3.3-2b-instruct",
    "adapter": "hf_granite.py",  # same adapter as 8B
},
```

### 6. Verified

```bash
python3 tests/test_adapter_cpu_accuracy.py granite2b
# ✓ All tokens match — padding is numerically invisible
```

The zero-padding produces identical results because:
- Q/K padded dimensions multiply with zero RoPE entries (identity rotation)
- V padded dimensions are zero, contribute nothing to attention output
- O projection's padded input rows are zero, contribute nothing to output

### Key Takeaway

Head_dim padding is handled entirely in `prepare_for_spyre()` — the
block forward function doesn't need any conditional logic. The adapter
works for both padded and non-padded variants of the same architecture.
