# Onboarding a New Model

Step-by-step guide for adding a HuggingFace Transformers model to the
Spyre adapter framework.

## Quick Start (with Claude Code)

In practice, Claude handles the adapter implementation. Your workflow:

1. Identify the model and HuggingFace checkpoint path
2. Check if `type(model.config)` already exists in
   `CONFIG_TO_ADAPTER_MODULE_MAPPING` in `hf_adapters/auto_spyre_model.py`.
   If it does, the adapter already exists — skip to Step 5 (register in tests)
3. Ask Claude to onboard it (e.g., "onboard Qwen2.5 using the
   qwen2 adapter" or "add an adapter for org/model-1b")
4. Review the adapter code and test results Claude produces
5. Run Spyre tests on the pod (Claude can drive this too if
   connected, or you run manually)

**Optional MCP servers** for enhanced onboarding (add to `~/.claude.json`):

```json
"mcpServers": {
  "web-search": {
    "type": "http",
    "url": "https://mcp.ete-server.vpc-int.res.ibm.com/mcp"
  },
  "spyre-kb": {
    "type": "stdio",
    "command": "<path-to>/spyre-knowledgebase/scripts/run-mcp-server",
    "args": [],
    "env": {
      "GITWIKI_ROOT": "<path-to>/spyre-knowledgebase/"
    }
  }
}
```

The detailed steps below describe what Claude does under the hood —
useful for reviewing its output, debugging failures, or onboarding
a model manually.

## Prerequisites

- Python environment with `pip install -e .` (editable install)
- Access to a HuggingFace checkpoint (or path to local weights)
- Familiarity with the model's architecture (attention type, norms,
  any special features)

## Step 1: Check Architecture Constraints

Before writing any code, verify the model is compatible. Currently supported:

- **Dense, decoder-only, autoregressive** causal LMs with RoPE (full or partial)
  or learned absolute position encoding — the main path this guide covers.
- **Encoder embedders** (BERT / XLM-RoBERTa / MPNet / ModernBERT) via
  `prefill_embed` / `prefill_encoder` and the `st_backend` — bidirectional, no KV
  cache. See `hf_bert.py` / `hf_modernbert.py`.
- **Vision-language (image→text) VLMs** — a vision tower plus a causal text
  decoder. See the Multimodal VLM Path in `ARCHITECTURE.md` and
  `hf_granite_vision_mm.py` / `hf_siglip_vision.py`. This guide's per-step
  causal-LM flow is the decoder half; the vision tower + feature-merge is extra.

Not compatible:
- MoE with dynamic routing (Mixtral, DBRX) — note: Granite 4.0 with `num_local_experts=1` is fine
- Encoder-decoder (T5, BART)
- Models requiring custom CUDA kernels with no PyTorch fallback

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("org/model")
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
print(f"head_dim={head_dim}, head_dim/2={head_dim // 2}")
```

### Constraint Checklist

| Check | Condition | Action |
|-------|-----------|--------|
| Stick alignment | `head_dim / 2 < 64` | Use `pad_attention_heads()` — handled automatically by `prepare_rope_and_heads()` |
| Fused QKV | Single `qkv_proj` weight | Split into separate `q_proj`, `k_proj`, `v_proj` at prepare time |
| Fused MLP | Single `gate_up_proj` weight | Split into separate `gate_proj`, `up_proj` at prepare time |
| Partial RoPE | `partial_rotary_factor < 1.0` | Use `PartialPrecomputedRotaryEmbedding` (see `hf_phi3.py`) |
| Large vocab | any | `pad_lm_head()` pads to a smooth stick count so the per-core span fits the 256 MB EAR limit — handles 200K–262K vocab in a single head. `chunk_lm_head()` is a fallback only if that can't fit (no current model) |
| Model multipliers | `embedding_multiplier`, `residual_multiplier`, `logits_scaling` | Preserve in block/forward functions |
| Non-standard norm | Post-norm, LayerNorm (no weight), Q/K norm | Custom block logic (see `hf_olmo.py`, `hf_olmo2.py`, `hf_qwen3.py`) |
| Gated model | Requires HF authentication | Use a non-gated alternative for CPU tests (e.g., TinyLlama for `model_type=llama`) |

## Step 2: Choose Your Starting Point

Most models fall into one of two categories:

### Standard GQA (simple path)

If the model has: separate Q/K/V projections, pre-norm (RMSNorm before
attention/MLP), standard SwiGLU MLP, no multipliers — use the shared
helpers. The adapter becomes trivial:

```python
from hf_adapters.hf_common import (
    prepare_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def prepare_for_spyre(model):
    from transformers.models.mymodel.modeling_mymodel import MyModelRMSNorm
    prepare_standard_gqa(model, MyModelRMSNorm)
```

That's the entire adapter — see `hf_llama.py`, `hf_qwen2.py`, `hf_mistral.py`
for real examples.

### Custom block (complex path)

If the model has multipliers, unusual norms, fused weights, or other
quirks — write custom `_make_compiled_block()`, `_run_forward()`,
`_run_backbone_forward()`, and `prepare_for_spyre()`.

Examples: `hf_granite.py`, `hf_phi3.py`, `hf_olmo2.py`

## Step 3: Create the Adapter File

Create `hf_adapters/hf_<model>.py`. Every adapter must expose:

| Export | Purpose |
|--------|---------|
| `_run_forward(model, input_ids, position_ids, attn_mask, key_caches, value_caches, is_filling, token_index, cache_position)` | Full forward → logits |
| `_run_backbone_forward(...)` | Same signature → last hidden state (no lm_head). Used by embedding callers |
| `prepare_for_spyre(model)` | Patch model in-place for Spyre |

For **standard GQA** adapters these are just assignments (see Step 2).

For **custom** adapters you also write:

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
    is_filling,         # bool — True=fill mode, False=expand mode
    token_index,        # int — current write position in cache
    cache_position,     # [S] — position indices for cache slice
) -> (hidden_states, key_cache, value_cache)
```

### `prepare_for_spyre(model)` (custom path)

Typical structure:
1. `prepare_rope_and_heads(model)` — checks head_dim, pads if needed, creates `PrecomputedRotaryEmbedding`
2. `pad_lm_head(model)`
3. Compile blocks: `model._spyre_compiled_blocks = [_make_compiled_block(l) for l in get_backbone(model).layers]`
4. Compile the final norm: `model._spyre_compiled_norm = torch.compile(get_backbone(model).norm, dynamic=False)`

Loading and generation are handled by `AutoSpyreModelForCausalLM`
— adapters no longer need `load_model`/`generate` wrappers.

## Step 4: Register in Auto-Loader

Add the model's config class and adapter module to
`CONFIG_TO_ADAPTER_MODULE_MAPPING` in `hf_adapters/auto_spyre_model.py`:

```python
from transformers import MyModelConfig
from hf_adapters import hf_mymodel

CONFIG_TO_ADAPTER_MODULE_MAPPING = {
    ...
    MyModelConfig: hf_mymodel,
}
```

This enables `AutoSpyreModelForCausalLM.from_pretrained()` to
automatically select the correct adapter.

## Step 5: Register in Tests

Add an entry to the right registry in `tests/model_registry.py` — the single
registry that every test file imports (`CAUSAL_LM_MODELS` for generative,
`EMBEDDING_MODELS` for embedders, `VISION_MODELS` for vision/multimodal):

```python
# tests/model_registry.py — CAUSAL_LM_MODELS
"mymodel": {
    "name": "MyModel 1B",
    "path": "org/mymodel-1b-instruct",
    "adapter": "hf_mymodel.py",
    "size": "1b",
},
```

Optional fields:
- `"dtype": "bfloat16"` / `"float32"` — set the test dtype if fp16 is wrong for
  the model (bf16-native models, or large multipliers that overflow fp16 on CPU)
- `"load_fn": True` — use if your adapter has a custom `load_hf_model()` function

## Step 6: Run CPU Accuracy Test

The CPU tests are pytest-parametrized off the registry — run from the repo root
and select your model with `-k <key>` (never `python tests/...`, which bypasses
the conftest patching):

```bash
# Both parametrized cases for one model (test_manual_path + test_auto_loader)
uv run pytest tests/test_adapter_cpu_accuracy.py -k mymodel

# Just one path
uv run pytest tests/test_adapter_cpu_accuracy.py -k "mymodel and manual"        # adapter alone
uv run pytest tests/test_adapter_cpu_accuracy.py -k "mymodel and auto_loader"   # via AutoSpyreModelForCausalLM
```

`test_manual_path` runs prefill + 4 decode steps through both stock HuggingFace
and your adapter, comparing top-1 token selections (all must match).
`test_auto_loader` verifies the end-to-end path through
`AutoSpyreModelForCausalLM`. (Embedding adapters use
`tests/test_embed_cpu_accuracy.py`; multimodal VLMs use `tests/test_vlm_e2e_cpu.py`.)

**Important:** The test runs the HF reference forward *before* calling
`prepare_for_spyre()`, because the RMSNorm patch modifies the class
globally.

## Step 7: Test on Spyre

Once CPU accuracy passes, test on hardware (requires Spyre pod access). The Spyre
lane lives under `tests/spyre/` and is also pytest-parametrized (`-k <key>`):

```bash
# End-to-end smoke test (load + generate, non-trivial output)
uv run pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k mymodel

# Token comparison (CPU vs Spyre, real weights, per-step top-1)
uv run pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k mymodel
```

## Step 8: Update Documentation

`ARCHITECTURE.md` is the **single source of truth** for the supported-model lists
(README.md links to it and must not duplicate them):

1. Add the model to the "Verified Checkpoints" table in `ARCHITECTURE.md`
2. Add the adapter to the "Model Family Coverage" table (and bump the
   "Coverage: N adapters · M verified checkpoints" line)
3. Add any model-specific features to the "Model-Specific Adaptations" table
4. Bump the badge counts at the top of `README.md` to match

## Common Gotchas

- **RMSNorm patch is global** — always run HF reference inference before `prepare_for_spyre()`
- **Zero-length tensors crash on Spyre** — create empty caches with `device=` param, never `.to("spyre")` on shape-0 tensor
- **CPU test compile overhead** — use `getattr(block, "_orig_mod", block)` to unwrap `torch.compile` in CPU-only paths
- **Test ordering** — run `test_adapter_cpu_accuracy.py` first; if tokens don't match on CPU, Spyre testing is pointless

---

## Worked Example: Granite 3.3 2B (head_dim padding)

Granite 3.3 2B shares the same adapter as 8B (`hf_granite.py`) but
requires head_dim padding: `head_dim=64` → `D/2=32 < 64` (one Spyre stick).

### Discovery

```python
config = AutoConfig.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
print(config.head_dim // 2)  # 32 — fails stick alignment!
```

Without padding, Spyre compile fails with:
`RuntimeError: Could not find a host dimension matching stick expr d4 in [...]`

### How it's handled

`prepare_rope_and_heads(model)` detects `head_dim/2 < BLOCK_SIZE` and
calls `pad_attention_heads()` to zero-pad Q/K/V/O weights from 64→128:

| Projection | Padding strategy |
|-----------|-----------------|
| Q, K | Interleaved per RoPE `[2, D/2]` groups (preserves rotation structure) |
| V | Simple end-padding (no RoPE applied) |
| O | Simple end-padding on input dim |

The block forward uses `attn.head_dim` (now 128) — no conditional logic needed.

### Why it's numerically exact

- Q/K padded dims multiply with zero RoPE entries (identity rotation)
- V/O padded dims are zero, contribute nothing to output

### Verification

```bash
uv run pytest tests/test_adapter_cpu_accuracy.py -k granite2b
# All tokens match — same adapter handles both 2B and 8B
```
