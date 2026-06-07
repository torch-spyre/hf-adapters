# Architecture

How the HuggingFace Transformers adapters work, what they change, and
which models are supported on Spyre.

## Verified Checkpoints

### Generative (causal-LM)

| Model | model\_type | head\_dim | D/2 | Stick Aligned | CPU Accurate | Spyre Compiles | Spyre Runs |
|-------|-----------|---------|-----|--------------|-------------|---------------|-----------|
| Qwen3 0.6B | qwen3 | 128 | 64 | Yes | Yes | Yes | Yes |
| Granite 3.3 8B | granite | 128 | 64 | Yes | Yes | Yes | Yes |
| Granite 3.3 2B | granite | 64→128 | 64 | Yes (padded) | Yes | Yes | Yes |
| Granite 4.0 1B | granitemoehybrid | 128 | 64 | Yes | Yes | Yes | Yes |
| SmolLM3 3B | smollm3 | 128 | 64 | Yes | Yes | Yes | Yes |
| Llama 3.2 3B | llama | 128 | 64 | Yes | Yes | Yes | Yes |
| TinyLlama 1.1B | llama | 64→128 | 64 | Yes (padded) | Yes | Yes | Yes |
| Qwen2.5 7B | qwen2 | 128 | 64 | Yes | Yes | Yes | Yes |
| Qwen2.5 1.5B | qwen2 | 128 | 64 | Yes | Yes | Yes | Yes |
| Mistral 7B v0.3 | mistral | 128 | 64 | Yes | Yes | Yes | Yes |
| Phi-4 mini | phi3 | 128 | 64 | Yes | Yes | Yes | Yes |
| OLMo 1B | olmo | 128 | 64 | Yes | Yes | Yes | Yes |
| OLMo2 1B | olmo2 | 128 | 64 | Yes | Yes | Yes | Yes |
| Falcon 3 1B | llama | 256 | 128 | Yes | Yes | Yes | Yes |
| DeepSeek-Coder 1.3B | llama | 128 | 64 | Yes | Yes | Yes | Yes |
| Yi 1.5 6B | llama | 128 | 64 | Yes | Yes | Yes | Yes |
| Granite Vision 4.1 4B | granite (text) | 64→128 | 64 | Yes (padded) | Yes | Yes | Yes |

**CPU Accurate** = adapter produces identical greedy tokens to stock HF on CPU.
**Spyre Compiles** = `torch.compile(block_forward)` succeeds on Spyre.
**Spyre Runs** = block produces output (no crash/NaN).

### Embedding

Use `st_backend` for the sentence-transformers API, or call `prefill_embed` / `prefill_encoder` directly from stock HF models.

| Model | model\_type | head\_dim | Stick Aligned | CPU Accurate | Spyre Compiles | Spyre Runs |
|-------|-----------|---------|--------------|-------------|---------------|-----------|
| Qwen3-Embedding 0.6B | qwen3 | 128 | Yes | Yes | Yes | Yes |
| GTE-Qwen2-1.5B | qwen2 | 128 | Yes | Yes | Yes | Yes (accuracy diverges from CPU) |
| E5-Mistral-7B | mistral | 128 | Yes | Yes | Yes | Yes (accuracy diverges from CPU) |
| Linq-Embed-Mistral | mistral | 128 | Yes | Yes | Yes | Yes (accuracy diverges from CPU) |
| SFR-Embedding-Mistral | mistral | 128 | Yes | Yes | Yes | Yes (accuracy diverges from CPU) |
| BGE-base-en-v1.5 | bert | 64 | Yes | Yes | Yes | Yes |
| all-MiniLM-L6-v2 | bert | 32→64 | Yes (padded) | Yes | Yes | Yes |
| BGE-M3 | xlm-roberta | 64 | Yes | Yes | Yes | Yes |
| all-mpnet-base-v2 | mpnet | 64 | Yes | Yes | Yes | Yes |
| ModernBERT-embed-base | modernbert | 64→128 | Yes (padded) | Yes | Yes | Yes |
| GTE-ModernBERT-base | modernbert | 64→128 | Yes (padded) | Yes | Yes | Yes |
| Granite-Embedding-97m-multilingual-r2 | modernbert | 32→128 | Yes (padded) | Yes | Yes | Yes |

**CPU Accurate** = adapter hidden-states have cosine similarity ≥ 0.9999 vs stock HF on CPU.
**Spyre Compiles / Spyre Runs** = via `test_e2e_embed_compare_spyre.py`. GTE-Qwen2 and the Mistral-backbone embedding models (E5-Mistral, Linq-Embed-Mistral, SFR-Embedding-Mistral) compile and execute end-to-end on Spyre but their pooled embeddings drift from the CPU reference; the Qwen3/BERT/XLM-RoBERTa/MPNet/ModernBERT encoder paths match within fp16 noise.

### Spyre Numerical Accuracy (torch-spyre @ 7c6ef99)

Per-layer block comparison (`test_block_cpu_vs_spyre.py`) with random
weights, measured as max absolute diff between CPU and Spyre output:

| Model | Prefill (seq=64) | Decode (seq=1) |
|-------|-----------------|----------------|
| Qwen3 0.6B | 0.01–0.02 | 0.3–5.5 |
| Llama 3.2 3B | 0.07–0.08 | 1.9–6.0 |

**Prefill is accurate** — errors are in the fp16 rounding range. The
first generated token matches the CPU reference (verified with Qwen3
E2E: `" Paris"` on both CPU and Spyre).

**Decode has large errors** — max diffs of 1–6 per block accumulate
across layers and autoregressive steps, causing token drift after the
first token. This is a torch-spyre compiler issue specific to the
single-token decode path (seq_len=1), not an adapter issue.

## Model Family Coverage

Each adapter handles a HuggingFace `model_type`. Once verified with
one checkpoint, all size variants and fine-tunes of that architecture
work with zero additional code — they share the same attention
pattern, norms, and weight layout.

| Adapter | model\_type | Verified | Also Compatible (same adapter, untested sizes/fine-tunes) |
|---------|-----------|----------|----------------------------------------------------------|
| hf\_llama.py | llama | 5 | Llama 2 7B/13B, Llama 3 8B, Llama 3.1 8B, Code Llama 7B/13B, Vicuna 7B/13B, OpenChat 3.5 7B, Nous Hermes 2, Solar 10.7B |
| hf\_qwen2.py | qwen2 | 3 | Qwen2 0.5B/1.5B/7B, Qwen2.5 0.5B/3B, Qwen2.5-Coder 0.5B–7B, Qwen2.5-Math 1.5B/7B |
| hf\_granite.py | granite | 3 | Granite 3.3 8B/2B Base, Granite 3.2 8B, Granite 3.1 8B/2B, Granite 3.0 8B, Granite Code 8B/3B |
| hf\_qwen3.py | qwen3 | 2 | Qwen3 1.7B, Qwen3 4B, Qwen3 8B |
| hf\_mistral.py | mistral | 2 | Mistral 7B v0.1/v0.2, Mistral 7B Instruct v0.1–v0.3, Zephyr 7B |
| hf\_phi3.py | phi3 | 1 | Phi-3 mini 4k/128k, Phi-3 small 8k |
| hf\_granitemoehybrid.py | granitemoehybrid | 1 | Granite 4.0 Micro |
| hf\_smollm3.py | smollm3 | 1 | — |
| hf\_olmo.py | olmo | 1 | OLMo 7B |
| hf\_olmo2.py | olmo2 | 1 | OLMo 2 7B |
| hf\_granite\_vision.py | granite (text) | 1 | — |
| hf\_bert.py | bert | 2 | BERT-base, BERT-large, RoBERTa-base/large, other BGE/MiniLM variants |
| hf\_xlm\_roberta.py | xlm-roberta | 1 | multilingual-e5-large, paraphrase-multilingual-mpnet-base-v2, other XLM-R fine-tunes |
| hf\_mpnet.py | mpnet | 1 | multi-qa-mpnet-base-{dot,cos}-v1, paraphrase-mpnet-base-v2, microsoft/mpnet-base, all-mpnet-base-v1 |
| hf\_modernbert.py | modernbert | 3 | answerdotai/ModernBERT-base, answerdotai/ModernBERT-large, other ModernBERT embed/classifier fine-tunes |

**Verified** = checkpoints tested in CI (appear in the matrix above).
**Also Compatible** = same `model_type` in HuggingFace config; expected
to work without code changes. Size constraints apply (must fit in
Spyre memory). Gated models require HF token access.

**Variant count: 15 adapters → 27 verified checkpoints, ~60+ compatible variants.**

## Public API

### Unified Auto API (Recommended)

```python
from hf_adapters import AutoSpyreModelForCausalLM
from transformers import AutoTokenizer

model = AutoSpyreModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
outputs = model.generate(tokenizer, ["What is 2+2?"], max_new_tokens=128)
```

`AutoSpyreModelForCausalLM` automatically selects the correct adapter based on the model's config type.

### Manual Control API

Each adapter also exposes `prepare_for_spyre(model)` for manual control:

```python
from transformers import AutoModelForCausalLM
from hf_adapters.hf_granite import prepare_for_spyre

model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-8b-instruct",
    dtype=torch.float16,
    device_map="cpu",
)
prepare_for_spyre(model)
model.to("spyre")
outputs = generate(
    model, tokenizer, ["Hello!"], max_new_tokens=32,
)
```

## How the Adapters Work

### Architecture

Each adapter follows the FMS `eager_spyre` compilation pattern:
compiled block functions with raw tensor KV caches, precomputed
RoPE rotation matrices, fp16 RMSNorm, and padded 64-block decode
generation loop.

```
hf_adapters/
├── hf_common.py          — shared utilities
│   DEVICE, BLOCK_SIZE,
│   PrecomputedRotaryEmbedding, apply_rope_matmul,
│   pad_attention_heads, patch_rmsnorm, pad_lm_head,
│   kv_cache_update, build_prefill_mask,
│   build_expansion_mask, load_model_common, generate
├── hf_granite.py          — Granite 3.3 adapter
├── hf_granite_vision.py   — Granite Vision 4.1 text backbone adapter
├── hf_qwen3.py            — Qwen3 adapter
├── hf_granitemoehybrid.py — Granite 4.0 dense adapter
├── hf_smollm3.py          — SmolLM3 adapter
├── hf_llama.py            — Llama adapter (Llama 1/2/3, Code Llama, Yi, TinyLlama)
├── hf_qwen2.py            — Qwen2 adapter (Qwen 1.5, Qwen 2, Qwen 2.5)
├── hf_mistral.py          — Mistral adapter (Mistral 7B v0.2, v0.3)
├── hf_phi3.py             — Phi-4 mini adapter
├── hf_olmo.py             — OLMo adapter (OLMo 1B, 7B)
├── hf_olmo2.py            — OLMo2 adapter (OLMo 2 7B)
├── hf_bert.py             — BERT-family encoder adapter (BGE, MiniLM)
├── hf_xlm_roberta.py      — XLM-RoBERTa encoder adapter (BGE-M3, multilingual-e5)
├── hf_mpnet.py            — MPNet encoder adapter (all-mpnet-base-v2 and variants)
├── hf_modernbert.py       — ModernBERT encoder adapter (RoPE, GeGLU, local/global attention)
└── __init__.py
```

```
┌───────────────────────────────────────────────┐
│  generate() — Python loop (hf_common.py)      │
│  ┌─────────────────────────────────────────┐  │
│  │  _run_forward() — per-model             │  │
│  │  ┌───────────────────────────────────┐  │  │
│  │  │  compiled block_forward()         │  │  │
│  │  │  • RMSNorm (fp16, patched)        │  │  │
│  │  │  • QKV projections                │  │  │
│  │  │  • RoPE (matmul, no slicing)      │  │  │
│  │  │  • KV cache update                │  │  │
│  │  │  • SDPA (enable_gqa=True)         │  │  │
│  │  │  • Output projection              │  │  │
│  │  │  • MLP (SwiGLU)                   │  │  │
│  │  └───────────────────────────────────┘  │  │
│  │  x N layers                             │  │
│  │  Final RMSNorm + LM head                │  │
│  └─────────────────────────────────────────┘  │
│  Token selection (CPU) + buffer management    │
└───────────────────────────────────────────────┘
```

### Deviations from Stock HuggingFace Transformers

#### 1. RoPE: Precomputed Rotation Matrices

| Stock HF | Adapter |
|---|---|
| `*RotaryEmbedding.forward()` | `PrecomputedRotaryEmbedding` |
| sin/cos computed every forward call | Precomputed once on CPU, cached as `[S, 2, 2, D/2]` rotation matrices |
| `rotate_half()` + slices | `apply_rope_matmul()` — reshape to `[B, L, H, 2, D/2]`, broadcast multiply, sum |
| `(cos, sin)` tuple output | `selected_freqs` tensor `[B, L, 2, 2, D/2]` |

**Why:** Spyre has no `sin`/`cos` ops and `aten.slice.Tensor` falls
back to CPU inside compiled graphs.

#### 2. RMSNorm: Class-Level Patch

| Stock HF | Adapter |
|---|---|
| Each model has its own RMSNorm class | `patch_rmsnorm(cls)` patches any RMSNorm class in-place |
| Casts to float32 for variance | Spyre: stays fp16. CPU: float32 (matches HF) |
| `hidden_states.pow(2).mean()` | Spyre: `(hidden_states * hidden_states).mean()`. CPU: same as HF |
| Python float epsilon | Spyre: `torch.ops.spyre.full((1,), eps, ...)` tensor. CPU: Python float |

**Why:** Spyre does not support dtype conversion on-device. `pow(2)`
is not well supported; element-wise multiply is native.

#### 3. LM Head Weight: Padded

| Stock HF | Adapter |
|---|---|
| Vocab dim as-is from model config | Padded to `ceil(vocab/64)*64 + 64` via `pad_lm_head()` |

**Why:** Spyre requires tensor dimensions aligned to 64-element
sticks for efficient matmul. The extra +64 avoids prime-number
multiples that cause poor work distribution.

#### 4. Decoder Layers: Custom Compiled Blocks

| Stock HF | Adapter |
|---|---|
| `*DecoderLayer.forward()` | `block_forward()` — plain function closure wrapping the same weights |
| `DynamicCache` Python object | Raw tensor lists passed as function args |
| `torch.cat` inside `DynamicCache.update()` | `torch.cat` (expand) or `spyre.overwrite` (fill) |
| Not compiled by default | `torch.compile(block_forward, dynamic=False)` |

**Why:** `DynamicCache` causes graph breaks in `torch.compile`. Raw
tensor args trace cleanly. `torch.ops.spyre.overwrite` must execute
inside the compiled graph to produce Spyre device code.

#### 5. Generation Loop: Custom Implementation

| Stock HF | Adapter |
|---|---|
| `GenerationMixin.generate()` | `generate()` in `hf_common.py` |
| Token-by-token with dynamic cache growth | 64-block padded decode: prefill, expand, fill (x63), expand cycle |
| Right-padded or unpadded prompts | Left-padded to multiple of 64 |
| Grows by 1 per token | Grows by 64 per expansion, then 63 single-slot overwrites |
| Full sampling, beam search, etc. | Greedy + top-k sampling, per-token timing |

**Why:** Spyre requires fixed-size block decode with
`spyre.overwrite` for KV cache updates. HF's generate has dynamic
shapes and DynamicCache incompatible with static-shape compilation.

#### 6. Attention Mask: Built Externally

| Stock HF | Adapter |
|---|---|
| `create_causal_mask()` using `torch.tril` | `build_prefill_mask()` / `build_expansion_mask()` on CPU |
| Model dtype (may be float32) | Always `float16` |
| On-device | Built on CPU, moved to Spyre |

**Why:** `torch.tril` is not supported on Spyre. Masks must be
`float16` (Spyre's native dtype).

#### 7. Embedding: No Change Required

HF's `nn.Embedding` automatically falls back to CPU via
torch-spyre's fallback mechanism. The result is transparently
transferred to the Spyre device. No adapter code needed.

### What Works As-Is (No Patching)

These HF Transformer components run natively on Spyre without
modification:

| Component | HF Class/Function | Spyre Support |
|---|---|---|
| Linear projections | `nn.Linear` | Native matmul |
| MLP activation | `nn.SiLU` (SwiGLU) | Native `silu` |
| Embedding multiplier | scalar-tensor mul | Native |
| Residual multiplier | scalar-tensor mul | Native |
| Logits scaling | tensor-scalar div | Native |
| SDPA | `F.scaled_dot_product_attention` | Decomposed by torch-spyre |
| GQA head expansion | `enable_gqa=True` | SDPA decomposition |
| Embedding lookup | `nn.Embedding` | CPU fallback (automatic) |

### Model-Specific Adaptations

| Feature | Granite 3.3 | Granite Vision 4.1 | Qwen3 | Granite 4.0 | SmolLM3 | Llama | Qwen2 | Mistral | Phi-4 mini | OLMo | OLMo2 |
|---------|------------|-------------------|-------|-------------|---------|-------|-------|---------|-----------|------|-------|
| Embedding multiplier | Yes | Yes | No | Yes | No | No | No | No | No | No | No |
| Residual multiplier | Yes | Yes | No | Yes | No | No | No | No | No | No | No |
| Logits scaling | Yes | Yes | No | Yes | No | No | No | No | No | No | No |
| Q/K RMSNorm | No | No | Yes (per-head) | No | No | No | No | No | No | No | Yes (flattened) |
| Fused QKV split | No | No | No | No | No | No | No | No | Yes | No | No |
| Fused MLP split | No | No | No | Yes | No | No | No | No | Yes | No | No |
| NoPE layers | No | No | No | No | Yes | No | No | No | No | No | No |
| Partial RoPE | No | No | No | No | No | No | No | No | Yes | No | No |
| Chunked LM head | No | No | No | No | No | No | No | No | Yes | No | No |
| Head-dim padding | 2B only | Yes (64→128) | No | No | No | TinyLlama | No | No | No | No | No |
| Custom model loading | No | Yes (safetensor remap) | No | No | No | No | No | No | No | No | No |
| Attention scaling | `config.attention_multiplier` | `config.attention_multiplier` | `head_dim**-0.5` | `config.attention_multiplier` | `head_dim**-0.5` | `head_dim**-0.5` | `head_dim**-0.5` | `head_dim**-0.5` | `head_dim**-0.5` | `head_dim**-0.5` | `head_dim**-0.5` |
| Norm type | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | RMSNorm (pre) | LayerNorm (pre, no weight) | RMSNorm (post) |

**Partial RoPE** (Phi-4): `PartialPrecomputedRotaryEmbedding` pads
the rotation matrix with identity `[[1,0],[0,1]]` entries so
`apply_rope_matmul` operates on full `head_dim` without slicing.
Avoids stickify non-zero offset assertion.

**Chunked LM head** (Phi-4): 200K+ vocab exceeds Spyre per-core
256 MB EAR limit. 8 smaller `nn.Linear` chunks along vocab dim,
cat results on CPU.

**Fused weight split** (Phi-4, Granite 4.0): QKV/gate_up_proj split
into separate linears at prepare time. Avoids stickify non-zero
offset assertions.

**Head-dim padding** (Granite 2B, TinyLlama, Granite Vision): `pad_attention_heads()`
zero-pads Q/K/V/O projections and RoPE freqs from 64→128 so
D/2 = 64 (one stick). Q/K use interleaved padding per RoPE
`[2, D/2]` group; V/O use simple end-padding.

**Custom model loading** (Granite Vision): The text backbone weights
are extracted from a multimodal checkpoint (vision+text) via safetensor
key remapping (`model.language_model.*` → `model.*`) into a standard
`GraniteForCausalLM`. No `trust_remote_code` required. Vision encoder
and projection layers are discarded — only the text backbone runs on
Spyre.

**OLMo LayerNorm** (OLMo): Uses weight-free `OlmoLayerNorm` (no
learnable parameters). Custom patch keeps it in fp16 on Spyre.

**Post-norm + Q/K RMSNorm** (OLMo2): Norm applied after attention/MLP
output, before residual add (not pre-norm). Q/K RMSNorm on flattened
projections before reshape and RoPE.

**Local/global attention + dual-theta RoPE** (ModernBERT): A pre-norm
RoPE encoder. Every `global_attn_every_n_layers`-th layer uses full
bidirectional attention with a high RoPE theta (160000); the rest use
local sliding-window attention (`±sliding_window` band) with a low
theta (10000). The adapter precomputes one `PrecomputedRotaryEmbedding`
per theta and builds both masks once per forward (the local mask is the
global mask intersected with the band via `add_sliding_window_band`),
selecting per layer by `layer.attention_type`. Fused `Wqkv` is split
into padded q/k/v (head_dim 64→128, like all-MiniLM); MLP is GeGLU
(`Wo(act(input) * gate)` from a single `Wi`). Layer 0's `attn_norm` is
`Identity` (the embedding LayerNorm already normalized the input). No
biases by default.

## Adding a New Model

See [ONBOARDING.md](ONBOARDING.md) for the full step-by-step guide,
constraint checklist, and a worked example (Granite 3.3 2B with
head_dim padding).

For a side-by-side comparison with the FMS `eager_spyre` approach,
see [docs/fms_comparison.md](docs/fms_comparison.md).

## Known Issues

### Spyre Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| No `sin`/`cos` ops | RoPE must be precomputed | `PrecomputedRotaryEmbedding` |
| No dtype conversion | RMSNorm must stay fp16 | Patched forward with device check |
| No `aten.slice` in compiled graphs | KV cache indexing falls back to CPU | `spyre.overwrite` for fill mode |
| `head_dim/2 < 64` (sub-stick) | Stickify assertion: `Could not find a host dimension matching stick expr d4 in [...]`. Rule: RoPE matmul requires `head_dim >= 128` (`D/2 >= 64`). | `pad_attention_heads()` pads Q/K/V/O and RoPE freqs to stick-aligned size (e.g. Granite 3.3 2B: 64→128) |
| `partial_rotary_factor < 1.0` | Non-zero offset assertion in stickify | Identity-padded rotation matrices in `PartialPrecomputedRotaryEmbedding` (implemented in `hf_phi3.py`) |
| Zero-length tensors crash `copy_host_to_device` | Segfault on `.to("spyre")` | Create empty tensors directly on device |
| fp16 overflow on CPU for large multipliers | NaN logits on CPU | Test in float32; runs fine on Spyre |

### Performance Issues

These affect speed but not correctness:

**Compilation overhead (first run):** The first invocation compiles
graphs per layer per mode (expand + fill). This takes several
minutes. Subsequent runs with the same shapes reuse cached compiled
graphs.

**`aten.slice` fallback in fill mode:** The KV cache fill operation
`k[:, :, token_index:token_index+1, :]` inside `spyre.overwrite`
triggers an `aten.slice.Tensor` CPU fallback per layer per fill
step.

**Recompilation per `token_index`:** Each unique `token_index`
value in fill mode triggers a new graph specialization (because
`torch.compile` specializes on Python int arguments). Over 63 fill
steps, this causes 63 recompilations on first use.

### Open Work

1. **Decode path numerical accuracy** — single-token decode (seq=1)
   has max diffs of 1–6 per block vs CPU. Prefill (seq=64) is
   accurate (0.01–0.08). Likely a torch-spyre stickify or layout
   issue specific to seq_len=1. This is the primary blocker for
   end-to-end correct generation on Spyre.
2. **Fix `token_index` recompilation** — pass as tensor to avoid
   specialization
3. **Fix `aten.slice` fallback in fill** — restructure overwrite
   call
4. **Multi-iteration benchmarking** — run 5+ iterations to measure
   steady-state latency (after compilation cache is warm)
