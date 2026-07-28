# Adapter Changes by Model Family

This document enumerates the specific changes each adapter applies to stock HuggingFace Transformers models to make them run on Spyre accelerators.

---

## Common Changes (All Adapters)

Every adapter applies the following transformations via shared utilities in `hf_common.py`:

| Change | Stock HuggingFace | Spyre Adapter |
|--------|------------------|---------------|
| **RoPE** | Computed on-the-fly per layer, element-wise sin/cos | Precomputed rotation matrices `[S, 2, 2, D/2]` on CPU, applied via matrix multiplication |
| **RMSNorm / LayerNorm** | Casts to float32 for numerical stability, then back | Stays in fp16 on Spyre (no dtype conversion allowed); uses float32 on CPU only |
| **LM head** | Original vocab dimension | Zero-padded to a stick-aligned, "smooth" vocab whose stick count has no large prime factor, so the per-core matmul span fits Spyre's 256 MB EAR limit (`pad_lm_head`) |
| **KV cache** | Dynamic, append-based (`DynamicCache`) | Pre-allocated at full size, written at a specific offset via native slice assignment (`cache[:, :, pos:pos+seq_len, :] = k`) |
| **Layer compilation** | Not compiled for inference | Each decoder layer compiled independently via `torch.compile(dynamic=False)` |
| **Generation loop** | 1 token per decode iteration | 64-token padded blocks: prefill → expand by 64 → fill 63 tokens → repeat |
| **Attention masks** | Boolean or simple format | Explicit float16 `[B, 1, L, S]` tensors with `-inf` for masked positions |
| **Input padding** | Variable-length | Left-padded to multiples of 64 (BLOCK_SIZE) |

---

## Per-Adapter Changes

### Llama (`hf_llama.py`)

**model\_type:** `llama`

**Unique changes:** None — applies only the common changes listed above. This is the simplest adapter pattern.

---

### Mistral (`hf_mistral.py`)

**model\_type:** `mistral`

**Unique changes:** None — architecturally identical to Llama. Applies only the common changes.

---

### Qwen2 (`hf_qwen2.py`)

**model\_type:** `qwen2`

**Unique changes:** None — same standard GQA architecture as Llama/Mistral. Applies only the common changes.

---

### Qwen3 (`hf_qwen3.py`)

**model\_type:** `qwen3`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Per-head Q/K RMSNorm** | Applies `q_norm` and `k_norm` to Q and K projections per-head before RoPE. Stock HF does this too, but the compiled block must inline it explicitly. |

**Not shared with others:** The Q/K normalization step before RoPE is unique to Qwen3 among the adapters.

---

### Granite (`hf_granite.py`)

**model\_type:** `granite`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Embedding multiplier** | Multiplies embedding output by `config.embedding_multiplier` |
| **Residual multiplier** | Multiplies both attention and MLP residual connections by `config.residual_multiplier` |
| **Logits scaling** | Divides final logits by `config.logits_scaling` |
| **Head-dim padding** | Pads Q/K/V/O projections when `head_dim // 2 < 64` (e.g., Granite 3.3 2B: 64→128) |

**Shared with Granite Vision and Granite 4.0:** The embedding/residual/logits multipliers are common to all Granite-family adapters.

---

### Granite Vision (`hf_granite_vision.py`)

**model\_type:** `granite` (text backbone only)

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Text backbone extraction** | Extracts text-only weights from multimodal checkpoint via safetensor key remapping (`model.language_model.*` → `model.*`) |
| **Custom model loading** | Loads into standard `GraniteForCausalLM` with `strict=False`; discards vision encoder and projection layers entirely |

**Reuses from Granite:** All forward pass changes (multipliers, head-dim padding) come from the Granite adapter directly via import.

---

### Granite 4.0 / MoE Hybrid (`hf_granitemoehybrid.py`)

**model\_type:** `granitemoehybrid`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Fused MLP weight splitting** | Splits fused `input_linear` weight `[2*intermediate, hidden]` into separate `gate_proj` and `up_proj` at prepare time. Spyre's stickify pass cannot handle non-zero offsets from fused weights. |
| **Embedding multiplier** | Same as Granite 3.3 |
| **Residual multiplier** | Same as Granite 3.3 |
| **Logits scaling** | Same as Granite 3.3 |

**Shared with Granite:** The multiplier pattern is identical. The fused MLP splitting is shared conceptually with Phi-3/4 (which also splits fused weights).

---

### Phi-3 / Phi-4 (`hf_phi3.py`)

**model\_type:** `phi3`

**Unique changes (most complex adapter):**

| Change | Description |
|--------|-------------|
| **Fused QKV splitting** | Splits combined `qkv_proj` into separate `q_proj`, `k_proj`, `v_proj` at prepare time |
| **Fused gate\_up splitting** | Splits combined `gate_up_proj` into separate `gate_proj` and `up_proj` at prepare time |
| **Partial RoPE** | Only rotates `partial_rotary_factor` of head\_dim (e.g., 0.75). Pads rotation matrix with identity entries so non-rotated dimensions pass through unchanged |
| **RoPE dimension permutation** | HF pairs `(j, j+rope_dim//2)` but Spyre's `apply_rope_matmul` pairs `(j, j+head_dim//2)`. Builds and applies permutation to Q/K weights at prepare time |
| **Chunked LM head** | Fallback (`chunk_lm_head`, currently unused) for when even a smooth-padded single head can't fit the 256 MB EAR limit: splits lm\_head into N chunks run separately on Spyre and concatenated on CPU. Not needed by any current model — smooth `pad_lm_head` handles 200K–262K vocab in one head |

**Shared with Granite 4.0:** The concept of fused weight splitting is shared, though the specific weights differ (QKV + gate\_up vs. input\_linear).

---

### SmolLM3 (`hf_smollm3.py`)

**model\_type:** `smollm3`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Conditional per-layer RoPE (NoPE)** | Some layers skip RoPE entirely based on `config.no_rope_layers`. The compiled block takes a `use_rope` flag; when False, RoPE is not applied to Q/K for that layer. |

**Note:** `config.no_rope_layers[i] = 1` means USE RoPE (counterintuitive naming from HF config).

---

### OLMo (`hf_olmo.py`)

**model\_type:** `olmo`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Weight-free LayerNorm** | OLMo uses LayerNorm without learnable weight/bias parameters (just `F.layer_norm` with eps). Requires a custom fp16 patch distinct from RMSNorm. |
| **Head-dim padding** | Same stick-alignment logic as Granite (pads if `head_dim // 2 < 64`) |

**Shared with Llama/Mistral/Qwen2:** Uses the same forward pass logic (standard GQA). The only difference is the norm class.

---

### OLMo2 (`hf_olmo2.py`)

**model\_type:** `olmo2`

**Unique changes:**

| Change | Description |
|--------|-------------|
| **Post-norm architecture** | Norm applied AFTER attention/MLP, before residual add. Pattern: `residual + post_ln(attn_out)` instead of pre-norm `attn(pre_ln(residual)) + residual` |
| **Q/K RMSNorm on flattened projections** | Like Qwen3's per-head norm, but applied to the flattened projection output BEFORE reshape to multi-head format |
| **Head-dim padding** | Same stick-alignment logic as Granite/OLMo |

**Difference from Qwen3:** Qwen3 normalizes after reshaping to heads; OLMo2 normalizes the flat projection then reshapes.

---

## Feature Matrix

| Feature | Llama | Mistral | Qwen2 | Qwen3 | Granite | Granite Vision | Granite 4.0 | Phi-3/4 | SmolLM3 | OLMo | OLMo2 |
|---------|:-----:|:-------:|:-----:|:-----:|:-------:|:--------------:|:-----------:|:-------:|:-------:|:----:|:-----:|
| Standard GQA (shared code) | ✓ | ✓ | ✓ | — | — | — | — | — | — | ✓ | — |
| Head-dim padding | auto | auto | auto | — | ✓ | ✓ | — | — | — | auto | auto |
| Embedding multiplier | — | — | — | — | ✓ | ✓ | ✓ | — | — | — | — |
| Residual multiplier | — | — | — | — | ✓ | ✓ | ✓ | — | — | — | — |
| Logits scaling | — | — | — | — | ✓ | ✓ | ✓ | — | — | — | — |
| Per-head Q/K norm | — | — | — | ✓ | — | — | — | — | — | — | ✓ |
| Fused weight splitting | — | — | — | — | — | — | ✓ (MLP) | ✓ (QKV+MLP) | — | — | — |
| Partial RoPE | — | — | — | — | — | — | — | ✓ | — | — | — |
| Conditional RoPE (NoPE) | — | — | — | — | — | — | — | — | ✓ | — | — |
| Post-norm architecture | — | — | — | — | — | — | — | — | — | — | ✓ |
| Weight-free LayerNorm | — | — | — | — | — | — | — | — | — | ✓ | — |
| Text backbone extraction | — | — | — | — | — | ✓ | — | — | — | — | — |

---

## DSpark Speculative-Decoding Drafters (`hf_dspark_qwen3`, `hf_dspark_gemma4`, `hf_dspark_granite`)

DSpark drafters are EAGLE-3-style **block proposers**, not causal-LMs: given the
target model's intermediate hidden states (concatenated over `target_layer_ids`)
plus a masked noise block, they propose a whole block of tokens in one shot. The
adapter's public forward is therefore `_run_draft_block` (context features + noise
block → block hidden states), consumed by DeepSpec's `build_dspark_proposal`, not
the token-by-token `generate` loop. Routing is by **architecture name**
(`*DSparkModel`) via `ARCH_TO_ADAPTER_MODULE_MAPPING`, since drafters reuse their
base model's config class.

| Change | Stock DSpark drafter | Spyre adapter |
|--------|----------------------|---------------|
| **Attention** | `K`/`V` = concat of projected context K/V and noise-block K/V; non-causal within the block | Same, at a **fixed** stick-aligned `ctx_pad + block` width with a CPU-built additive `-inf` mask over the context zero-pad and kv tail (`build_ctx_block_mask`); no per-token KV cache |
| **RoPE** | element-wise per layer | Precomputed rotation matrices applied by matmul (`apply_rope_matmul`), sliced on the sequence axis for the concat-KV window |
| **Context projection `fc`** | `hidden_norm(fc(ctx))` on device | Run on CPU (fp32) — the wide reduction dim (`num_layers·hidden`) has no Spyre matmul layout for a non-canonical (e.g. cross-card) context tensor |
| **LM head / markov `w2`** | full-vocab matmuls | Stick-padded (`pad_lm_head`); `compute_logits` sliced back to the true vocab and family softcap / logits-scaling reapplied |
| **Markov head** | `sample_block_tokens` runs `markov_w1` (gather) on device | Sample loop on host; per-step bias via a CPU-snapshot `markov_w1` gather + the padded on-device `markov_w2` |
| **Token embedding** | `nn.Embedding` on device | CPU gather against a weight snapshot, moved to device (with the family `embedding_multiplier`) |
| **Confidence head** | trims the block to a confident prefix | Disabled at `confidence_threshold=0` (full block kept) — its path also runs the device `markov_w1` against CPU ids |

Per-family specifics mirror the corresponding *target* adapters: Qwen 3 has
per-head Q/K RMSNorm; Granite carries the attention / embedding / residual
multipliers and logits scaling; Gemma 4 has its own four-LayerNorm block with a
per-layer `layer_scalar`, scale-free V-norm, and `attention_k_eq_v` V-aliasing.
