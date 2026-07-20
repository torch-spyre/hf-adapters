# HF Adapters — Model Coverage

## Summary

| # | Adapter | model\_type | Verified Checkpoints | Compatible Models |
|---|---------|------------|---------------------|-------------------|
| 1 | hf\_llama.py | `llama` | 5 | 14+ |
| 2 | hf\_qwen2.py | `qwen2` | 2 | 10+ |
| 3 | hf\_granite.py | `granite` | 2 | 6+ |
| 4 | hf\_granite\_vision.py | `granite` (text backbone) | 1 | — |
| 5 | hf\_qwen3.py | `qwen3` | 1 | 3+ |
| 6 | hf\_mistral.py | `mistral` | 1 | 4+ |
| 7 | hf\_phi3.py | `phi3` | 1 | 3+ |
| 8 | hf\_granitemoehybrid.py | `granitemoehybrid` | 2 | 1+ |
| 9 | hf\_smollm3.py | `smollm3` | 1 | — |
| 10 | hf\_olmo.py | `olmo` | 1 | 1+ |
| 11 | hf\_olmo2.py | `olmo2` | 1 | 1+ |

---

## 1. hf\_llama.py

**HF model\_type:** `llama`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Llama 3.2 3B | 128 | — |
| TinyLlama 1.1B | 64→128 | Padded via `pad_attention_heads()` |
| Falcon 3 1B | 256 | — |
| DeepSeek-Coder 1.3B | 128 | — |
| Yi 1.5 6B | 128 | — |

### Compatible Models

- Llama 2 7B / 13B
- Llama 3 8B
- Llama 3.1 8B
- Code Llama 7B / 13B
- Vicuna 7B / 13B
- OpenChat 3.5 7B
- Nous Hermes 2
- Solar 10.7B
- Any model registering as `model_type=llama` in HF Transformers

---

## 2. hf\_qwen2.py

**HF model\_type:** `qwen2`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Qwen2.5 7B | 128 | — |
| Qwen2.5 1.5B | 128 | — |

### Compatible Models

- Qwen 1.5 0.5B / 1.5B / 7B
- Qwen 2 0.5B–7B
- Qwen 2.5 0.5B / 3B
- Qwen2.5-Coder 0.5B–7B
- Qwen2.5-Math 1.5B / 7B

---

## 3. hf\_granite.py

**HF model\_type:** `granite`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Granite 3.3 8B | 128 | — |
| Granite 3.3 2B | 64→128 | Padded via `pad_attention_heads()` |

### Compatible Models

- Granite 3.0 8B
- Granite 3.1 8B / 2B
- Granite 3.2 8B
- Granite Code 8B / 3B

---

## 4. hf\_granite\_vision.py

**HF model\_type:** `granite` (text backbone only)

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Granite Vision 4.1 4B | 64→128 | Padded; text backbone extracted from multimodal checkpoint |

### Compatible Models

Vision-specific adapter; extracts and adapts the text backbone from Granite Vision multimodal checkpoints via safetensor key remapping.

---

## 5. hf\_qwen3.py

**HF model\_type:** `qwen3`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Qwen3 0.6B | 128 | — |

### Compatible Models

- Qwen3 1.7B
- Qwen3 4B
- Qwen3 8B

---

## 6. hf\_mistral.py

**HF model\_type:** `mistral`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Mistral 7B v0.3 | 128 | — |

### Compatible Models

- Mistral 7B v0.1 / v0.2
- Mistral 7B Instruct v0.1–v0.3
- Zephyr 7B

---

## 7. hf\_phi3.py

**HF model\_type:** `phi3`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Phi-4 mini | 128 | Partial RoPE (`partial_rotary_factor=0.75`), chunked LM head (vocab > 200K) |

### Compatible Models

- Phi-3 mini 4k / 128k
- Phi-3 small 8k

---

## 8. hf\_granitemoehybrid.py

**HF model\_type:** `granitemoehybrid`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| Granite 4.0 1B Base (`granite-4.0-1b-base`) | 128 | Dense variant only (`num_local_experts=1`, no Mamba) |
| Granite 4.0 1B Instruct (`granite-4.0-1b`) | 128 | Dense variant only (`num_local_experts=1`, no Mamba) |

### Compatible Models

- Granite 4.0 Micro

---

## 9. hf\_smollm3.py

**HF model\_type:** `smollm3`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| SmolLM3 3B | 128 | Conditional per-layer RoPE (NoPE layers) |

### Compatible Models

No additional compatible models listed.

---

## 10. hf\_olmo.py

**HF model\_type:** `olmo`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| OLMo 1B | 128 | Weight-free LayerNorm (no learnable parameters) |

### Compatible Models

- OLMo 7B

---

## 11. hf\_olmo2.py

**HF model\_type:** `olmo2`

### Verified Checkpoints

| Checkpoint | head\_dim | Notes |
|-----------|---------|-------|
| OLMo2 1B | 128 | Post-norm architecture (norm after attention/MLP) |

### Compatible Models

- OLMo 2 7B
