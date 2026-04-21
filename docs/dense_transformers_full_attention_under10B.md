# Dense Full-Attention Transformers Under 10B — Model Catalog

Comprehensive catalog of dense, decoder-only transformer models with full attention in every layer, under 10B parameters, available on HuggingFace Hub.

**Criteria:**
- Dense transformer (no MoE)
- Full attention in every layer (no sliding window, no SSM/Mamba, no hybrid)
- Under 10B parameters (at least one variant)
- Causal/autoregressive (decoder-only)
- Available on HuggingFace Hub

## Models Already Tracked in Issue #4

### Tier 1: Adapters Done

| Adapter | model_type | Models | head_dim | Spyre |
|---------|-----------|--------|----------|-------|
| `hf_qwen3.py` | `qwen3` | Qwen3 0.6B, 1.7B, 4B, 8B | 128 | Yes |
| `hf_granite.py` | `granite` | Granite 3.3 8B | 128 | Yes |
| `hf_granitemoehybrid.py` | `granitemoehybrid` | Granite 4.0 1B, Micro | 128 | Yes |
| `hf_smollm3.py` | `smollm3` | SmolLM3 3B | 128 | Yes |

### Tier 2–5: Planned in Issue #4

| model_type | Models | head_dim | Status |
|-----------|--------|----------|--------|
| `llama` | Llama 1/2/3, Code Llama, Yi, TinyLlama | 128 (64 for 1B/tiny) | Tier 2 |
| `qwen2` | Qwen 1.5/2/2.5 (0.5B–7B) | 128 (64 for 0.5B) | Tier 2 |
| `mistral` | Mistral 7B v0.2/v0.3 | 128 | Tier 2 |
| `internlm2` | InternLM2 1.8B, 7B | 128 | Tier 2 |
| `internlm3` | InternLM3 8B | 128 | Tier 2 |
| `olmo` | OLMo 1B, 7B | 128 | Tier 2 |
| `olmo2` | OLMo 2 7B | 128 | Tier 2 |
| `phi3` | Phi-4 mini 3.8B | 128 | Tier 2 (blocked on partial_rotary_factor) |
| `bloom` | BLOOM 560M–7.1B | varies | Tier 3 (ALiBi) |
| `mpt` | MPT 7B | varies | Tier 3 (ALiBi) |
| `opt` | OPT 1.3B–6.7B | 64 | Tier 3 (learned pos emb) |
| `gpt2` | GPT-2 XL 1.6B | 64 | Tier 3 (learned pos emb) |
| — | Granite 3.3 2B, Llama 3.2 1B, TinyLlama, Qwen2-0.5B, Falcon 7B, StableLM 2 1.6B | 64 | Tier 4 (blocked, sub-stick) |
| — | OpenELM, Pythia, GPT-J, Cerebras-GPT, RedPajama, DeepSeek v1, Baichuan2, Tiny Aya 3B | varies | Tier 5 (needs investigation) |

## NEW — Models Missing from Issue #4

### Spyre-Compatible (head_dim >= 128, RoPE)

| Model | model_type | Sizes <10B | head_dim | KV Heads | Pos. Encoding | Notes |
|-------|-----------|-----------|----------|----------|--------------|-------|
| **Gemma 1** | `gemma` | 2B, 7B | 256 | 2B: 1 (MQA), 7B: 16 (MHA) | RoPE | Gated access. New adapter needed. |
| **CodeGemma** | `gemma` | 7B | 256 | 16 | RoPE | Gemma 1 fine-tune for code. Same adapter as Gemma 1. |
| **Falcon 3** | `llama` | 1B, 3B, 7B, 10B | 256 | 4 (GQA) | RoPE | Uses `model_type=llama`. May work with Llama adapter. |
| **DeepSeek v1** | `llama` | 7B | 128 | 32 (MHA) | RoPE | Standard Llama arch. Covered by Llama adapter. |
| **DeepSeek-Coder v1** | `llama` | 1.3B, 6.7B | 128 | 32 (MHA) | RoPE | Standard Llama arch. Covered by Llama adapter. |
| **GLM-4-9B** | `chatglm` | 9B | 128 | 2 (MQA) | RoPE | Needs `trust_remote_code`. New adapter needed. |
| **Aquila2-7B** | `aquila` | 7B | 128 | 32 (MHA) | RoPE (theta=1M) | New adapter needed. |
| **MAP-Neo 7B** | `llama` | 7B | 128 | 16 (MHA) | RoPE | Llama-based. Covered by Llama adapter. |
| **Amber 6.7B** | `llama` | 6.7B | 128 | 32 (MHA) | RoPE | Llama-based. Covered by Llama adapter. |
| **Pythia** | `gpt_neox` | 70M–6.9B | 128 | 32 (MHA) | Partial RoPE (25%) | New adapter needed. |
| **RedPajama-INCITE** | `gpt_neox` | 3B, 7B | 128 | 32 (MHA) | RoPE (100%) | GPT-NeoX arch. Shares adapter with Pythia. |
| **Ministral 3B** | `mistral` | 3B | 128 | 8 (GQA) | RoPE (YaRN, 256K ctx) | Full attention (`sliding_window: null`). Covered by Mistral adapter. |
| **GPT-J 6B** | `gptj` | 6B | 256 | 16 (MHA) | Partial RoPE (25%) | New adapter needed. |

### Non-RoPE (New)

| Model | model_type | Sizes <10B | head_dim | Pos. Encoding | Notes |
|-------|-----------|-----------|----------|--------------|-------|
| **Baichuan2-7B** | `baichuan` | 7B | 128 | Absolute (learned) | Needs `trust_remote_code`. New adapter needed. |
| **Cerebras-GPT** | `gpt2` | 111M–6.7B | 128 | Absolute (GPT-2 style) | Shares adapter with GPT-2. |

### Blocked on Spyre (head_dim < 128)

| Model | model_type | head_dim | head_dim/2 | Blocker |
|-------|-----------|----------|-----------|---------|
| **Phi-2 2.7B** | `phi` | 80 | 40 | Sub-stick + partial_rotary_factor=0.4 |
| **Phi-1.5 1.3B** | `phi` | 64 | 32 | Sub-stick + partial_rotary_factor=0.5 |
| **Phi-1 1.3B** | `phi` | 64 | 32 | Sub-stick |
| **SmolLM2 1.7B** | `llama` | 64 | 32 | Sub-stick (Llama adapter) |
| **SmolLM2 360M** | `llama` | 64 | 32 | Sub-stick (Llama adapter) |
| **SmolLM2 135M** | `llama` | 64 | 32 | Sub-stick (Llama adapter) |
| **Persimmon-8B** | `persimmon` | 64 | 32 | Sub-stick + QK layer norm |
| **StableLM 2 1.6B** | `stablelm` | 64 | 32 | Sub-stick + partial_rotary_factor=0.25 |

### Needs Investigation

| Model | model_type | Notes |
|-------|-----------|-------|
| **Cohere Aya-23-8B** | `cohere` | Gated. Custom `CommandForCausalLM`. head_dim/pos encoding TBD. |
| **MiniCPM 2B** | `minicpm` | Custom scaling params (scale_emb, scale_depth). RoPE. Full attention. |
| **XVERSE-7B** | `xverse` | RoPE, full MHA. Less widely used. |
| **CodeGen 2B/6B** | `codegen` | Partial RoPE (rotary_dim=64). Full attention. Older code model. |
| **SantaCoder** | `gpt_bigcode` | MQA. head_dim=128. Older code model. |
| **OpenELM 3B** | `openelm` | Variable GQA per layer (num_heads changes layer-by-layer). |

## Models That Do NOT Qualify

| Model | Reason |
|-------|--------|
| Gemma 2 (2B, 9B) | Alternating sliding window (4096) + full attention layers |
| Gemma 3 (1B, 4B, 12B, 27B) | 5:1 local (sliding window 1024) / global attention ratio |
| Gemma 4 (1B, 4B) | Hybrid sliding window + global attention |
| Ministral 8B | Interleaved sliding window (3 sliding : 1 full per 4-layer cycle). Note: Ministral 3B DOES qualify (full attention). |
| Mistral 7B v0.1 | sliding_window=4096 |
| StarCoder2 (3B, 7B) | sliding_window=4096 |
| RWKV | RNN architecture, not transformer |
| RecurrentGemma | Griffin/RNN architecture |
| Jamba | Hybrid SSM-Transformer-MoE |
| Nemotron-H 8B | Hybrid Mamba2 + attention |
| Falcon Mamba 7B | SSM, not transformer |
| Cohere Command R | >10B only (35B+) |
| Skywork-13B | >10B only |

## Fine-Tunes (No Separate Adapter Needed)

These share `model_type` with their base model and are covered by existing adapters:

| Fine-Tune | Base model_type |
|-----------|----------------|
| Zephyr-7B | `mistral` |
| GritLM-7B | `mistral` |
| Nous-Hermes-2-Mistral-7B | `mistral` |
| Neural-Chat-7B | `mistral` |
| Nous-Hermes-2-Llama-2-7B | `llama` |
| Vicuna, Alpaca, etc. | `llama` |

## Net New Adapters Needed (Beyond Issue #4)

| Priority | Adapter | model_type | Models Covered | head_dim | Spyre? |
|----------|---------|-----------|---------------|----------|--------|
| High | `hf_gemma.py` | `gemma` | Gemma 1 2B/7B, CodeGemma 7B | 256 | Yes |
| Medium | `hf_gpt_neox.py` | `gpt_neox` | Pythia 70M–6.9B, RedPajama 3B/7B | 128 | Yes |
| Medium | `hf_chatglm.py` | `chatglm` | GLM-4-9B | 128 | Yes |
| Medium | `hf_aquila.py` | `aquila` | Aquila2-7B | 128 | Yes |
| Low | `hf_gptj.py` | `gptj` | GPT-J 6B | 256 | Yes |
| Low | `hf_baichuan.py` | `baichuan` | Baichuan2-7B | 128 | Likely |
| Low | `hf_cohere.py` | `cohere` | Aya-23-8B | TBD | TBD |
| Low | `hf_minicpm.py` | `minicpm` | MiniCPM 2B | TBD | TBD |
| Blocked | `hf_phi.py` (expand) | `phi` | Phi-1, Phi-1.5, Phi-2 | 64–80 | No |
| Blocked | `hf_persimmon.py` | `persimmon` | Persimmon-8B | 64 | No |
| Blocked | `hf_stablelm.py` | `stablelm` | StableLM 2 1.6B | 64 | No |

**Note:** Falcon 3 uses `model_type=llama` — verify whether `hf_llama.py` handles it without modification (head_dim=256 instead of 128 is the main difference).
