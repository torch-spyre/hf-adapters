# CLAUDE.md

## Project

HF Adapters for Spyre — runtime monkey-patches that make stock HuggingFace Transformers models run on IBM Spyre accelerators. No forks, no custom model classes.

## Build & Test

```bash
# Install (editable)
pip install -e .

# CPU accuracy tests (causal-LM logits + embedding hidden-states)
# IMPORTANT: Always use pytest from repo root, never `python tests/test_*.py`
# conftest.py patches hf_adapters modules at collection time for CPU testing
pytest tests/test_adapter_cpu_accuracy.py
pytest tests/test_adapter_cpu_accuracy.py -k qwen3   # one model (selects both paths)
pytest tests/test_adapter_cpu_accuracy.py -k "qwen3 and manual"  # one path
pytest tests/test_embed_cpu_accuracy.py

# Load tests (verify models load without errors)
pytest tests/test_load_cpu.py      # CPU load test
pytest tests/test_load_spyre.py    # Spyre load test (requires hardware)

# Spyre tests (on pod only — requires torch_spyre)
python3 tests/test_e2e_smoke_spyre.py qwen3
python3 tests/test_e2e_token_compare_spyre.py qwen3
```

## Spyre Pod

- Namespace: `a5-deepview`
- Connect: `kubectl exec -it -n $NS $POD -- bash -l`
- Run command: `kubectl exec -n $NS $POD -- bash -lc "<cmd>"`
- Copy project: `kubectl cp . $NS/$POD:$HOME/hf_adapters`
- PYTHONPATH must include the hf_adapters project root (not pip-installed on pod)

## Writing a New Adapter

Every adapter follows the same pattern (see `hf_granite.py` as the canonical example):

1. `_make_compiled_block(layer)` — closure over layer weights, returns `torch.compile(block_forward, dynamic=False)`
2. `_run_forward(model, ...)` — embedding → RoPE → N compiled blocks → final norm → lm_head
3. `prepare_for_spyre(model)` — patches RMSNorm, creates PrecomputedRotaryEmbedding, pads LM head, compiles blocks
4. `load_model(model_path, dtype)` / `generate(model, tokenizer, prompts, **kwargs)` — thin wrappers

Import shared utilities from `hf_common.py`: `PrecomputedRotaryEmbedding`, `apply_rope_matmul`, `kv_cache_update`, `patch_rmsnorm`, `pad_lm_head`, `pad_attention_heads`, `load_model_common`, `generate`.

### Definition of Done (per adapter)

- [ ] Adapter file in `hf_adapters/`
- [ ] CPU accuracy test passes (identical greedy tokens vs stock HF)
- [ ] Per-layer block comparison on Spyre (compiles, no NaN)
- [ ] Added to model compatibility matrix in `ARCHITECTURE.md`
- [ ] Registry entries in all test files
- [ ] At least one model size tested end-to-end on Spyre

## Critical Rules

### head_dim and stick alignment
`head_dim / 2` must be >= 64 (one Spyre stick) for the RoPE matmul to compile. Models with smaller head_dim can use `pad_attention_heads()` to zero-pad Q/K/V/O projections and RoPE frequencies to a stick-aligned size (e.g. Granite 3.3 2B: 64→128). Always check before Spyre testing:
```python
config = AutoConfig.from_pretrained("org/model")
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
if head_dim // 2 < 64:
    # Use pad_attention_heads() — see hf_granite.py for example
```

### Test ordering matters
Run HF reference forward BEFORE calling `prepare_for_spyre()`. The RMSNorm patch modifies the class globally.

### Zero-length tensors crash Spyre
Create empty KV caches with `device=` parameter, never `.to("spyre")` on a zero-length tensor.

### Unwrap torch.compile for CPU tests
Use `getattr(compiled_block, "_orig_mod", compiled_block)` to skip compilation overhead in CPU-only test paths.

### Gated models
Llama models require HF authentication. Use non-gated alternatives for CPU tests (e.g., TinyLlama for model_type=llama). HF token is configured on the Spyre pod.

## Current Work

Epic issue #4 tracks adapters for 73 dense full-attention models <10B. Priority:
1. Llama (model_type=llama, covers 16+ models including Yi, CodeLlama)
2. Qwen2 (model_type=qwen2, like Qwen3 but without Q/K RMSNorm)
3. Mistral (model_type=mistral, trivial given Llama)
4. Phi-4 unblock (partial_rotary_factor=0.75 needs compiler fix — head_dim=128 is fine)
5. InternLM2/3, OLMo/OLMo2
6. Non-RoPE: BLOOM, MPT, OPT, GPT-2
