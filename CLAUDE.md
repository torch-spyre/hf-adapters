# CLAUDE.md

## Project

HF Adapters for Spyre — runtime monkey-patches that make stock HuggingFace Transformers models run on IBM Spyre accelerators. No forks, no custom model classes.

## Build & Test

```bash
# Install (editable)
pip install -e .

# Module config tests (on pod — uses OOT framework, requires torch_spyre + oot_framework)
# First time: uv sync --group oot
bash tests/run_oot_module_configs.sh tests/configs/module_tests/granite_3_3_8b_instruct_spyre.yaml -v
bash tests/run_oot_module_configs.sh tests/configs/module_tests/  # run all configs

# Spyre tests (on pod only — requires torch_spyre)
python3 tests/test_e2e_smoke_spyre.py qwen3
python3 tests/test_e2e_token_compare_spyre.py qwen3
# Spyre tests (on pod only — requires torch_spyre). Pytest-parametrized off the
# model registry; select one model with -k <key>, or run the file for all.
pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k qwen3
pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k qwen3
pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py -k bge_base  # Text embedder
pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm   # multimodal VLM
pytest -s -vvv tests/spyre/test_load_spyre.py    # Spyre load test
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
- [ ] Registry entry in `tests/model_registry.py`
- [ ] Compiles + runs end-to-end on Spyre (`tests/spyre/test_e2e_*_spyre.py`, no crash/NaN)
- [ ] Added to the Verified Checkpoints + Model Family Coverage tables in `ARCHITECTURE.md` (the single source of truth; bump the README badge counts)

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

### Gated models
Llama models require HF authentication. HF token is configured on the Spyre pod.

## Current Work

Epic issue #4 tracks adapters for 73 dense full-attention models <10B. Priority:
1. Llama (model_type=llama, covers 16+ models including Yi, CodeLlama)
2. Qwen2 (model_type=qwen2, like Qwen3 but without Q/K RMSNorm)
3. Mistral (model_type=mistral, trivial given Llama)
4. Phi-4 unblock (partial_rotary_factor=0.75 needs compiler fix — head_dim=128 is fine)
5. InternLM2/3, OLMo/OLMo2
6. Non-RoPE: BLOOM, MPT, OPT, GPT-2
