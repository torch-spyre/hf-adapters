# HF Adapters for Spyre

Minimal runtime patches that make stock [HuggingFace Transformers](https://github.com/huggingface/transformers) models run on [Spyre](https://research.ibm.com/blog/ibm-spyre) accelerators.

No forks, no custom model classes — each adapter monkey-patches the
standard HF model at load time, replacing only the operations Spyre
cannot execute natively (RoPE, RMSNorm, KV cache management, generation
loop). Everything else — weights, tokenizer, config — comes straight
from `transformers`.

## Supported Models

9 adapters covering 12 models (Granite, Granite Vision, Qwen3, Qwen2,
SmolLM3, Llama, Mistral, Phi-4, OLMo, OLMo2). All compile and run on
Spyre.

See [ARCHITECTURE.md](ARCHITECTURE.md#model-compatibility-matrix) for
the full compatibility matrix with head\_dim, stick alignment, and test
status.

## Quick Start

```python
from hf_adapters.hf_granite import load_model, generate
from transformers import AutoTokenizer

model = load_model("ibm-granite/granite-3.3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")

outputs = generate(model, tokenizer, ["What is 2+2?"], max_new_tokens=128)
print(outputs[0])
```

Replace `hf_granite` with `hf_granite_vision`, `hf_qwen3`,
`hf_granitemoehybrid`, `hf_smollm3`, `hf_llama`, `hf_qwen2`,
`hf_mistral`, or `hf_phi3` for other model families.

## Repo Structure

```
README.md
ARCHITECTURE.md                        Detailed status, architecture docs

hf_adapters/
├── hf_common.py              Shared utilities: RoPE precomputation,
│                              RMSNorm patching, LM head padding,
│                              head-dim padding, mask builders,
│                              KV cache helpers, generate loop
├── hf_granite.py              Granite 3.3 adapter
├── hf_granite_vision.py       Granite Vision 4.1 text backbone adapter
├── hf_qwen3.py                Qwen3 adapter
├── hf_granitemoehybrid.py     Granite 4.0 dense adapter
├── hf_smollm3.py              SmolLM3 adapter
├── hf_llama.py                Llama adapter (Llama 1/2/3, Code Llama, Yi, TinyLlama)
├── hf_qwen2.py                Qwen2 adapter (Qwen 1.5, Qwen 2, Qwen 2.5)
├── hf_mistral.py              Mistral adapter (Mistral 7B v0.2, v0.3)
├── hf_phi3.py                 Phi-4 mini adapter
└── __init__.py

tests/
├── test_adapter_cpu_accuracy.py       CPU: adapter vs stock HF
├── test_block_cpu_vs_spyre.py         Per-layer CPU vs Spyre comparison
├── test_e2e_smoke_spyre.py            E2E: load + generate on Spyre
└── test_e2e_token_compare_spyre.py    E2E: HF CPU vs adapter Spyre tokens
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers`
- `sentencepiece` (for some tokenizers)
- `torch_spyre` (for Spyre tests only — not needed for CPU tests)

## Running Tests

Two classes: **CPU-only** (adapter vs stock HF on CPU) and **Spyre**
(requires Spyre hardware + `torch_spyre`).

Every test script accepts model aliases as arguments. Run with no args
to test all models.

### CPU Tests (no Spyre required)

Compares adapter's patched forward pass against stock HF on CPU.
Greedy tokens must match at every step. Downloads weights on first run.

```bash
python tests/test_adapter_cpu_accuracy.py            # all models
python tests/test_adapter_cpu_accuracy.py granite     # one model
```

### Spyre Tests (requires Spyre hardware)

**Per-layer block comparison** (random weights, no download):

```bash
python tests/test_block_cpu_vs_spyre.py all
python tests/test_block_cpu_vs_spyre.py granite
```

**E2E smoke test** (real weights, verify non-trivial output):

```bash
python tests/test_e2e_smoke_spyre.py granite
```

**E2E token comparison** (HF CPU vs adapter Spyre, greedy tokens):

```bash
python tests/test_e2e_token_compare_spyre.py granite
```

Note: Spyre has known numerical accuracy limitations. Token mismatches
between CPU and Spyre are expected until torch\_spyre fixes land.

## License

Apache 2.0
