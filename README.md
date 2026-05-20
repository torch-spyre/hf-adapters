# HF Adapters for Spyre

![adapters](https://img.shields.io/badge/adapters-11-blue)
![verified](https://img.shields.io/badge/verified_checkpoints-17-green)
![compatible](https://img.shields.io/badge/compatible_models-60%2B-orange)

Minimal runtime patches that make stock [HuggingFace Transformers](https://github.com/huggingface/transformers) models run on [Spyre](https://research.ibm.com/blog/ibm-spyre) accelerators.

No forks, no custom model classes — each adapter monkey-patches the
standard HF model at load time, replacing only the operations Spyre
cannot execute natively (RoPE, RMSNorm, KV cache management, generation
loop). Everything else — weights, tokenizer, config — comes straight
from `transformers`.

## Supported Models

**11 adapters · 17 verified checkpoints · 60+ compatible models**

| Adapter | Verified | Also Compatible |
|---------|----------|-----------------|
| hf\_llama.py | Llama 3.2 3B, TinyLlama, Falcon 3 1B, DeepSeek-Coder 1.3B, Yi 1.5 6B | Llama 2/3 7–13B, Code Llama 7B/13B, Vicuna, OpenChat, Solar |
| hf\_qwen2.py | Qwen2.5 7B, 1.5B | Qwen2 0.5–7B, Qwen2.5 0.5B/3B, Qwen2.5-Coder, Qwen2.5-Math |
| hf\_granite.py | Granite 3.3 8B/2B | Granite 3.0–3.2, Granite Code 8B/3B |
| hf\_granite\_vision.py | Granite Vision 4.1 4B | — |
| hf\_qwen3.py | Qwen3 0.6B | Qwen3 1.7B, 4B, 8B |
| hf\_mistral.py | Mistral 7B v0.3 | Mistral v0.1/v0.2, Instruct variants, Zephyr 7B |
| hf\_phi3.py | Phi-4 mini | Phi-3 mini 4k/128k, Phi-3 small 8k |
| hf\_granitemoehybrid.py | Granite 4.0 1B | Granite 4.0 Micro |
| hf\_smollm3.py | SmolLM3 3B | — |
| hf\_olmo.py | OLMo 1B | OLMo 7B |
| hf\_olmo2.py | OLMo2 1B | OLMo 2 7B |

Each adapter covers all size variants and fine-tuned checkpoints sharing the same
HuggingFace `model_type`. See [ARCHITECTURE.md](ARCHITECTURE.md#verified-checkpoints)
for head\_dim details, stick alignment, and Spyre numerical accuracy.

## Quick Start

```python
from hf_adapters import AutoSpyreModelForCausalLM
from transformers import AutoTokenizer

model = AutoSpyreModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")

outputs = model.generate(tokenizer, ["What is 2+2?"], max_new_tokens=128)
print(outputs[0])
```

The `AutoSpyreModelForCausalLM` class automatically selects the correct adapter module based on the model's config type.

## Embedding Models

For embedding models, use the `sentence-transformers` library with the `backend="spyre"` parameter:

```python
import hf_adapters.st_backend  # Register Spyre backend
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
embeddings = model.encode(["hello world", "how are you"])
```

The `st_backend` module automatically patches `sentence-transformers` to apply the relevant Spyre adapter when loading the model. All standard SentenceTransformer methods (`encode()`, `similarity()`, etc.) work unchanged.
Currently just decoder-only models are supported.

## Repo Structure

```
README.md
ARCHITECTURE.md                        Detailed status, architecture docs

hf_adapters/
├── auto_spyre_model.py        Unified auto-loading interface (AutoSpyreModel, AutoSpyreModelForCausalLM)
├── hf_common.py               Shared utilities: RoPE precomputation,
│                               RMSNorm patching, LM head padding,
│                               head-dim padding, mask builders,
│                               KV cache helpers, generate loop
├── hf_granite.py               Granite 3.3 adapter
├── hf_granite_vision.py        Granite Vision 4.1 text backbone adapter
├── hf_qwen3.py                 Qwen3 adapter
├── hf_granitemoehybrid.py      Granite 4.0 dense adapter
├── hf_smollm3.py               SmolLM3 adapter
├── hf_llama.py                 Llama adapter (Llama 1/2/3, Code Llama, Yi, Falcon 3)
├── hf_qwen2.py                 Qwen2 adapter (Qwen2, Qwen2.5, Coder, Math)
├── hf_mistral.py               Mistral adapter (Mistral 7B v0.1–v0.3)
├── hf_phi3.py                  Phi-4 / Phi-3 adapter
├── hf_olmo.py                  OLMo adapter (OLMo 1B, 7B)
├── hf_olmo2.py                 OLMo2 adapter (OLMo 2 1B, 7B)
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
- `sentencepiece`
- `accelerate`
- `sentence_transformers`
- `torch_spyre` (for Spyre hardware only — not needed for CPU tests)

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

## Development

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality checks before each commit. The following hooks are configured:

- **Trailing whitespace / end-of-file fixer / mixed line endings**
- **File checks**: YAML, TOML, JSON validation; large file guard (>1 MB); merge conflict markers; debug statements
- **[Black](https://github.com/psf/black)** — code formatting
- **[Ruff](https://github.com/astral-sh/ruff-pre-commit)** — linting with auto-fix
- **[mypy](https://github.com/pre-commit/mirrors-mypy)** — static type checking (runs on `hf_adapters/` only)

### Setup

```bash
pip install -e ".[dev]"     # installs pre-commit, black, ruff, mypy as dev deps
pre-commit install          # activate hooks in your local clone
```

### Usage

Hooks run automatically on `git commit`. To run manually against all files:

```bash
pre-commit run --all-files
```

## License

Apache 2.0
