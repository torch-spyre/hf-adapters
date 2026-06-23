# HF Adapters for Spyre

![adapters](https://img.shields.io/badge/adapters-17-blue)
![verified](https://img.shields.io/badge/verified_checkpoints-31-green)
![compatible](https://img.shields.io/badge/compatible_models-100%2B-orange)

Minimal runtime patches that make stock [HuggingFace Transformers](https://github.com/huggingface/transformers) models run on [Spyre](https://research.ibm.com/blog/ibm-spyre) accelerators.

No forks, no custom model classes — each adapter monkey-patches the
standard HF model at load time, replacing only the operations Spyre
cannot execute natively (RoPE, RMSNorm, KV cache management, generation
loop). Everything else — weights, tokenizer, config — comes straight
from `transformers`.

## Supported Models

**17 adapters · 31 verified checkpoints · 100+ compatible models**

| Adapter | Verified | Also Compatible | Usage |
|---------|----------|-----------------|-------|
| hf\_llama.py | Llama 3.2 3B, TinyLlama, Falcon 3 1B, DeepSeek-Coder 1.3B, Yi 1.5 6B | Llama 2/3 7–13B, Code Llama 7B/13B, Vicuna, OpenChat, Solar | Generative |
| hf\_qwen2.py | Qwen2.5 7B, 1.5B, GTE-Qwen2-1.5B | Qwen2 0.5–7B, Qwen2.5 0.5B/3B, Qwen2.5-Coder, Qwen2.5-Math | Generative + Embedding |
| hf\_granite.py | Granite 3.3 8B/2B | Granite 3.0–3.2, Granite Code 8B/3B | Generative |
| hf\_granite\_vision.py | Granite Vision 4.1 4B | — | Generative |
| hf\_qwen3.py | Qwen3 0.6B, Qwen3-Embedding 0.6B | Qwen3 1.7B, 4B, 8B | Generative + Embedding |
| hf\_mistral.py | Mistral 7B v0.3, E5-Mistral-7B, Linq-Embed-Mistral, SFR-Embedding-Mistral | Mistral v0.1/v0.2, Instruct variants, Zephyr 7B | Generative + Embedding |
| hf\_phi3.py | Phi-4 mini | Phi-3 mini 4k/128k, Phi-3 small 8k | Generative |
| hf\_gemma3.py | Gemma 3 1B, EmbeddingGemma 300M | Gemma 3 4B/12B/27B text | Generative + Embedding |
| hf\_gemma4.py | Gemma 4 12B (fp16) | — | Generative |
| hf\_granitemoehybrid.py | Granite 4.0 1B | Granite 4.0 Micro | Generative |
| hf\_smollm3.py | SmolLM3 3B | — | Generative |
| hf\_olmo.py | OLMo 1B | OLMo 7B | Generative |
| hf\_olmo2.py | OLMo2 1B | OLMo 2 7B | Generative |
| hf\_bert.py | BGE-base-en-v1.5, all-MiniLM-L6-v2 | BERT-family encoder models | Embedding |
| hf\_xlm\_roberta.py | BGE-M3 | multilingual-e5-large, paraphrase-multilingual-mpnet-base-v2, other XLM-R fine-tunes | Embedding |
| hf\_mpnet.py | all-mpnet-base-v2 | multi-qa-mpnet-base-{dot,cos}-v1, paraphrase-mpnet-base-v2, microsoft/mpnet-base | Embedding |
| hf\_modernbert.py | ModernBERT-embed-base, GTE-ModernBERT-base, Granite-Embedding-97m-multilingual-r2 | ModernBERT-base/large, other ModernBERT embed/classifier fine-tunes | Embedding |

Each adapter covers all size variants and fine-tuned checkpoints sharing the same
HuggingFace `model_type`. See [ARCHITECTURE.md](ARCHITECTURE.md#verified-checkpoints)
for head\_dim details, stick alignment, and Spyre numerical accuracy.

## Installation

```bash
# Install core deps
uv sync

# Install core + dev deps
uv sync --group dev

# Install core + torch-spyre deps
uv sync --group spyre

# Install core + test deps
uv sync --group test

# Install everything
uv sync --group dev --group spyre --group test
```

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

Note that `model.generate()` is a modified version of the stock HF `generate()` method, with a different signature and functionality (See [docs/generate_vs_stock_hf.md](docs/generate_vs_stock_hf.md)).

## Embedding Models

For embedding models, use the `sentence-transformers` library with the `backend="spyre"` parameter:

```python
import hf_adapters.st_backend  # Register Spyre backend
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
embeddings = model.encode(["hello world", "how are you"])
```

The `st_backend` module automatically patches `sentence-transformers` to apply the relevant Spyre adapter when loading the model. All standard SentenceTransformer methods (`encode()`, `similarity()`, etc.) work unchanged.

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
├── hf_bert.py                  BERT-family encoder adapter (BGE, MiniLM)
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
├── hf_gemma3.py                Gemma 3 adapter (Gemma 3 text, EmbeddingGemma)
├── hf_gemma4.py                Gemma 4 adapter (unified text backbone)
├── hf_xlm_roberta.py           XLM-RoBERTa encoder adapter (BGE-M3, multilingual-e5)
├── hf_mpnet.py                 MPNet encoder adapter (all-mpnet-base-v2 and variants)
├── hf_modernbert.py            ModernBERT encoder adapter (RoPE, GeGLU, local/global attention)
├── st_backend.py               sentence-transformers Spyre backend (all decoder adapters)
└── __init__.py

tests/
├── test_adapter_cpu_accuracy.py       CPU: adapter vs stock HF (causal-LM)
├── test_embed_cpu_accuracy.py         CPU: embedding hidden-states vs stock HF
├── test_block_cpu_vs_spyre.py         Per-layer CPU vs Spyre comparison
├── test_e2e_smoke_spyre.py            E2E: load + generate on Spyre
├── test_e2e_token_compare_spyre.py    E2E: HF CPU vs adapter Spyre tokens
└── test_e2e_embed_compare_spyre.py    E2E: HF CPU vs adapter Spyre embeddings
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

### CPU Tests (no Spyre required)

Compares adapter's patched forward pass against stock HF on CPU.
Greedy tokens must match at every step. Downloads weights on first run.

**Important**: CPU tests must be run from the repository root with pytest to ensure proper module patching:

```bash
# Adapter accuracy tests (causal-LM logits)
uv run pytest tests/test_adapter_cpu_accuracy.py                  # all causal-LM models
uv run pytest tests/test_adapter_cpu_accuracy.py -k qwen3         # one model (manual + auto-loader)
uv run pytest tests/test_adapter_cpu_accuracy.py -k "qwen3 and manual"    # manual adapter only

# Embedding accuracy tests (hidden-states)
uv run pytest tests/test_embed_cpu_accuracy.py                    # all embedding models
uv run pytest tests/test_embed_cpu_accuracy.py -k bge_base        # one model

# Load tests (verify models load without errors)
uv run pytest tests/test_load_cpu.py                              # CPU load test
uv run pytest tests/test_load_spyre.py                            # Spyre load test (requires hardware)
```

**Note**: Do not run CPU tests with `python tests/test_*.py` — this bypasses pytest's conftest.py setup and will cause import errors. Always use `pytest` (or `uv run pytest`).

### Spyre Tests (requires Spyre hardware)

**Per-layer block comparison** (random weights, no download):

```bash
python tests/test_block_cpu_vs_spyre_DEPRECATED.py all
python tests/test_block_cpu_vs_spyre_DEPRECATED.py granite
```

**E2E smoke test** (real weights, verify non-trivial output):

```bash
python tests/test_e2e_smoke_spyre.py granite
```

**E2E token comparison** (HF CPU vs adapter Spyre, greedy tokens):

```bash
python tests/test_e2e_token_compare_spyre.py granite
```

**E2E embedding comparison** (HF CPU vs adapter Spyre, hidden-states cosine):

```bash
python tests/test_e2e_embed_compare_spyre.py bge-base
python tests/test_e2e_embed_compare_spyre.py minilm
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
uv sync --group dev

pre-commit install          # activate hooks in your local clone
```

### Usage

Hooks run automatically on `git commit`. To run manually against all files:

```bash
pre-commit run --all-files
```

## License

Apache 2.0
