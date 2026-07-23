# HF Adapters for Spyre

![adapters](https://img.shields.io/badge/adapters-27-blue)
![verified](https://img.shields.io/badge/verified_checkpoints-45-green)
![compatible](https://img.shields.io/badge/compatible_models-100%2B-orange)

Minimal runtime patches that make stock [HuggingFace Transformers](https://github.com/huggingface/transformers) models run on [Spyre](https://research.ibm.com/blog/ibm-spyre) accelerators.

No forks, no custom model classes — each adapter monkey-patches the
standard HF model at load time, replacing only the operations Spyre
cannot execute natively (RoPE, RMSNorm, KV cache management, generation
loop). Everything else — weights, tokenizer, config — comes straight
from `transformers`.

## Supported Models

**27 adapters · 45 verified checkpoints · 100+ compatible models**

Coverage spans **generative** (causal-LM), **embedding** (sentence-transformers),
and **vision-language** (image→text) models — from Llama / Qwen / Granite / Mistral /
Phi / Gemma / OLMo / GPT decoders to BERT / XLM-RoBERTa / MPNet / ModernBERT
encoders and the Granite Vision 4.1 (SigLIP tower + Granite text), Mistral3 Vision
(Pixtral tower + Mistral text), and Gemma 4 (encoder-free) multimodal VLMs.

Each adapter covers all size variants and fine-tuned checkpoints sharing the same
HuggingFace `model_type`. The **canonical, per-adapter model lists** — verified
checkpoints, also-compatible models, `head_dim` / stick-alignment details, and
Spyre numerical accuracy — live in **[ARCHITECTURE.md](ARCHITECTURE.md#verified-checkpoints)**.

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

## Multimodal Models (image → text)

For vision-language models, use `AutoSpyreModelForImageTextToText`. It loads the
full VLM via `AutoModelForImageTextToText`, prepares **both** towers (vision +
text decoder) for Spyre, and exposes a multimodal `generate`:

```python
from hf_adapters import AutoSpyreModelForImageTextToText
from transformers import AutoProcessor
from PIL import Image

# --- Granite Vision 4.1 ---
model = AutoSpyreModelForImageTextToText.from_pretrained("ibm-granite/granite-vision-4.1-4b")
processor = AutoProcessor.from_pretrained("ibm-granite/granite-vision-4.1-4b")
processor.tokenizer.padding_side = "left"  # matches the decode loop's right-aligned prompts

# Build the batch the official way — the chat template tokenizes and expands the
# image tokens in one call (the two-step text/images path mis-tiles anyres images).
image = Image.open("cat.jpg").convert("RGB")
conv = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Briefly describe this image."},
]}]
batch = processor.apply_chat_template(
    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)

texts = model.generate(
    processor,
    batch["input_ids"], batch["attention_mask"],
    batch["pixel_values"], batch["image_sizes"],
    max_new_tokens=64,
)
print(texts[0])

```

A multimodal checkpoint's config is registered under both auto classes:
`AutoSpyreModelForCausalLM` selects the text-only adapter (vision tower
discarded), while `AutoSpyreModelForImageTextToText` selects the combined
multimodal adapter. This works for Granite Vision (`Granite4VisionConfig`),
Mistral3 Vision (`Mistral3Config`), and Gemma 4 (`Gemma4UnifiedConfig`, an
encoder-free VLM — no vision tower; see [ARCHITECTURE.md](ARCHITECTURE.md#multimodal-vlm-path-vision-tower--text-decoder)).

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
├── hf_*.py                    One adapter per model family (see the Model Family
│                               Coverage table in ARCHITECTURE.md for the full list)
├── st_backend.py               sentence-transformers `backend="spyre"` integration for embedding models
└── __init__.py

tests/                                 CPU tests (no Spyre required)
├── test_adapter_cpu_accuracy.py       CPU: adapter vs stock HF (causal-LM)
├── test_embed_cpu_accuracy.py         CPU: embedding hidden-states vs stock HF
├── test_vlm_e2e_cpu.py                CPU: multimodal adapter vs stock generate
├── test_load_cpu.py                   CPU: models load without errors
└── spyre/                             Spyre tests (require hardware + torch_spyre)
    ├── test_e2e_smoke_spyre.py        E2E: load + generate on Spyre
    ├── test_e2e_token_compare_spyre.py E2E: HF CPU vs adapter Spyre tokens
    ├── test_e2e_embed_compare_spyre.py E2E: HF CPU vs adapter Spyre embeddings
    ├── test_vlm_e2e_spyre.py          E2E: multimodal adapter on Spyre (teacher-forced)
    └── test_load_spyre.py             Spyre: models load without errors
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

# Load test (verify models load without errors)
uv run pytest tests/test_load_cpu.py                              # CPU load test
```

**Note**: Do not run CPU tests with `python tests/test_*.py` — this bypasses pytest's conftest.py setup and will cause import errors. Always use `pytest` (or `uv run pytest`).

### Spyre Tests (requires Spyre hardware)

The Spyre lane lives under `tests/spyre/` and is also pytest-driven (not
`python tests/...`). Each test is parametrized off the model registry, so a
single model is selected with `-k <key>` (e.g. `granite2b`, `qwen3`, `bge_base`).
Run the whole file to cover every registered model. Run from the repository root.

```bash
# E2E smoke test (real weights, verify non-trivial output)
uv run pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py                  # one representative model per adapter
uv run pytest -s -vvv tests/spyre/test_e2e_smoke_spyre.py -k granite2b     # one model

# E2E token comparison (HF CPU vs adapter Spyre, per-step greedy tokens)
uv run pytest -s -vvv tests/spyre/test_e2e_token_compare_spyre.py -k granite2b

# E2E embedding comparison (HF CPU vs adapter Spyre, hidden-states cosine)
uv run pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py -k bge_base

# E2E multimodal VLM (image→text; teacher-forced per-step logit comparison)
uv run pytest -s -vvv tests/spyre/test_vlm_e2e_spyre.py -k granite_vision_mm

# Load test (verify a model loads on Spyre without errors)
uv run pytest -s -vvv tests/spyre/test_load_spyre.py
```

`-s -vvv` matches each test's documented usage and shows the per-step comparison
tables the token / embedding / VLM tests print.

Note: Spyre has known numerical accuracy limitations. Greedy token mismatches
between CPU and Spyre are expected on the single-token decode path until
torch\_spyre fixes land — which is why the VLM lane asserts a per-step logit
cosine floor rather than exact tokens (see
[ARCHITECTURE.md](ARCHITECTURE.md#vision-language-imagetext)).

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
