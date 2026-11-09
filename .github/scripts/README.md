# GitHub Actions Scripts

## generate_test_matrix.py

Dynamically generates test matrices for GitHub Actions workflows from the model registry.

### Purpose

This script ensures that CI test matrices stay synchronized with the model registry (`tests/model_registry.py`) and adapter mappings (`hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`). When new adapters are added, the CI automatically tests them without manual workflow updates.

### How It Works

1. Imports the representative path lists from `tests/model_registry.py`.
2. Generates one JSON array per task suite, including `causal_matrix`,
   `embed_matrix`, `masked_lm_matrix`, `vision_matrix`, and `reranker_matrix`.
   `combined_matrix` remains the causal + embedding matrix used by shared jobs.
3. Outputs matrices in GitHub Actions format for consumption by test jobs.

### Usage

```bash
# Generate all matrices
python .github/scripts/generate_test_matrix.py

# Exclude specific models (e.g., temporarily broken models)
python .github/scripts/generate_test_matrix.py --exclude granite-vision phi4

# Restrict every matrix to an explicit allowlist (flips each category from its
# representative set to ALL registered paths, then filters to these)
python .github/scripts/generate_test_matrix.py --only Qwen/Qwen3-0.6B ibm-granite/granite-4.1-8b

# Perf tiers only: fail instead of dropping a model perf cannot benchmark
python .github/scripts/generate_test_matrix.py --strict-perf --only ibm-granite/granite-4.1-8b

# In GitHub Actions workflow
- name: Generate matrices
  id: generate
  run: |
    python .github/scripts/generate_test_matrix.py --exclude granite-vision
```

### Excluding Models

To temporarily exclude models from all test matrices:

1. Edit `.github/workflows/test_pull_request.yaml`
2. Find the `generate-matrix` job
3. Add model keys to the `--exclude` list:
   ```yaml
   python .github/scripts/generate_test_matrix.py --exclude granite-vision phi4 other-model
   ```

### The perf matrix and `--strict-perf`

`perf_matrix` is built from the **causal-LM** paths of an explicit `--only` allowlist
(`tests/model_lists/perf.yaml`), never from the whole registry. The restriction is
deliberate: spyre-perf-suite's generative harness measures prefill+decode, so an embedding
or encoder path has no decode phase to benchmark.

Without `--strict-perf` such a path is dropped silently. That is unsafe for a perf run: an
empty `perf_matrix` makes `spyre-perf-tests`' `if:` skip every leg, and because perf cells
are `gating: false` the run reports success having measured nothing. Pass `--strict-perf`
and the script raises instead, naming each path to move.

It is **opt-in** because one `generate-matrix` job emits all 12 matrices for every tier, and
the non-perf tiers legitimately test encoder and embedding models. `_test_matrix.yaml` passes
it only when `test_type` is `perf`.

### Adding New Models

To add a new model to CI:

1. Add the model to the appropriate task registry in `tests/model_registry.py`.
2. Ensure the model's adapter is in `hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`
3. The CI will automatically include it in the next run

No workflow changes needed!

### Dependencies

- Python 3.11+
- transformers
- torch

These are installed in the `generate-matrix` job before running the script.
