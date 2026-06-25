# GitHub Actions Scripts

## generate_test_matrix.py

Dynamically generates test matrices for GitHub Actions workflows from the model registry.

### Purpose

This script ensures that CI test matrices stay synchronized with the model registry (`tests/model_registry.py`) and adapter mappings (`hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`). When new adapters are added, the CI automatically tests them without manual workflow updates.

### How It Works

1. Imports `select_representative_models()` from `tests/model_registry.py`
2. Generates three JSON arrays:
   - `causal_matrix`: Causal LM models only (one per adapter)
   - `embed_matrix`: Embedding models only (one per adapter)
   - `combined_matrix`: All models (causal + embedding)
3. Outputs matrices in GitHub Actions format for consumption by test jobs

### Usage

```bash
# Generate all matrices
python .github/scripts/generate_test_matrix.py

# Exclude specific models (e.g., temporarily broken models)
python .github/scripts/generate_test_matrix.py --exclude granite-vision phi4

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

### Adding New Models

To add a new model to CI:

1. Add the model to `tests/model_registry.py` (either `CAUSAL_LM_MODELS` or `EMBEDDING_MODELS`)
2. Ensure the model's adapter is in `hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`
3. The CI will automatically include it in the next run

No workflow changes needed!

### Dependencies

- Python 3.11+
- transformers
- torch

These are installed in the `generate-matrix` job before running the script.
