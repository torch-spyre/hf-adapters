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

## Weekly vision model support

`push-to-clickhouse.yaml` includes `vision` (full image-to-text models) and
`vision_tower` (registered standalone towers). Selecting `model_type=vision`
also includes the tower checks. Like generative and embedding models, VLMs use
`top_k_downloads` and `top_k_trending` (1000 each by default). The catalog
merges both rankings, `resources/vision_models_curated.txt`, and every eligible
registered VLM checkpoint. Ranked discovery uses the Hub's `image-text-to-text`
tag and expands the candidate pool until the requested post-filter count is
reached or the ranking is exhausted. Tower discovery is registry-only.
Tower metadata reports the full checkpoint's parameter count, so tower checks
bypass the checkpoint-size gate and always use the x1 runner tier. Architecture
exclusions still apply. x1 vision and tower shards hold `--shard-size-vision`
checkpoints (default 25), smaller than text shards because each vision
checkpoint runs alone in a fresh worker.

Full VLMs compare stock-HF and Spyre logits through prefill and four
teacher-forced decode steps (cosine floor 0.99), then generate over the five
existing sample images with the same Spyre model. Towers use the shared checkpoint
dtype policy for both paths and compare stock-HF
features with the compiled adapter, checking shapes, finite values, per-patch
cosine, and relative L2 error. Each vision evaluation uses a fresh process and
runs the stock reference before Spyre preparation. Results are written to the
existing v2 capability tables; tower and VLM results remain distinct even for
shared checkpoint paths and early worker failures. Existing access, architecture
exclusion, remote-code, and model-size policies still apply.

Tower configuration resolves an immutable Hub commit, and both the checkpoint
index and weight shards are downloaded at that revision. Reference checkpoint
loading failures are reported as `cpu_load_failed`; forward and input preparation
failures are reported as `cpu_generate_failed`. The VLM verifier clears unwound
exception frames, including chained errors, before releasing models, so retained
pytest tracebacks do not retain callee-owned models or KV caches. Error messages
and traceback locations remain available.

To launch a small scan from an implementation branch:

```bash
gh workflow run push-to-clickhouse.yaml --ref <branch> \
  -f model_type=vision -f top_k_downloads=10 -f top_k_trending=10 \
  -f clickhouse_db=dev
```

For a local catalog/sharding check without database access:

```bash
python .github/scripts/generate_weekly_shards.py --model-type vision \
  --top-k-downloads 10 --top-k-trending 10 --snapshot-date 2026-11-25 \
  --write-to-csv /tmp/vision-verdicts.csv --output-dir /tmp/vision-shards
```

`--write-to-csv` requires new output files. The workflow shares the VLM CI stock
reference cache (`HF_ADAPTERS_VLM_REF_CACHE`); its revision/input/config keys
prevent reuse across incompatible references. Latency is diagnostic, and these
checks establish adapter parity and runtime support rather than caption quality.
