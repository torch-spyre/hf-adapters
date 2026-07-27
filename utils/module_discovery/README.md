# vLLM module-config generation & standalone testing

This directory contains `auto_generate_module_config_vllm.py`, which captures the
per-module forward inputs of a **vLLM (v1)** model and emits a YAML config so each
module can be re-run standalone on Spyre by the `test_vllm` test in
`tests/test_modules_custom.py`.

There are two steps:

1. **Generate** a YAML from a model name (runs the model once under vLLM, prefill only).
2. **Test**: run each captured module standalone (built from `AutoConfig`, **not** via
   `LLM()`) and compare CPU eager vs `spyre` + `torch.compile`.

Assumptions: vLLM v1, single GPU, `tensor_parallel_size=1`. Only layer 0 is captured
(one representative decoder layer, including its submodules); KV-cache / decode phase
is out of scope for this version (prefill only).

---

## 1. Generate the YAML

`auto_generate_module_config_vllm.py` loads the model with
`LLM(model=..., enforce_eager=True, tensor_parallel_size=1)`, registers forward
pre-hooks via `llm.apply_model()`, runs a single prefill, and writes the YAML.

**Requirements:** generation must run in a **CUDA environment** (single GPU) with
**vLLM 0.24** installed — the standard GPU vLLM runtime is used to load and run the
model. (This differs from step 2, which runs the generated YAML on CPU or the Spyre
pod and does not use `LLM()`.)

```bash
python utils/module_discovery/auto_generate_module_config_vllm.py \
    --model ibm-granite/granite-3.3-8b-instruct \
    --seq-len 128 \
    --dtype bfloat16
```

Arguments:

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | *(required)* | HuggingFace model path/id |
| `--seq-len` | `128` | Prefill sequence length |
| `--dtype` | `bfloat16` | Model load dtype (`bfloat16` / `float16` / `float32`); also written to the YAML `supported_dtypes` |
| `--model-impl` | `native` | vLLM backend (`native` / `transformers`) |
| `--output` | `./tests/configs/module_tests/<model>_vllm_spyre.yaml` | Output YAML path |

The generated YAML:

- registers each captured module under a `*TestModuleCustom*::test_vllm` test entry,
- records each module's real vLLM class in `module_path` (e.g.
  `vllm.model_executor.models.granite.GraniteMLP`),
- carries the model's config as a config-type constructor arg so the test can rebuild
  the config with `AutoConfig`,
- stores each module's captured prefill forward-input shapes/dtypes in `forward_inputs`.

`enforce_eager=True` is required for generation: with v1's default `torch.compile` +
CUDA-graph capture, graph replay bypasses the Python submodule hooks and nothing is
captured.

---

## 2. Run the standalone module tests

Tests are driven by the OOT framework via `tests/run_oot_module_configs.sh`, which
consumes the generated YAML. `test_vllm` (in `tests/test_modules_custom.py`) rebuilds
each **vLLM-native** module under a `VllmConfig` + a TP=1 distributed group, initializes
its weights deterministically (xavier), and compares a CPU-eager reference against a
`spyre` + `torch.compile` run. Non-vLLM modules (e.g. PyTorch-standard `nn.GroupNorm`,
or `transformers.*` modules from HF-generated YAMLs) are skipped.

### Prerequisite: install `torch-spyre` and `oot_framework` (once)

The documented run below does **not** use `uv run --with-editable`. Instead, install the
two local packages into the project venv up front, so the ordinary `uv run` picks them
up. Because `oot_framework` is otherwise pinned in `uv.lock` as a git dependency and
installed as a real-file copy (which wins over the source tree), reinstall it editable:

```bash
# from the hf-adapters repo root
uv pip install -e ../torch-spyre
uv pip uninstall oot_framework            # drop the git-pinned copy if present
uv pip install -e ../torch-spyre/tests/oot_framework
```

Verify the editable install points at the source tree (not `.venv/.../site-packages`):

```bash
uv run --no-sync python -c "import oot_framework, os; print(oot_framework.__file__)"
# expect: .../torch-spyre/tests/oot_framework/__init__.py
```

> Note: a later `uv sync` (or a lockfile update) reverts these editable installs back to
> the pinned copies, silently dropping local edits to `torch-spyre` / `oot_framework`.
> Re-run the `uv pip install -e` steps after any such sync. For a permanent change,
> commit to `torch-spyre` and bump the pinned sha in `uv.lock`.

### Run

With the editable installs in place, run the YAML without any `--with-editable` flags:

```bash
uv run \
    --index-strategy unsafe-best-match \
    --no-default-groups --group dev --no-group spyre \
    --with 'vllm==0.24.0' \
    --with 'torch==2.11.0+cpu' \
    --with 'torchaudio==2.11.0+cpu' \
    --with 'torchvision==0.26.0+cpu' \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    tests/run_oot_module_configs.sh \
    "$(pwd)/tests/configs/module_tests/granite_3_3_8b_instruct_spyre_vllm.yaml" \
    -v -s -rsadp
```

Notes on the flags:

- `--index-strategy unsafe-best-match` + `--extra-index-url .../whl/cpu` let uv pull the
  `+cpu` torch wheels alongside the default index.
- The pinned `torch`/`torchvision`/`torchaudio` versions must be mutually compatible;
  a mismatch surfaces as `RuntimeError: operator torchvision::nms does not exist` at
  import time (torchvision linked against a different torch build).
- `--no-group spyre` runs on CPU; drop it (and the `+cpu` pins) to run on the Spyre pod.
- Trailing pytest args: `-v` verbose, `-s` no capture, `-rsadp` summary for
  skipped/failed/etc.

To run a whole directory of configs instead of one file, pass the directory:

```bash
uv run ... tests/run_oot_module_configs.sh "$(pwd)/tests/configs/module_tests/" -v -s -rsadp
```

### What the results mean

- **PASS** — the vLLM module's CPU-eager and `spyre`-compiled outputs agree within the
  YAML `supported_dtypes` tolerance.
- **SKIP** — the module is not a vLLM-native standalone target (`nn.*` / `transformers.*`),
  is forward-context dependent (`*Attention` / `*DecoderLayer`, deferred), has no
  resolvable config arg, or vLLM is unavailable.
- **FAIL** — outputs diverge beyond tolerance, or the module could not be built/run.
