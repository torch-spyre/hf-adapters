# HF / HF-with-hf-adapters module-config generation & standalone testing

This directory contains `auto_generate_module_config.py`, which captures the
per-module forward inputs of a **HuggingFace Transformers** model and emits a YAML
config so each module can be re-run standalone on Spyre by the `test_with_cpu`,
`test_eager_vs_compile` and `test_layout_stride` tests in
`tests/test_modules_custom.py` (plus upstream's `test_forward`).

It has two capture paths, selected with `--loader`:

| `--loader` | Loads with | Captures | Drives the forward via |
|------------|-----------|----------|------------------------|
| `hf` *(default)* | `AutoModel` / `AutoModelForCausalLM` | stock `transformers` modules (`GraniteAttention`, `GraniteMLP`, …) | `model(**inputs)` — a `StaticCache` prefill, then one decode step |
| `spyre` | `AutoSpyreModelForCausalLM` | the **hf-adapters** modules that `prepare_for_spyre()` substitutes into the model tree (`StandardGQAAttention`, …) | `model.generate()` — the real padded 64-block loop |

Use `hf` to test what upstream Transformers runs; use `spyre` to test what actually
executes on Spyre. **The two are mutually exclusive within one process**:
`prepare_for_spyre()` calls `patch_rmsnorm`, which rewrites the RMSNorm class
globally, so run one loader per invocation.

There are two steps:

1. **Generate** a YAML from a model path (loads the model once and runs it).
2. **Test**: run each captured module standalone and compare a CPU reference against
   `spyre` (eager and/or `torch.compile`).

---

## 1. Generate the YAML

### Prerequisite: install `torch-spyre` and others (once)

```bash
uv sync --group dev --group spyre --group test
```

### Run

```bash
# Stock HuggingFace modules
python utils/module_discovery/auto_generate_module_config.py \
    --model_path ibm-granite/granite-3.3-8b-instruct \
    --seq_len 128

# hf-adapters (Spyre) modules
python utils/module_discovery/auto_generate_module_config.py \
    --model_path ibm-granite/granite-3.3-8b-instruct \
    --seq_len 128 --loader spyre
```

Arguments:

| Flag | Default | Meaning |
|------|---------|---------|
| `--model_path` | *(required)* | HuggingFace model path/id |
| `--seq_len` | `128` | Prompt sequence length. Under `--loader spyre` this is a *target*: `generate()` block-pads it to a multiple of `BLOCK_SIZE`, and the padded length is what gets recorded |
| `--output` | `<model>.yaml` (hf) / `<model>adapter.yaml` (spyre) | Output YAML path, under `./tests/configs/module_tests/`. The defaults differ so the two loaders never overwrite each other |
| `--loader` | `hf` | Capture path (see the table above) |
| `--dtype` | `bfloat16` (hf) / `float16` (spyre) | Load dtype. The spyre default matches `AutoSpyreModelForCausalLM`; `auto` consults the adapter registry's per-model dtype |
| `--device` | `spyre` | **`--loader spyre` only.** Patches `hf_common.DEVICE`. Pass `cpu` for an off-pod dry run (no `torch_spyre` needed) |
| `--max_new_tokens` | `3` | **`--loader spyre` only.** Decode steps to run — see "Why 3" below |
| `--no_static_cache` | off | **`--loader hf` only.** Use the model's default dynamic KV cache instead of a `StaticCache` |
| `--max_cache_len` | `2048` | **`--loader hf` only.** `StaticCache` capacity |

The same work is available programmatically, so a caller can drive capture without
the CLI:

```python
from auto_generate_module_config import generate_spyre_module_config

generate_spyre_module_config(
    "ibm-granite/granite-3.3-8b-instruct",
    seq_len=128,
    device="cpu",                  # off-pod dry run
    output="/tmp/granite_adapter.yaml",
)
```

`main()` is a thin wrapper over this function, so the CLI and the API produce
identical YAML.

The generated YAML:

- registers each captured module under the `test_forward` and
  `*TestModuleCustom*::{test_with_cpu,test_eager_vs_compile,test_layout_stride}`
  test entries, tagged `model__<model_name>`,
- records each module's real class in `module_path`,
- carries the model's config as a config-type constructor arg so the test can rebuild
  it with the captured dimensions,
- stores each module's captured forward-input shapes/dtypes in `forward_inputs`, one
  entry per distinct invocation pattern.

### `--loader spyre` specifics

**Wrapper modules are emitted as a nested module arg.** `StandardGQAAttention.__init__`
takes an already-constructed HF attention (it adopts its q/k/v/o projections) and keeps
no config of its own, so it cannot be described by a config arg alone. The generator
records the HF class it wrapped — snapshotted *before* `prepare_for_spyre()` replaces
the layer — and emits it as a `module_path`-keyed constructor arg:

```yaml
- name: StandardGQAAttention_<hash>
  module_path: hf_adapters.hf_common.StandardGQAAttention
  apply_device_layout: true
  constructor_inputs:
    args:
      - module_path: transformers.models.granite.modeling_granite.GraniteAttention
        config_path: transformers.models.granite.configuration_granite.GraniteConfig
        config_kwargs: {hidden_size: 2048, num_attention_heads: 32, head_dim: 128}
        module_kwargs: {layer_idx: 0}
```

This needs the OOT framework's `InputArgModule` support.

**`head_dim` is the padded value.** When `pad_attention_heads()` lifts `head_dim` to a
Spyre stick boundary, the generator emits `model._spyre_head_dim` (e.g. `128` where the
checkpoint has `64`) so the rebuilt projections get the widths the adapter actually
runs, not the checkpoint's.

**`apply_device_layout: true`** asks the test to transfer parameters with
`torch_spyre.model_utils.load_model_to_spyre` — each `nn.Linear` weight laid out
`dim_order=[1, 0]` — rather than a plain `.to(device)`, which cannot express a device
layout.

**`StandardGQABlock` is intentionally not emitted.** Its constructor takes a live
decoder layer plus an `is_res_mul` flag, and it owns submodules (`mlp`, both norms,
`self_attn`) that are already captured as standalone entries. Its own arithmetic
(residual order, `residual_multiplier`, norm placement) is covered end-to-end by
`tests/spyre/test_e2e_*`. `PrecomputedRotaryEmbedding` / `InvFreqShim` are skipped for
the same reason — the HF rotary module they wrap is captured instead.

**Why `--max_new_tokens 3`.** `generate()` produces three distinct forward shapes, and
3 is the minimum that reaches all of them:

| step | `is_filling` | `token_index` | `cache_position` | |
|------|--------------|---------------|------------------|--|
| `i=0` | `False` | `0` | `0` | PREFILL — the whole prompt |
| `i=1` | `False` | `0` | `BLOCK_SIZE` | EXPANSION — claim the next 64-token block |
| `i=2` | **`True`** | `1` | | FILL — write one token into that block |

FILL matters disproportionately: `is_filling` selects a different branch of
`kv_cache_update` (write one token at `token_index` vs. write the whole block), and
`torch.compile` specializes on it, so FILL is a separately compiled binary — and it is
the shape nearly every generated token uses. Raising the value further only adds more
FILL invocations at successive `token_index` values, each another compiled binary in
the module test.

---

## 2. Run the standalone module tests

Tests are driven by the OOT framework via `tests/run_oot_module_configs.sh`, which
consumes the generated YAML. Each module is rebuilt from its `constructor_inputs`,
given the captured `forward_inputs`, and compared CPU-vs-device:

| Test | What it checks |
|------|----------------|
| `test_with_cpu` | CPU eager is the golden reference; the device run (compile by default, eager via `TEST_EAGER_WITH_CPU=1`) must match |
| `test_eager_vs_compile` | Spyre eager and Spyre compiled must agree with each other and with CPU, in a single pass |
| `test_layout_stride` | Same comparison, exercising the YAML's `device_layout` input specs; entries with `apply_device_layout` also get the production parameter layout |
| `test_forward` (upstream) | The module builds and runs at all |

All three custom tests build the device module from the **same** weights as the CPU
reference, so a mismatch means a real numerical divergence rather than different random
init.

### Prerequisite: install `torch-spyre`, `oot`, and others (once)

```bash
uv sync --group dev --group spyre --group test --group oot
```

If you want to do edtable install, see the vLLM section below for the one-time `torch-spyre` / `oot_framework` editable install — the prerequisite is identical.

### Run

```bash
uv run \
    tests/run_oot_module_configs.sh \
    "$(pwd)/tests/configs/module_tests/granite_3_3_8b_instruct.yaml" \
    -v -s -rsadp
```

Add `-k <pattern>` to narrow to one module while iterating, e.g. `-k Attention`.

`--no-group spyre` runs on CPU; drop it to run on the Spyre pod. A YAML generated with
`--loader spyre` carries `device: spyre` input specs and `apply_device_layout`, so it is
meant for the pod — on CPU the layout request degrades to a plain `.to()` rather than
failing.

To run a whole directory of configs instead of one file, pass the directory:

```bash
uv run ... tests/run_oot_module_configs.sh "$(pwd)/tests/configs/module_tests/" -v -s -rsadp
```

### What the results mean

- **PASS** — the module's CPU and device outputs agree within the YAML
  `supported_dtypes` tolerance.
- **SKIP** — the module was filtered out by the config (`unlisted_test_mode: skip`), or
  the dtype variant is outside `supported_dtypes`.
- **FAIL** — outputs diverge beyond tolerance, or the module could not be built/run. A
  `TypeError: forward() missing N required positional arguments` points at the *capture*
  side rather than the module: the generator dropped an argument it could not describe.

---

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
| `--output` | `./tests/configs/module_tests/<model>_vllm.yaml` | Output YAML path |

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
    "$(pwd)/tests/configs/module_tests/granite_3_3_8b_instruct_vllm.yaml" \
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
