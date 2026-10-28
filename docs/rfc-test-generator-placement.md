# RFC: Put each test-config generator in the repo that owns its runtime

| | |
|---|---|
| **Status** | Draft |
| **Date** | 2026-10-15 |
| **Repos affected** | `hf-adapters`, `spyre-inference`, `torch-spyre` |
| **Tools affected** | model-ops YAML generator (`utils/model_ops`), module-config generator (`utils/module_discovery`), vLLM module test (`test_vllm`) |

## 1. Summary

Today `hf-adapters` holds every test-config generator: the HuggingFace ones, the
vLLM / spyre-inference ones, and the shared code that writes the YAML. This RFC
splits them along one guiding principle:

> **No script may cause a downstream package to be installed in an upstream
> package.**

The stack is `torch-spyre` (upstream) ← `hf-adapters` and `torch-spyre` ←
`spyre-inference` (both downstream). The two downstream repos are siblings and
neither depends on the other. In practice:

- Code that needs `transformers` or `hf_adapters` lives in **hf-adapters**: HF
  model discovery, HF op-generation drivers, and HF module tests and module
  YAMLs.
- Code that needs `vllm` or `spyre_inference` lives in **spyre-inference**:
  vLLM model discovery, vLLM op-generation drivers, and vLLM module tests and
  module YAMLs.
- **torch-spyre** holds the generated op YAMLs, the op tests, the YAML schema,
  and only generator code that needs nothing beyond `torch`.
- It follows that no repo installs a library its own code doesn't import.
  Sibling repos never install each other.

Op tests and module tests are split differently:

- **Op tests belong only to torch-spyre.** The captured ops are torch ops
  (`aten` / `torch.*`), whichever framework ran the model. So every op-test
  YAML, and the test that runs it, lives in torch-spyre. hf-adapters and
  spyre-inference keep only the small driver scripts that load a model (they
  need `transformers` or `vllm` to do that). The YAML those scripts write is
  committed to torch-spyre, **never to the repo that holds the driver**.
- **Module tests are the only per-repo part.** Modules are framework classes:
  `transformers.*` and `hf_adapters.*` modules on one side, `vllm.*` modules on
  the other. Their YAMLs and tests live with the repo that owns those classes.

The result:

| What | Lives in |
|---|---|
| **Op-test YAMLs (all models, all sources)**, op tests (`test_model_ops_v2.py`), YAML schema | `torch-spyre` only (unchanged) |
| Op-test driver scripts, **HuggingFace** models (output goes to torch-spyre) | `hf-adapters` |
| Op-test driver scripts, **spyre-inference / vLLM** models (output goes to torch-spyre) | `spyre-inference` |
| Module discovery + module tests + module YAMLs, **HuggingFace / hf-adapters** modules | `hf-adapters` |
| Module discovery + module tests (`test_vllm`) + module YAMLs, **vLLM** modules | `spyre-inference` |
| Shared, framework-agnostic generator core (`TorchOpCollector`, module-entry YAML builders) | `torch-spyre`, as `oot_framework.generators` |

`hf-adapters` and `spyre-inference` never depend on each other. Both already
depend on `torch-spyre`, which is where the shared code goes.

The shared core is the only generation code that does not sit downstream. Both
downstream repos need it, and since they can't depend on each other, the only
repo both can import it from is the one they both already depend on. It still
follows the principle: it imports only `torch`, `pyyaml` and `regex`, so
torch-spyre installs nothing downstream (section 4.2).

## 2. Motivation

### 2.1 Current layout

| Component | Path in `hf-adapters` | Third-party imports |
|---|---|---|
| Op collector (shared) | `utils/model_ops/utils/torchop_yaml.py` | `torch`, `yaml`, `regex`, `dotenv`; `transformers` (version lookup + a debug `main()`) |
| HF op drivers (11 models) | `utils/model_ops/models/*/run_huggingface.py` | `torch`, `transformers`, `mistral_common`, `huggingface_hub` |
| spyre-inference op driver (Gemma4) | `utils/model_ops/models/gemma4-26b-a4b/run_spyre_inference.py` | **`vllm`**, `spyre_inference` plugin |
| HF module discovery | `utils/module_discovery/auto_generate_module_config.py` | `torch`, `transformers`, `yaml`, `hf_adapters`, `torch_spyre` |
| vLLM module discovery | `utils/module_discovery/auto_generate_module_config_vllm.py` | **`vllm`**, plus private helpers from the HF generator |
| vLLM module test | `tests/test_modules_custom.py::test_vllm` (+ `_run_vllm_module`, `_setup_distributed`, …) | **`vllm`** |
| vLLM module configs | `tests/configs/module_tests_vllm/*.yaml` | n/a |

### 2.2 Problems

1. **`hf-adapters` carries vLLM code it can't run.** vLLM isn't a dependency
   here, so the vLLM pieces only run with an ad-hoc `uv run --with 'vllm==0.24.0'
   --with 'torch==2.11.0+cpu' …` command or inside the spyre-inference image. CI
   gives up on them: `.github/workflows/_test_matrix.yaml:544` notes that
   "`*vllm*` configs self-skip (vllm package not importable here)". So the vLLM
   module tests never run in the repo that holds them.

2. **The vLLM tooling has drifted from vLLM.** The module-discovery README pins
   vLLM 0.24 and torch 2.11. `spyre-inference` is on vLLM 0.28 and torch 2.13.
   Nothing catches this, because the generator doesn't live where vLLM is
   installed and tested.

3. **Adding vLLM to `hf-adapters` as an optional group would make things worse.**
   `uv` resolves every dependency group into one `uv.lock`, so a `vllm` group
   would have to resolve together with `transformers==5.15.0` and the torch pin.
   A conflict would break `uv lock` for every contributor, not just the people
   using that group.

4. **Producer and consumer are in different repos.** The op YAML hard-codes
   `${TORCH_DEVICE_ROOT}/tests/models/test_model_ops_v2.py`
   (`torchop_yaml.py:1861`), and the files are committed to
   `torch-spyre/tests/configs/model_ops_tests/`. The schema is defined in
   `torch-spyre` (`oot_framework.oot_test_config_models`), but the code that
   writes it lives in `hf-adapters` and has no access to that schema at
   generation time.

5. **The two module generators are coupled through private helpers.**
   `auto_generate_module_config_vllm.py:50` imports `PrettyDumper`,
   `MODULE_TEST_TIER_LABELS`, `_build_module_entry_dict` and
   `_process_pytree_structure` from the HF generator. It also inlines a copy of
   `ModuleInfoCapture._create_invocation_signature`. The vLLM generator can't be
   moved until this shared core has a proper home.

6. **It isn't clear where op YAMLs belong.** `utils/model_ops/README.md` says
   the Gemma4 YAML produced through spyre-inference "is kept in
   spyre-inference", while the HF-produced op YAMLs are in torch-spyre. An op
   test exercises torch ops, so where the YAML lives should not depend on which
   framework traced the model. Splitting op YAMLs across repos would also split
   the op-test results across three CI systems.

## 3. Goals and non-goals

**Goals**

- Each repo's lockfile contains only the libraries its own code imports.
- Each generator lives next to the runtime it traces, so it is tested against
  the same library versions that ship.
- The shared YAML-emitting code exists once, in the repo that owns the schema.
- Every op-test YAML lives in torch-spyre. hf-adapters and spyre-inference
  never commit one.
- No period during the migration where a generator is unavailable.

**Non-goals**

- Changing the op-test or module-test YAML schema, or the meaning of any test.
- Rewriting the generators. This RFC moves code and makes it public; it does
  not change behavior.
- Automating cross-repo YAML PRs (see Open questions).

## 4. Design

### 4.1 Dependency layering

```
  UPSTREAM       torch-spyre
   (torch, oot_framework, op tests, op YAMLs, YAML schema,
    oot_framework.generators  <- torch-only shared core)
            ^                         ^
            |                         |
  DOWNSTREAM hf-adapters        spyre-inference
  (transformers, HF drivers,   (vllm, vLLM drivers,
   HF module discovery,         vLLM module discovery,
   HF module tests + YAMLs)     test_vllm + module YAMLs)
```

Arrows mean "depends on" and point only upstream, toward `torch-spyre`. There
are no arrows between `hf-adapters` and `spyre-inference`. Every move in this
RFC either keeps an arrow pointing upstream or removes one that pointed the
wrong way: today hf-adapters effectively installs vLLM, a sibling's runtime,
through `--with vllm`.

### 4.2 Shared core in `torch-spyre`: `oot_framework.generators`

Add a subpackage to the existing `oot-framework` distribution
(`torch-spyre/tests/oot_framework/`), which both downstream repos already
install, or can install, from the same git source:

```
torch-spyre/tests/oot_framework/
  generators/
    __init__.py
    torchop_collector.py   # TorchOpCollector and op-YAML emission
    module_config.py       # backend-independent module-entry builders
```

`oot_framework/pyproject.toml` must list `oot_framework.generators` in
`packages`.

**`torchop_collector.py`** is `utils/model_ops/utils/torchop_yaml.py` with these
changes:

| Today | After the move | Reason |
|---|---|---|
| `from dotenv import load_dotenv` at module scope | Lazy `try: import dotenv` inside `setup_logging()`; no-op if missing | Avoid adding `python-dotenv` to torch-spyre |
| `_get_transformers_version()` does `import transformers` | `importlib.metadata.version(...)`, which reads package metadata without importing the package | No `transformers` import in torch-spyre |
| `_convert_transformers_path_to_url` covers only `transformers` | A small table of resolvers `{transformers, vllm}` that turn `site-packages/<pkg>/…:LINE` into a GitHub link at the installed version | spyre-inference YAMLs get source links too |
| Debug `main()` that loads a HF model | Removed. The per-model drivers already cover this | It was the only hard `transformers` import |

Runtime deps stay `torch`, `pyyaml`, `regex`, which are already
`oot-framework` dependencies. torch-spyre gains no new requirements.

**`module_config.py`** takes the backend-independent part of
`auto_generate_module_config.py` and drops the leading underscores, since these
become a public API:

- `MODULE_TEST_TIER_LABELS`, `PrettyDumper`
- `process_pytree_structure`, `extract_tensor_info`, `extract_scalar_info`,
  `extract_device_layout`
- `build_module_entry_dict`, plus the sample-input converters it calls
  (`convert_captured_input_to_sample_input`, `tensor_info_to_spec`,
  `convert_constructor_arg_to_sample_input`)
- `invocation_signature` (replaces the copy inlined in the vLLM generator)

Everything specific to HF (`StaticCache` prefill/decode, cache-info
extraction, config-kwarg extraction, the `--loader spyre` path, transformers
source references) stays in `hf-adapters`.

**Optional: validate on write.** Both writers can validate the dict against
`oot_framework.oot_test_config_models.OOTTestConfig` before writing. That turns
schema drift into a failure at generation time instead of at test time. This is
only possible when the writer and the schema live in the same package.

### 4.3 `hf-adapters` after the change

```
hf-adapters/
  utils/model_ops/
    models/<model>/run_huggingface.py      # import from oot_framework.generators.torchop_collector
    README.md                              # HF drivers only
  utils/module_discovery/
    auto_generate_module_config.py         # HF + --loader spyre; shared parts imported
    README.md                              # vLLM section removed, links to spyre-inference
  tests/test_modules_custom.py             # test_vllm and its helpers removed
  tests/configs/module_tests/              # HF / adapter configs only
```

Removed: `utils/model_ops/utils/torchop_yaml.py`,
`utils/model_ops/models/gemma4-26b-a4b/run_spyre_inference.py`,
`utils/module_discovery/auto_generate_module_config_vllm.py`,
`tests/configs/module_tests_vllm/`.

### 4.4 `spyre-inference` after the change

```
spyre-inference/
  scripts/model_ops/
    gemma4_26b_a4b.py                      # was run_spyre_inference.py
    README.md
  scripts/module_discovery/
    auto_generate_module_config_vllm.py    # imports oot_framework.generators.module_config
    README.md
  tests/module_tests/
    test_modules_vllm.py                   # test_vllm + _run_vllm_module, _setup_distributed, ...
    run_oot_module_configs.sh
  tests/configs/module_tests/
    *.yaml                                 # was hf-adapters/tests/configs/module_tests_vllm/
```

Add a dependency group, pinned to the **same torch-spyre revision** as the
existing `torch-spyre` source so the two can't disagree:

```toml
[dependency-groups]
test-gen = ["oot-framework"]

[tool.uv.sources]
oot-framework = { git = "https://github.com/torch-spyre/torch-spyre",
                  subdirectory = "tests/oot_framework",
                  rev = "<same sha as torch-spyre>" }
```

`spyre-inference` does **not** gain `hf-adapters`, `mistral_common`, or
`python-dotenv`.

### 4.5 `torch-spyre` after the change

- Adds `oot_framework/generators/`, described in 4.2.
- Takes over the collector's unit tests from `hf-adapters`:
  `tests/test_torchop_collector_compile_fx.py` and
  `tests/test_model_ops_op_names.py`.
- `tests/configs/model_ops_tests/` remains the single home for op-test YAMLs
  from both sources. A YAML produced through spyre-inference gets a name suffix,
  following the existing `-fms` / `-with-device-layout` convention, e.g.
  `gemma-4-26B-A4B-it-spyre-inference.yaml`.
- `test_model_ops_v2.py` is unchanged.
- No new runtime dependency. `transformers` and `vllm` are never imported.

### 4.6 Dependency matrix

| Library | hf-adapters | spyre-inference | torch-spyre |
|---|---|---|---|
| `torch`, `torch-spyre` | yes (today) | yes (today) | owner |
| `oot-framework` | yes, `oot` group (today) | **new**, `test-gen` group | owner |
| `transformers` | yes (core) | yes (via vLLM, today) | no |
| `vllm` | **removed** (ad-hoc `--with`) | yes (core, today) | no |
| `hf-adapters` | owner | no | no |
| `mistral_common`, `huggingface_hub` | yes | no | no |
| `python-dotenv` | optional | no | no (lazy import) |

### 4.7 Where outputs go

| Generator | Writes | Committed to |
|---|---|---|
| HF op drivers (`hf-adapters`) | op-test YAML | `torch-spyre/tests/configs/model_ops_tests/<model>.yaml` |
| spyre-inference op drivers | op-test YAML | `torch-spyre/tests/configs/model_ops_tests/<model>-spyre-inference.yaml` |
| HF module discovery | module-test YAML | `hf-adapters/tests/configs/module_tests/` |
| vLLM module discovery | module-test YAML | `spyre-inference/tests/configs/module_tests/` |

**Rule: op-test YAMLs exist only in torch-spyre.** The captured ops are torch
ops, so the op test doesn't depend on which framework produced the trace.
hf-adapters and spyre-inference hold driver scripts, never their output.
Module-test YAMLs are the only per-repo YAMLs, because they rebuild framework
classes (`transformers.*`, `hf_adapters.*`, `vllm.*`) that only that repo can
import.

To enforce the rule:

- The op drivers write to the current directory, or to `--output`. Their
  READMEs document the destination as
  `<torch-spyre>/tests/configs/model_ops_tests/`, and no driver defaults to a
  path inside its own repo.
- hf-adapters and spyre-inference each add a CI or pre-commit check that fails
  if a committed YAML references `test_model_ops_v2.py` /
  `TestSpyreModelOps`. (Neither repo has such a file today, so the check
  starts green.)

Each op YAML header should record its provenance: generator repo and SHA,
model, and `transformers` / `vllm` version. The existing parameters header is
where this goes. Reviewers in torch-spyre can then trace a YAML back to the
script that made it.

## 5. Migration plan

Each phase is a separate PR and can be merged on its own. Phase 3 lands before
Phase 4, so the vLLM tooling is never missing.

| Phase | Repo | Change | Done when |
|---|---|---|---|
| 0 | hf-adapters | This RFC | Accepted by maintainers of all three repos |
| 1 | torch-spyre | Add `oot_framework.generators` (4.2) and move the collector unit tests | Unit tests pass; nothing downstream changes yet |
| 2 | hf-adapters | Switch HF drivers and HF module discovery to `oot_framework.generators`; delete `torchop_yaml.py` and the extracted helpers | Regenerating `granite-3.3-8b-instruct.yaml` (op and module) gives a byte-identical file |
| 3 | spyre-inference | Add `scripts/model_ops/`, `scripts/module_discovery/`, `tests/module_tests/`, the module configs, the `test-gen` group, and the "no op YAMLs" check (4.7). The Gemma4 op YAML goes to torch-spyre as `gemma-4-26B-A4B-it-spyre-inference.yaml` | vLLM module generator and `test_vllm` run on vLLM 0.28 / torch 2.13 without `--with` overrides; no op YAML in the repo |
| 4 | hf-adapters | Remove the vLLM generator, `run_spyre_inference.py`, `test_vllm` and its helpers, `module_tests_vllm/`, and the vLLM sections of the READMEs (replace with links). Delete the "kept in spyre-inference" line from `utils/model_ops/README.md`. Add the "no op YAMLs" check. Drop the `*vllm*` self-skip comment in CI | `rg -i vllm utils tests` finds nothing but links; READMEs name torch-spyre as the only home for op YAMLs |
| 5 (optional) | spyre-inference | CI job that runs `tests/module_tests/` on a Spyre runner | Results reach ClickHouse (see Risks) |

**Proving that Phase 2 changes nothing:** regenerate one op YAML and one module
YAML before and after the switch, then `diff` them. Any difference is a bug in
the extraction.

**Git history:** move files with `git mv` inside a repo. Between repos, link the
source path and commit in the PR description so `git log` on the old repo still
leads to the history.

## 6. Alternatives considered

| Option | Why rejected |
|---|---|
| **A. Keep everything in hf-adapters and add a `vllm` dependency group** | `uv` resolves all groups into one lock, so vLLM's torch and transformers pins would constrain every hf-adapters user (problem 3). It still wouldn't use spyre-inference's vLLM version. |
| **B. Copy the shared core into each repo** | Two copies of the YAML writer drift apart, and neither is next to the schema. Problem 5 already shows the start of this (the inlined `_create_invocation_signature`). |
| **C. New standalone repo or package for generators** | Adds a release and pinning cycle with no benefit over `oot-framework`, which both repos already install from torch-spyre and which already holds the schema. Worth revisiting if the generators grow beyond test-config generation. |
| **D. Move everything into torch-spyre** | torch-spyre would need `transformers`, `vllm` and `hf-adapters` (the `--loader spyre` path imports it). That installs downstream packages in the upstream repo, which is exactly what the guiding principle forbids, and creates a cycle with hf-adapters. |

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `oot_framework.generators` becomes an API two repos depend on | Keep it small and public (section 4.2); everything else stays private. hf-adapters tracks torch-spyre `main`, so breakage shows up within a day; spyre-inference pins a SHA and upgrades on purpose. |
| `test_vllm` shares helpers with the HF tests in `test_modules_custom.py` (`randomize_weights_xavier`, `_run_forward`, `_extract_all_tensors`) | Phase 3 copies what the vLLM test needs. If a third copy shows up, promote the generic helpers into `oot_framework` as well. |
| ClickHouse ingestion (`.github/scripts/parse_module_test_logs.py`) assumes module configs are under `hf-adapters/tests/configs/module_tests/` | No loss today, since the vLLM configs self-skip in hf-adapters CI. Phase 5 adds ingestion in spyre-inference, reusing the torch-spyre `clickhouse-ingest` extension. |
| The `models-ops` group in hf-adapters is also used by non-generator CI (`push-to-clickhouse.yaml:196` prefilter, model-support scan) | Phase 2 trims only what the generators alone needed (`python-dotenv`, and `PyYAML` / `regex`, which now come via `oot-framework`). Keep anything the prefilter imports. Check each removal with `rg` over `.github/`. |
| spyre-inference's vLLM is the Spyre platform build, but the vLLM module generator assumed a CUDA host | The generator only needs `enforce_eager=True` and forward hooks. Phase 3 checks that it runs on the Spyre platform; if not, it keeps a documented CUDA path (see Open questions). |

## 8. Open questions

1. **Cross-repo handoff of op YAMLs.** Today a contributor runs a generator and
   opens a PR to `torch-spyre/tests/configs/model_ops_tests/` by hand. Should a
   scheduled job in hf-adapters and spyre-inference open those PRs
   automatically?
2. **Where vLLM module discovery runs.** Spyre platform only, or keep the CUDA
   path for models that don't fit on one Spyre card yet?
3. **Name of the spyre-inference dependency group.** This RFC uses `test-gen`.
   An existing group in that repo might be a better fit.
4. **Deprecation shim.** Should hf-adapters keep a one-line
   `utils/model_ops/utils/torchop_yaml.py` that re-exports from
   `oot_framework.generators` for one release, so in-flight branches don't
   break?
