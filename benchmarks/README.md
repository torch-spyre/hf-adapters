# Benchmark definitions

Declarative `bench/v1alpha1` definitions read by the `spyre-bench` engine. They run beside the existing perf path,
never instead of it unless asked: `BENCH_MODE` is `off` by default, the `perf` tier and its Makefile targets behave as
before, and no required check depends on this directory.

## Layout

| Path | Content |
|---|---|
| `project.yaml` | Project, Suites `hf-models` (perf) and `model-correctness` (quality), Selection `model-correctness` |
| `systems.yaml` | the System every model benchmark runs on |
| `workloads.yaml` | `greedy-parity`, the quality Workload (see below) |
| `environments.yaml` | `one-device`, the request the on-demand workflow uses |
| `hf-models/<model>.yaml` | one Benchmark per model, generated |
| `selections.yaml` | `smoke`, `hf-models-nightly`, `hf-models-weekly`, generated |
| `model-correctness/` | quality Benchmarks over the `evals/` question samples |
| `stores.yaml`, `assets-eval.yaml`, `datasets.yaml`, `models.yaml` | where the question samples and the weights come from, by digest or revision |
| `ingest-ids.golden.tsv` | the ids the ingest writes for every case, report, kernel and total row (synthetic fixtures) |
| `tools/`, `tests/` | case runners, the identity conformance check, the collector, and their tests |

Further Environments and node classes live with the caller (`BENCH_CENTRAL`, `BENCH_ENVIRONMENT`).

## Perf: from `perf.yaml`

`tests/model_lists/perf.yaml` stays the single model list. Regenerate after editing it or
`tests/model_lists/perf_bench.yaml`:

```bash
python scripts/generate_bench_definitions.py          # write
python scripts/generate_bench_definitions.py --check  # fail when stale
python scripts/generate_bench_definitions.py --prune  # also delete hf-models files not generated
```

| Source | Definition |
|---|---|
| `nightly.models` | tier `core`; Selection `hf-models-nightly` |
| `weekly.models` | tier `extended`; Selection `hf-models-weekly` |
| `perf_bench.yaml` `sanity_model` | tier `sanity`; Selection `smoke` holds one of its cases |
| model path | benchmark name: last path segment, lower-cased, unless `benchmark_names` maps it |
| `exclude.yaml` | not mapped; `perf-tests` keeps applying it per architecture |

Each model expands to six cases: `config` (`full-model`, `one-block`) by `mode` (`prefill-only`,
`decode-only`, `generate`), the vocabulary the existing model rows already use. The cases use the
`kernel/layer` workload of the `kernel` pack; the neutral runner name is `kernel-bench`.

### Identity

Every Benchmark carries an `identity` block (`mode: ingest`, `profile: perf-kernel`), so the engine's stored
`benchmark_id` is the id the ingest writes for the same row. The name is the model's bare name as the perf suite
reports it, lower-cased (the ingest lower-cases it too); `ibm-granite/granite-4.1-8b` is `granite` because the
ingest folds `granite_` ops into that name. The six discriminators are the ones the report parser hashes:
`record_type: model`, `config_name` and `run_mode` from the case, and empty `input_shapes`, `kernel_name`,
`is_total` for the model-level row.

`make bench-check-ids` (and the `bench-conformance` workflow) derives each id twice, once from the definitions
through the engine and once by running the ingest's own parsers over a synthetic report (benchmark report row; for the
kernel breakdown a raw-named kernel, two compiled kernels and a Total), and fails when they differ, when an id in the
golden table moves, or when a definition declares a `schedule` or lacks `est_hours`. Changing an id on purpose means
`python benchmarks/tools/check_identity_ids.py --update` and a reviewed golden diff, which starts a new series.

The legacy model rows with `run_mode` `null` (reports written before the mode tag existed) are not declared: no new
rows are expected for them.

Names of models the suite has not reported yet are the bare-name rule applied; the first real report decides
whether the suite spells one differently (a `granite_` op would fold into `granite`), and the definitions then
need a `benchmark_names` entry.

## Cost and cadence

`spec.est_hours` on a Benchmark is the only cost hint (`perf_bench.yaml`, placeholder until measured). When and how
often a Selection runs belongs to whoever invokes it, never to these files.

## Running

`BENCH_MODE` selects how `make tests TEST_TYPE=perf` uses the engine:

| `BENCH_MODE` | Effect |
|---|---|
| `off` (default) | nothing changes |
| `side-by-side` | the legacy `perf-tests` runs as before; then `make bench-perf` runs the engine over the same models, reading the legacy reports (`BENCH_PERF_RESULTS`), into `$(RESULTS_DIR)-bench`. Nothing is measured twice and the legacy results are not touched; an engine failure is reported, never fatal |
| `replace` | only the engine path; each case's run command runs `make perf-tests` (`BENCH_MODE=off`) once per model |

`BENCH_MODE` is read from the environment as well as the command line, as every `BENCH_*` variable is (a CI cell sets
it through its environment). `replace` therefore skips the legacy perf run wherever it is set.

A case's run command is `tools/run_model_case.py`: it keeps the testcases of its own model, config and mode from the
report, writes them to the case's `raw/` directory (the layout the unchanged XML ingest reads) and a `metrics.json`
with the wall-clock time for a native bundle. The engine hands a command only the variables named in
`BENCH_PASS_ENV`; `bench-perf` adds `BENCH_PERF_RESULTS` itself, other variables the suite needs (`HF_TOKEN`, ...)
have to be named for `replace` to work.

Two ways to ingest the engine's output, both next to the legacy ingest:

| Route | How | Notes |
|---|---|---|
| `xml` | `make bench-collect`, then the ingest action with `xml-dir` | same measurement keys and series as the legacy rows; this is what Jenkins hands to its unchanged ingest step |
| `native` | the ingest action with `bench-dir` and `channel-trust: ci` | the engine's Results as a `bench` bundle; `workflow: integration-tests` files them in the legacy GHA series. `ci` only from a job that runs this repository's code with its own secrets, never on `pull_request`, `pull_request_target` or `workflow_run`; a comment-triggered run passes `lab` |

`.github/workflows/bench-on-demand.yml` is the manual route (`workflow_dispatch`, dev instance only, advisory).
While both paths write rows, a series is counted twice by anything that reads both (the HUD projection, until the
schema owner filters bench rows out).

## Quality

`greedy-parity` is a reference-free Workload: greedy decoding compared with a reference output set bound where a case
is instantiated. It carries no threshold, result or reference.

`model-correctness/` scores the two question samples vendored for `evals/` (50 questions each of slot-12 and slot-14,
declared as Assets with their sha256) with Qwen3-0.6B on the Spyre adapter and records the accuracy
(`make bench-quality`, Selection `model-correctness`; needs a device and `BENCH_TYPE=quality`). Greedy decoding, so no
seed is part of the identity; the ids are the generic profile over `dataset`, `harness`, `model`, `num_fewshot`,
`num_questions`, `record_type` and `task`, pinned in the golden table. The recorder (`tools/run_eval_case.py`) uses
the current public model API; the older `evals/*` scripts are left as they are. No score appears in a log
(`BENCH_QUIET`), and the per-answer text stays in the case directory.

The lm-eval integration under review in hf-adapters#698 is independent: its protocol fingerprint is computed at run
time, which an identity block cannot express yet, so its cases are not declared here.

## Make targets

`mk/bench.mk` wraps the CLI: `bench-validate`, `bench-check-ids`, `bench-list`, `bench-explain CASE=...`, `bench-plan`,
`bench-run` (plans first unless `BENCH_PLAN` names a plan), `bench-collect`, `bench-verdict`, `bench-perf`,
`bench-quality`, `bench-smoke`, `bench-report`. Parameters are `BENCH_*` variables (`BENCH_DIR`, `BENCH_CENTRAL`,
`BENCH_PACK`, `BENCH_SELECTION`, `BENCH_CASES`, `BENCH_TIER`, `BENCH_TYPE`, `BENCH_PURPOSE`, `BENCH_ENVIRONMENT`,
`BENCH_SUBJECT_ID`, `BENCH_RUN_ID`, `BENCH_RESULTS_DIR` (alias `BENCH_OUT`), `BENCH_MODE`). `list`, `report` and
`verdict` need an engine that has the verb.
When the engine, or a verb in the installed version, is missing, a target prints the install command and exits 2; `BENCH_OPTIONAL=1` turns that into a notice and exit 0. `BENCH_ENVIRONMENT` defaults to `one-device` (`environments.yaml`).

## CPU pinning

Pinning is a plan-time request, applied by the engine when it runs the plan; the Makefile contains no `taskset`.

| Variable | Meaning |
|---|---|
| `BENCH_CPU_PINNING` | `none`, `device_local`, `per_device` or `explicit` (empty: the Environment's own request, else none) |
| `BENCH_CPU_SET` | CPU list, only with `explicit` |
| `BENCH_PIN_OPTS` | comma-separated `key=value`: `cores_per_device`, `spread`, `smt`, `layout`, `launcher`, ... |
| `BENCH_REQUIRE_PINNING` | `1`: an unverified or violated placement caps the result at suspect |

```bash
make bench-plan BENCH_CPU_PINNING=per_device BENCH_PIN_OPTS=cores_per_device=4,layout=ranks BENCH_REQUIRE_PINNING=1
make bench-run  BENCH_PLAN=results/bench/plan.json
```

A tensor-parallel case starts its ranks with `torchrun`, one process per rank. With `layout=ranks` the case command must
run the ranks behind the engine's generated `{pin_wrapper}` (`torchrun --no-python ... {pin_wrapper} python script.py`),
which pins each rank by `LOCAL_RANK`; for every other policy the wrapper only passes the command through. Do not use
`BENCH_PREFIX` for pinning: it prefixes the engine process, not the ranks. Pinned and unpinned runs are separate series
through the derived label `env.label.cpu_pinning`; add it to `series_labels` in `project.yaml` to track both.
Thread pools should match the pinned set: record `OMP_NUM_THREADS` and `SPYRE_NUM_CPUS` for the run. Whether pinning
helps a model is measured per node class with the engine's `bench-pinning-matrix`.

The pre-commit hook `bench-validate` checks the generated files and validates; it passes with a notice
when the engine is absent, and fails instead when `BENCH_REQUIRED=1`.
