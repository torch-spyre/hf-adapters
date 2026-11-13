# Benchmark definitions

Declarative `bench/v1alpha1` definitions read by the `spyre-bench` engine. Nothing here runs in CI today:
the `perf` tier and its Makefile targets are unchanged, and no check depends on this directory.

## Layout

| Path | Content |
|---|---|
| `project.yaml` | Project, Suites `hf-models` (perf) and `model-correctness` (quality) |
| `systems.yaml` | the System every model benchmark runs on |
| `workloads.yaml` | `greedy-parity`, the quality Workload (see below) |
| `hf-models/<model>.yaml` | one Benchmark per model, generated |
| `selections.yaml` | `smoke`, `hf-models-nightly`, `hf-models-weekly`, generated |

Environments and node classes are not defined here; a case gets its Environment from the caller
(`BENCH_ENVIRONMENT`).

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
| `nightly.models` | tier `core`, schedule preferred `nightly`, fallback `weekly`; Selection `hf-models-nightly` |
| `weekly.models` | tier `extended`, schedule preferred `weekly`; Selection `hf-models-weekly` |
| `perf_bench.yaml` `sanity_model` | tier `sanity`; Selection `smoke` holds one of its cases |
| model path | benchmark name: last path segment, lower-cased, unless `benchmark_names` maps it |
| `exclude.yaml` | not mapped; `perf-tests` keeps applying it per architecture |

Each model expands to six cases: `config` (`full-model`, `one-block`) by `mode` (`prefill-only`,
`decode-only`, `generate`), the vocabulary the existing model rows already use. The cases use the
`kernel/layer` workload of the `kernel` pack; the neutral runner name is `kernel-bench`.

`schedule.*` annotations carry cadence (`preferred`, `fallback`, `est_hours`, `priority`) until the
engine has a `schedule` block. `est_hours` is a placeholder. Change a cadence in `perf_bench.yaml`
(`schedule.overrides`, keyed by model path) and regenerate.

### Identity

A benchmark id hashes the component, the benchmark name, identity tags and discriminators; `backend` is
not hashed. Existing model rows hash the discriminators `record_type=model`, `config_name`, `run_mode`
(plus three empty ones). The definitions carry the same name, config and mode, and
`scripts/bench_identity_parity.py` prints the existing-style id next to the id the engine derives:

```bash
python scripts/bench_identity_parity.py [--known ids.tsv]
```

The engine hashes every flattened case and system attribute, so its ids differ from the existing ones
until it can pin a legacy discriminator set. Do not switch a producer to the engine ids before that.

## Quality

`greedy-parity` is a reference-free Workload: greedy decoding (temperature 0) compared with a reference
output set bound where a case is instantiated. It carries no threshold, result or reference. The
existing `evals/` scripts and any lm-eval integration are untouched and independent: lm-eval cases would
be a separate Suite and Workload type.

## Make targets

`mk/bench.mk` wraps the CLI: `bench-validate`, `bench-list`, `bench-explain CASE=...`, `bench-plan`,
`bench-run`, `bench-smoke`, `bench-report`. Parameters are `BENCH_*` variables (`BENCH_DIR`,
`BENCH_CENTRAL`, `BENCH_SELECTION`, `BENCH_TIER`, `BENCH_PURPOSE`, `BENCH_ENVIRONMENT`, `BENCH_OUT`).
When the engine, or a verb in the installed version, is missing a target prints a notice and exits 0; `BENCH_REQUIRED=1` makes that exit 2. `BENCH_*` reach only the `bench-*` recipes.

The pre-commit hook `bench-validate` checks the generated files and validates; it passes with a notice
when the engine is absent, and fails instead when `BENCH_REQUIRED=1`.
