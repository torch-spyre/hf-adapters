# Weekly scan with randomly initialized weights

The default weekly scan loads pretrained checkpoints. Add `--random-weights` to
construct the same Transformers model classes from their configs instead:

```bash
python -m tests.spyre.weekly_generation.weekly_test \
  --mode embedding --fetch --top-k 10 --snapshot-date 2026-11-04 \
  --random-weights --seed 42 --models-per-process 1 \
  --write-to-csv results/random.csv
```

Run on a Spyre host with the repository root on `PYTHONPATH`. Configs, tokenizer
files, SentenceTransformer module configs and any required custom model code must
still be available. Standard Transformers and SentenceTransformer weight loaders
are replaced only inside the evaluation child process. Custom loaders bypassing
those APIs and tensor-parallel initialization are not supported by this mode.

Initialization is deterministic per model, subfolder, model class and seed. Each
load starts in float32 before casting to the requested dtype, so the CPU and Spyre
paths use matching initial values. This requires full model RAM during
initialization, as well as the usual accelerator memory and compilation time.

The existing verification criteria are unchanged: generative top-1 token agreement
and embedding token cosine similarity. Random weights can change numerical
behavior and token margins, so a pass/fail difference deserves investigation.
The scan tests architecture execution, not pretrained checkpoint correctness or
loading. Random mode therefore requires CSV output; it cannot write to the
current ClickHouse checkpoint tables. A neighboring `.run.json` identifies the
weight mode and seed for both pretrained and random CSV runs.

## Reproduce the top-10 comparison

```bash
python -m tests.spyre.weekly_generation.compare_weight_modes \
  --output-dir results/top10 --snapshot-date 2026-11-04
```

This freezes the first ten rows of each `resources/*_DEPRECATED.csv`, applies the
weekly parameter ceiling, runs each mode with one model per worker, and writes
input lists, logs, per-mode CSVs and `comparison.json`. Models above the ceiling
are reported as skipped in both modes, not as matching successful evaluations.
Use `--weights pretrained` or `--weights random` to run the halves separately, or
`--summarize-only` to rebuild the comparison from their existing results. Use a
new output directory for a new experiment; completed CSVs are not overwritten.
Incomplete pairs are reported as pending and excluded from the comparison count.

To compare only the first 100 generative models from the same catalog:

```bash
python -m tests.spyre.weekly_generation.compare_weight_modes \
  --mode generative --top-k 100 --output-dir results/generative-top100 \
  --snapshot-date 2026-11-04 --seed 42
```

The ranking is taken before filtering, so this includes any size skips in the
100-model accounting. The larger models require correspondingly more host and
accelerator memory; the parameter ceiling does not guarantee that they fit a
particular test host.
