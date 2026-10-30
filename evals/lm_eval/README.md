# Periodic LM evaluation

`run_benchmarks.py` runs the same YAML/JSON suite with an lm-eval harness model wrapper - either `SpyreHFLM`, which uses HF adapters, or `HFLM` for native Transformers.
The native path does not import `hf_adapters` or require torch-spyre.

## Suite definition

Edit [benchmarks.yaml](benchmarks.yaml) to change the workload.
Each entry can have literal lm-eval task/group IDs, sample limit, few-shot count, context length, generation budget, dtype, batch size, prompting settings, seed, and timeout.
Entry values override `defaults`.
The YAML/JSON config is the single source of benchmark defaults; the runner has
no fallback values. Every setting listed in the shipped `defaults` must be
provided either there or in each benchmark entry. Missing or unknown settings
fail validation before model loading. Set `generation: null` to leave generation
settings to the task; an explicit generation mapping requires `max_gen_toks` and
`do_sample: false`. Generation mappings are replaced as a whole, not deep-merged.
A generation budget must be smaller than `max_length`; HFLM may truncate long prompts to fit the remaining context.

The default suite uses 50 questions per task, except AIME24, which has only 30:

| Benchmark | Questions | Generation cap |
|---|---:|---:|
| GSM8K | 50 | 1,024 |
| MMLU: five explicitly listed subjects | 50 per subject (250 total) | scoring only |
| GPQA Diamond zero-shot | 50 | scoring only |
| AIME24 | 30 (complete dataset) | 2,048 |
| IFEval | 50 | 1,280 |

`limit` selects up to N documents uniformly without replacement **per leaf task**, including when a group ID is used. A private RNG derived from `seed` and the leaf task ID selects the same subset across models, backends, and runs, independently of task order. Smaller datasets are evaluated in full. The selected indices are passed to lm-eval's `samples` API and evaluated in dataset order; the harness's prefix-only `limit` is not used.

All four harness RNG seeds and IFEval's language detector also use `seed`.
The shipped suite uses greedy, zero-shot generation with chat templates enabled.
Change the YAML to adjust prompting or task counts, and
regenerate affected references.
Task definitions, stop strings, and scoring/extraction rules come from the
pinned lm-eval version unless the suite explicitly overrides them.

GPQA zero-shot answer-choice ordering is a runner-local exception: lm-eval
0.4.13 shuffles with global RNG state that is not captured in the dataset cache
key. Both backends instead use a private per-document RNG derived from SHA-256
of the suite seed, original question/answers, and preprocessing version. The
seed and version are included in the dataset cache key and result protocol;
answer cleaning, prompts, and scoring remain unchanged.

With `dtype: auto`, Spyre uses the normal HF adapters dtype resolutions
(bf16/fp16); Note that native transformers may use a different dtype
(e.g., fp32 where Spyre uses fp16), depending on the target checkpoint.

Install dependencies in an environment with the appropriate torch build:

```bash
python -m pip install -r evals/lm_eval/requirements.txt
python -m evals.lm_eval.run_benchmarks --dry-run
```

Authenticate with `hf auth login` or `HF_TOKEN`. GPQA requires access to
`Idavidrein/gpqa`; gated model repositories need their own access grants.
Datasets and IFEval's NLTK resources are downloaded on first use.

## One-time GPU references

Use an environment with CUDA-enabled torch and enough GPU memory for the
selected model. Do not use the repository's Spyre/CPU torch lockfile to create
the GPU environment. Install the requirements above, then run:

```bash
python -m evals.lm_eval.run_benchmarks \
  --backend hf --device cuda --write-references \
  --output-dir evals/lm_eval/runs/gpu-reference
```

This reads `nightly.models` from `tests/model_lists/perf.yaml`. Models run
sequentially. To run one model, add `--model ibm-granite/granite-4.1-8b`;
repeat `--model` to select several. `--device cuda:1` selects another GPU.

Compact reference files are written to `evals/lm_eval/references/`, one per
model, with successful benchmarks only. Commit those JSON files to make them
available to scheduled CI. Large run artifacts stay in the ignored `runs/`
directory. Existing reference files are protected; use `--overwrite-references`
when deliberately replacing one. A partial rerun replaces that model's reference
file with the successful benchmarks from that run; it does not merge old data.
`--reference-dir` can select a different reference directory.

## Spyre runs and CI

In the configured Spyre runtime environment:

```bash
python -m evals.lm_eval.run_benchmarks \
  --backend spyre --model ibm-granite/granite-4.1-8b \
  --output-dir evals/lm_eval/runs/spyre-check
```

[LM evaluation](../../.github/workflows/lm-eval.yaml) is scheduled daily at
03:00 UTC and supports manual dispatch. It reads the nightly model list at the
same commit tested by every matrix job. Each model gets a single-card Spyre
runner using the existing build action and the image's installed torch-spyre.
`HF_TOKEN` is supplied from the repository secret. Manual dispatch accepts an
alternate config path and a space-separated model list.

Every benchmark runs in a fresh process. Errors and timeouts are
recorded, remaining benchmarks continue, and any execution failure makes the
command/job fail. A missing
reference does not fail the job, and quality changes are reported without an
automatic regression threshold.

Each run writes a measurement payload, `benchmarks.json`. Diagnostics live under
`attachments/`: the frozen `suite.json`, compact `report.json`, rendered
`summary.md`, and per-benchmark output. Per-benchmark `results.json` contains
the detailed harness output, and `run.log` captures execution logs. Worker stdout
and stderr are also streamed live to the console, including loading and evaluation
progress bars. Each completed benchmark prints its metric scores and, when
compatible references exist, reference scores and relative percentage changes.
Silent periods can still occur during compilation; benchmark timeouts remain
enforced independently of whether the worker prints output. A temporary
`.worker-result.json` transfers each worker's result to the supervisor and is
consumed after the worker exits.
CI attaches the
Markdown summary to the job and retains artifacts for 30 days, including on
failure. Per-sample model outputs, metrics, document IDs, and provenance hashes
are saved in `samples.json` attachments. Dataset documents, input prompts, and
targets are omitted to avoid publishing gated source text.
Output directories must be empty; omitting `--output-dir` creates a dated run.

Each CI evaluation job also derives its tested software artifact ID on that
job's runner, using the shared torch-spyre identity script and library from
`main`. The base image is read from `/home/senuser/spyre_artifact.json`, falling
back to `/home/senuser/spyre_artifact_id.txt`. The installed change is
`hf-adapters@<checked-out-sha12>`; torch-spyre stays from the image.

The job summary and capture-step outputs expose `artifact_id` and
`base_artifact_id`. The downloaded `lm-eval-<model-key>` artifact includes
`attachments/artifact-id.txt`, containing the full record:

```text
<artifact_id>|<base_artifact_id>|hf-adapters@<sha12>
```

The attachments also retain `hf-adapters-sha.txt`, `identity-library-sha.txt`,
and any image stamp files available on the runner. Capture is staged outside
the evaluation output and preserved after evaluation, including failures.
Each matrix job retains its own record. An unstamped image produces a warning
and no invented ID; identity capture errors do not prevent evaluation.

Use the first field of the record as the bundle's `--artifact id:<UUID>`.
Keep the full record for database registration: a derived UUID is not proof
that the artifact is already registered. This workflow captures identities
and publishes GitHub result archives; it does not register artifacts or
upload measurements to ClickHouse/Artifactory.

## Dashboard bundle

The runner produces measurements for **kind `benchmarks`, tier `quality`**.
Its output directory is a run archive; use the sealing command below to create
the `bundle.json` and `results/` layout expected by the ingestion CLI.
CI retains the run archive as a GitHub artifact. Database upload is not wired.

```text
output/
  benchmarks.json
  attachments/
    suite.json
    report.json
    summary.md
    <model-key>/<benchmark-id>/
      results.json
      samples.json
      run.log
```

`benchmarks.json` contains `{"benchmarks": [...]}`. Each entry has:

- `name`: the YAML benchmark ID, such as `gsm8k` or `mmlu`.
- `props`: full model ID, leaf task ID, actual `num_fewshot`, extraction `filter`,
  and a `protocol` fingerprint of the leaf's settings, checkpoint, precision,
  task/scoring versions, preprocessing, and sampled document/prompt/target hashes.
  This keeps different workloads distinct without including the backend, device
  ordinal, runtime versions, or run timestamp in the benchmark identity.
  IFEval also has `sample_unit: prompt` or `sample_unit: instruction`.
  Metrics for the same task/filter/unit with different observation counts get
  `observation_count` to distinguish their identities.
- `backend`: `spyre`, or the native device type (`cuda`, `cpu`, `mps`, etc.).
- `measurements`: raw scores using metric names such as `exact_match` and
  `acc_norm`; the lm-eval filter suffix is represented by `props.filter` instead.
- `iterations`: actual observations behind those scores, not the configured limit.

Entries with the same task, filter, observation unit, and count share a
measurements mapping. Different denominators get separate identities with
disjoint metrics: IFEval prompt scores count prompts, while instruction scores
count instructions, even when both counts happen to be equal. This distinction
is required because the ingestion writer merges equal benchmark identities and
sums their iterations; splitting only the JSON rows would lose the denominators.
Failed benchmarks and group-level aggregates are retained in the diagnostic
report but not exported as leaf-task measurements. Reference scores and percentage
changes remain in the diagnostic report; they are not ingested as quality scores.
The payload is updated after every benchmark, retaining successful results when
another benchmark fails. Everything under `attachments/` is kept with the bundle
but must not be ingested as measurements.

### Seal a run for upload

On the machine preparing the bundle, install the optional ingestion dependency:

```bash
python -m pip install -r evals/lm_eval/requirements-ingest.txt
```

After the evaluation finishes, name the software artifact that was actually
tested and its architecture:

```bash
python -m evals.lm_eval.bundle \
  --run-dir evals/lm_eval/runs/spyre-check \
  --out evals/lm_eval/runs/spyre-check.tgz \
  --artifact 'id:<tested-artifact-uuid>' --arch x86_64

python -m spyre_clickhouse_ingest results \
  --from-bundle evals/lm_eval/runs/spyre-check.tgz --validate-only
```

Find the UUID on the tested artifact's dashboard page under **Add results →
Identity**. Numeric GitHub result-archive IDs are not software artifact UUIDs.
Alternatively, pass `--artifact 'image:<repository>@sha256:<digest>'` for a
recorded image. Architecture is required and describes where evaluation ran,
not the machine packaging the results. Local validation cannot establish that
an artifact exists in the database or matches the actual tested environment.

`--component` defaults to `hf-adapters`; it identifies the suite's owner and
participates in benchmark identity. `--run-url` can preserve the original CI
run link. `--out` accepts a new directory, `.tgz`, or `.tar.gz` outside the input
run. Existing destinations are protected.

The sealed output contains:

```text
bundle.json             # kind=benchmarks, test_type=quality, artifact, arch, hashes
results/benchmarks.json
attachments/...         # all diagnostic files copied from the run
```

The command uses the official ingestion validator and an in-memory writer
preview, rejecting empty results, invalid metadata, and identities that would
merge measurement rows. It copies every attachment and lists its SHA-256 in
the manifest before sealing. This avoids the upstream `--benchmarks-dir`
packaging path's omission of attachments. The source run is unchanged.

The manifest uses the bundle schema version (currently **1**); the database is
schema **v2**. Execution timestamps come from `report.json`'s `created_at` and
`updated_at`; packaging time is recorded separately as `sealed_at`. Historical
runs without execution timestamps are left undated.

For an older run that the writer preview rejects because of colliding
identities, regenerate `benchmarks.json` from its compatible `attachments/report.json`
using `dashboard_results(report)` before sealing. The existing recorded scores
and counts are sufficient; rerunning inference is unnecessary. Old reports
missing actual observation counts or device information need those facts
recovered first, as in the historical combined reference bundle.

Once validation passes, the standard ingestion CLI can upload the sealed
bundle with `results --upload <bundle.tgz> --wait`, using the configured
Artifactory relay settings and token. Sealing and `--validate-only` perform no
network requests, uploads, or database writes.

## Comparing with a reference

For each task/metric/filter, the report calculates:

```text
percent_change = 100 * (current - reference) / abs(reference)
```

For example, 0.70 versus a reference of 0.80 is **−12.5%**, not −10 percentage
points. The JSON also contains the absolute score delta and metric direction.
A zero baseline has an undefined percentage (`null` / `N/A`) but retains its
absolute delta. Missing metrics or references are explicitly marked.

Comparison requires the same effective suite settings, pinned harness and scoring
library versions, task versions, loaded checkpoint SHA, model dtype, and hashes
of the actual sampled documents, prompts, and targets. The sampling algorithm and
seed are recorded in the protocol; original document indices are recorded alongside
the hashes in `attachments/report.json` and reference files. Unknown checkpoint SHAs
or any mismatch produce `incompatible_reference` with the differing fields.
References remain reusable when unrelated benchmarks are changed. Torch and
torch-spyre versions are recorded but are not required to match between GPU
and Spyre. Local checkpoints without a Hub commit SHA can run, but cannot be
automatically compared as a verified matching reference.

Reports and references use the same version-1 JSON format:
`models[model][benchmark]` contains the execution status, protocol, and a flat
`metrics` list. Each metric row contains `task`, `metric` (including its filter),
`value`, `higher_is_better`, `iterations`, and `num_fewshot`. When comparable, that same row also contains
`reference`, `delta`, `percent_change`, and `comparison`. Reference files are
reports restricted to one model's successful benchmarks.

## Local checks

```bash
python -m pytest -q evals/lm_eval/test_benchmarks.py
python -m pytest -q evals/lm_eval/test_dashboard.py
python -m pytest -q evals/lm_eval/test_workflow_identity.py
python -m pytest -q evals/lm_eval/test_bundle.py  # requires requirements-ingest.txt
python -m pytest -q evals/lm_eval/test_gpqa.py
python -m unittest evals.lm_eval.test_spyre_lm_eval_backend
```

The suite tests include offline native HFLM scoring and generation with a tiny
local model, reference compatibility, percentage calculations, model selection,
and continuation after failures. The optional bundle tests use the official
ingestion validator and writer to check archive layout, attachment hashes,
metadata rejection, and preservation of IFEval observation counts.
