# Transformers 5.22 upgrade validation

The initial validation below describes commit `fbc95c3`. The
[PR review follow-up](#pr-771-review-follow-up) records additional issues
found and fixed during independent review.

Validated on 2026-11-11. The dependency pin is `transformers==5.22.0`;
`uv.lock` also updates its required `tokenizers` dependency to 0.23.3.
No other locked package versions changed.

## Compatibility changes

- Replace the removed TP integration API with the 5.22 distributed styles,
  DTensor loading placeholders, shard operations, and forward transforms.
  Loaded adapter parameters remain local tensors with Spyre-aware DMA layouts.
- Preserve exact layer overrides, complete explicit plans, grouped KV-head
  replication, uneven vocabulary gathers, rowwise bias handling, and embedding
  masking/reduction.
- Preserve packed expert gate/up slicing for fused and separate checkpoint
  tensors. Keep CPU-staged weights on CPU through converters' final placement,
  and preserve expert reductions after replacing DTensor placeholders.
- Stage Phi3's fused projections as full CPU weights before the adapter splits
  them. The attention output and MLP down projections retain rowwise TP.
- Read partial rotary fractions from `rope_parameters` for Phi3 and Qwen3.5.
- Use the new Muse Glimmer vision interpolation helper and its new tensor axes.
- Correct CPU test collection, canonical prompt encoding, reference-before-
  preparation ordering, and a stale token-classification helper import.

The TP bridge temporarily patches Transformers' loader functions inside
`spyre_tp_loading()`. Tests verify restoration after exceptions and that
ordinary device placement is retained for parameters without CPU staging.

## Environments

| Environment | Versions / configuration |
| --- | --- |
| CPU | macOS, Python 3.13, Torch 2.14.1, Transformers 5.22.0, tokenizers 0.23.3 |
| Spyre | Linux x86_64, Python 3.12.15, Torch 2.14.1+cpu, torch-spyre `0.6.0.dev432+head.g52aeb859`, Transformers 5.22.0 |
| Device allocation | One-card development pod for the real checkpoint; temporary two-card pod for TP |

Spyre dependencies were installed under `/tmp/hf522-deps`, with source under
`/tmp/hf522-src`. Tests used `PYTHONPATH=/tmp/hf522-deps:/tmp/hf522-src`.
The temporary two-card pod was deleted after collecting the logs; the existing
development pod remains running.

## CPU checks

The focused TP/rotary suite passed **27 tests**:

```bash
.venv/bin/python -m pytest \
  tests/cpu/test_tensor_parallel_loading.py \
  tests/cpu/test_spyre_tensor_parallel.py \
  tests/cpu/test_rotary_config_compatibility.py -q
```

The distributed test launches real Gloo workers at TP2 and TP4. It checks
checkpoint slicing, safetensors reads, grouped KV replication, scalar conversion
state, uneven linear/embedding shards, packed and separate expert tensors,
tied embeddings, exact plan overrides, and numerical parity for Qwen3, Granite,
Gemma3, Phi3, Qwen3.5 dense, and Qwen3.5 MoE. Every family is checked through
both the HF checkpoint loader and prepared adapter generation.

CPU TP tests use float32 with compilation disabled. Reference forward logits
use `atol=2e-6, rtol=2e-5`; generation logits use `atol=2e-5, rtol=2e-4`, with
exact greedy-token agreement. The CPU MoE emulator converts transported expert
indices to int64 because CPU indexing does not accept Spyre's float transport.
Phi3 and Qwen3.5 additionally cover rotary fractions 0.25, 0.75, and 1.0.

The broad CPU/unit and adapter/module coverage suite passed **385 tests**
(139 model-harness cases deselected) in 79.21 seconds:

```bash
.venv/bin/python -m pytest tests/cpu \
  tests/test_adapter_coverage.py tests/test_module_config_identity.py \
  tests/test_generate_module_test_matrix.py -m 'not model_harness' -q
```

The real `Qwen/Qwen3-0.6B` CPU auto-loader accuracy and multibatch generation
tests also passed. These exercise canonical chat-template input encoding.

## Spyre checks

`tests/spyre/test_e2e_token_compare_spyre.py` passed for the real
`Qwen/Qwen3-0.6B` checkpoint on one card, covering prefill and four greedy
decode steps (160.75 seconds).

The new `tests/spyre/test_tp_transformers_loading_spyre.py` uses small random
checkpoints, fp16, batch size two, and three generated tokens. It loads actual
checkpoints through the public auto-loader and runs real Spyre compilation.
It requires exact generated tokens, finite logits, and HF-reference logit parity
at `atol=0.02, rtol=0.02` on every rank.

Two-card runs passed for Qwen3, Granite, Gemma3, Phi3, Qwen3.5 dense, and
Qwen3.5 MoE: **six model cases plus the embedding collective, on both ranks**.
Gemma3 exercises a single KV head replicated across two query-head shards.
Phi3 and both Qwen3.5 variants use a 0.75 rotary fraction. The embedding
collective requires exact output equality. The MoE rerun passed in
173.53 seconds after correcting converter placement; no discovered test
failures remain unresolved.

Reproduce the complete device suite on a two-card allocation:

```bash
SPYRE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m pytest \
  tests/spyre/test_tp_transformers_loading_spyre.py \
  tests/spyre/test_tp_embedding_collective_spyre.py -sv --tb=short
```

## Static checks and scope

Ruff and Black passed for all changed Python files. Mypy passed for all
58 production Python files. `uv lock --check` and `git diff --check` passed.

This is targeted upgrade validation, not a run of every registered production
checkpoint or the full OOT hardware matrix. Four-rank validation ran on CPU;
physical Spyre validation used two cards. No throughput benchmark was run.

Local validation logs are retained in `/tmp/hf522-*.log`, including
`hf522-tp-final.log`, `hf522-validation-final.log`, `hf522-model-checks.log`,
`hf522-qwen-device.log`, `hf522-tp2-device.log`, `hf522-tp2-phi3.log`,
`hf522-tp2-qwen-extra.log`, `hf522-tp2-moe.log`, and `hf522-mypy-all.log`.
The Qwen extra log includes the initial MoE staging failure;
`hf522-tp2-moe.log` records the successful rerun.

## PR #771 review follow-up

Reviewed `fbc95c3d2ebc9473adcc0fc069732da53380f8db` against
`188aec96f61927e9a76b183d33dddb4639edd92b`. The original PR was not ready to
merge: its CI exposed failures that the initial small-checkpoint tests missed.

### Findings and fixes

| Finding | Fix and regression coverage |
| --- | --- |
| Automatic TP initialization fails with `No distributed backend found for device type 'spyre'`. Pre-initialized test groups hid this. | Initialize the combined Gloo/SpyreCCL group inside the scoped loading bridge. Omit `device_id`, which would trigger unsupported `split_group` calls during mesh construction. Test fresh initialization, preservation of existing groups, restoration after errors, and compiled loading after destroying the fixture's setup group. |
| Pixtral no longer stores a position-indexed frequency table; indexing its new vector crashes on real image positions. | Build matrix RoPE from the axial inverse frequencies and explicit row/column coordinates. Compare complete towers against HF on rectangular images, singly and packed together. |
| Granite Vision's `pack_image_features` now returns a list of image tensors. Unpacking two values crashes for one image and loses the second image when there are two. | Concatenate the returned list. Check all six deepstack/spatial injection points for one and two images against HF's real projectors and packing. |
| Gemma 4 vision's default RoPE type is now `axial`; the old `default` guard rejects valid checkpoints. | Accept axial RoPE and compare the complete tower with HF, including padded image positions and both standardization settings. |
| Gemma 4 MoE CPU staging only matches multimodal checkpoint names. Text-only checkpoints retain DTensor expert weights. | Add the `model.layers.*.experts` staging rules. Run real TP2/TP4 checkpoint conversion and numerical forward checks, plus compiled TP2 generation. |
| Vocabulary sharding leaves `padding_idx` in global coordinates, potentially outside the local embedding. | Translate the index on its owning rank and clear it elsewhere. Cover nonzero padding IDs in the final vocabulary shard, including real uneven TP2/TP4 shards. |
| Dense Qwen3.5's scalar DeltaNet gates use narrow column shards, and batched decode fails with a sub-stick copy/layout error. | Keep `in_proj_a` and `in_proj_b` replicated; retain sharding of the larger projections. Check actual gate/QKV shard sizes in Gloo TP2/TP4 and run compiled hybrid attention on TP2. |

The Gloo tests additionally cover Gemma 4 dense generation and Qwen3.5
dense/MoE hybrid linear/full attention. Gemma 4 MoE's CPU test covers loading
and native TP forward; its packed adapter execution requires Spyre hardware.

The dense and MoE `spyre_tp_replicated_linear_modules` functions are live
adapter hooks: `hf_common._resolve_tp_plan` discovers and calls them with
`getattr` for `tp_plan="auto"`. The dense hook replicates only the two small
DeltaNet gates; the pre-existing MoE hook replicates attention and shared-expert
projections while leaving routed experts sharded. Explicit plans bypass these
automatic policies. Neither hook can be removed as unused code.

The full repository-level run also uncovered stale weekly-scan test fixtures
already present in the PR base: the fake sink lacked
`get_models_at_snapshot_date`, and constructor expectations omitted
`model_list_file`. Those fixtures now implement the current interface; no
weekly-scan production behavior changed. The existing `models-ops` dependency
`python-dotenv` was installed to run these tests.

### Review validation

- Full CPU/unit suite: **572 passed, 6 skipped, 139 model-harness cases
  deselected**. This includes the repository-level tests and Gloo TP2/TP4.
- Real CPU auto-loader accuracy and multibatch generation: **6 passed** across
  Qwen3-0.6B, Phi-3.5-mini-instruct, and Qwen3.5-2B.
- Real CPU token classification: **1 passed** for `dslim/bert-base-NER`.
- Six offline vision regression cases passed with HF numerical parity.
- The original seven compiled TP2 cases passed on **both ranks** (760.97 s),
  including automatic initialization and the embedding collective.
- Gemma 4 dense and Qwen3.5 dense hybrid compiled generation also passed on
  both ranks: **nine distinct passing TP2 cases** across the focused runs.
- Production TP2 smoke tests passed on **both ranks** for
  `ibm-granite/granite-3.3-8b-instruct` and `Qwen/Qwen3.6-35B-A3B`, producing
  `The capital of France is Paris.` and `The capital of France is **Paris**.`
  respectively. These smoke tests do not establish full-checkpoint logit parity.
- Production Granite Vision and Ministral/Pixtral VLM tests passed on Spyre.
- Production Gemma 4 E2B VLM passed on the two-card pod using one card:
  all five teacher-forced steps met the existing cosine threshold, with 4/5
  top-token agreement (the test reports, rather than requires, top-token equality).
- Production Gemma 4 26B-A4B VLM also passed using one card: all five
  teacher-forced steps had cosine similarity between 0.999324 and 0.999928,
  with 4/5 top-token agreement. The combined Gemma VLM run passed both tests
  in 1358.09 seconds; its log is `/tmp/pr771-gemma-vision-fixed.log`.
- All pre-commit hooks passed, including Black, Ruff, lockfile validation, and
  mypy; a separate mypy run passed across all 58 production files.

### Unresolved TP correctness findings

The review is **not all green**. Two added random-checkpoint hardware cases
remain failing, with the assertions retained (no xfails or relaxed tolerances):

| Case | Observed failure |
| --- | --- |
| `qwen3_5_moe_hybrid` | At generation step 2, the largest logit error is about 0.134. A routing probe found rank 0 selecting experts `[3, 0]` while rank 1 and the CPU reference selected `[3, 2]` for the same token. The reference router's selection margin was about 0.00171. Reducing partial outputs from different experts violates the TP contract. The linear-attention stages themselves were close to CPU results; routing needs consistency across ranks. |
| `gemma4_moe` | Both ranks generated one wrong token in the two-sequence, three-token test. Maximum absolute logit errors grew from 0.00586 to 0.03760 to 0.60535 over the three generation steps. This is substantial numerical divergence, not simply a near-tie in the final argmax. The root cause has not been isolated. |

Both hybrid Qwen fixtures pass the identical fp16 checks with real Gloo TP2,
and the broader Gloo TP2/TP4 checks pass in fp32. The Gemma 4 MoE CPU checks
cover checkpoint loading and native TP forward, not packed adapter execution.
The first Gemma 4 MoE hardware fixture incorrectly enabled PLE, which the
adapter explicitly rejects; the final fixture disables PLE, as in the supported
MoE checkpoint. The numerical failure above is from that corrected fixture.

The gather probe independently reproduced the dense Qwen failure for a
`[2, 1, 2]` local output. Equal-shard reshape and transpose alternatives also
failed on sub-stick copies; a `[2, 64, 2]` prefill output passed. Replicating the
small scalar gates resolves the model's decode failure without changing its
larger TP projections.

These findings establish current failures on the reviewed tree. The same two
fixtures were also run on base commit `188aec9` with Transformers 5.15.0,
tokenizers 0.22.2, and the same two cards and torch-spyre revision. Both failed:
Gemma 4 MoE failed before generation with `Unsupported copy types: src on
spyre:0 dst on spyre:0` (and the equivalent on rank 1), while hybrid Qwen3.5
MoE produced three wrong tokens. This confirms that the text-only Gemma staging
gap predates the upgrade. The earlier base failures alone do not establish the
origin of the remaining numerical failures. Log: `/tmp/pr771-base-moe.log`.

A second base run backported only the two text-only Gemma expert-staging
patterns so generation could execute. Both ranks then failed at the same
Gemma token position (sequence 1, generation step 2), with the same absolute
token-ID difference of 25. For Qwen, the same run set the old config's
`partial_rotary_factor` field to the fixture's explicit 0.75 value, matching
the old adapter's configuration interface. Both ranks then passed token
equality but failed the logit comparison at generation step 2, as on 5.22.
Thus both numerical test failures also occur before the Transformers upgrade
when those prerequisites are corrected. This comparison does not prove that
their internal numerical causes are identical across versions.
Log: `/tmp/pr771-base-isolation.log`; the isolated base source and dependency
overlays remain on the pod under `/tmp/pr771-base` and `/tmp/pr771-base-deps`.

An experimental Gemma FP32-router change did not compile: Spyre reported
`No mechanism to scatter elements from one stick to multiple sticks` for a
`[128, 4]` FP32 routing tensor. That experiment remains outside the repository;
it is not a validated fix. Log: `/tmp/pr771-gemma-router-probe.log`.

The outstanding failures prevent claiming that all Spyre tests pass.

The CPU/unit command was:

```bash
.venv/bin/python -m pytest tests/cpu tests \
  --ignore=tests/spyre --ignore=tests/test_modules.py \
  --ignore=tests/test_modules_custom.py -m 'not model_harness' -q --tb=short
```

The expanded two-card suite is reproducible with the original device-suite
command above. It now includes Gemma 4 dense/MoE and Qwen3.5 dense/MoE hybrid
attention. Device runs use `COLL_ALLREDUCE_ALGO=PairwisePow2` and
`HF_DEACTIVATE_ASYNC_LOAD=1`, matching the production multicard CI settings.

Review logs are retained locally under `/tmp/pr771-*.log`. Hardware logs live
under `/tmp/pr771-*.log` on `a5-deepview/benjams-dev-pf-x2` and
source is isolated in `/tmp/pr771-src`.
The two-card pod uses the user-provided `benjams-spyre-dev-pf-x2.yaml` and the
same torch-spyre commit as CI, `21f5bddf47a1e73901ffbaf04d10e2a1fb145536`.
The existing one-card pod was evicted during vision validation for exceeding
its ephemeral-storage limit. Its successful Granite/Ministral results were
observed before eviction, but their remote logs were not recovered. Subsequent
runs use the healthy two-card pod with model checkpoints on the existing PVC.
The one-card pod has not been deleted or recreated; both supplied manifests are
unchanged.

The original CI RoBERTa load failure was a PCIe/BusFence hardware error. Its
[fresh-pod retry](https://github.com/torch-spyre/hf-adapters/actions/runs/43026516706/job/130551323038)
passed. The remote PR still points to `fbc95c3`; the review edits are local and
have not been committed or pushed, so the remote checks do not test these fixes.

The 139-case production CPU model matrix and the complete OOT hardware matrix
were not rerun locally. Existing PR CI results provide broader checkpoint/OOT
coverage; the review reruns target the failures and the changed loader paths.
The dedicated TP loading/collective test files are currently manual checks:
the multicard CI jobs invoke `run_multicard_smoke.py`, not these pytest files.
The general CPU workflow is commented out in `_test_matrix.yaml` (issue #449).
The new regression tests therefore do not yet provide automatic PR gating.
No performance claim or four-card hardware claim is made.
