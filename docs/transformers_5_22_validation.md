# Transformers 5.22 upgrade validation

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
