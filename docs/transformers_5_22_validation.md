# Transformers 5.22 compatibility checks

This project pins Transformers 5.22.0 and tokenizers 0.23.3. Run the commands
below from the repository root. Validation results and historical investigations
belong in [PR #771](https://github.com/torch-spyre/hf-adapters/pull/771), rather
than in this document.

## CPU regressions

```bash
uv sync --frozen --no-default-groups --group test
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run --no-sync python -m pytest \
  tests/cpu/test_tensor_parallel_loading.py \
  tests/cpu/test_spyre_tensor_parallel.py \
  tests/cpu/test_transformers_tp_contract.py \
  tests/cpu/test_rotary_config_compatibility.py \
  tests/cpu/test_vision_transformers_compatibility.py \
  tests/cpu/test_muse_glimmer_mm.py -q --tb=short
```

The loading test launches real Gloo workers at TP2 and TP4, requiring local
socket access. It covers checkpoint slicing, grouped KV replication, packed
and separate expert weights, CPU staging, tied embeddings, exact plan overrides,
and incomplete checkpoints. Native HF forwards and supported CPU adapter paths
are compared in float32; packed Gemma MoE adapter execution requires Spyre.
Separate tests check repeated fullgraph compilation with rowwise bias, private
upstream API signatures, rotary configuration, and vision parity.

## Two-card Spyre regressions

Use an environment with torch-spyre and the pinned project dependencies.
Keep the checkpoint-loading file first: its first case tests automatic process
group initialization before any compiled collectives run.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SPYRE_DEVICES=0,1 COLL_ALLREDUCE_ALGO=PairwisePow2
export HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_OFFLINE=1
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --max-restarts=0 --log-dir=tp-logs --tee=3 -m pytest \
  tests/spyre/test_tp_transformers_loading_spyre.py \
  tests/spyre/test_tp_rowwise_bias_spyre.py \
  tests/spyre/test_tp_embedding_collective_spyre.py -sv --tb=short
```

The ten small-checkpoint cases use fp16, two input sequences, and three generated
tokens. They require finite logits, exact token agreement, and HF-reference
logit parity (`atol=0.02`, `rtol=0.02`) on every rank. The biased linear test
covers repeated compiled prefill and decode; the embedding test checks exact
collective output.

Two numerical cases remain strict expected failures:

- Hybrid Qwen3.5 MoE rank-dependent expert routing: [#772](https://github.com/torch-spyre/hf-adapters/issues/772).
- Gemma 4 MoE decode divergence: [#773](https://github.com/torch-spyre/hf-adapters/issues/773).

Only token/logit mismatches qualify for these xfails. Loading, compilation,
non-finite outputs, and unexpected passes fail the suite. Full-attention
Qwen3.5 MoE coverage does not establish hybrid MoE correctness; production
text smoke tests do not establish reference-logit parity.
[#770](https://github.com/torch-spyre/hf-adapters/issues/770) remains open.

## CI and coverage limits

`_test_matrix.yaml` runs the CPU command in `cpu-tp-tests` and the hardware
command in `spyre-tp-regression-tests` for regression, trunk, unit, and explicit
multicard-smoke runs. Both feed the existing merge-gating result job. Per-rank
hardware logs and CPU JUnit results are uploaded as artifacts.

The loader rejects missing or mismatched Spyre-planned checkpoint weights,
excluding legitimate tied aliases, before HF can allocate generic device
buffers. Its scoped patches affect the process: independent HF model loads
must be serialized while `spyre_tp_loading()` is active.

These tests do not cover every registered production checkpoint, the complete
OOT matrix, or four-card Spyre execution. Vocab-parallel embedding stages integer
masking on CPU at every call, including decode. Measure that cost on the target
hardware with the same environment as above:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/benchmark_tp_embedding.py --iterations=100
```

The benchmark reports per-rank full-call and pre-staged compiled-body latency
for batch sizes one/two and sequence lengths one/64. Each call copies its output
to CPU to synchronize. Compilation and warmup are excluded. The delta includes
host masking, input transfers, and Python dispatch; it is not a pure DMA
measurement or an end-to-end model throughput result.
