# Spyre Numerical Accuracy Findings

Date: 2026-04-24
Model: Qwen/Qwen3-0.6B (28 layers, hidden=1024, head_dim=128)
torch-spyre: upstream main @ `223312d`

## MMLU Evaluation: GPU vs Spyre

| Metric | GPU (A100) | Spyre |
|--------|:-:|:-:|
| Accuracy | 22/50 (44.0%) | 22/50 (44.0%) |
| Same prediction | 50/50 (100%) | — |
| Avg time/sample | 0.2s | 22s (cached), 110s (first compile) |

**All 50 MMLU predictions were identical.** MMLU requires only 1 generated token (A/B/C/D), so the answer is determined by prefill logits. Prefill accuracy matches GPU exactly.

## Decode Divergence

Decode tokens 2+ diverge from GPU due to error accumulation through layers. Verified with `test_e2e_token_compare_v2.py` (fill+expand mode):

| Step | GPU Token | Spyre Token | Match | Max Logit Diff |
|------|-----------|-------------|:-----:|:-:|
| prefill | Paris | Paris | OK | 0.20 |
| decode-1 | . | . | OK | 0.18 |
| decode-2 | The | . | FAIL | 18.5 |
| decode-3 | capital | , | FAIL | 17.3 |
| decode-4+ | ... | ... | FAIL | 12-18 |

### Root Cause: RMSNorm dl16 Precision

Isolated by running the CPU fill-mode path with individual ops swapped to Spyre:

| Variant | Token Match (9 steps) |
|---------|:-:|
| All CPU (control) | 9/9 |
| Spyre softmax only | 9/9 |
| Spyre RMSNorm only | 6/9 |
| Spyre softmax + RMSNorm | 6/9 |

**Spyre's dl16 RMSNorm accounts for the divergence.** The fp16-only variance/rsqrt computation without f32 upcast produces ~2-5% mean relative error per layer, which compounds across 28 layers and multiple decode steps.

### RMSNorm Error by Model (real weights, seq=128)

| Model | hidden | fp16 vs f32 | Spyre dl16 vs f32 | dl16 extra |
|-------|:------:|:-----------:|:-----------------:|:----------:|
| Granite 3.3 8B | 4096 | 3.72% | 4.99% | +1.27% |
| Granite 3.3 2B | 2048 | 1.48% | 2.51% | +1.03% |
| Qwen 2.5 7B | 3584 | 3.15% | 5.07% | +1.92% |
| Qwen 3 0.6B | 1024 | 0.66% | 1.73% | +1.07% |
| Llama 3.2 3B | 3072 | 2.47% | 3.78% | +1.31% |
| Mistral 7B v0.3 | 4096 | 4.11% | 4.84% | +0.73% |

Measured with vanilla fp16 RMSNorm (manual Python loop forcing true fp16 accumulation) vs Spyre, using real model weights and input scale 1/sqrt(hidden_size).

## Softmax Leak (Secondary)

Spyre softmax does not fully zero `-inf` masked positions:

```
CPU masked region max:   0.00000000  (exact zero)
Spyre masked region max: 0.00004900  (~2e-5 per position)
```

With 62 masked positions leaking ~2e-5 each, the total leaked probability is ~0.13%. When multiplied by large garbage KV values (100+), this can contribute to output error. However, the isolation test showed **softmax alone doesn't cause token divergence** (9/9 match with Spyre softmax only).

## torch-spyre Bug Fixes and Issues

### Fixed: torch.cat size-1 (#1760)

`torch.cat([cache, new], dim=2)` produced incorrect results when `new` had size 1 along the concat dimension. Fixed on upstream main.

```
Before fix: new_seq=1: max_diff=4.9248
After fix:  new_seq=1: max_diff=0.0020
```

### Open: spyre.overwrite (#1765)

`torch.ops.spyre.overwrite` produces incorrect results in eager mode. The overwritten position shows ~2000x more error than untouched positions.

```python
# Reproducer
cache = torch.randn(1, 8, 64, 128, dtype=torch.float16)
new_val = torch.randn(1, 8, 1, 128, dtype=torch.float16)
ref = cache.clone(); ref[:, :, 0:1, :] = new_val
cache_sp = cache.to("spyre"); new_val_sp = new_val.to("spyre")
torch.ops.spyre.overwrite(input=new_val_sp, output=cache_sp, dims=[2], offsets=[0])
# written row: max_diff=4.4102  (expected: 0.002)
```

Also returns `None` in eager mode (in-place), requiring a workaround:
```python
result = torch.ops.spyre.overwrite(input=..., output=cache, ...)
if result is not None:
    cache = result
```

**Adapter status (migrated):** `hf_common.py`'s `kv_cache_update` no
longer uses `torch.ops.spyre.overwrite`. It was deprecated (torch-spyre#2488)
and the write was migrated to a native slice assignment
(`cache[:, :, pos:pos+seq_len, :] = k`), which sidesteps both the eager
numerical error and the `None`-return issue above. This finding is
retained as a record of the op's behavior; it no longer affects the
adapter. (The separate per-offset compile specialization is a property of
the offset being a compile-time constant, not of the op — it persists
after the migration; see ARCHITECTURE.md "Open Work".)

### Open: RMSNorm wrong on non-contiguous inputs (#1781)

Synthetic fp16 RMSNorm on Spyre is numerically wrong when the input is a
non-contiguous `[B, H, S, D]` view, while the same values become accurate after
`.contiguous()`.

Observed on Spyre with `tests/test_rmsnorm_noncontig_repro.py`:

```text
RMSNorm compare
  noncontig_vs_cpu     max=  7.9531 mean=  1.0955
  contig_vs_cpu        max=  0.0312 mean=  0.0015
  tok_vs_cpu           max=  0.0156 mean=  0.0017

Patched module compare
  mod_noncontig        max=  7.1602 mean=  0.8328
  mod_contig           max=  0.0312 mean=  0.0015
  mod_tok              max=  0.0156 mean=  0.0017

Compiled wrapper compare
  compiled_contig      max=  0.0312 mean=  0.0015
  compiled_clone       max=  7.9531 mean=  1.0955
```

Issue filed: `torch-spyre/torch-spyre#1781`

### New: model-side decode bug is upstream of RMSNorm

The latest layer-0 decode-fill repro shows that Spyre `q_proj/k_proj` is
already wrong on identical inputs before per-head RMSNorm runs, so the current
Qwen3 decode divergence cannot be attributed to RMSNorm alone.

From `tests/test_qwen3_layer0_kpath_repro.py`:

```text
q_proj_raw               full_max=  7.9688 ti_max=  4.6494 ti_mean=  0.2394
k_proj_raw               full_max=  7.7734 ti_max=  7.7715 ti_mean=  0.3039
q_norm                   full_max= 48.7188 ti_max= 46.9219 ti_mean=  1.8808
k_norm                   full_max=389.3750 ti_max=328.6250 ti_mean=  5.0129
```

Interpretation:
- non-contiguous RMSNorm is a real upstream bug and now has its own issue
- the active model decode failure still needs a separate `q_proj/k_proj`
  investigation, because raw Q/K already diverges before RMSNorm amplifies it

### Open: compiled head-layout transform returns wrong values (#1783)

The compiled Qwen-style head shaping transform

```python
x.view(B, S, H, D).transpose(1, 2)
```

is numerically wrong on Spyre for 64-token fp16 shapes, even without linear,
RMSNorm, RoPE, or any model code.

Observed with `tests/test_qwen3_head_layout_repro.py`:

```text
seq1_q / seq1_k:
  heads_only max ~= 0.001

seq64_q_contig:
  heads_only full_max=3.1758

seq64_k_contig:
  heads_only full_max=2.9580

seq64_q_strided:
  heads_only full_max=3.0830

seq64_k_strided:
  heads_only full_max=2.9893
```

Issue filed: `torch-spyre/torch-spyre#1783`

## Fill-Mode Decode Logic

Verified correct on CPU (9/9 token match vs HF DynamicCache). The fill+expand block-based decode produces identical results to HF's single-token DynamicCache when run in f32 on CPU. All divergence is from Spyre's dl16 numerics.

## GSM8K Generation Comparison (Short Sample)

Question: "Marin and his neighbor Nancy each eat 4 apples a day. How many apples do they eat in 30 days?"

**GPU (stock HF, A100):**
```
Answer: There are 4 apples each day, so in 30 days, they eat $4 \times 30 = 120$ apples.

Answer: 120
```
43 tokens, coherent reasoning, pred=120

**Spyre (adapter, dl16):**
```
Answer::  5:::55

::: :::::: the


 6: :

:

  There



: :  5:
6::6  :3omething

:: 6:6
```
47 tokens, incoherent garbage after first token, pred=6

The GPU output shows correct chain-of-thought reasoning. The Spyre output degrades into garbage within the first few tokens — colons, random digits, broken words. This is the dl16 RMSNorm error compounding across 28 layers and multiple decode steps.

**Key insight:** Prefill (first token) matches GPU. Multi-token generation rapidly degrades. Single-token tasks (MMLU) are unaffected; multi-token tasks (GSM8K, open-ended generation) are broken.

## Performance Characteristics

| Phase | Time | Notes |
|-------|------|-------|
| Model load + prepare | ~10s | Qwen3 0.6B |
| First compilation (warmup) | ~70-210s | Depends on max_new_tokens |
| Per new prompt length | ~100s | torch.compile(dynamic=False) recompiles |
| Cached fill step | ~4.6s/token | Within existing 64-token block |
| Cached expansion step | ~20s/token | New block, restickify on batchmatmul |
| MMLU (cached, 5 tokens) | ~22s/sample | Mostly prefill |
