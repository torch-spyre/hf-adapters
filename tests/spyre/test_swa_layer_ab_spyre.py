"""Device A/B: Gemma 4's sliding layer, band-masked SDPA versus the SWA op.

End-to-end token equality is a useful integration signal, but it cannot bound the
numeric error introduced by one attention replacement. This test can: one layer,
identical inputs, identical cache contents, and both paths measured against a
common float32 reference.

**Why not compare the two device paths to each other at a tight tolerance.** Both
paths reduce in fp16 in a different order — the op over a compact buffer, the band
over the full cache — so they disagree by fp16 reduction noise that is no op
defect and that the shipped path would fail too. The comparison that is meaningful
is therefore against a common float32 reference, asserting the op is no less
accurate than the band path it replaces (plus an fp16 floor; see ``_fp16_floor``).

Gemma 4 attends **unscaled** (``scaling == 1.0``) at ``head_dim == 256``, where
``q . k`` on iid-random weights would have std ``sqrt(256) == 16`` and drive the
softmax nearly one-hot — an argmax that flips under fp16 rounding and inflates the
error at a single element, an artifact of random weights the real model never sees
(its learned norms keep scores moderate). ``make_sliding_attention`` stands in for
those learned norms with a ``head_dim ** -0.25`` Q/K gain that pins score std at 1,
so the softmax here is non-degenerate and the residual gap is ordinary reduction
noise. ``scaling`` stays 1.0, so the op is still exercised in its production regime.

So the assertion is the one that matters: **the op must be no less accurate than
the path it replaces.**

Run (on the Spyre pod)::

    source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
    python3 -m pytest -s -vvv tests/spyre/test_swa_layer_ab_spyre.py
"""

import copy
import math

import pytest
import torch
from _swa_helpers import FakeKVModel, identity_freqs, make_sliding_attention

from hf_adapters.hf_common import (
    add_causal_sliding_window_band,
    allocate_kv_caches,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    anchored_step,
    compact_after_prefill,
    roll_compact_buffer,
)

# Gemma 4 12B's sliding layers, at one quarter the head count so a test fits.
WINDOW = 1024
HEAD_DIM = 256
Q_HEADS = 4
KV_HEADS = 2
DTYPE = torch.float16
# The op may be up to this multiple of the band path's own float32 error, plus a
# floor so the ratio stays meaningful when both errors are near zero.
ERROR_RATIO = 2.0
ERROR_FLOOR = 5e-3


def _spyre_caches(capacity):
    """Caches with the pinned device layout the indirect scatter requires."""
    model = FakeKVModel([(KV_HEADS, HEAD_DIM, HEAD_DIM)])
    keys, values = allocate_kv_caches(model, 1, capacity, DTYPE, device="spyre")
    return keys[0], values[0]


def _cpu_caches(capacity):
    return (
        torch.zeros(1, KV_HEADS, capacity, HEAD_DIM),
        torch.zeros(1, KV_HEADS, capacity, HEAD_DIM),
    )


def _run(module, hidden, freqs, mask, key_cache, value_cache, index, **kwargs):
    """One compiled forward on device. Returns (output on CPU as float32, caches)."""
    compiled = torch.compile(module, dynamic=False)
    with torch.no_grad():
        out, key_cache, value_cache = compiled(
            hidden, freqs, mask, key_cache, value_cache, index, **kwargs
        )
    return out.to("cpu").float(), key_cache, value_cache


def _run_cpu32(module, hidden, freqs, mask, key_cache, value_cache, index):
    """The same layer in float32 on CPU, eager — the closest thing to truth."""
    with torch.no_grad():
        out, key_cache, value_cache = module(
            hidden.float(), freqs.float(), mask.float(), key_cache, value_cache, index
        )
    return out, key_cache, value_cache


def _fp16_floor(reference, terms):
    """Analytic fp16 noise floor for a ``terms``-long reduction at this output scale.

    ``sqrt(n) * eps * scale`` is the usual random-walk estimate of accumulated
    rounding in an n-term fp16 reduction. It exists because a pure ratio test
    demands the op be more accurate than fp16 allows whenever SDPA happens to be
    unusually accurate: measured at decode, the band path lands at 0.010 where
    fp16's floor for the same reduction is 0.059, so requiring 2x0.010 asks the op
    to beat its own arithmetic.
    """
    scale = reference.abs().max().item()
    return math.sqrt(terms) * torch.finfo(torch.float16).eps * scale


def _assert_no_worse(op_out, band_out, reference, terms, rows_from=0):
    """The op must beat the band path, or come within fp16's floor — whichever is
    more permissive.

    Both halves are load-bearing. The ratio catches an op that is materially less
    accurate than the path it replaces. The floor keeps the test from failing an op
    that is merely fp16-accurate on a case where SDPA got lucky.

    All three tensors are ``[B, L, hidden]`` — the layer returns its ``o_proj``
    output, not per-head attention — so ``rows_from`` slices dim 1, the query rows.
    It drops leading rows whose attention is undefined (see the left-padding test).
    ``terms`` is how many KV columns the op reduces over.
    """
    op_out = op_out[:, rows_from:]
    band_out = band_out[:, rows_from:]
    reference = reference[:, rows_from:]
    assert torch.isfinite(op_out).all(), "op output must be finite"
    band_error = (band_out - reference).abs().max().item()
    op_error = (op_out - reference).abs().max().item()
    floor = _fp16_floor(reference, terms)
    allowed = max(ERROR_RATIO * band_error, floor) + ERROR_FLOOR
    print(
        f"\n  op {op_error:.4f} vs band {band_error:.4f} "
        f"(allowed {allowed:.4f} = max({ERROR_RATIO}x band, fp16 floor {floor:.4f}) "
        f"+ {ERROR_FLOOR}, ref scale {reference.abs().max().item():.3f})"
    )
    assert op_error <= allowed, (
        f"op error {op_error:.4f} exceeds {allowed:.4f}: neither within "
        f"{ERROR_RATIO}x the band path's own error {band_error:.4f} nor within "
        f"fp16's floor {floor:.4f} for a {terms}-term reduction"
    )


def _modules():
    """A band-path module and an op-path module with identical weights, plus a
    float32 CPU twin of the band path to measure both against."""
    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=DTYPE
    )
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"
    reference = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=torch.float32
    )
    return band.to("spyre"), op.to("spyre"), reference


@pytest.mark.parametrize("seqlen_q", [64, 512])
def test_prefill_op_no_worse_than_band_mask(seqlen_q):
    torch._dynamo.reset()
    capacity = 1088
    torch.manual_seed(3)
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(0, seqlen_q)

    mask = build_prefill_mask(1, seqlen_q, capacity, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.arange(seqlen_q)[None, :], WINDOW)

    band, op, reference = _modules()
    band_out, _, _ = _run(
        band,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    ref_out, _, _ = _run_cpu32(
        reference, hidden, freqs, mask, *_cpu_caches(capacity), index
    )
    # A 64-row query block spans WINDOW + 64 columns, capped by the allocation.
    _assert_no_worse(op_out, band_out, ref_out, terms=min(capacity, WINDOW + 64))


def test_runtime_mask_decode_op_no_worse_than_band_mask():
    """A real prefill, then one decode step reading the 1024 columns behind it.

    The cache is filled by running prefill through each module rather than by an
    eager ``index_copy_``: that mirrors production, and an eager index write on
    device falls back to CPU with a warning.
    """
    torch._dynamo.reset()
    capacity, written = 1088, 512
    torch.manual_seed(4)
    prompt = torch.randn(1, written, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    prompt_freqs = identity_freqs(1, written, HEAD_DIM, dtype=DTYPE)
    prompt_index = make_cache_index(0, written)
    prompt_mask = build_prefill_mask(1, written, capacity, 0, dtype=DTYPE)
    prompt_mask = add_causal_sliding_window_band(
        prompt_mask, torch.arange(written)[None, :], WINDOW
    )

    band, op, reference = _modules()
    _, band_k, band_v = _run(
        band,
        prompt.to("spyre"),
        prompt_freqs.to("spyre"),
        prompt_mask.to("spyre"),
        *_spyre_caches(capacity),
        prompt_index.to("spyre"),
    )
    _, op_k, op_v = _run(
        op,
        prompt.to("spyre"),
        prompt_freqs.to("spyre"),
        prompt_mask.to("spyre"),
        *_spyre_caches(capacity),
        prompt_index.to("spyre"),
    )
    _, ref_k, ref_v = _run_cpu32(
        reference,
        prompt,
        prompt_freqs,
        prompt_mask,
        *_cpu_caches(capacity),
        prompt_index,
    )

    token = torch.randn(1, 1, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    token_freqs = identity_freqs(1, 1, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(written, 1)
    # build_decode_mask, not build_prefill_mask: a one-token step at a non-zero
    # cache position must allow every real column up to that position, whereas
    # build_prefill_mask masks everything past the query's *relative* row index
    # and so allows only column 0. The band is combined additively, so an
    # over-restrictive base cannot be widened back — every column ends up masked.
    mask = build_decode_mask(1, capacity, written, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.tensor([[written]]), WINDOW)

    band_out, _, _ = _run(
        band,
        token.to("spyre"),
        token_freqs.to("spyre"),
        mask.to("spyre"),
        band_k,
        band_v,
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        token.to("spyre"),
        token_freqs.to("spyre"),
        mask.to("spyre"),
        op_k,
        op_v,
        index.to("spyre"),
    )
    ref_out, _, _ = _run_cpu32(reference, token, token_freqs, mask, ref_k, ref_v, index)
    # One query row has no stagger, so it spans WINDOW + 1 columns.
    _assert_no_worse(op_out, band_out, ref_out, terms=min(capacity, WINDOW + 1))


def test_left_padding_op_no_worse_than_band_mask():
    """17 pad columns inside the window: the runtime mask must be preserved.

    This is the case the batch>1 Qwen3 decode bug lived in, so it is also the
    canary for window_band_mask's -inf fill (see hf_common._mask_fill_value).

    Compared from row ``offset`` on, plus finiteness everywhere. Rows below the
    threshold have their whole window excluded, so all three paths disagree there
    by construction and none of the answers is meaningful — those rows are padding
    whose outputs are discarded. What matters is that they stay finite, since a
    non-finite value would reach the next layer's KV cache.
    """
    torch._dynamo.reset()
    capacity, seqlen_q, offset = 1088, 512, 17
    torch.manual_seed(5)
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(0, seqlen_q)

    mask = build_prefill_mask(1, seqlen_q, capacity, offset, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.arange(seqlen_q)[None, :], WINDOW)

    band, op, reference = _modules()
    band_out, _, _ = _run(
        band,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    ref_out, _, _ = _run_cpu32(
        reference, hidden, freqs, mask, *_cpu_caches(capacity), index
    )
    assert torch.isfinite(op_out).all(), "fully-masked pad rows must stay finite"
    _assert_no_worse(
        op_out, band_out, ref_out, terms=min(capacity, WINDOW + 64), rows_from=offset
    )


def test_anchored_decode_matches_band_mask_across_a_shift():
    """The shipped geometry, on device, over a 64-row roll.

    Exercises the production shift path: on the shift step the harness rolls the
    compact buffer with an eager ``roll_compact_buffer`` — fresh allocations, the
    same proven pattern as ``compact_after_prefill`` — *before* the compiled block,
    exactly as the driver does. The earlier in-graph roll was an in-place
    ``index_select`` into ``index_copy_`` on the *same* cache; device Inductor fused
    the read into the write and clobbered live rows (op 0.28 vs band 0.001 at the
    shift), while the CPU lane could not catch it because eager materializes the
    select first. This test is what proved the fresh-allocation roll fixes it.
    """
    torch._dynamo.reset()
    window, prompt, steps = 1024, 512, 70
    capacity, full_capacity = 1088, 1152

    # Deep-copy on CPU, then move both to device: deepcopy of an already-"spyre"
    # module fails (no aten::set_.source_Storage kernel for that backend), the
    # same reason _modules() above deep-copies before .to("spyre").
    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, window, swa_mode=None, dtype=DTYPE
    )
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"
    band = band.to("spyre")
    op = op.to("spyre")
    reference = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, window, swa_mode=None, dtype=torch.float32
    )

    band_k, band_v = _spyre_caches(full_capacity)
    op_k, op_v = _spyre_caches(max(prompt, capacity))
    ref_k, ref_v = _cpu_caches(full_capacity)

    torch.manual_seed(5)
    hidden = torch.randn(1, prompt, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
    freqs = identity_freqs(1, prompt, HEAD_DIM, dtype=DTYPE).to("spyre")
    index = make_cache_index(0, prompt, "spyre")

    mask = build_prefill_mask(1, prompt, full_capacity, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(
        mask, torch.arange(prompt)[None, :], window
    ).to("spyre")

    compiled_band = torch.compile(band, dynamic=False)
    with torch.no_grad():
        _, band_k, band_v = compiled_band(hidden, freqs, mask, band_k, band_v, index)
        compiled_op = torch.compile(op, dynamic=False)
        _, op_k, op_v = compiled_op(
            hidden,
            freqs,
            mask[..., :capacity],
            op_k,
            op_v,
            index,
        )
        _, ref_k, ref_v = _run_cpu32(
            reference,
            hidden.to("cpu"),
            freqs.to("cpu"),
            mask.to("cpu"),
            ref_k,
            ref_v,
            make_cache_index(0, prompt),
        )

    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    op_k, op_v = compact_after_prefill(op_k, op_v, state, prompt)
    assert op_k.shape[2] == capacity

    shifts = 0
    for step_index in range(steps):
        slot = prompt + step_index
        token = torch.randn(1, 1, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
        token_freqs = identity_freqs(1, 1, HEAD_DIM, dtype=DTYPE).to("spyre")
        # build_decode_mask for a one-token step at a non-zero position; see the
        # note in the decode A/B above.
        band_mask = build_decode_mask(1, full_capacity, slot, 0, dtype=DTYPE)
        band_mask = add_causal_sliding_window_band(
            band_mask, torch.tensor([[slot]]), window
        ).to("spyre")

        step = anchored_step(state, "spyre", DTYPE)
        shifts += int(step.do_shift)
        if step.do_shift:
            op_k, op_v = roll_compact_buffer(op_k, op_v)
        with torch.no_grad():
            expected, band_k, band_v = compiled_band(
                token,
                token_freqs,
                band_mask,
                band_k,
                band_v,
                make_cache_index(slot, 1, "spyre"),
            )
            actual, op_k, op_v = compiled_op(
                token,
                token_freqs,
                step.attention_mask,
                op_k,
                op_v,
                step.cache_index,
            )
            # The float32 twin keeps its own full-length cache through the same
            # token stream, so every step is measured against truth rather than
            # against the other fp16 path. See _assert_no_worse.
            ref_out, ref_k, ref_v = _run_cpu32(
                reference,
                token.to("cpu"),
                token_freqs.to("cpu"),
                band_mask.to("cpu"),
                ref_k,
                ref_v,
                make_cache_index(slot, 1),
            )
        state.advance()
        _assert_no_worse(
            actual.to("cpu").float(),
            expected.to("cpu").float(),
            ref_out,
            terms=capacity,
        )

    assert shifts == 1
