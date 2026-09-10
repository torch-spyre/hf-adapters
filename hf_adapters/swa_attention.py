# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sliding-window attention for the Spyre adapters.

The only module in hf-adapters that calls ``torch.ops.spyre.*``. Gemma 3 and
Gemma 4 alternate sliding and full-attention layers; the sliding ones can read
just their window out of a compact KV buffer instead of scoring the whole cache
behind a band mask (torch-spyre#3405).

Two things make that a module rather than a call site:

1. ``spyre::sliding_window_attention`` is registered for the spyre device only and
   has an empty eager body, so CPU needs the definition computed literally.
2. Both prefill and anchored decode pass their complete geometry in one runtime
   mask, so position and padding values never specialize a compiled graph.
"""

import dataclasses

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _mask_fill_value,
    allocate_kv_cache_tensor,
    kv_cache_shapes,
    text_config,
)


def sliding_capacity(window_size, q_block=BLOCK_SIZE):
    """Rows an anchored compact KV buffer allocates for ``window_size``.

    ``window_size + q_block - 1`` rounded up to a stick: a block of ``q_block``
    query rows has staggered windows spanning ``window_size + q_block - 1``
    columns, and ``rejection_reason`` refuses a capacity that is not
    stick-aligned. 1088 for Gemma 4 (W=1024), 576 for Gemma 3 (W=512).
    """
    return -(-(window_size + q_block - 1) // BLOCK_SIZE) * BLOCK_SIZE


def fit_attention_mask(attention_mask, cache_capacity):
    """Crop or mask-pad an attention mask to a layer's physical cache width.

    The generic generation driver sizes masks for the global cache. Sliding
    caches can be either narrower (long contexts) or wider (short prompts whose
    global cache is below the minimum window allocation). Build this small piece
    of per-layer input outside compiled blocks so the custom op always receives
    its exact ``[B, 1, Lq, Lk]`` contract without adding work to every layer.
    """
    width = attention_mask.size(-1)
    if width == cache_capacity:
        return attention_mask
    if width > cache_capacity:
        return attention_mask[..., :cache_capacity]

    original_device = attention_mask.device
    padded = F.pad(
        attention_mask.to("cpu"),
        (0, cache_capacity - width),
        value=_mask_fill_value(attention_mask.dtype),
    )
    return padded.to(original_device)


def physical_cache_capacity(cache):
    """Return the cache width preserved by a compiled in-place update.

    Chunked prefill can pass a prefix view of a larger pinned Spyre cache. The
    eager tensor reports the view width, but the compiled in-place ``index_copy_``
    returns the owning allocation. Use that allocation width when fitting the
    mask passed to attention later in the same graph.
    """
    if cache.device.type != "spyre":
        return cache.size(2)

    owner = cache
    while isinstance(getattr(owner, "_base", None), torch.Tensor):
        base = owner._base
        if base.ndim != 4:
            break
        owner = base
    return owner.size(2)


def allocate_swa_caches(model, batch_size, max_cache_len, dtype, device):
    """Allocate compact-capable caches for models with sliding attention layers.

    Global layers retain the complete generation capacity. Sliding layers only
    need enough rows for prefill and the anchored decode buffer; they are compacted
    to ``sliding_capacity`` immediately before the first decode step. KV-sharing
    consumers alias their producer's allocation instead of retaining unused caches.
    """
    cfg = text_config(model.config)
    shapes = kv_cache_shapes(model)
    if len(shapes) != len(cfg.layer_types):
        raise ValueError("KV shapes and layer_types must have the same length")
    prompt_len = getattr(model, "_spyre_padded_prompt_len", max_cache_len)
    capacities = [
        (
            max(sliding_capacity(cfg.sliding_window), prompt_len)
            if layer_type == "sliding_attention"
            else max_cache_len
        )
        for layer_type in cfg.layer_types
    ]
    producer_of = getattr(model, "_spyre_producer_of", [None] * len(shapes))
    if len(producer_of) != len(shapes):
        raise ValueError(
            "_spyre_producer_of and _spyre_kv_shapes must have the same length"
        )

    key_caches = []
    value_caches = []
    for i, ((n_kv, head_dim, value_dim), rows) in enumerate(zip(shapes, capacities)):
        producer = producer_of[i]
        if producer is not None:
            if producer < 0 or producer >= i or shapes[producer] != shapes[i]:
                raise ValueError(
                    f"invalid KV-sharing producer {producer} for layer {i}: "
                    "the producer must precede the consumer and have the same KV shape"
                )
            # Shared Gemma 4 layers always read key_caches[producer] in the driver.
            # Alias the list entry too, both to avoid allocating an unused full cache
            # and to let compaction release the producer's large prefill buffer.
            key_caches.append(key_caches[producer])
            value_caches.append(value_caches[producer])
            continue
        key_caches.append(
            allocate_kv_cache_tensor(batch_size, n_kv, rows, head_dim, dtype, device)
        )
        value_caches.append(
            allocate_kv_cache_tensor(batch_size, n_kv, rows, value_dim, dtype, device)
        )
    return key_caches, value_caches


def sliding_window_attention(
    query,
    key_cache,
    value_cache,
    attention_mask,
    *,
    window_size,
    is_causal,
    scale,
):
    """Attend ``query`` against the window of a KV cache.

    Args:
        query: ``[B, Hq, Lq, D]``.
        key_cache, value_cache: ``[B, Hkv, capacity, D]``. GQA is expanded inside
            the op, so ``Hq`` need only be a whole multiple of ``Hkv``. Both must
            have the same shape (``check_window_read`` requires it) and the
            allocation must be **zero-filled**: a window may overshoot the written
            prefix, and an additive mask cannot rescue a NaN.
        window_size: configured backward window, with exclusive lower bound
            ``c - window_size``. The mask carries the exact boundary and any
            permitted non-causal pairs.
        is_causal: static promise that the mask never allows a future key. The
            Spyre kernel may use narrow causal reads for square prefill when
            true; false scans the complete physical cache and leaves all
            attention semantics to ``attention_mask``.
        scale: ``Q·Kᵀ`` multiplier. ``None`` means ``1/sqrt(D)``. Gemma 4 attends
            **unscaled** and must pass ``1.0``.
        attention_mask: additive ``[B, 1, Lq, capacity]`` mask carrying all
            position-dependent state: causality, the window, unwritten cache rows,
            and padding. Changing its values does not specialize the graph.

    Returns ``[B, Hq, Lq, D]``.
    """
    if query.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            query,
            key_cache,
            value_cache,
            attention_mask,
            window_size,
            is_causal,
            scale,
        )
    return F.scaled_dot_product_attention(
        query,
        key_cache,
        value_cache,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=True,
    )


@dataclasses.dataclass
class SlidingWindowCache:
    """Anchored compact-buffer state for one generation's sliding layers.

    The invariant, at the start of every 64-token stick period:

      * physical rows ``[0, anchor)`` hold the most recent ``anchor`` tokens, all
        real once the buffer has filled
      * rows ``[anchor, capacity)`` are empty and take the next 64 writes

    Token ``m`` of a period is written at row ``anchor + m``; after the 64th, rows
    ``[64, capacity)`` shift down to ``[0, anchor)`` and the invariant is restored.
    The shift happens *before* a stick of writes rather than after, which is what
    keeps every row below the write cursor real and so keeps ``valid_start`` at 0
    in the steady state.

    The compact buffer bounds memory: without it a sliding layer would carry a KV
    allocation the size of the whole context, gigabytes across Gemma 4 12B's 40
    sliding layers. The current write row and valid prefix travel in a fixed-shape
    runtime mask, not Python geometry, so the 64 physical rows in the write stick
    all reuse one decode graph. The 64-row roll is an eager driver step (see
    ``roll_compact_buffer``), not a graph branch.
    """

    window_size: int
    capacity: int
    write_row: int
    valid_start: list

    @property
    def anchor(self):
        """First physical row of the 64-row stick currently being written."""
        return self.capacity - BLOCK_SIZE

    @classmethod
    def after_prefill(cls, window_size, prompt_len, offsets):
        """State for the first decode step, i.e. just after compaction.

        ``offsets`` is ``generate``'s per-sequence left padding, in the prefill
        buffer's coordinates. Compaction keeps the newest ``min(prompt_len,
        anchor)`` rows, right-aligned at the anchor, so:

          * a prompt longer than the buffer pushes its pad columns off the front
            and needs no threshold at all;
          * a shorter one leaves unwritten rows at the front, and any pad that
            travelled with it sits directly above them.
        """
        capacity = sliding_capacity(window_size)
        anchor = capacity - BLOCK_SIZE
        kept = min(prompt_len, anchor)
        valid_start = [
            (anchor - kept) + max(0, int(offset) - (prompt_len - kept))
            for offset in offsets
        ]
        return cls(window_size, capacity, anchor, valid_start)

    def stick_offset(self):
        """Index of the current token within its 64-row query stick."""
        return self.write_row - self.anchor

    def needs_shift(self):
        """True when the write stick is full, so the buffer must roll first."""
        return self.write_row >= self.capacity

    def shift(self):
        """Advance the bookkeeping past a 64-row roll of the buffer."""
        self.write_row = self.anchor
        self.valid_start = [max(0, start - BLOCK_SIZE) for start in self.valid_start]

    def advance(self):
        """Move the write cursor on by the one token this step wrote."""
        self.write_row += 1


def compact_after_prefill(key_cache, value_cache, state, prompt_len):
    """Move a prefill-sized cache's newest rows into a fresh anchored buffer.

    Prefill needs ``max(sliding_capacity(W), prompt)`` rows; decode needs only
    ``capacity``. Rather than carry the prefill allocation for the whole
    generation — at Gemma 4 12B's 40 sliding layers and an 8192-token context that
    is gigabytes — copy the newest ``min(prompt_len, anchor)`` rows into
    ``[anchor - kept, anchor)`` of a compact buffer and let the big one go.

    Returns the new ``(key_cache, value_cache)``; the caller must replace its
    references, since nothing else keeps the compact buffers alive.
    """
    anchor = state.anchor
    kept = min(prompt_len, anchor)
    device = key_cache.device
    batch, num_kv_heads, _, head_dim = key_cache.shape
    compact_key = allocate_kv_cache_tensor(
        batch, num_kv_heads, state.capacity, head_dim, key_cache.dtype, device
    )
    compact_value = allocate_kv_cache_tensor(
        batch,
        num_kv_heads,
        state.capacity,
        value_cache.shape[3],
        value_cache.dtype,
        device,
    )
    _compact_copy(
        compact_key,
        key_cache,
        dst_start=anchor - kept,
        src_start=prompt_len - kept,
        length=kept,
    )
    _compact_copy(
        compact_value,
        value_cache,
        dst_start=anchor - kept,
        src_start=prompt_len - kept,
        length=kept,
    )
    return compact_key, compact_value


def _compact_copy(destination, source, *, dst_start, src_start, length):
    """Copy contiguous cache rows while preserving the destination allocation.

    The row ranges are contiguous, so a sliced ``copy_`` is both simpler and
    faster than ``index_select`` followed by ``index_copy_``. On Spyre, ``copy_``
    dispatches to the offset-aware ``spyre::copy_from_d2d`` helper; this keeps the
    transfer on device and preserves the destination's pinned cache layout. The
    index-copy form falls back through CPU when called eagerly, making every
    64-token roll an avoidable host round trip.
    """
    destination.narrow(2, dst_start, length).copy_(source.narrow(2, src_start, length))
    return destination


def roll_compact_buffer(key_cache, value_cache):
    """Roll a compact buffer down one 64-row stick, into fresh allocations.

    The shipped shift: rows ``[BLOCK_SIZE, capacity)`` become
    ``[0, capacity - BLOCK_SIZE)`` of a freshly zeroed buffer, restoring the
    invariant that ``[0, anchor)`` holds the most recent ``anchor`` tokens; the
    trailing ``BLOCK_SIZE`` rows stay zero for the next stick of writes.

    Fresh buffers rather than an in-place copy on the same tensor. The
    source rows ``[64, capacity)`` and destination rows ``[0, capacity - 64)``
    overlap, and on device Inductor fuses the out-of-place ``index_select`` into
    the in-place scatter, so the write clobbers rows still being read — silently,
    and only on device (CPU eager materializes the select first). Writing into a
    *different* tensor removes the aliasing, the pattern ``kv_cache_update`` and
    ``compact_after_prefill`` already rely on. Run eager by the driver once per 64
    decode steps, like ``compact_after_prefill`` — not inside the compiled block,
    so ``allocate_kv_cache_tensor``'s device-layout pin is applied on the proven eager
    path rather than under compile, where an unhonored pin would scatter to the
    wrong rows silently (torch-spyre#3705). Zero-fill in the trailing rows is safe:
    they sit above the write cursor and the window never reads them before they are
    overwritten.

    Returns the new ``(key_cache, value_cache)``; the caller must replace its
    references, since nothing else keeps the rolled buffers alive.
    """
    capacity = key_cache.size(2)
    device = key_cache.device
    batch, num_kv_heads, _, head_dim = key_cache.shape
    rolled_key = allocate_kv_cache_tensor(
        batch, num_kv_heads, capacity, head_dim, key_cache.dtype, device
    )
    rolled_value = allocate_kv_cache_tensor(
        batch, num_kv_heads, capacity, value_cache.shape[3], value_cache.dtype, device
    )
    length = capacity - BLOCK_SIZE
    _compact_copy(
        rolled_key,
        key_cache,
        dst_start=0,
        src_start=BLOCK_SIZE,
        length=length,
    )
    _compact_copy(
        rolled_value,
        value_cache,
        dst_start=0,
        src_start=BLOCK_SIZE,
        length=length,
    )
    return rolled_key, rolled_value


@dataclasses.dataclass(frozen=True)
class AnchoredStep:
    """What one anchored decode step passes into a compiled sliding block.

    ``attention_mask`` carries position and padding as tensor data. ``cache_index``
    is likewise a tensor, so neither changing value can specialize the graph.
    ``do_shift`` tells the driver to roll the buffer eagerly before the compiled
    block; an in-graph in-place self-copy was the aliasing the compiler fused
    unsafely.
    """

    do_shift: bool
    cache_index: torch.Tensor
    attention_mask: torch.Tensor


def _anchored_attention_mask(state, dtype, device):
    """Build one runtime mask shared by every sliding layer in this step."""
    batch = len(state.valid_start)
    mask = torch.zeros((batch, 1, 1, state.capacity), dtype=dtype)
    fill = _mask_fill_value(dtype)
    window_start = max(0, state.write_row - state.window_size + 1)
    for batch_index, valid_start in enumerate(state.valid_start):
        mask[batch_index, :, :, : max(window_start, valid_start)] = fill
    mask[:, :, :, state.write_row + 1 :] = fill
    return mask.to(device)


def anchored_step(state, device, dtype):
    """Geometry for the next anchored decode step, rolling the buffer if due.

    Mutates ``state`` when a shift is due — the tensor roll itself is a separate
    eager ``roll_compact_buffer`` the caller runs before the compiled block when
    ``do_shift`` is set. The caller must call ``state.advance()`` after the step
    completes.

    The query is written at ``write_row``. The fixed-shape mask exposes exactly
    its resident causal window, so no Python position enters the compiled block.
    """
    do_shift = state.needs_shift()
    if do_shift:
        state.shift()
    return AnchoredStep(
        do_shift=do_shift,
        cache_index=torch.tensor([state.write_row], dtype=torch.long).to(device),
        attention_mask=_anchored_attention_mask(state, dtype, device),
    )


def valid_start_for(model, batch_size):
    """Initial valid cache column per sequence -- ``generate``'s left padding.

    ``generate`` stashes ``_spyre_prompt_offsets``; a caller driving a forward
    directly (the layer tests) has none. Gemma 3 and Gemma 4 use this to initialize
    the compact cache's runtime attention mask.
    """
    offsets = getattr(model, "_spyre_prompt_offsets", None)
    if offsets is None:
        return [0] * batch_size
    return [int(offset) for offset in offsets]


def roll_sliding_buffers(layer_types, key_caches, value_caches, *, producer_of=None):
    """Roll every sliding layer's compact buffer down one stick, in place.

    The eager pre-block half of a shift step: replaces each sliding layer's
    ``(key, value)`` in the caller's lists with freshly-allocated rolled buffers
    (see ``roll_compact_buffer``). Run before the compiled blocks so the roll
    never enters the graph. Shared by the Gemma 3 and Gemma 4 drivers.
    """
    if producer_of is None:
        producer_of = [None] * len(layer_types)
    for i, (layer_type, producer) in enumerate(zip(layer_types, producer_of)):
        if layer_type == "sliding_attention" and producer is None:
            key_caches[i], value_caches[i] = roll_compact_buffer(
                key_caches[i], value_caches[i]
            )
    rebind_shared_caches(key_caches, value_caches, producer_of)


def compact_sliding_buffers(
    layer_types,
    key_caches,
    value_caches,
    state,
    prompt_len,
    *,
    producer_of=None,
):
    """Compact every sliding layer down to its anchored buffer after prefill.

    The post-prefill half: replaces each sliding layer's prompt-sized ``(key,
    value)`` with a compact anchored buffer (see ``compact_after_prefill``) and
    lets the big allocations go. Shared by the Gemma 3 and Gemma 4 drivers.
    """
    if producer_of is None:
        producer_of = [None] * len(layer_types)
    for i, (layer_type, producer) in enumerate(zip(layer_types, producer_of)):
        if layer_type == "sliding_attention" and producer is None:
            key_caches[i], value_caches[i] = compact_after_prefill(
                key_caches[i], value_caches[i], state, prompt_len
            )
    rebind_shared_caches(key_caches, value_caches, producer_of)


def rebind_shared_caches(key_caches, value_caches, producer_of):
    """Keep unused KV-sharing list entries aliased to their producer buffers."""
    for i, producer in enumerate(producer_of):
        if producer is not None:
            key_caches[i] = key_caches[producer]
            value_caches[i] = value_caches[producer]
