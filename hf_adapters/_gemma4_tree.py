# Copyright 2026 The Torch-Spyre Authors.
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

"""Token-tree verification for the Gemma 4 MoE adapter (speculative decoding).

A drafter proposes a tree of ``T`` candidate tokens. Node 0 is the token
accepted last; every other node extends its parent by one token, and parents
precede their children. One ``verify`` forward scores the whole tree, and
``commit`` keeps the accepted root-to-node path in the KV caches. Batch 1.

Caches:

- **Global layers** keep one full-length cache. The tree occupies rows
  ``[start, start + T)``, one per node: node ``j`` has position
  ``start + depth(j)`` but sits in physical row ``start + j``.
- **Sliding layers** keep the anchored compact buffer of ``swa_attention``
  (``capacity = window + 64``; rows ``[0, write_row)`` hold the newest
  committed tokens). The tree occupies rows ``[write_row, write_row + T)``.
  When it would not fit, the buffer first moves its newest
  ``anchor = capacity - 64`` rows (at least ``window``) down to ``[0, anchor)``,
  so no row a node's window can reach is lost.

Masks are additive and built on the host, so position never specializes a
graph. A node reads the committed rows (only the newest ``window`` positions
on a sliding layer) and its own ancestors in the tree block. Attention is SDPA
with that mask on both layer types: the sliding-window op takes a causal
promise, and a tree is not causal.

``commit(path)`` copies the accepted nodes to the rows a plain decode would
have written (``start + depth``). It is a gather and a scatter, in two graphs,
so every source row is read before an overlapping destination is written.

The FFN is ``Gemma4MoEBlock``'s decode FFN. With ``SPYRE_MOE_UNION_CAP`` set,
a ``T``-row verify reads each of at most ``cap`` distinct experts once instead
of ``T * top_k`` slabs (lossy below the batch's expert union).

``tap_layers`` copies the listed decoder layers' outputs into persistent
``[1, T, H]`` buffers inside each layer's compiled graph, for a drafter that
reads target hidden states (XPress). Node order; read after ``verify``.

With greedy acceptance against the verify's own predictions, the output is
the target's greedy continuation, up to the floating-point differences
between a ``T``-row forward and ``T`` single-row forwards.
"""

from dataclasses import dataclass, field

import torch

from hf_adapters import hf_common
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _mask_fill_value,
    allocate_kv_cache_tensor,
    embed_text_tokens,
    kv_cache_shapes,
    make_cache_index,
    text_config,
)
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    compact_after_prefill,
    sliding_capacity,
)

SLIDING = "sliding_attention"


def tree_depths(parents):
    """Depth of every node of an ancestor-closed tree (``parents[0] == -1``)."""
    if not parents or parents[0] != -1:
        raise ValueError("node 0 must be the root (parent -1)")
    depths = [0]
    for i, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < i:
            raise ValueError(f"node {i}: parent {parent} must precede it")
        depths.append(depths[parent] + 1)
    return depths


def tree_ancestors(parents):
    """``ancestors[j]``: the nodes on the root-to-``j`` path, ``j`` included."""
    tree_depths(parents)
    paths = [[0]]
    for i, parent in enumerate(parents[1:], 1):
        paths.append(paths[parent] + [i])
    return paths


def accept_greedy(tokens, parents, predictions):
    """Longest tree path matching the target's greedy predictions.

    ``predictions[j]`` is the target's argmax after node ``j``. Returns
    ``(path, bonus)``: the accepted node indices, root first, and the
    target's token after the last accepted node (always emitted). Sibling
    tokens must be distinct, so the path is unique.
    """
    children = {}
    for i, parent in enumerate(parents[1:], 1):
        key = (parent, int(tokens[i]))
        if key in children:
            raise ValueError(f"duplicate sibling token {key[1]} under node {parent}")
        children[key] = i
    path = [0]
    while True:
        bonus = int(predictions[path[-1]])
        child = children.get((path[-1], bonus))
        if child is None:
            return path, bonus
        path.append(child)


def global_tree_mask(ancestors, start, valid_start, capacity, dtype):
    """Committed columns ``[valid_start, start)`` plus each node's ancestors."""
    mask = torch.full(
        (1, 1, len(ancestors), capacity), _mask_fill_value(dtype), dtype=dtype
    )
    mask[..., valid_start:start] = 0
    for row, path in enumerate(ancestors):
        mask[0, 0, row, [start + p for p in path]] = 0
    return mask


def sliding_tree_mask(
    ancestors, depths, write_row, valid_start, capacity, window, dtype
):
    """Ring columns a node at position ``start + depth`` may read.

    Ring row ``r < write_row`` holds position ``start - (write_row - r)``: it is
    inside node ``j``'s window iff ``(write_row - r) + depth_j < window``.
    """
    mask = torch.full(
        (1, 1, len(ancestors), capacity), _mask_fill_value(dtype), dtype=dtype
    )
    for row, (path, depth) in enumerate(zip(ancestors, depths)):
        lower = max(valid_start, write_row + depth - window + 1)
        if lower < write_row:
            mask[0, 0, row, lower:write_row] = 0
        mask[0, 0, row, [write_row + p for p in path]] = 0
    return mask


def _read_rows(keys, values, rows):
    return keys.index_select(2, rows), values.index_select(2, rows)


def _write_rows(keys, values, k, v, rows):
    keys.index_copy_(2, rows, k)
    values.index_copy_(2, rows, v)
    return keys, values


@dataclass
class TreeState:
    """Host bookkeeping of one request."""

    capacity: int
    keys: list
    values: list
    position: int = 0  # next free global row (committed rows end here)
    valid_start: int = 0  # first real global column (left padding before it)
    pad: int = 0  # left padding of the prompt
    ring_write: int = 0  # sliding layers: next free ring row
    ring_valid: int = 0  # sliding layers: first real ring row
    rolls: int = 0
    pending: int | None = None  # start row of the verified, uncommitted tree
    taps: dict = field(default_factory=dict)


def _backbone(model):
    from hf_adapters.hf_gemma4 import _gemma4_backbone

    return _gemma4_backbone(model)


def _final_norm(model, hidden):
    from hf_adapters.hf_gemma4 import _compiled_gemma4_rms_norm

    norm = _backbone(model).norm
    return _compiled_gemma4_rms_norm(
        hidden, norm.weight if norm.with_scale else None, norm.eps
    )


class TreeVerifier:
    """Prefill, verify and commit for a prepared ``hf_gemma4_moe`` model."""

    def __init__(self, model, width, *, tap_layers=(), prefill_chunk=512):
        cfg = text_config(model.config)
        if not getattr(cfg, "enable_moe_block", False):
            raise ValueError("TreeVerifier drives the Gemma 4 MoE adapter")
        if any(p is not None for p in getattr(model, "_spyre_producer_of", [])):
            raise ValueError("KV-sharing layers are not supported in tree verify")
        if not 2 <= width <= BLOCK_SIZE:
            raise ValueError(f"tree width must be in [2, {BLOCK_SIZE}]")
        if prefill_chunk % BLOCK_SIZE:
            raise ValueError("prefill_chunk must be a BLOCK_SIZE multiple")
        self.model = model
        self.width = int(width)
        self.layer_types = list(cfg.layer_types)
        self.window = int(cfg.sliding_window)
        self.ring_capacity = sliding_capacity(self.window)
        self.prefill_chunk = int(prefill_chunk)
        self.tap_layers = tuple(sorted(int(i) for i in tap_layers))
        self.dtype = hf_common.get_model_dtype(_backbone(model))
        self.blocks = list(_backbone(model).layers)
        for i, block in enumerate(self.blocks):
            block.enable_tree_verify(tap=i in self.tap_layers)
        self._gather = torch.compile(_read_rows, dynamic=False, fullgraph=True)
        self._scatter = torch.compile(_write_rows, dynamic=False, fullgraph=True)
        self.state = None

    def _sliding(self, i):
        return self.layer_types[i] == SLIDING

    # -- prefill ---------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, input_ids, max_new_tokens):
        """Prefill a ``[1, P]`` prompt; returns the last row's logits ``[1, 1, V]``.

        Every layer prefills into a full-length cache with explicit causal and
        window masks (the adapter's own prefill graphs). The sliding layers'
        newest rows are then compacted into their rings.
        """
        model = self.model
        chunk = self.prefill_chunk
        norm = hf_common.normalize_generation_inputs(input_ids, pad_to_multiple=chunk)
        padded = norm.padded_len
        pad = int(norm.prompt_offsets[0])
        capacity = hf_common.generation_cache_len(
            padded, max_new_tokens + 2 * self.width
        )
        keys, values = [], []
        for n_kv, hd, vd in kv_cache_shapes(model):
            keys.append(allocate_kv_cache_tensor(1, n_kv, capacity, hd, self.dtype))
            values.append(allocate_kv_cache_tensor(1, n_kv, capacity, vd, self.dtype))
        kv_len = hf_common._sdpa_compatible_kv_length(padded)
        pre_k = hf_common._prefill_cache_inputs(keys, kv_len, True)
        pre_v = hf_common._prefill_cache_inputs(values, kv_len, True)
        builder = hf_common._ChunkedPrefillMaskBuilder(
            1,
            chunk,
            kv_len,
            norm.prompt_offsets,
            dtype=self.dtype,
            device=hf_common.DEVICE,
        )
        saved = (
            getattr(model, "_spyre_swa_mode", None),
            getattr(model, "_spyre_padded_prompt_len", None),
            getattr(model, "_spyre_prompt_offsets", None),
            getattr(model, "_spyre_active_prefill_chunk_size", None),
        )
        # Explicit band masks over full-length caches for this prefill.
        model._spyre_swa_mode = None
        model._spyre_swa_state = None
        model._spyre_padded_prompt_len = padded
        model._spyre_prompt_offsets = norm.prompt_offsets
        model._spyre_active_prefill_chunk_size = chunk
        from hf_adapters.hf_gemma4_moe import _run_backbone_forward

        taps = {i: [] for i in self.tap_layers}
        compiled = model._spyre_compiled_blocks
        saved_blocks = list(compiled)
        for i in self.tap_layers:

            def tapped(*args, _block=compiled[i], _rows=taps[i]):
                out = _block(*args)
                _rows.append(out[0].to("cpu"))
                return out

            compiled[i] = tapped
        try:
            for begin in range(0, padded, chunk):
                hidden = _run_backbone_forward(
                    model,
                    norm.input_ids[:, begin : begin + chunk].to(hf_common.DEVICE),
                    norm.position_ids[:, begin : begin + chunk].to(hf_common.DEVICE),
                    builder.build(begin),
                    pre_k,
                    pre_v,
                    cache_index=make_cache_index(begin, chunk, hf_common.DEVICE),
                )
        finally:
            (
                model._spyre_swa_mode,
                model._spyre_padded_prompt_len,
                model._spyre_prompt_offsets,
                model._spyre_active_prefill_chunk_size,
            ) = saved
            model._spyre_swa_state = None
            compiled[:] = saved_blocks
        logits = hf_common.run_lm_head(model, hidden, logits_to_keep=1)

        ring = SlidingWindowCache.after_prefill(self.window, padded, [pad])
        for i in range(len(keys)):
            if self._sliding(i):
                keys[i], values[i] = compact_after_prefill(
                    keys[i], values[i], ring, padded
                )
        self.state = TreeState(
            capacity=capacity,
            keys=keys,
            values=values,
            position=padded,
            valid_start=pad,
            pad=pad,
            ring_write=ring.anchor,
            ring_valid=int(ring.valid_start[0]),
            taps={i: torch.cat(t, dim=1)[:, pad:] for i, t in taps.items() if t},
        )
        return logits

    # -- verify / commit -------------------------------------------------------

    def _make_room(self, n):
        """Roll the sliding rings so ``n`` more rows fit after the committed ones.

        Keeps exactly the newest ``anchor = capacity - BLOCK_SIZE`` rows, moved
        to ``[0, anchor)``. ``anchor >= window``, so every row a node's window
        can reach survives; a whole-stick roll (``roll_compact_buffer``) would
        keep fewer whenever the write row is not at a stick boundary. Gather and
        scatter are separate graphs, so the overlapping copy reads every source
        row before writing.
        """
        st = self.state
        if st.ring_write + n <= self.ring_capacity:
            return
        anchor = self.ring_capacity - BLOCK_SIZE
        shift = st.ring_write - anchor
        if shift <= 0 or anchor + n > self.ring_capacity:
            raise RuntimeError(f"tree of {n} rows does not fit the sliding ring")
        device = hf_common.DEVICE
        src = make_cache_index(shift, anchor, device)
        dst = make_cache_index(0, anchor, device)
        for i in range(len(st.keys)):
            if self._sliding(i):
                k, v = self._gather(st.keys[i], st.values[i], src)
                st.keys[i], st.values[i] = self._scatter(
                    st.keys[i], st.values[i], k, v, dst
                )
        st.ring_write = anchor
        st.ring_valid = max(0, st.ring_valid - shift)
        st.rolls += 1

    @torch.no_grad()
    def verify(self, tokens, parents):
        """Score a ``width``-node tree. Returns logits ``[1, width, V_padded]``."""
        st = self.state
        n = len(parents)
        if n != self.width or len(tokens) != n:
            raise ValueError(f"tree must have exactly {self.width} nodes")
        if st.pending is not None:
            raise RuntimeError("verify called twice without a commit")
        depths = tree_depths(parents)
        ancestors = tree_ancestors(parents)
        start = st.position
        if start + n > st.capacity:
            raise RuntimeError("global KV cache full")
        self._make_room(n)
        full = global_tree_mask(
            ancestors, start, st.valid_start, st.capacity, self.dtype
        )
        slid = sliding_tree_mask(
            ancestors,
            depths,
            st.ring_write,
            st.ring_valid,
            self.ring_capacity,
            self.window,
            self.dtype,
        )
        positions = (torch.tensor(depths) + (start - st.pad))[None]
        ids = torch.tensor([int(t) for t in tokens], dtype=torch.long)[None]
        device = hf_common.DEVICE
        h = embed_text_tokens(self.model, ids.to(device))
        freqs = {lt: rope(h, positions) for lt, rope in self.model._spyre_rope.items()}
        full, slid = full.to(device), slid.to(device)
        g_index = make_cache_index(start, n, device)
        r_index = make_cache_index(st.ring_write, n, device)
        for i, block in enumerate(self.blocks):
            sliding = self._sliding(i)
            h, st.keys[i], st.values[i] = block.tree_forward(
                h,
                freqs[self.layer_types[i]],
                slid if sliding else full,
                st.keys[i],
                st.values[i],
                r_index if sliding else g_index,
                block.layer_scalar,
            )
        st.pending = start
        return hf_common.run_lm_head(self.model, _final_norm(self.model, h))

    def tap_buffers(self):
        """``{layer: [1, width, H]}`` device buffers of the last verify (node order)."""
        return {i: self.blocks[i].tree_tap for i in self.tap_layers}

    @torch.no_grad()
    def commit(self, path):
        """Keep the accepted root-to-node ``path`` (node indices, root first)."""
        st = self.state
        start = st.pending
        if start is None:
            raise RuntimeError("commit without a verify")
        n, m = self.width, len(path)
        if not path or path[0] != 0 or m > n:
            raise ValueError("an accepted path starts at the root")
        if path != list(range(m)):
            # Fixed n-row geometry: rows past the path repeat the root and land
            # beyond the committed prefix, where they stay masked.
            src = list(path) + [0] * (n - m)
            device = hf_common.DEVICE
            g_src = torch.tensor([start + p for p in src], dtype=torch.long).to(device)
            r_src = torch.tensor([st.ring_write + p for p in src], dtype=torch.long)
            r_src = r_src.to(device)
            g_dst = make_cache_index(start, n, device)
            r_dst = make_cache_index(st.ring_write, n, device)
            for i in range(len(st.keys)):
                src_i, dst_i = (r_src, r_dst) if self._sliding(i) else (g_src, g_dst)
                k, v = self._gather(st.keys[i], st.values[i], src_i)
                st.keys[i], st.values[i] = self._scatter(
                    st.keys[i], st.values[i], k, v, dst_i
                )
        st.position = start + m
        st.ring_write += m
        st.pending = None
        return m
