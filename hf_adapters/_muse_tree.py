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

"""Token-tree verification for the Muse Glimmer adapter (speculative decoding).

Batch 1. A drafter proposes a tree of ``width`` tokens (``_spec_tree``); one
``verify`` forward of ``hf_muse_glimmer``'s compiled decoder layers scores
every node, and ``commit`` keeps the accepted root-to-node path.

The verifier always runs the adapter's plain-SDPA blocks
(``model._spyre_muse_full_blocks``) over its own full-length KV cache, on
every layer, under explicit host-built masks. Plain ``generate`` keeps its
per-request choice between those blocks and the anchored sliding-window
blocks; the verifier neither reads nor changes it. A tree is then only a
different mask:

- node ``j`` is written to cache row ``start + j`` and has position
  ``root_position + depth(j)``;
- it reads every committed row (on a sliding layer, only those whose
  position is less than ``sliding_window`` behind its own) and its tree
  ancestors.

**Mask-only commit.** Accepted nodes stay in their rows; a host row-position
map records them, and the rejected rows become holes that every later mask
excludes. The next tree starts after the last accepted row. A commit is host
bookkeeping only. When the cache runs out of rows, ``_make_room`` compacts
the committed rows after the prompt with one fixed-shape gather and one
scatter per layer (two graphs, so every source row is read before an
overlapping destination is written).

Prefill follows ``hf_muse_glimmer.generate``'s policy by default: one
64-aligned chunk for prompts up to 512 tokens, 512-token chunks beyond.

``tap_layers`` keeps the listed decoder layers' outputs (host copies, node
order) for a drafter that reads target hidden states (DFlash).

With greedy acceptance on the verify's own argmax, the output is the target's
greedy continuation up to the floating-point differences between a
``width``-row forward and single-row forwards. Logits are returned with the
LM head's padded vocabulary; callers crop on the host.
"""

import math
import types
from dataclasses import dataclass, field

import torch

from hf_adapters import hf_common, hf_muse_glimmer
from hf_adapters._spec_tree import tree_ancestors, tree_depths
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _mask_fill_value,
    allocate_kv_cache_tensor,
    get_backbone,
    get_model_dtype,
    kv_cache_shapes,
    make_cache_index,
    run_lm_head,
    text_config,
)

SLIDING = "sliding_attention"


def own_graph(fn, tag):
    """Compile ``fn``'s original function under its own code object.

    The adapter compiles every layer from one ``block_forward`` code object.
    The tree graphs get one code object per layer, so their shapes (tree
    width, prefill chunk, tree capacity) never count against, or scan, the
    plain decode's cache entries, and each layer's guards are checked alone.
    """
    orig = getattr(fn, "_torchdynamo_orig_callable", fn)
    code = orig.__code__.replace(co_name=f"{orig.__code__.co_name}_{tag}")
    clone = types.FunctionType(
        code, orig.__globals__, code.co_name, orig.__defaults__, orig.__closure__
    )
    clone.__kwdefaults__ = orig.__kwdefaults__
    return torch.compile(clone, dynamic=False, fullgraph=True)


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
    prompt_rows: int  # padded prompt extent; rows below hold the prompt
    row_pos: torch.Tensor  # [capacity] position of each committed row, -1 if none
    start: int = 0  # first free row: the next tree goes to [start, start + width)
    position: int = 0  # position of the next tree's root
    pending: tuple | None = None  # (start, depths) of a verified, uncommitted tree
    compactions: int = 0
    taps: dict = field(default_factory=dict)


class MuseTreeVerifier:
    """Prefill, verify and commit for a prepared ``hf_muse_glimmer`` model."""

    def __init__(
        self,
        model,
        width,
        *,
        tap_layers=(),
        prefill_chunk=None,
        capacity_multiple=2048,
        capacity=None,
    ):
        blocks = getattr(model, "_spyre_muse_full_blocks", None)
        if blocks is None:
            raise ValueError("prepare the model with hf_muse_glimmer first")
        if not 2 <= width <= BLOCK_SIZE:
            raise ValueError(f"tree width must be in [2, {BLOCK_SIZE}]")
        if prefill_chunk is not None and prefill_chunk % BLOCK_SIZE:
            raise ValueError("prefill_chunk must be a BLOCK_SIZE multiple")
        if capacity_multiple % hf_common._SDPA_MAX_SEQUENCE_TILE_SIZE:
            raise ValueError("capacity_multiple must be a multiple of 512")
        cfg = text_config(model.config)
        self.model = model
        self.width = int(width)
        self.layer_types = list(cfg.layer_types)
        self.window = int(cfg.sliding_window or 0)
        self.prefill_chunk = None if prefill_chunk is None else int(prefill_chunk)
        self.capacity_multiple = int(capacity_multiple)
        # A fixed capacity keeps every graph's shape across requests (one
        # compile per benchmark instead of one per prompt-length bucket).
        self.fixed_capacity = None if capacity is None else int(capacity)
        self.tap_layers = tuple(sorted(int(i) for i in tap_layers))
        if any(not 0 <= i < len(self.layer_types) for i in self.tap_layers):
            raise ValueError(f"tap layers {self.tap_layers} outside the decoder")
        self.dtype = get_model_dtype(get_backbone(model))
        self.rope = model._spyre_rope[SLIDING]  # one table; NoPE blocks ignore it
        # The adapter's layer-0 input layout (falls back to the shared one).
        self.embed = getattr(model, "_spyre_embed_tokens", hf_common.embed_text_tokens)
        self.blocks = [own_graph(fn, f"tree{i}") for i, fn in enumerate(blocks)]
        self._gather = torch.compile(_read_rows, dynamic=False, fullgraph=True)
        self._scatter = torch.compile(_write_rows, dynamic=False, fullgraph=True)
        self.state = None

    # -- masks -----------------------------------------------------------------

    def _additive(self, ok):
        mask = torch.zeros(ok.shape, dtype=self.dtype)
        return mask.masked_fill_(~ok, _mask_fill_value(self.dtype))[None, None]

    def _masks(self, query_pos, committed, extra=None):
        """``(full, sliding)`` additive masks ``[1, 1, Q, capacity]`` on the device.

        Query ``q`` reads row ``c`` when ``0 <= committed[c] <= query_pos[q]``
        (sliding layers: also ``query_pos[q] - committed[c] < window``) or when
        ``extra[q, c]`` (its tree ancestors, always inside the window).
        """
        ok = (committed[None, :] >= 0) & (committed[None, :] <= query_pos[:, None])
        slid = ok
        if self.window:
            slid = ok & ((query_pos[:, None] - committed[None, :]) < self.window)
        if extra is not None:
            ok, slid = ok | extra, slid | extra
        device = hf_common.DEVICE
        return self._additive(ok).to(device), self._additive(slid).to(device)

    # -- forward ---------------------------------------------------------------

    def _forward(self, h, positions, full, slid, cache_index):
        st = self.state
        freqs = self.rope(h, positions)
        taps = dict.fromkeys(self.tap_layers)
        for i, block in enumerate(self.blocks):
            mask = slid if self.layer_types[i] == SLIDING else full
            h, st.keys[i], st.values[i] = block(
                h, freqs, mask, st.keys[i], st.values[i], cache_index
            )
            if i in taps:
                taps[i] = h.to("cpu")
        return h, taps

    def _chunk(self, input_ids):
        """``(chunk, normalized inputs)``: the adapter's prefill policy by default."""
        chunk = self.prefill_chunk
        if chunk is None:
            norm = hf_common.normalize_generation_inputs(input_ids)
            if norm.padded_len <= hf_muse_glimmer._PREFILL_CHUNK_SIZE:
                return norm.padded_len, norm
            chunk = hf_muse_glimmer._PREFILL_CHUNK_SIZE
        return chunk, hf_common.normalize_generation_inputs(
            input_ids, pad_to_multiple=chunk
        )

    @torch.no_grad()
    def prefill(self, input_ids, max_new_tokens):
        """Prefill a ``[1, P]`` prompt; returns the last row's logits ``[1, 1, V]``.

        Chunks run under explicit causal and window masks over the verifier's
        full-length cache.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("tree verify is batch-1 only")
        model = self.model
        chunk, norm = self._chunk(input_ids)
        padded, pad = norm.padded_len, int(norm.prompt_offsets[0])
        reserve = 2 * max_new_tokens + 2 * self.width + BLOCK_SIZE
        capacity = self.fixed_capacity or (
            math.ceil((padded + reserve) / self.capacity_multiple)
            * self.capacity_multiple
        )
        if padded + 2 * self.width > capacity:
            raise ValueError(
                f"prompt of {padded} rows does not fit capacity {capacity}"
            )
        keys, values = [], []
        for n_kv, hd, vd in kv_cache_shapes(model):
            keys.append(allocate_kv_cache_tensor(1, n_kv, capacity, hd, self.dtype))
            values.append(allocate_kv_cache_tensor(1, n_kv, capacity, vd, self.dtype))
        row_pos = torch.full((capacity,), -1, dtype=torch.long)
        self.state = st = TreeState(
            capacity=capacity,
            keys=keys,
            values=values,
            prompt_rows=padded,
            row_pos=row_pos,
            start=padded,
            position=padded - pad,
        )
        taps = {i: [] for i in self.tap_layers}
        rows = torch.arange(capacity)
        prompt_pos = torch.where(rows >= pad, rows - pad, -1)
        prompt_pos[padded:] = -1
        for begin in range(0, padded, chunk):
            ids = norm.input_ids[:, begin : begin + chunk]
            h = self.embed(model, ids)
            positions = norm.position_ids[:, begin : begin + chunk]
            # Causal by position (positions grow with rows); a padding query
            # (position -1) reads nothing and its output is never read.
            committed = torch.where(rows < begin + chunk, prompt_pos, -1)
            full, slid = self._masks(prompt_pos[begin : begin + chunk], committed)
            h, chunk_taps = self._forward(
                h,
                positions,
                full,
                slid,
                make_cache_index(begin, chunk, hf_common.DEVICE),
            )
            for i, t in chunk_taps.items():
                taps[i].append(t)
        row_pos[:padded] = prompt_pos[:padded]
        st.taps = {i: torch.cat(t, dim=1)[:, pad:] for i, t in taps.items()}
        return run_lm_head(model, model._spyre_compiled_norm(h), logits_to_keep=1)

    # -- verify / commit -------------------------------------------------------

    def _make_room(self):
        """Compact the committed rows after the prompt so ``width`` rows fit."""
        st = self.state
        if st.start + self.width <= st.capacity:
            return
        tail = torch.arange(st.prompt_rows, st.capacity)
        kept = tail[st.row_pos[st.prompt_rows :] >= 0]
        n = int(kept.numel())
        if st.prompt_rows + n + self.width > st.capacity:
            raise RuntimeError("tree KV cache full")
        # Fixed-shape geometry: the source list repeats the last kept row.
        fill = kept[-1:] if n else tail[:1]
        src = torch.cat([kept, fill.expand(tail.numel() - n)])
        device = hf_common.DEVICE
        src_d, dst_d = src.to(device), tail.to(device)
        for i in range(len(st.keys)):
            k, v = self._gather(st.keys[i], st.values[i], src_d)
            st.keys[i], st.values[i] = self._scatter(
                st.keys[i], st.values[i], k, v, dst_d
            )
        positions = st.row_pos[kept].clone()
        st.row_pos[st.prompt_rows :] = -1
        st.row_pos[st.prompt_rows : st.prompt_rows + n] = positions
        st.start = st.prompt_rows + n
        st.compactions += 1

    @torch.no_grad()
    def verify(self, tokens, parents):
        """Score a ``width``-node tree. Returns logits ``[1, width, V]``."""
        st = self.state
        n = len(parents)
        if n != self.width or len(tokens) != n:
            raise ValueError(f"tree must have exactly {self.width} nodes")
        if st.pending is not None:
            raise RuntimeError("verify called twice without a commit")
        depths = tree_depths(parents)
        self._make_room()
        start = st.start
        q_pos = torch.tensor(depths) + st.position
        extra = torch.zeros(n, st.capacity, dtype=torch.bool)
        for row, path in enumerate(tree_ancestors(parents)):
            extra[row, [start + p for p in path]] = True
        full, slid = self._masks(q_pos, st.row_pos, extra)
        ids = torch.tensor([[int(t) for t in tokens]], dtype=torch.long)
        h = self.embed(self.model, ids)
        h, st.taps = self._forward(
            h, q_pos[None], full, slid, make_cache_index(start, n, hf_common.DEVICE)
        )
        st.pending = (start, depths)
        return run_lm_head(self.model, self.model._spyre_compiled_norm(h))

    def tap_features(self):
        """``[rows, len(taps) * H]`` host features of the last prefill / verify."""
        st = self.state
        return torch.cat([st.taps[i] for i in self.tap_layers], dim=-1)[0]

    @torch.no_grad()
    def commit(self, path):
        """Keep the accepted root-to-node ``path`` (node indices, root first)."""
        st = self.state
        if st.pending is None:
            raise RuntimeError("commit without a verify")
        start, depths = st.pending
        if not path or path[0] != 0 or len(path) > self.width:
            raise ValueError("an accepted path starts at the root")
        for p in path:
            st.row_pos[start + p] = st.position + depths[p]
        st.position += len(path)
        st.start = start + max(path) + 1
        st.pending = None
        return len(path)
