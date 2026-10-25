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

"""Cached-context tree proposals for the XPress drafter (``hf_xpress_qwen3``).

``hf_xpress_qwen3.propose_block`` re-projects a fixed 48-row context window
for every block and refines one chain with Jacobi passes. A decode loop that
drafts every step needs the training-time context and a tree, so this module
adds, on the same ``XPressDrafter`` and folded refiner:

- **A draft-side KV cache.** Every context row (target hidden states at the
  drafter's ``target_layer_ids``) is projected once (``fc`` + ``hidden_norm``)
  and written to each draft layer's own K/V cache. A block then attends to all
  cached rows, as in training: sliding layers to those within
  ``sliding_window`` of the anchor (and the block causally), the full layer to
  all of them (and the whole block).
- **One graph per draft layer.** The block's 16 query rows sit in fixed cache
  rows after the context; masks are host-built. Layer 0 also projects the new
  context rows, and the last layer also runs the final norm and the head.
- **A chunked head.** The head (the target's tied embedding, optionally only
  the rows of a draft vocabulary) is reduced on the device to each 64-column
  chunk's max and first index, so a block returns one ``[block, 2 * chunks + H]``
  tensor instead of full-vocabulary logits.
- **Path-conditioned tree search on the host.** The refiner's slot mix is
  causal: slot ``k``'s bias depends only on the tokens before ``k``. A tree
  node's bias is therefore computed exactly for its own root-to-node path; no
  Jacobi passes. The candidates of each slot are its chunk winners. The tree
  grows best-first (frontier, branching, depth) and keeps the ``budget`` best
  nodes by path log-probability.
- **fp16-safe norms.** The projected context (~2e4) and the residual stream
  (~1.3e4) square past the fp16 maximum, so every draft RMSNorm runs on
  ``x / 256`` with ``eps / 256**2``, which is the same normalization.

The tree is ancestor-closed, parents precede children, and siblings carry
distinct tokens, as ``_gemma4_tree.TreeVerifier`` requires.
"""

import math

import torch
import torch.nn.functional as F

from hf_adapters import hf_common
from hf_adapters.hf_common import allocate_kv_cache_tensor, apply_rope_matmul

CHUNK = 64
NORM_PRESCALE = 256.0
SLIDING = "sliding_attention"


def _rms(x, weight, eps, prescale=NORM_PRESCALE):
    """RMSNorm on ``x / prescale`` with ``eps / prescale**2``: the same result,
    without squaring fp16 values past their maximum."""
    xs = (x * (1.0 / prescale)).float()
    y = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps / prescale**2)
    return y.to(x.dtype) * weight


class XPressTreeDrafter:
    """Tree proposals from an ``hf_xpress_qwen3.XPressDrafter`` with a context cache.

    ``model`` is a loaded (not Spyre-prepared) drafter. ``draft_vocab`` is an
    optional 1-D tensor of target token ids the head may propose (all of the
    vocabulary by default). Call ``reset(max_rows)`` per request, ``append``
    context rows as the target commits them, then ``propose`` per step.
    """

    def __init__(
        self,
        model,
        *,
        draft_vocab=None,
        frontier=4,
        branching=4,
        depth=7,
        dtype=torch.float16,
        device=None,
        compile=True,
        head_slice_chunks=512,
    ):
        self.device = device or hf_common.DEVICE
        self.dtype = dtype
        cfg = model.config
        self.block = int(model.block_size)
        self.mask_id = int(model.mask_token_id)
        self.layer_types = list(model.layer_types)
        self.window = int(cfg.sliding_window or 0)
        self.eps = float(cfg.rms_norm_eps)
        self.target_layer_ids = list(model.target_layer_ids)
        attn0 = model.layers[0].self_attn
        self.head_dim = int(attn0.head_dim)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.scaling = float(attn0.scaling)
        self.options = dict(
            frontier=int(frontier), branching=int(branching), depth=int(depth)
        )
        if not 1 <= self.options["depth"] < self.block:
            raise ValueError(f"depth must be in [1, {self.block - 1}]")

        dev, dt = self.device, dtype

        def put(t):
            return t.detach().to(dev, dt).contiguous()

        def kn(linear):
            return put(linear.weight.t())

        hidden = cfg.hidden_size
        fc = model.fc.weight.detach().t()  # [taps * H, H]
        self.fc = [
            put(fc[j * hidden : (j + 1) * hidden])
            for j in range(len(self.target_layer_ids))
        ]
        self.hidden_norm = put(model.hidden_norm.weight)
        self.norm = put(model.norm.weight)
        self.layers = []
        for layer in model.layers:
            a, m = layer.self_attn, layer.mlp
            self.layers.append(
                dict(
                    ln_in=put(layer.input_layernorm.weight),
                    ln_post=put(layer.post_attention_layernorm.weight),
                    q=kn(a.q_proj),
                    k=kn(a.k_proj),
                    v=kn(a.v_proj),
                    o=kn(a.o_proj),
                    q_norm=put(a.q_norm.weight),
                    k_norm=put(a.k_norm.weight),
                    gate=kn(m.gate_proj),
                    up=kn(m.up_proj),
                    down=kn(m.down_proj),
                )
            )
        embed = model.embed_tokens.weight.detach().to("cpu")
        self.embed_cpu = embed.to(dt)
        vocab = (
            torch.arange(embed.shape[0])
            if draft_vocab is None
            else torch.as_tensor(draft_vocab, dtype=torch.long).unique(sorted=True)
        )
        self.vocab = vocab
        self.count = int(vocab.numel())
        chunks = -(-self.count // CHUNK)
        chunks = -(-chunks // CHUNK) * CHUNK  # whole sticks of chunk stats
        # The head is reduced in slices of at most ``head_slice_chunks``
        # chunks: a full-vocabulary [block, 4096] chunk-stat tensor needs a
        # restickify whose address step the Spyre backend cannot encode.
        per_slice = min(int(head_slice_chunks), chunks)
        chunks = -(-chunks // per_slice) * per_slice
        self.chunks = chunks
        head = F.pad(embed[vocab].to(dt), (0, 0, 0, chunks * CHUNK - self.count))
        bias = torch.zeros(chunks * CHUNK, dtype=dt)
        bias[self.count :] = -1e4
        width = per_slice * CHUNK
        self.head = [
            put(head[s : s + width].t()) for s in range(0, chunks * CHUNK, width)
        ]
        self.head_bias = [
            put(bias[s : s + width]) for s in range(0, chunks * CHUNK, width)
        ]
        # Chunk positions are small integers, exact in fp16.
        self.positions_in_chunk = put(torch.arange(CHUNK, dtype=torch.float32))

        # Folded refiner on the host (fp32): hidden_proj / summary_proj fold
        # down_h / down_g and the hidden part of in_proj; token_embed folds the
        # token part; mix = tril(L) + I stored [k, j, r].
        head_r = model.xpress_head
        self.r_hidden = head_r.hidden_proj.weight.detach().float()
        self.r_summary = head_r.summary_proj.weight.detach().float()
        self.r_token = head_r.token_embed.weight.detach().float()
        self.r_mix = head_r.mix.detach().float()
        self.r_gate = head_r.mlp_gate.weight.detach().float()
        self.r_up = head_r.mlp_up.weight.detach().float()
        self.r_down = head_r.mlp_down.weight.detach().float()
        self.r_readout = head_r.readout.weight.detach().float()

        inv = model.rotary_emb.inv_freq.detach().to("cpu").double()
        self._inv_freq = inv
        self._compile = compile
        self._fns = {}
        self.capacity = 0

    # -- device graphs ---------------------------------------------------------

    def _rope_table(self, positions):
        f = positions.double()[:, None] * self._inv_freq[None]
        c, s = f.cos().float(), f.sin().float()
        rot = torch.stack([torch.stack([c, -s], 1), torch.stack([s, c], 1)], 1)
        return rot[None].to(self.device, self.dtype)  # [1, L, 2, 2, D/2]

    def _project(self, feats):
        """``hidden_norm(fc(cat(feats)))`` as a sum of per-tap blocks."""
        y = feats[0] @ self.fc[0]
        for j in range(1, len(feats)):
            y = y + feats[j] @ self.fc[j]
        return _rms(y, self.hidden_norm, self.eps)

    def _context_kv(self, ctx, weights, rope):
        m = ctx.shape[1]
        k = (ctx @ weights["k"]).view(1, m, self.n_kv, self.head_dim)
        k = _rms(k, weights["k_norm"], self.eps, 1.0).transpose(1, 2)
        v = (ctx @ weights["v"]).view(1, m, self.n_kv, self.head_dim).transpose(1, 2)
        return apply_rope_matmul(k, rope), v

    def _layer_body(
        self, i, h, ctx, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx
    ):
        L, B = self.layers[i], self.block
        x = _rms(h, L["ln_in"], self.eps)
        q = (x @ L["q"]).view(1, B, self.n_heads, self.head_dim)
        q = apply_rope_matmul(
            _rms(q, L["q_norm"], self.eps, 1.0).transpose(1, 2), rope_blk
        )
        kb = (x @ L["k"]).view(1, B, self.n_kv, self.head_dim)
        kb = apply_rope_matmul(
            _rms(kb, L["k_norm"], self.eps, 1.0).transpose(1, 2), rope_blk
        )
        vb = (x @ L["v"]).view(1, B, self.n_kv, self.head_dim).transpose(1, 2)
        if ctx is not None:
            kc, vc = self._context_kv(ctx, L, rope_ctx)
            keys.index_copy_(2, c_idx, kc)
            values.index_copy_(2, c_idx, vc)
        keys.index_copy_(2, b_idx, kb)
        values.index_copy_(2, b_idx, vb)
        o = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        h = h + o.transpose(1, 2).reshape(1, B, -1) @ L["o"]
        y = _rms(h, L["ln_post"], self.eps)
        return h + (F.silu(y @ L["gate"]) * (y @ L["up"])) @ L["down"]

    def _head_stats(self, h):
        """Final norm, head, and per-64-column chunk max / first index."""
        hid = _rms(h, self.norm, self.eps)
        maxima, firsts = [], []
        for head, bias in zip(self.head, self.head_bias):
            logits = (hid @ head)[0] + bias  # [B, slice_chunks * 64]
            grouped = logits.reshape(self.block, -1, CHUNK)
            cmax = grouped.amax(dim=-1, keepdim=True)
            hit = (grouped == cmax).to(logits.dtype)
            first = hit * self.positions_in_chunk + (1 - hit) * float(CHUNK)
            maxima.append(cmax[..., 0])
            firsts.append(first.amin(dim=-1))
        # One packed tensor in the model dtype: a materialized fp16 -> fp32
        # cast on Spyre permutes lanes, so the host casts after the copy.
        return torch.cat([*maxima, *firsts, hid[0]], dim=-1)

    def _fn(self, kind, i):
        key = (kind, i, self.capacity)
        fn = self._fns.get(key)
        if fn is not None:
            return fn
        last = i == len(self.layers) - 1
        if kind == "block":

            def fn(
                h, feats, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx, ctx=None
            ):
                if i == 0:
                    ctx = self._project(feats)
                h = self._layer_body(
                    i, h, ctx, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx
                )
                if last:
                    return self._head_stats(h), ctx
                return h, ctx

        else:

            def fn(feats, rope_ctx, keys, values, c_idx, ctx=None):
                if i == 0:
                    ctx = self._project(feats)
                kc, vc = self._context_kv(ctx, self.layers[i], rope_ctx)
                keys.index_copy_(2, c_idx, kc)
                values.index_copy_(2, c_idx, vc)
                return ctx

        if self._compile:
            code = fn.__code__.replace(co_name=f"xpress_{kind}{i}_c{self.capacity}")
            fn = type(fn)(
                code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
            )
            fn = torch.compile(fn, dynamic=False, fullgraph=True)
        self._fns[key] = fn
        return fn

    # -- request state ---------------------------------------------------------

    def reset(self, max_rows):
        """Allocate caches for ``max_rows`` context rows plus one block."""
        capacity = (
            -(-(max_rows + self.block) // hf_common.BLOCK_SIZE) * hf_common.BLOCK_SIZE
        )
        if capacity != self.capacity:
            self.capacity = capacity
            rows = capacity + hf_common.BLOCK_SIZE
            self.keys = [
                allocate_kv_cache_tensor(
                    1, self.n_kv, rows, self.head_dim, self.dtype, self.device
                )
                for _ in self.layers
            ]
            self.values = [
                allocate_kv_cache_tensor(
                    1, self.n_kv, rows, self.head_dim, self.dtype, self.device
                )
                for _ in self.layers
            ]
            self.block_index = hf_common.make_cache_index(
                capacity, self.block, self.device
            )
        self.rows = 0
        self.last_row = None

    def _split(self, features):
        """``[m, taps * H]`` host rows -> per-tap ``[1, m, H]`` device tensors."""
        H = features.shape[-1] // len(self.target_layer_ids)
        return [
            features[None, :, j * H : (j + 1) * H]
            .to(self.device, self.dtype)
            .contiguous()
            for j in range(len(self.target_layer_ids))
        ]

    def _padded(self, features, width):
        """Pad context rows to ``width`` by repeating the last one (``rms(0)`` is 0/0)."""
        m = features.shape[0]
        if m < width:
            features = torch.cat([features, features[-1:].expand(width - m, -1)])
        return features

    @torch.no_grad()
    def append(self, features, chunk=128):
        """Write ``[m, taps * H]`` context rows (target hidden states, host)."""
        features = features.reshape(-1, features.shape[-1])
        for s in range(0, features.shape[0], chunk):
            part = features[s : s + chunk]
            m = part.shape[0]
            if self.rows + m > self.capacity:
                raise RuntimeError("XPress context cache full")
            rows = self._padded(part, chunk)
            index = hf_common.make_cache_index(self.rows, chunk, self.device)
            rope = self._rope_table(torch.arange(self.rows, self.rows + chunk))
            ctx = None
            for i in range(len(self.layers)):
                ctx = self._fn("append", i)(
                    self._split(rows), rope, self.keys[i], self.values[i], index, ctx
                )
            self.rows += m
            self.last_row = part[-1:]

    def _masks(self, n):
        """Additive masks for the block queries at positions ``n .. n + B - 1``."""
        cap, B = self.capacity, self.block
        cols = torch.arange(cap + hf_common.BLOCK_SIZE)[None]
        q = torch.arange(n, n + B)[:, None]
        ctx = cols < n
        blk = (cols >= cap) & (cols < cap + B)
        block_pos = cols - cap + n
        fill = hf_common._mask_fill_value(self.dtype)
        full = ctx | blk
        masks = {"full_attention": full}
        if self.window:
            # As in training (and hf_xpress_qwen3.build_draft_masks): context
            # within ``window`` of the anchor for every query; the block causal.
            masks[SLIDING] = (ctx & (cols >= n - self.window)) | (
                blk & (block_pos <= q)
            )

        def additive(ok):
            m = torch.zeros(1, 1, B, cols.shape[1], dtype=self.dtype)
            return m.masked_fill(~ok[None, None], fill).to(self.device)

        return {k: additive(v) for k, v in masks.items()}

    @torch.no_grad()
    def draft(self, anchor, new_features=None):
        """One backbone forward; returns host ``(chunk_max, chunk_first, hidden)``.

        ``new_features`` (``[m, taps * H]``, ``m <= block``) are the context
        rows committed since the previous call; they are projected inside the
        layer-0 graph and written before the block attends.
        """
        B = self.block
        m = 0 if new_features is None else int(new_features.shape[0])
        if m > B:
            raise ValueError(f"at most {B} new context rows per draft")
        start = self.rows
        if m:
            rows = self._padded(new_features.reshape(m, -1), B)
            self.last_row = new_features.reshape(m, -1)[-1:]
        else:
            if self.last_row is None:
                raise RuntimeError("draft before any context row")
            # Rewrite the last row onto itself: same graph, same geometry.
            rows, start = self._padded(self.last_row, B), start - 1
        n = self.rows + m
        if n + B > self.capacity:
            raise RuntimeError("XPress context cache full")
        c_idx = hf_common.make_cache_index(start, B, self.device)
        rope_ctx = self._rope_table(torch.arange(start, start + B))
        rope_blk = self._rope_table(torch.arange(n, n + B))
        ids = torch.full((B,), self.mask_id, dtype=torch.long)
        ids[0] = int(anchor)
        h = F.embedding(ids, self.embed_cpu)[None].to(self.device)
        masks = self._masks(n)
        feats = self._split(rows)
        ctx = None
        for i, lt in enumerate(self.layer_types):
            h, ctx = self._fn("block", i)(
                h,
                feats,
                rope_ctx,
                rope_blk,
                masks[lt],
                self.keys[i],
                self.values[i],
                c_idx,
                self.block_index,
                ctx,
            )
        self.rows = n
        out = h.to("cpu").float()
        c = self.chunks
        return out[:, :c], out[:, c : 2 * c], out[:, 2 * c :]

    # -- host tree search ------------------------------------------------------

    def _candidates(self, cmax, first):
        cols = (torch.arange(self.chunks) * CHUNK)[None] + first.round().long()
        valid = cols < self.count
        ids = self.vocab[cols.clamp(max=self.count - 1)]
        return ids, cmax.masked_fill(~valid, -1e4)

    def _slot_logprobs(self, hidden_part, prev_paths, k, cand_ids, cand_vals):
        """Log-probs over slot ``k``'s candidates for each path of previous tokens.

        ``prev_paths`` is ``[L, k + 1]``: the token before each slot ``0..k``.
        """
        a = hidden_part[None, : k + 1] + self.r_token[prev_paths]  # [L, k+1, r]
        c = (self.r_mix[k, : k + 1][None] * a).sum(dim=1)  # [L, r]
        z = c + (F.silu(c @ self.r_gate.T) * (c @ self.r_up.T)) @ self.r_down.T
        logits = cand_vals[k][None] + z @ self.r_readout[cand_ids[k]].T
        return logits.log_softmax(-1)

    @torch.no_grad()
    def propose(self, anchor, predecessor, budget, new_features=None):
        """A ``budget``-node tree rooted at ``anchor``.

        ``predecessor`` is the token before the anchor (the refiner's slot-0
        input). Returns ``(tokens, parents)``; node 0 is the anchor.
        """
        cmax, first, hidden = self.draft(anchor, new_features)
        cand_ids, cand_vals = self._candidates(cmax, first)
        summary = hidden.mean(dim=0, keepdim=True) @ self.r_summary.T
        hidden_part = hidden @ self.r_hidden.T + summary  # [B, r]
        o = self.options
        nodes = [
            dict(
                token=int(anchor),
                parent=-1,
                depth=0,
                score=0.0,
                path=(int(predecessor), int(anchor)),
            )
        ]
        frontier = [0]
        for d in range(min(o["depth"], self.block - 1)):
            prevs = torch.tensor([nodes[i]["path"] for i in frontier], dtype=torch.long)
            logp = self._slot_logprobs(hidden_part, prevs, d + 1, cand_ids, cand_vals)
            values, index = logp.topk(o["branching"], dim=-1)
            grown = []
            for lane, parent in enumerate(frontier):
                for j in range(o["branching"]):
                    token = int(cand_ids[d + 1, int(index[lane, j])])
                    nodes.append(
                        dict(
                            token=token,
                            parent=parent,
                            depth=d + 1,
                            score=nodes[parent]["score"] + float(values[lane, j]),
                            path=nodes[parent]["path"] + (token,),
                        )
                    )
                    grown.append(len(nodes) - 1)
            frontier = sorted(grown, key=lambda i: (-nodes[i]["score"], i))[
                : o["frontier"]
            ]
            if len(nodes) > budget:
                kth = sorted((x["score"] for x in nodes), reverse=True)[budget - 1]
                if kth >= max(nodes[i]["score"] for i in frontier):
                    break  # no deeper node can enter the top ``budget``
        keep = sorted(
            range(len(nodes)), key=lambda i: (-nodes[i]["score"], nodes[i]["depth"], i)
        )
        keep = sorted(keep[:budget], key=lambda i: (nodes[i]["depth"], i))
        # Ancestor-closed: a node's score is below its parent's (log-probs),
        # so every kept node's parent ranks above it and is kept too.
        remap = {old: new for new, old in enumerate(keep)}
        tokens = [nodes[i]["token"] for i in keep]
        parents = [
            -1 if nodes[i]["parent"] < 0 else remap[nodes[i]["parent"]] for i in keep
        ]
        return _dedupe_siblings(tokens, parents)


def _dedupe_siblings(tokens, parents):
    """Drop a node (and its subtree) whose token repeats an earlier sibling's.

    Chunk winners are distinct tokens per slot, so this only triggers when two
    vocabulary chunks share a winner after the draft-vocabulary mapping.
    """
    seen, drop = set(), set()
    for i, p in enumerate(parents):
        if p in drop or (i and (p, tokens[i]) in seen):
            drop.add(i)
        elif i:
            seen.add((p, tokens[i]))
    keep = [i for i in range(len(tokens)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    return [tokens[i] for i in keep], [
        -1 if parents[i] < 0 else remap[parents[i]] for i in keep
    ]


def path_logprob(drafter, hidden, cmax, first, predecessor, anchor, path_tokens):
    """Reference: the refiner's log-prob of ``path_tokens`` after ``anchor``,
    computed slot by slot on the full path (for tests)."""
    cand_ids, cand_vals = drafter._candidates(cmax, first)
    summary = hidden.mean(dim=0, keepdim=True) @ drafter.r_summary.T
    hidden_part = hidden @ drafter.r_hidden.T + summary
    total, prev = 0.0, [int(predecessor), int(anchor)]
    for k, token in enumerate(path_tokens, start=1):
        logp = drafter._slot_logprobs(
            hidden_part, torch.tensor([prev]), k, cand_ids, cand_vals
        )[0]
        where = (cand_ids[k] == token).nonzero()
        total += float(logp[where[0, 0]]) if where.numel() else -math.inf
        prev.append(int(token))
    return total
