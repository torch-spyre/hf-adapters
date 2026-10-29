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

"""Cached-context tree proposals from the Muse Glimmer DFlash assistant.

``transformers``' ``MuseGlimmerAssistantModel`` (``meta-models/Muse-Glimmer-30B-assistant``)
drafts a block of 16 tokens per forward: the last emitted token (the anchor)
and 15 mask tokens, attending bidirectionally to each other and to context
rows projected from the target's hidden states at ``target_layer_ids``. It
has no embedding or head of its own: the block embeds through the target's
raw token table (without Muse's embedding norm) and its rows ``1..15`` go
through the target's bare ``lm_head`` (no multiplier or softcap).

As in ``transformers``' ``DFlashCache`` loop, a block whose anchor has position
``n`` (``n`` context rows) reads, at query position ``n + j``, every context
row ``p`` with ``n + j - p <= sliding_window``, and the whole block.

This module runs it on Spyre with the layout of ``_xpress_tree``:

- **A draft-side KV cache per layer.** Every context row is projected once
  (``fc`` + ``output_norm_enc``, as a sum of per-tap blocks so no 33280-wide
  fp16 sum is formed), and its keys / values are written to each layer's
  cache. Block rows sit in fixed rows after the context capacity, so one
  graph per layer serves every step; masks are host-built.
- **The target's head**, called inside the last layer's graph, so an fp8
  ``lm_head`` (``_muse_fp8``) runs in a compiled region too.
- **fp16-safe norms.** Hidden-size RMSNorms run on ``x / 256`` with
  ``eps / 256**2``: the same normalization without squaring large values.
- **A best-first tree on the host** (``_spec_tree.best_first_tree``): DFlash
  slots are independent of the path, so a node's score is the sum of its
  slots' log-probs and the tree keeps the ``budget`` best nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters import hf_common
from hf_adapters._spec_tree import best_first_tree
from hf_adapters.hf_common import (
    allocate_kv_cache_tensor,
    apply_rope_matmul,
    get_backbone,
    to_default_device_layout,
)

NORM_PRESCALE = 256.0


def _rms(x, weight, eps, prescale=NORM_PRESCALE):
    """``weight * rmsnorm(x)`` computed on ``x / prescale`` (fp16-safe)."""
    xs = (x * (1.0 / prescale)).float()
    y = xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + eps / prescale**2)
    return y.to(x.dtype) * weight


class MuseDFlashDrafter:
    """Tree proposals from a ``MuseGlimmerAssistantModel`` for a prepared target.

    ``assistant`` is a loaded (not Spyre-prepared) assistant; ``target`` the
    ``hf_muse_glimmer``-prepared target whose embedding table and
    ``lm_head`` the drafter shares. Call ``reset(max_rows)`` per request,
    ``append`` the prompt's context rows, then ``propose`` per step.
    """

    def __init__(
        self,
        assistant,
        target,
        *,
        top_k=4,
        depth=None,
        dtype=torch.float16,
        device=None,
        compile=True,
    ):
        self.device = device or hf_common.DEVICE
        self.dtype = dtype
        cfg = assistant.config
        if any(t != "sliding_attention" for t in cfg.layer_types):
            raise ValueError("the Muse DFlash assistant has sliding layers only")
        self.block = int(cfg.block_size)
        self.mask_id = int(cfg.mask_token_id)
        self.window = int(cfg.sliding_window)
        self.eps = float(cfg.rms_norm_eps)
        self.target_layer_ids = list(cfg.target_layer_ids)
        self.head_dim = int(cfg.head_dim)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.scaling = float(assistant.layers[0].self_attn.scaling)
        self.top_k = int(top_k)
        self.depth = self.block - 1 if depth is None else int(depth)
        if not 1 <= self.depth < self.block:
            raise ValueError(f"depth must be in [1, {self.block - 1}]")

        dev, dt = self.device, dtype

        def put(t):
            return t.detach().to(dev, dt).contiguous()

        def kn(linear):
            return put(linear.weight.t())

        hidden = cfg.hidden_size
        fc = assistant.encoder.fc.weight.detach().t()  # [taps * H, H]
        self.fc = [
            put(fc[j * hidden : (j + 1) * hidden])
            for j in range(len(self.target_layer_ids))
        ]
        self.norm_enc = put(assistant.encoder.output_norm_enc.weight)
        self.norm = put(assistant.norm.weight)
        self.layers = []
        for layer in assistant.layers:
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
        backbone = get_backbone(target)
        if backbone.embed_tokens.weight.shape[1] != hidden:
            raise ValueError("assistant and target hidden sizes differ")
        self.embed = backbone.embed_tokens  # raw lookup, no embed_norm
        self.head = target.lm_head
        self.vocab = int(hf_common.text_config(target.config).vocab_size)
        self._inv_freq = assistant.rotary_emb.inv_freq.detach().to("cpu").double()
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
        """``output_norm_enc(fc(cat(feats)))`` as a sum of per-tap blocks."""
        y = feats[0] @ self.fc[0]
        for j in range(1, len(feats)):
            y = y + feats[j] @ self.fc[j]
        return _rms(y, self.norm_enc, self.eps)

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
                    return self.head(_rms(h, self.norm, self.eps)), ctx
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
            # One code object per layer and kind: compiled entries are cached
            # per code object, and layers sharing one scan each other's guards.
            code = fn.__code__.replace(
                co_name=f"muse_dflash_{kind}{i}_c{self.capacity}"
            )
            fn = type(fn)(
                code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
            )
            fn = torch.compile(fn, dynamic=False, fullgraph=True)
        self._fns[key] = fn
        return fn

    # -- request state ---------------------------------------------------------

    def reset(self, max_rows):
        """Allocate caches for ``max_rows`` context rows plus one block."""
        stick = hf_common.BLOCK_SIZE
        capacity = -(-(max_rows + self.block) // stick) * stick
        if capacity != self.capacity:
            self.capacity = capacity
            rows = capacity + stick
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

    @staticmethod
    def _padded(features, width):
        """Pad rows to ``width`` by repeating the last one (``rms(0)`` is 0/0)."""
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
                raise RuntimeError("DFlash context cache full")
            index = hf_common.make_cache_index(self.rows, chunk, self.device)
            rope = self._rope_table(torch.arange(self.rows, self.rows + chunk))
            feats, ctx = self._split(self._padded(part, chunk)), None
            for i in range(len(self.layers)):
                ctx = self._fn("append", i)(
                    feats, rope, self.keys[i], self.values[i], index, ctx
                )
            self.rows += m
            self.last_row = part[-1:]

    def _mask(self, n):
        """Additive mask of the block queries at positions ``n .. n + B - 1``."""
        cap, B = self.capacity, self.block
        cols = torch.arange(cap + hf_common.BLOCK_SIZE)[None]
        q = torch.arange(n, n + B)[:, None]
        ok = ((cols < n) & (q - cols <= self.window)) | (
            (cols >= cap) & (cols < cap + B)
        )
        fill = hf_common._mask_fill_value(self.dtype)
        mask = torch.zeros(1, 1, B, cols.shape[1], dtype=self.dtype)
        return mask.masked_fill(~ok[None, None], fill).to(self.device)

    @torch.no_grad()
    def draft(self, anchor, new_features=None):
        """One block forward; returns host logits ``[block, V]`` (float).

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
            raise RuntimeError("DFlash context cache full")
        c_idx = hf_common.make_cache_index(start, B, self.device)
        rope_ctx = self._rope_table(torch.arange(start, start + B))
        rope_blk = self._rope_table(torch.arange(n, n + B))
        ids = torch.full((1, B), self.mask_id, dtype=torch.long)
        ids[0, 0] = int(anchor)
        h = to_default_device_layout(
            nn.Embedding.forward(self.embed, ids.to(self.embed.weight.device))
        )
        mask, feats, ctx = self._mask(n), self._split(rows), None
        for i in range(len(self.layers)):
            h, ctx = self._fn("block", i)(
                h,
                feats,
                rope_ctx,
                rope_blk,
                mask,
                self.keys[i],
                self.values[i],
                c_idx,
                self.block_index,
                ctx,
            )
        self.rows = n
        # Copy in the device dtype, cast on the host (a materialized fp16 ->
        # fp32 cast on Spyre permutes lanes).
        # Copy whole in the device dtype, then crop and cast on the host (a
        # device slice or fp16 -> fp32 cast of the logits is not exact).
        return h.to("cpu")[0, :, : self.vocab].float()

    @torch.no_grad()
    def propose(self, anchor, budget, new_features=None):
        """A ``budget``-node tree rooted at ``anchor``: ``(tokens, parents)``."""
        logits = self.draft(anchor, new_features)
        logp = logits[1 : self.depth + 1].log_softmax(-1)
        values, ids = logp.topk(min(self.top_k, budget - 1), dim=-1)
        return best_first_tree(anchor, ids.tolist(), values.tolist(), budget)
