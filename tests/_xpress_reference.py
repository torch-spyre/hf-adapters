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

"""
Plain-PyTorch oracle for the XPress drafter, written from the training code.

Mirrors vllm-project/speculators ``models/dflash`` (backbone) and
``models/xpress`` (refiner) at inference: no Spyre idioms (no stick padding,
no matmul RoPE, no CPU/device split), boolean masks, ``rotate_half`` RoPE and
the training-layout refiner weights (``mix_l`` unfolded, ``x + (L*tril) x``).
The adapter is tested against this, not against itself.
"""

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[:, None] + _rotate_half(x) * sin[:, None]


def reference_allowed(
    ctx_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Boolean ``[block, ctx + block]`` attention pattern for one draft layer.

    Training (``create_anchor_block_mask_mod``): every query of the block sees
    context keys before the anchor — within ``sliding_window`` of the anchor
    position on sliding layers — and the block itself: all of it on full
    layers, only earlier-or-equal slots on sliding layers.
    """
    anchor_pos = int(ctx_positions[-1]) + 1
    ctx_ok = ctx_positions < anchor_pos
    if sliding_window is not None:
        ctx_ok = ctx_ok & (ctx_positions >= anchor_pos - sliding_window)
    ctx_part = ctx_ok[None, :].expand(block_size, -1)
    if sliding_window is None:
        block_part = torch.ones(block_size, block_size, dtype=torch.bool)
    else:
        block_part = torch.ones(block_size, block_size, dtype=torch.bool).tril()
    return torch.cat([ctx_part, block_part], dim=1)


@torch.no_grad()
def reference_block_hidden(
    model,
    ctx_features: torch.Tensor,
    ctx_positions: torch.Tensor,
    anchor_id: int,
) -> torch.Tensor:
    """Post-norm block hidden states ``[1, block, hidden]`` in float32."""
    cfg = model.config
    block = model.block_size
    ctx = model.hidden_norm(model.fc(ctx_features.float()))
    ids = torch.full((1, block), model.mask_token_id, dtype=torch.long)
    ids[0, 0] = anchor_id
    h = F.embedding(ids, model.embed_tokens.weight.float())

    block_positions = int(ctx_positions[-1]) + 1 + torch.arange(block)
    positions = torch.cat([ctx_positions, block_positions])[None]
    cos, sin = model.rotary_emb(h, positions)
    ctx_len = ctx.shape[1]

    for layer, layer_type in zip(model.layers, cfg.layer_types):
        attn = layer.self_attn
        window = cfg.sliding_window if layer_type == "sliding_attention" else None
        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1, block, -1, attn.head_dim)
        k = torch.cat([attn.k_proj(ctx), attn.k_proj(x)], 1)
        v = torch.cat([attn.v_proj(ctx), attn.v_proj(x)], 1)
        k = k.view(1, ctx_len + block, -1, attn.head_dim)
        v = v.view(1, ctx_len + block, -1, attn.head_dim)
        q = _rope(attn.q_norm(q).transpose(1, 2), cos[:, ctx_len:], sin[:, ctx_len:])
        k = _rope(attn.k_norm(k).transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        allowed = reference_allowed(ctx_positions, block, window)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[None, None], scale=attn.scaling
        )
        h = residual + attn.o_proj(out.transpose(1, 2).reshape(1, block, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.norm(h)


def reference_refiner_bias(
    w: dict[str, torch.Tensor],
    prev_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """speculators ``XPressRefinerHead.block_bias`` on training-layout weights."""
    block = hidden.shape[1]
    g = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    hcache = torch.cat([hidden @ w["down_h.weight"].T, g @ w["down_g.weight"].T], -1)
    lat = w["token_embed.weight"][prev_ids]
    x = torch.cat([hcache, lat], -1) @ w["in_proj.weight"].T
    l_eff = w["mix_l"] * torch.ones(block, block).tril()
    x = x + torch.einsum("ckj,njc->nkc", l_eff, x)
    gate = F.silu(x @ w["mlp_gate.weight"].T) * (x @ w["mlp_up.weight"].T)
    x = x + gate @ w["mlp_down.weight"].T
    return x @ w["readout.weight"].T


def reference_jacobi(
    w: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_id: int,
    predecessor_id: int,
    num_passes: int,
) -> torch.Tensor:
    """speculators' free-running refine rounds; returns the ``block-1`` drafts."""
    pred = base_logits.argmax(-1)
    for _ in range(num_passes):
        pred = pred.clone()
        pred[:, 0] = anchor_id
        prev = torch.cat([torch.tensor([[predecessor_id]]), pred[:, :-1]], dim=1)
        pred = (base_logits + reference_refiner_bias(w, prev, hidden)).argmax(-1)
    return pred[:, 1:]


def reference_sequential(
    w: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_id: int,
    predecessor_id: int,
) -> torch.Tensor:
    """Left-to-right greedy with the refiner — the Jacobi fixed point."""
    block = base_logits.shape[1]
    tokens = [anchor_id] + [0] * (block - 1)
    for k in range(1, block):
        prev = torch.tensor([[predecessor_id] + tokens[:-1]])
        refined = base_logits + reference_refiner_bias(w, prev, hidden)
        tokens[k] = int(refined[0, k].argmax())
    return torch.tensor([tokens[1:]])
