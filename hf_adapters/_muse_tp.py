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

"""Tensor parallelism for the Muse Glimmer text decoder (TP 2 or 4).

Opt-in: active only when ``torch.distributed`` is initialized with more than
one rank (``torchrun``) and the model was loaded unsharded on the host (no
``tp_plan``). Every rank loads the checkpoint on the host and keeps only its
slices before anything reaches the device. ``hf_muse_glimmer`` calls
``maybe_shard`` once, after preparing its LM head and before compiling its
decoder layers, so the shared Gemma 3 blocks see the local head counts:

- q, attention gate, MLP gate and up are split by output (whole heads);
- k and v are split by KV head; with fewer KV heads than ranks (Muse has 2,
  so TP 4) each head is replicated across the ranks whose query heads read
  it, as ``SpyreGroupedColwiseParallel`` does;
- o and down are split by input and become ``AllReduceLinear``: the block
  calls them as modules, so their partial sums are all-reduced inside the
  compiled block, before the post-attention / post-FFN norms;
- the LM head is split by vocabulary into equal 128-row-aligned shards; the
  logits are all-gathered inside the compiled head and returned *padded*:
  callers crop to the vocabulary on the host, because a device slice of the
  gathered tensor corrupts every column past the first rank's shard;
- embeddings and norms stay replicated.

Sharding on the host keeps unsharded weights off the device. The DFlash
assistant can be sharded the same way (``shard_assistant``). A model that
Transformers already sharded (``tp_plan``) is left alone.
"""

import dataclasses

import torch
import torch.nn as nn

from hf_adapters import hf_common


@dataclasses.dataclass(frozen=True)
class TPContext:
    rank: int
    world: int
    group_name: str


def tp_context():
    """The active TP group, or ``None`` when running on one device."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return None
    world = dist.get_world_size()
    if world == 1:
        return None
    return TPContext(dist.get_rank(), world, dist.group.WORLD.group_name)


class AllReduceLinear(nn.Linear):
    """A rowwise-sharded linear whose output is summed over the TP group."""

    group_name: str

    def forward(self, x):
        return all_reduce(super().forward(x), self.group_name)


def _rows(linear, start, end):
    linear.weight = nn.Parameter(
        linear.weight.detach()[start:end].contiguous(), requires_grad=False
    )
    linear.out_features = end - start


def _cols(linear, start, end, group_name=None):
    linear.weight = nn.Parameter(
        linear.weight.detach()[:, start:end].contiguous(), requires_grad=False
    )
    linear.in_features = end - start
    if group_name is not None:
        linear.__class__ = AllReduceLinear
        linear.group_name = group_name


def _hf_sharded(module):
    return getattr(module, "_hf_device_mesh", None) is not None or (
        getattr(module.weight, "device_mesh", None) is not None
    )


def maybe_shard(model):
    """Shard ``model`` for the active TP group; a no-op on one device."""
    ctx = tp_context()
    if ctx is None:
        return None
    layer0 = hf_common.get_backbone(model).layers[0]
    if _hf_sharded(layer0.self_attn.q_proj):
        return None  # Transformers' tp_plan already placed the shards
    shard_text(model, ctx)
    return ctx


def shard_text(model, ctx):
    """Slice the text decoder and LM head for ``ctx`` in place (host tensors)."""
    cfg = hf_common.text_config(model.config)
    rank, world = ctx.rank, ctx.world
    heads, kv_heads, hd = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    if heads % world:
        raise ValueError(f"{heads} attention heads do not split over {world} ranks")
    if kv_heads % world and world % kv_heads:
        raise ValueError(f"{kv_heads} KV heads do not map onto {world} ranks")
    if cfg.intermediate_size % world:
        raise ValueError("intermediate size does not split over the ranks")
    q_local = heads // world
    if kv_heads >= world:
        kv_local, kv_first = kv_heads // world, rank * (kv_heads // world)
    else:
        kv_local, kv_first = 1, rank // (world // kv_heads)
    # Each rank's query heads must read only its own KV heads.
    group = heads // kv_heads
    if (rank * q_local) // group < kv_first or (
        (rank + 1) * q_local - 1
    ) // group >= kv_first + kv_local:
        raise ValueError("query heads and KV heads do not shard consistently")
    inter = cfg.intermediate_size // world
    for layer in hf_common.get_backbone(model).layers:
        attn, mlp = layer.self_attn, layer.mlp
        for proj in (
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            attn.o_proj,
            attn.gate_proj,
        ):
            if getattr(proj, "bias", None) is not None:
                raise ValueError("Muse TP supports bias-free attention only")
        q0 = rank * q_local * hd
        _rows(attn.q_proj, q0, q0 + q_local * hd)
        _rows(attn.gate_proj, q0, q0 + q_local * hd)
        kv0 = kv_first * hd
        _rows(attn.k_proj, kv0, kv0 + kv_local * hd)
        _rows(attn.v_proj, kv0, kv0 + kv_local * hd)
        _cols(attn.o_proj, q0, q0 + q_local * hd, ctx.group_name)
        _rows(mlp.gate_proj, rank * inter, (rank + 1) * inter)
        _rows(mlp.up_proj, rank * inter, (rank + 1) * inter)
        _cols(mlp.down_proj, rank * inter, (rank + 1) * inter, ctx.group_name)
    model._spyre_tp = ctx
    _shard_lm_head(model, ctx)


def _shard_lm_head(model, ctx):
    """Split the head by vocabulary, one equal shard per rank, and install
    the gathered head as the adapter's ``_spyre_lm_head_forward``."""
    cfg = hf_common.text_config(model.config)
    vocab = int(cfg.vocab_size)
    head = model.lm_head
    w = head.weight.detach()[:vocab]
    # Equal, stick-aligned shards: the gathered logits are the plain rank-major
    # concatenation (no per-rank crop, whose 202048 / 2 = 101024 = 1578.5
    # sticks would leave a half-stick write offset Spyre cannot lower); the
    # padding columns after the real vocabulary are cropped by the caller.
    # 128-row shards (two 64-element sticks) also suit a W8A8 head.
    stick = 2 * hf_common.BLOCK_SIZE
    per = -(-vocab // (ctx.world * stick)) * stick
    while (
        hf_common._largest_prime_factor(per // hf_common.BLOCK_SIZE)
        * w.shape[1]
        * (hf_common.BLOCK_SIZE * w.element_size())
        > hf_common._EAR_LIMIT_BYTES
    ):
        per += stick
    full = torch.nn.functional.pad(w, (0, 0, 0, per * ctx.world - vocab))
    local = full[ctx.rank * per : (ctx.rank + 1) * per]
    head.weight = nn.Parameter(local.contiguous(), requires_grad=False)
    head.out_features = per

    multiplier, cap = cfg.output_multiplier, cfg.final_logit_softcapping

    def gathered_head(h):
        logits = gather_vocab(model, head(h)) * multiplier
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    model._spyre_lm_head_forward = torch.compile(
        gathered_head, dynamic=False, fullgraph=True
    )

    def raw_head(h):
        return gather_vocab(model, head(h))

    # The bare gathered head (no multiplier / softcap), for the DFlash drafter.
    model._spyre_tp_raw_head = torch.compile(raw_head, dynamic=False, fullgraph=True)


def shard_assistant(assistant, ctx):
    """Slice a ``MuseGlimmerAssistantModel``'s layers for ``ctx`` (host, in place).

    q / k / v by head and gate / up by output, o / down by input; the caller
    all-reduces o and down. The context projection (``fc``), the norms and
    the head stay replicated. Returns ``(local q heads, local KV heads)``.
    """
    cfg = assistant.config
    world, rank, hd = ctx.world, ctx.rank, cfg.head_dim
    heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    if heads % world or kv_heads % world or cfg.intermediate_size % world:
        raise ValueError(f"the assistant does not split evenly over {world} ranks")
    q_local, kv_local = heads // world, kv_heads // world
    inter = cfg.intermediate_size // world
    for layer in assistant.layers:
        a, m = layer.self_attn, layer.mlp
        q0, kv0 = rank * q_local * hd, rank * kv_local * hd
        _rows(a.q_proj, q0, q0 + q_local * hd)
        _rows(a.k_proj, kv0, kv0 + kv_local * hd)
        _rows(a.v_proj, kv0, kv0 + kv_local * hd)
        _cols(a.o_proj, q0, q0 + q_local * hd)
        _rows(m.gate_proj, rank * inter, (rank + 1) * inter)
        _rows(m.up_proj, rank * inter, (rank + 1) * inter)
        _cols(m.down_proj, rank * inter, (rank + 1) * inter)
    return q_local, kv_local


def gather_vocab(model, logits):
    """``[..., padded_local]`` rank logits -> ``[..., padded vocab]`` on every rank."""
    from hf_adapters.spyre_tensor_parallel import spyre_compiled_all_gather_last_dim

    ctx = model._spyre_tp
    return spyre_compiled_all_gather_last_dim(logits, ctx.world, ctx.group_name)


def all_reduce(tensor, group_name):
    from hf_adapters.spyre_tensor_parallel import spyre_compiled_all_reduce

    return spyre_compiled_all_reduce(tensor, group_name)


def broadcast(obj, src=0):
    """Rank ``src``'s host object on every rank (decisions stay in lockstep)."""
    import torch.distributed as dist

    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
