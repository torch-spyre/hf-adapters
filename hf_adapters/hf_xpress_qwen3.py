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
Spyre adapter for the **XPress speculative-decoding drafter** (``Qwen3XPressModel``).

XPress (arXiv 2608.02438, vllm-project/speculators ``models/xpress``) is a DFlash
block drafter plus a small causal refiner. It is a block *proposer*, not a
causal-LM, so like the DSpark drafters (``_dspark_common``) its public surface
is block-propose — ``run_draft_block`` / ``propose_block`` — not ``generate``.

One forward proposes ``block_size - 1`` tokens after an anchor:

1. **Backbone** (Qwen3-style layers, whatever the target family): the target's
   hidden states at ``target_layer_ids`` are concatenated and projected
   (``fc`` + ``hidden_norm``) into a context; the block ``[anchor, MASK...]`` is
   embedded with the *target's* raw ``embed_tokens`` (no Gemma sqrt(H) scale —
   training loads the bare weight) and every layer attends over
   ``concat(K/V(context), K/V(block))`` (``_dspark_common.make_dspark_block``).
   Sliding layers see context within ``sliding_window`` of the anchor and the
   block *causally*; full layers see all context and the whole block
   (speculators ``create_anchor_block_mask_mod``, ``sliding_window_non_causal``
   False). ``norm`` + the target's ``lm_head`` give base logits — unscaled, no
   softcap, as in training. Tied targets (Gemma4) reuse the embedding as the
   head; untied ones (Qwen3-8B) supply their own ``lm_head``.
2. **Refiner** (``_xpress_refiner``): ``num_passes`` Jacobi passes, each
   re-feeding the previous pass's argmax as the slot predecessors. With top-C,
   every pass scores only the base logits' top-C candidates.

Spyre/CPU split (each boundary is a known Spyre limitation, not a choice):

- Spyre: backbone blocks, final norm, LM head, refiner hidden path and latent.
- CPU: embedding gathers (no Spyre gather kernel), the wide ``fc`` context
  projection (``_dspark_common.project_context``), argmax / top-k (device
  argmax falls back, fp16 top-k corrupts large ids), and the top-C readout —
  ``C x rank`` per slot, cheaper than a device round trip and exact in fp32.

The context is a fixed-width window of the ``ctx_pad`` most recent target
positions, re-projected every block (no draft-side KV cache yet), so shapes are
static and every block compiles once. This is narrower than training: there the
full-attention layer sees the whole context and the sliding layers see
``sliding_window`` (2048) positions. Longer contexts are truncated with a
warning; pass a larger ``ctx_pad`` to ``prepare_for_spyre`` to keep more.

Usage::

    from hf_adapters import hf_xpress_qwen3 as xp
    from hf_adapters.hf_common import move_model_to_spyre

    model = xp.load_model("UIUC-SSAIL/gemma4-26b-a4b-XPress-b16",
                          target_path="google/gemma-4-26B-A4B-it",
                          dtype=torch.bfloat16)
    move_model_to_spyre(model, xp, torch.bfloat16)
    draft_ids = xp.propose_block(model, ctx_features, ctx_positions, anchor_id)
"""

import copy
import glob
import json
import os
import warnings
from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from hf_adapters._dspark_common import (
    compile_dspark_blocks,
    embed_noise_block,
    kv_pad_for,
    project_context,
    snapshot_cpu_embeddings,
    snapshot_cpu_fc,
)
from hf_adapters._xpress_refiner import (
    PUBLISHED_PREFIX,
    TRAINING_PREFIX,
    XPressRefinerHead,
)
from hf_adapters.hf_common import (
    DEVICE,
    PrecomputedRotaryEmbedding,
    _mask_fill_value,
    pad_lm_head,
    padded_vocab_size,
)

# 48 context rows + a 16-token block = kv_pad 64, the attention width the
# DSpark drafters already compile at.
CTX_PAD = 48
SLIDING = "sliding_attention"
FULL = "full_attention"
_SHARED_WITH_TARGET = ("embed_tokens.", "lm_head.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _validate_config(config: Qwen3Config) -> None:
    """Reject checkpoint layouts this adapter does not implement."""
    dflash = getattr(config, "dflash_config", None) or {}
    if "mask_token_id" not in dflash or not dflash.get("target_layer_ids"):
        raise ValueError(
            "XPress config needs dflash_config.mask_token_id/target_layer_ids"
        )
    if getattr(config, "sample_from_anchor", False):
        raise ValueError(
            "Only the fill-in layout (sample_from_anchor=False) is supported"
        )
    block = getattr(config, "block_size", None)
    if getattr(config, "xpress_block_size", block) != block or not block:
        raise ValueError("xpress_block_size must equal the DFlash block_size")
    if getattr(config, "draft_vocab_size", config.vocab_size) != config.vocab_size:
        raise ValueError("Only full-vocabulary drafters (no d2t mapping) are supported")
    if not hasattr(config, "xpress_rank"):
        raise ValueError("Not an XPress checkpoint: missing xpress_rank")
    layer_types = config.layer_types or [FULL] * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries for "
            f"{config.num_hidden_layers} draft layers"
        )
    unsupported = sorted(set(layer_types) - {SLIDING, FULL})
    if unsupported:
        raise ValueError(
            f"Unsupported draft layer_types {unsupported}; expected {SLIDING!r} "
            f"or {FULL!r}"
        )
    if SLIDING in layer_types and not config.sliding_window:
        raise ValueError("Sliding draft layers need config.sliding_window")


class XPressDrafter(nn.Module):
    """DFlash backbone + XPress refiner, with the target's vocab weights."""

    # Set by prepare_for_spyre.
    _spyre_rope: PrecomputedRotaryEmbedding
    _spyre_cpu_submodules: list[str]
    _spyre_xpress: dict[str, int]
    _spyre_compiled_blocks: list[Callable[..., torch.Tensor]]
    _spyre_xpress_fns: dict[str, Callable[..., torch.Tensor]]
    xpress_readout: nn.Linear

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        _validate_config(config)
        config = copy.deepcopy(config)  # leave the caller's config untouched
        config._attn_implementation = "sdpa"
        dflash = config.dflash_config
        hidden = config.hidden_size
        self.config = config
        self.block_size = int(config.block_size)
        self.mask_token_id = int(dflash["mask_token_id"])
        self.target_layer_ids = list(dflash["target_layer_ids"])
        self.layer_types = list(config.layer_types or [FULL] * config.num_hidden_layers)
        self.num_passes = int(getattr(config, "xpress_num_passes", 6))
        self.topc = int(getattr(config, "xpress_topc", 512))

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.fc = nn.Linear(len(self.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = Qwen3RMSNorm(hidden, eps=config.rms_norm_eps)
        self.norm = Qwen3RMSNorm(hidden, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)
        self.xpress_head = XPressRefinerHead(
            config.vocab_size,
            hidden,
            self.block_size,
            rank=int(config.xpress_rank),
            mlp_hidden=int(
                getattr(config, "xpress_mlp_hidden", 2 * config.xpress_rank)
            ),
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def build_drafter(config: Qwen3Config) -> XPressDrafter:
    return XPressDrafter(config)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_drafter_weights(
    model: XPressDrafter,
    weights: Mapping[str, torch.Tensor],
    target_embed_weight: torch.Tensor,
    target_lm_head_weight: torch.Tensor | None = None,
) -> None:
    """Strictly load a drafter checkpoint plus the target's vocab matrices.

    XPress checkpoints omit ``embed_tokens``/``lm_head``: both are the
    target's. ``target_lm_head_weight`` is the target's own ``lm_head`` when it
    is untied (e.g. Qwen3-8B); ``None`` means tied, so the embedding serves as
    the head. Works on a ``meta``-built model (``assign=True``).
    """
    head_prefixes = (PUBLISHED_PREFIX, TRAINING_PREFIX)
    expected = {
        n
        for n, _ in model.named_parameters()
        if not n.startswith(("xpress_head.", *_SHARED_WITH_TARGET))
    }
    backbone = {k: v for k, v in weights.items() if not k.startswith(head_prefixes)}
    missing, unexpected = expected - backbone.keys(), backbone.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Invalid XPress backbone weights: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    shape = (model.config.vocab_size, model.config.hidden_size)
    lm_head = (
        target_embed_weight
        if target_lm_head_weight is None
        else (target_lm_head_weight)
    )
    for name, tensor in (("embedding", target_embed_weight), ("lm_head", lm_head)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Target {name} {tuple(tensor.shape)} != drafter {shape}")
    head_state = model.xpress_head.checkpoint_state(weights)
    state = {
        **backbone,
        **{f"xpress_head.{k}": v for k, v in head_state.items()},
        "embed_tokens.weight": target_embed_weight,
        "lm_head.weight": lm_head,
    }
    model.load_state_dict(state, strict=True, assign=True)


def _local_dir(path: str, revision: str | None, patterns: list[str]) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path, revision=revision, allow_patterns=patterns)


def _embedding_key(names: list[str]) -> str:
    """The text-decoder token embedding among a target checkpoint's tensors."""
    keys = [
        n
        for n in names
        if n.endswith(".embed_tokens.weight") or n == "embed_tokens.weight"
    ]
    if len(keys) > 1:
        keys = [n for n in keys if "language_model" in n] or keys
    if len(keys) != 1:
        raise ValueError(f"Cannot identify the target token embedding among {keys}")
    return keys[0]


def _lm_head_key(names: list[str]) -> str | None:
    """The text decoder's output head among a target's tensors; None if tied."""
    keys = [n for n in names if n.endswith("lm_head.weight")]
    if len(keys) > 1:
        keys = [n for n in keys if "language_model" in n] or keys
    if len(keys) > 1:
        raise ValueError(f"Cannot identify the target lm_head among {keys}")
    return keys[0] if keys else None


def _read_target_tensor(
    target_path: str,
    revision: str | None,
    pick_key: Callable[[list[str]], str | None],
) -> torch.Tensor | None:
    """Read one tensor of a (possibly sharded) target, fetching only its shard."""
    root = _local_dir(target_path, revision, ["*.json"])
    index_path = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        key = pick_key(list(weight_map))
        if key is None:
            return None
        shard = weight_map[key]
    else:
        key, shard = None, "model.safetensors"
    if not os.path.exists(os.path.join(root, shard)):
        root = _local_dir(target_path, revision, [shard])  # fetch just this shard
    with safe_open(os.path.join(root, shard), framework="pt") as f:
        key = key or pick_key(list(f.keys()))
        return None if key is None else f.get_tensor(key)


def load_target_embedding(
    target_path: str, revision: str | None = None
) -> torch.Tensor:
    """Read only the token-embedding tensor of a (possibly sharded) target."""
    embed = _read_target_tensor(target_path, revision, _embedding_key)
    if embed is None:  # unreachable: _embedding_key raises instead
        raise ValueError(f"No token embedding in {target_path}")
    return embed


def load_target_lm_head(
    target_path: str, revision: str | None = None
) -> torch.Tensor | None:
    """The target's own ``lm_head`` if its checkpoint stores one (untied, e.g.
    Qwen3-8B); ``None`` for tied targets, whose embedding is the head."""
    return _read_target_tensor(target_path, revision, _lm_head_key)


def read_checkpoint(
    model_path: str, revision: str | None = None
) -> tuple[Qwen3Config, dict[str, torch.Tensor]]:
    """Config and raw tensors of a drafter checkpoint (local dir or HF id)."""
    root = _local_dir(model_path, revision, ["*.json", "*.safetensors"])
    shards = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors in {root}")
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        weights.update(load_file(shard))
    return Qwen3Config.from_pretrained(root), weights


def build_model(
    config: Qwen3Config,
    weights: Mapping[str, torch.Tensor],
    target_embed_weight: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    target_lm_head_weight: torch.Tensor | None = None,
) -> XPressDrafter:
    """A ready (eval, frozen) drafter from already-read checkpoint tensors."""
    with torch.device("meta"):
        model = build_drafter(config)
    load_drafter_weights(model, weights, target_embed_weight, target_lm_head_weight)
    model = model.to(dtype).eval().requires_grad_(False)
    if target_lm_head_weight is None:
        # The cast split the shared vocab matrix into two copies; re-tie it
        # (the Spyre move unties it again, for the device lm_head only).
        model.lm_head.weight = model.embed_tokens.weight
    # Rebuild the rotary after the cast: its buffers were on meta, and a bf16
    # inv_freq skews the relative RoPE angles by ~1e-3 rad per position of key
    # span: negligible at kv_pad 64, radians at the 2048-token training window.
    model.rotary_emb = Qwen3RotaryEmbedding(model.config)
    return model


def load_model(
    model_path: str,
    *,
    target_embed_weight: torch.Tensor | None = None,
    target_lm_head_weight: torch.Tensor | None = None,
    target_path: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> XPressDrafter:
    """Load an XPress drafter; vocab weights come from the target model.

    With ``target_path`` both the embedding and (for an untied target) the
    ``lm_head`` are read from the target checkpoint; with explicit tensors,
    pass ``target_lm_head_weight`` for an untied target.
    """
    if target_embed_weight is None:
        if target_path is None:
            raise ValueError("Pass target_embed_weight or target_path")
        target_embed_weight = load_target_embedding(target_path)
        target_lm_head_weight = load_target_lm_head(target_path)
    config, weights = read_checkpoint(model_path, revision)
    return build_model(
        config, weights, target_embed_weight, dtype, target_lm_head_weight
    )


# ---------------------------------------------------------------------------
# Spyre preparation
# ---------------------------------------------------------------------------


def _compile(fn):
    return torch.compile(fn, dynamic=False)


def _padded_readout(readout: torch.Tensor) -> nn.Linear:
    """Device copy of the ``[vocab, rank]`` readout, padded like ``lm_head``.

    Built on ``meta`` and given the padded weight, so no full-vocab random
    init is allocated just to be replaced.
    """
    vocab, rank = readout.shape
    padded = padded_vocab_size(vocab, rank, readout.element_size())
    with torch.device("meta"):
        linear = nn.Linear(rank, padded, bias=False)
    linear.weight = nn.Parameter(
        F.pad(readout.detach(), (0, 0, 0, padded - vocab)), requires_grad=False
    )
    return linear


def prepare_for_spyre(model: XPressDrafter, ctx_pad: int = CTX_PAD) -> None:
    """Compile the backbone + refiner and mark the CPU-resident modules.

    Called by ``hf_common.move_model_to_spyre`` before the device move; the
    modules in ``_spyre_cpu_submodules`` stay on CPU: the gather-only weights,
    and ``fc``/``hidden_norm``, which only run from their CPU snapshot.
    """
    if ctx_pad < 1:
        raise ValueError(f"ctx_pad must be positive, got {ctx_pad}")
    pad_lm_head(model)
    snapshot_cpu_embeddings(model)
    snapshot_cpu_fc(model)
    model._spyre_rope = PrecomputedRotaryEmbedding(model.rotary_emb)
    model._spyre_cpu_submodules = [
        "embed_tokens",
        "fc",
        "hidden_norm",
        "xpress_head.token_embed",
        "xpress_head.readout",
    ]
    model.xpress_readout = _padded_readout(model.xpress_head.readout.weight)

    kv_pad = kv_pad_for(ctx_pad, model.block_size)
    model._spyre_xpress = {"ctx_pad": ctx_pad, "kv_pad": kv_pad}
    model._spyre_compiled_blocks = compile_dspark_blocks(
        model, ctx_pad=ctx_pad, kv_pad=kv_pad, use_qk_norm=True
    )
    head = model.xpress_head
    model._spyre_xpress_fns = {
        "norm": _compile(model.norm),
        "lm_head": _compile(model.lm_head),
        "project_hidden": _compile(head.project_hidden),
        "latent": _compile(head.latent),
        "readout": _compile(model.xpress_readout),
    }


# ---------------------------------------------------------------------------
# Block propose
# ---------------------------------------------------------------------------


def build_draft_masks(
    ctx_positions: torch.Tensor,
    *,
    ctx_pad: int,
    kv_pad: int,
    block_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Additive ``[1, 1, block, kv_pad]`` masks per draft layer type.

    Key layout: ``[context (ctx_pad, real rows first) | block | tail pad]``.
    Built on CPU with a finite fill (``_mask_fill_value``) — the Spyre-safe
    idiom shared with the target adapters.
    """
    valid = ctx_positions.shape[0]
    anchor_pos = int(ctx_positions[-1]) + 1
    allowed = torch.zeros(block_size, kv_pad, dtype=torch.bool)
    allowed[:, :valid] = True
    allowed[:, ctx_pad : ctx_pad + block_size] = True
    masks = {FULL: allowed}
    if sliding_window is not None:
        sliding = allowed.clone()
        sliding[:, :valid] &= (ctx_positions >= anchor_pos - sliding_window)[None]
        sliding[:, ctx_pad : ctx_pad + block_size] = torch.ones(
            block_size, block_size, dtype=torch.bool
        ).tril()
        masks[SLIDING] = sliding
    fill = _mask_fill_value(dtype)
    return {
        kind: torch.zeros(1, 1, block_size, kv_pad, dtype=dtype).masked_fill(~ok, fill)
        for kind, ok in masks.items()
    }


def _check_context(
    model: XPressDrafter,
    ctx_features: torch.Tensor,
    ctx_positions: torch.Tensor,
    anchor_id: int,
) -> None:
    """Fail fast on inputs that would otherwise give silently wrong drafts."""
    width = len(model.target_layer_ids) * model.config.hidden_size
    if ctx_features.dim() != 3 or ctx_features.shape[0] != 1:
        raise ValueError(
            f"ctx_features must be [1, L, {width}], got {tuple(ctx_features.shape)}"
        )
    if ctx_features.shape[2] != width:
        raise ValueError(
            f"ctx_features width {ctx_features.shape[2]} != {width} "
            f"({len(model.target_layer_ids)} target layers x hidden size)"
        )
    if ctx_positions.dim() != 1 or ctx_positions.shape[0] != ctx_features.shape[1]:
        raise ValueError(
            f"ctx_positions {tuple(ctx_positions.shape)} must be 1-D with one "
            f"position per ctx_features row ({ctx_features.shape[1]})"
        )
    if ctx_positions.shape[0] == 0:
        raise ValueError("XPress needs at least one target context position")
    if ctx_positions.shape[0] > 1 and not bool((ctx_positions.diff() > 0).all()):
        raise ValueError("ctx_positions must be strictly increasing")
    if not 0 <= anchor_id < model.config.vocab_size:
        raise ValueError(f"anchor_id {anchor_id} outside the vocabulary")


def _window(ctx_features, ctx_positions, ctx_pad):
    """Keep the ``ctx_pad`` most recent context rows; zero-pad to ``ctx_pad``."""
    if ctx_positions.shape[0] > ctx_pad:
        warnings.warn(
            f"XPress context of {ctx_positions.shape[0]} positions truncated to "
            f"the {ctx_pad} most recent (ctx_pad); training attends to more. "
            "Pass a larger ctx_pad to prepare_for_spyre to keep more.",
            stacklevel=3,
        )
    feats = ctx_features[:, -ctx_pad:].to("cpu", torch.float32)
    positions = ctx_positions[-ctx_pad:].to("cpu")
    pad = ctx_pad - feats.shape[1]
    return F.pad(feats, (0, 0, 0, pad)), positions, F.pad(positions, (0, pad))


def run_draft_block(
    model: XPressDrafter,
    ctx_features: torch.Tensor,
    ctx_positions: torch.Tensor,
    anchor_id: int,
) -> torch.Tensor:
    """Backbone forward: target context + ``[anchor, MASK...]`` -> post-norm
    block hidden states ``[1, block, H]`` on device.

    ``ctx_features``: ``[1, L, len(target_layer_ids) * H]`` target hidden states
    (decoder-layer outputs at ``target_layer_ids``; HF ``hidden_states`` index
    ``id + 1``) for positions ``ctx_positions`` ``[L]``, ending right before the
    anchor.
    """
    _check_context(model, ctx_features, ctx_positions, anchor_id)
    spec = model._spyre_xpress
    ctx_pad, kv_pad, block = spec["ctx_pad"], spec["kv_pad"], model.block_size
    feats, valid_positions, padded_positions = _window(
        ctx_features, ctx_positions, ctx_pad
    )
    dtype = model.norm.weight.dtype
    ctx = project_context(model, feats)

    ids = torch.full((1, block), model.mask_token_id, dtype=torch.long)
    ids[0, 0] = anchor_id
    h = embed_noise_block(model, ids)  # raw embedding: no multiplier, as trained

    block_positions = int(valid_positions[-1]) + 1 + torch.arange(block)
    positions = torch.cat([padded_positions, block_positions])[None]
    freqs = model._spyre_rope(h, positions)
    masks = build_draft_masks(
        valid_positions,
        ctx_pad=ctx_pad,
        kv_pad=kv_pad,
        block_size=block,
        sliding_window=model.config.sliding_window,
        dtype=dtype,
    )
    masks = {kind: m.to(DEVICE) for kind, m in masks.items()}
    for compiled_block, kind in zip(model._spyre_compiled_blocks, model.layer_types):
        h = compiled_block(h, ctx, freqs, masks[kind])
    hidden: torch.Tensor = model._spyre_xpress_fns["norm"](h)
    return hidden


def _to_host(x: torch.Tensor, width: int) -> torch.Tensor:
    # Transfer in the device dtype, cast on CPU: a materialized fp16->fp32
    # cast on Spyre has been seen to permute lanes (spyre-inference XPress).
    return x.to("cpu").float()[..., :width]


@torch.no_grad()
def propose_block(
    model: XPressDrafter,
    ctx_features: torch.Tensor,
    ctx_positions: torch.Tensor,
    anchor_id: int,
    *,
    predecessor_id: int | None = None,
    num_passes: int | None = None,
    topc: int | None = None,
) -> torch.Tensor:
    """Propose ``block_size - 1`` greedy draft tokens (CPU ``LongTensor``).

    ``predecessor_id`` is the token before the anchor; the default (the anchor
    itself) is the vLLM#54448 serving convention, while training feeds the real
    preceding token. ``num_passes``/``topc`` default to the checkpoint's
    ``xpress_num_passes``/``xpress_topc``; ``topc=0`` scores the full vocab.
    """
    passes = model.num_passes if num_passes is None else num_passes
    topc = model.topc if topc is None else topc
    if passes < 0 or topc < 0:
        raise ValueError("num_passes and topc must be non-negative")
    fns, head, vocab = (
        model._spyre_xpress_fns,
        model.xpress_head,
        model.config.vocab_size,
    )

    hidden = run_draft_block(model, ctx_features, ctx_positions, anchor_id)
    base = _to_host(fns["lm_head"](hidden), vocab)[:, 1:]  # [1, block-1, V]
    draft = base.argmax(-1)
    if passes == 0:
        return draft[0]

    projected = fns["project_hidden"](hidden)
    if topc:
        base, candidates = base.topk(min(topc, vocab), dim=-1)
        readout = F.embedding(candidates, head.readout.weight).float()  # [1,S,C,r]
    prefix = torch.tensor(
        [[anchor_id if predecessor_id is None else predecessor_id, anchor_id]]
    )
    for _ in range(passes):
        prev = torch.cat([prefix, draft[:, :-1]], dim=1)
        token_latent = F.embedding(prev, head.token_embed.weight).to(
            DEVICE, projected.dtype
        )
        latent = fns["latent"](projected, token_latent)
        if topc:
            scores = base + torch.einsum(
                "bscr,bsr->bsc", readout, _to_host(latent, head.rank)[:, 1:]
            )
            draft = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        else:
            draft = (base + _to_host(fns["readout"](latent), vocab)[:, 1:]).argmax(-1)
    return draft[0]
