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
XPress causal refiner head (arXiv 2608.02438), in a Spyre-friendly layout.

The head turns the drafter's block hidden states ``h`` ``[B, block, H]`` and the
previous token of every slot into a logit bias ``[B, block, V]``. Training form
(vllm-project/speculators ``XPressRefinerHead.block_bias``)::

    x = in_proj([down_h(h) ; down_g(mean_block(h)) ; token_embed(prev)])
    x = x + (L * tril) x              # per-rank-channel causal mix over slots
    x = x + mlp_down(silu(mlp_gate(x)) * mlp_up(x))
    bias = readout(x)

``load_checkpoint_weights`` folds this once, in fp32, into an equivalent form
with fewer device ops (exact up to rounding):

- ``in_proj`` is split by input block and folded into its neighbours:
  ``hidden_proj = in_proj_h @ down_h``, ``summary_proj = in_proj_g @ down_g`` and
  ``token_embed = token_embed @ in_proj_t.T`` — so the token path is a pure
  gather (CPU) and the pass-invariant hidden path is two ``H -> r`` matmuls.
- the mixer becomes ``tril(L) + I`` stored ``[k, j, r]`` (rank innermost, the
  Spyre stick dimension), applied as a broadcast-multiply + sum over ``j``.

Accepts both the published serving layout (``xpress_head.*``, e.g.
``UIUC-SSAIL/*-XPress-b16``) and the speculators training layout
(``refiner_head.*``).
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical (training) name -> published serving name.
_PUBLISHED_NAMES = {
    "token_embed.weight": "w1.weight",
    "down_h.weight": "down_h.weight",
    "down_g.weight": "down_g.weight",
    "in_proj.weight": "in_proj.weight",
    "mix_l": "mix.L",
    "mlp_gate.weight": "mlp.gate_proj.weight",
    "mlp_up.weight": "mlp.up_proj.weight",
    "mlp_down.weight": "mlp.down_proj.weight",
    "readout.weight": "w2.weight",
}
PUBLISHED_PREFIX = "xpress_head."
TRAINING_PREFIX = "refiner_head."


def canonical_head_weights(
    weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Select the refiner tensors from a checkpoint and return training names.

    Exactly one of the two layouts must be present, and it must be complete.
    """
    published = any(k.startswith(PUBLISHED_PREFIX) for k in weights)
    training = any(k.startswith(TRAINING_PREFIX) for k in weights)
    if published == training:
        raise ValueError(
            "Expected exactly one XPress head layout "
            f"({PUBLISHED_PREFIX}* or {TRAINING_PREFIX}*) in the checkpoint"
        )
    if published:
        prefix, names = PUBLISHED_PREFIX, _PUBLISHED_NAMES
    else:
        prefix, names = TRAINING_PREFIX, {k: k for k in _PUBLISHED_NAMES}
    present = {k[len(prefix) :] for k in weights if k.startswith(prefix)}
    expected = set(names.values())
    if present != expected:
        raise ValueError(
            f"Invalid XPress head weights: missing={sorted(expected - present)}, "
            f"unexpected={sorted(present - expected)}"
        )
    return {canon: weights[prefix + src] for canon, src in names.items()}


class XPressRefinerHead(nn.Module):
    """Folded XPress refiner; see the module docstring for the algebra."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        *,
        rank: int = 256,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size, rank, mlp_hidden) <= 0 or block_size < 2:
            raise ValueError("XPress needs positive dimensions and a block of >= 2")
        self.block_size = block_size
        self.rank = rank
        self.token_embed = nn.Embedding(vocab_size, rank)
        self.hidden_proj = nn.Linear(hidden_size, rank, bias=False)
        self.summary_proj = nn.Linear(hidden_size, rank, bias=False)
        self.mix = nn.Parameter(torch.zeros(block_size, block_size, rank))
        self.mlp_gate = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, rank, bias=False)
        self.readout = nn.Linear(rank, vocab_size, bias=False)

    def checkpoint_state(
        self, weights: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Folded ``state_dict`` for this head from raw checkpoint tensors."""
        w = {k: v.float() for k, v in canonical_head_weights(weights).items()}
        r = self.rank
        mix_l = w["mix_l"]
        if mix_l.shape != (r, self.block_size, self.block_size):
            raise ValueError(
                f"XPress mixer shape {tuple(mix_l.shape)} does not match "
                f"rank={r}, block_size={self.block_size}"
            )
        in_h, in_g, in_t = w["in_proj.weight"].split(r, dim=1)
        eye = torch.eye(self.block_size)
        folded_mix = mix_l * torch.ones_like(eye).tril() + eye  # [r, k, j]
        return {
            "token_embed.weight": w["token_embed.weight"] @ in_t.T,
            "hidden_proj.weight": in_h @ w["down_h.weight"],
            "summary_proj.weight": in_g @ w["down_g.weight"],
            "mix": folded_mix.permute(1, 2, 0).contiguous(),  # [k, j, r]
            "mlp_gate.weight": w["mlp_gate.weight"],
            "mlp_up.weight": w["mlp_up.weight"],
            "mlp_down.weight": w["mlp_down.weight"],
            "readout.weight": w["readout.weight"],
        }

    def load_checkpoint_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        state = self.checkpoint_state(weights)
        dtype = self.mix.dtype
        self.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)

    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pass-invariant hidden path ``[B, block, H] -> [B, block, r]``."""
        summary = self.summary_proj(hidden.mean(dim=1, keepdim=True))
        projected: torch.Tensor = self.hidden_proj(hidden) + summary
        return projected

    def latent(
        self, projected: torch.Tensor, token_latent: torch.Tensor
    ) -> torch.Tensor:
        """One refine pass up to the readout: fuse, causal mix, SwiGLU MLP."""
        x = projected + token_latent
        x = (self.mix[None] * x[:, None]).sum(dim=2)  # [B, k, j, r] -> [B, k, r]
        out: torch.Tensor = x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))
        return out

    def bias(self, prev_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Eager full-vocab bias ``[B, block, V]`` (reference / CPU path)."""
        token_latent = self.token_embed(prev_ids)
        latent = self.latent(self.project_hidden(hidden), token_latent)
        bias: torch.Tensor = self.readout(latent)
        return bias
