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

"""CPU coverage for cached-context XPress tree proposals (``_xpress_tree``).

The oracle is the speculators-math reference of ``tests/_xpress_reference``:
block hidden states over the whole context and the training-layout refiner.
"""

import pytest
import torch
from test_xpress_cpu import (
    BLOCK,
    HIDDEN,
    TAPS,
    VOCAB,
    WINDOW,
    _loaded_drafter,
    _training_head_weights,
)

from hf_adapters import hf_xpress_qwen3
from hf_adapters._xpress_tree import XPressTreeDrafter, _dedupe_siblings
from tests._xpress_reference import reference_block_hidden, reference_refiner_bias


def _drafter(**kwargs):
    model = _loaded_drafter(hf_xpress_qwen3)
    return model, XPressTreeDrafter(
        model, dtype=torch.float32, device="cpu", compile=False, **kwargs
    )


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, TAPS * HIDDEN, generator=g)


@pytest.mark.parametrize("prompt_rows, new_rows", [(5, 3), (20, 1), (40, 0)])
def test_cached_block_matches_the_full_context_reference(prompt_rows, new_rows):
    """Prompt rows appended in chunks, plus new rows inside the block graph,
    give the reference block hidden states over the whole context (and the
    sliding layers keep only their window)."""
    model, drafter = _drafter()
    drafter.reset(max_rows=128)
    prompt = _rows(prompt_rows, 0)
    drafter.append(prompt, chunk=16)
    new = _rows(new_rows, 1) if new_rows else None
    _, _, hidden = drafter.draft(anchor=7, new_features=new)
    context = prompt if new is None else torch.cat([prompt, new])
    expected = reference_block_hidden(
        model, context[None], torch.arange(context.shape[0]), anchor_id=7
    )
    torch.testing.assert_close(hidden, expected[0], rtol=1e-4, atol=1e-4)
    assert context.shape[0] > WINDOW  # the sliding window really cuts


@pytest.mark.parametrize("head_slice_chunks, slices", [(512, 1), (16, 4)])
def test_chunk_stats_find_each_slot_argmax(head_slice_chunks, slices):
    """Per-chunk winners give every slot's argmax, also when the head is
    reduced in several slices (as the full Gemma 4 vocabulary is on Spyre)."""
    model, drafter = _drafter(head_slice_chunks=head_slice_chunks)
    assert len(drafter.head) == slices
    drafter.reset(max_rows=64)
    drafter.append(_rows(12, 2))
    cmax, first, hidden = drafter.draft(anchor=3)
    logits = hidden @ model.embed_tokens.weight.float().T
    ids, vals = drafter._candidates(cmax, first)
    best = vals.argmax(-1)
    assert ids.gather(-1, best[:, None])[:, 0].tolist() == logits.argmax(-1).tolist()


def test_path_conditioned_bias_matches_the_training_refiner():
    """The refiner's slot-k bias for a tree path equals the training formula on
    the path's own previous tokens (its causal mix ignores later slots)."""
    model, drafter = _drafter()
    drafter.reset(max_rows=64)
    drafter.append(_rows(12, 3))
    cmax, first, hidden = drafter.draft(anchor=11)
    w = _training_head_weights(0)
    prev = [5, 11, 42, 17, 9]  # predecessor, anchor, three path tokens
    full = reference_refiner_bias(
        w,
        torch.tensor([prev + [0] * (BLOCK - len(prev))]),
        hidden[None],
    )[0]
    summary = hidden.mean(dim=0, keepdim=True) @ drafter.r_summary.T
    part = hidden @ drafter.r_hidden.T + summary
    for k in range(1, len(prev)):
        a = part[None, : k + 1] + drafter.r_token[torch.tensor([prev[: k + 1]])]
        c = (drafter.r_mix[k, : k + 1][None] * a).sum(dim=1)
        z = (
            c
            + (torch.nn.functional.silu(c @ drafter.r_gate.T) * (c @ drafter.r_up.T))
            @ drafter.r_down.T
        )
        torch.testing.assert_close(
            (z @ drafter.r_readout.T)[0], full[k], rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("budget", [8, 16])
def test_proposed_tree_is_ancestor_closed_with_distinct_siblings(budget):
    _, drafter = _drafter(frontier=3, branching=3, depth=5)
    drafter.reset(max_rows=64)
    drafter.append(_rows(10, 4))
    tokens, parents = drafter.propose(anchor=9, predecessor=2, budget=budget)
    assert len(tokens) == len(parents) <= budget
    assert parents[0] == -1 and tokens[0] == 9
    assert all(0 <= p < i for i, p in enumerate(parents) if i)
    pairs = [(p, t) for p, t in zip(parents[1:], tokens[1:])]
    assert len(pairs) == len(set(pairs))
    assert all(0 <= t < VOCAB for t in tokens)


def test_the_tree_spine_is_the_greedy_refined_chain():
    """With frontier 1 and branching 1 the tree is the left-to-right greedy
    chain (the Jacobi fixed point of hf_xpress_qwen3.propose_block), over each
    slot's candidates: the winner of every 64-column chunk of the base logits."""
    model, drafter = _drafter(frontier=1, branching=1, depth=BLOCK - 1)
    drafter.reset(max_rows=64)
    drafter.append(_rows(10, 5))
    tokens, parents = drafter.propose(anchor=9, predecessor=2, budget=BLOCK)
    _, _, hidden = drafter.draft(anchor=9)  # same context: rewrites the last row
    w = _training_head_weights(0)
    base = hidden @ model.embed_tokens.weight.float().T
    padded = torch.nn.functional.pad(base, (0, drafter.chunks * 64 - VOCAB), value=-1e4)
    winners = (
        padded.reshape(BLOCK, -1, 64).argmax(-1) + torch.arange(drafter.chunks) * 64
    )
    chain = [9]
    for k in range(1, BLOCK):
        prev = torch.tensor([[2] + chain + [0] * (BLOCK - len(chain) - 1)])
        bias = reference_refiner_bias(w, prev, hidden[None])[0]
        cand = winners[k][winners[k] < VOCAB]
        chain.append(int(cand[(base[k, cand] + bias[k, cand]).argmax()]))
    assert parents == [-1] + list(range(BLOCK - 1))
    assert tokens == chain


def test_duplicate_siblings_and_their_subtrees_are_dropped():
    tokens, parents = _dedupe_siblings([1, 5, 5, 7, 8], [-1, 0, 0, 2, 1])
    assert tokens == [1, 5, 8] and parents == [-1, 0, 1]
