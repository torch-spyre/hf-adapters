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

"""CPU coverage for Muse Glimmer token-tree verify and mask-only commit.

A tiny random Muse Glimmer (real ``[s, s, s, f-NoPE]`` pattern, 48-token
window) goes through ``hf_muse_glimmer.prepare_for_spyre``. A tree verify
must agree with the stock Hugging Face model on every root-to-node path, and
commits must leave a cache from which later verifies continue exactly as
plain decoding would, through the sliding window and a compaction. The
adapter's own ``generate`` (which picks plain SDPA or the anchored
sliding-window cache per request) must be unaffected by a tree request in
between.
"""

import itertools

import pytest
import torch

from hf_adapters import _muse_tree, hf_muse_glimmer
from hf_adapters._muse_tree import MuseTreeVerifier
from hf_adapters._spec_tree import (
    accept_greedy,
    best_first_tree,
    pad_tree,
    tree_ancestors,
    tree_depths,
)
from tests._muse_helpers import (
    HIDDEN,
    VOCAB,
    WINDOW,
    prepared_pair,
    stock_forward,
    text_pattern_ok,
)

# A 12-node tree: a spine 0-1-2-3-4-5 plus side branches.
PARENTS = [-1, 0, 1, 2, 3, 4, 0, 1, 6, 2, 3, 8]
TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture
def muse_pair(monkeypatch):
    return prepared_pair(monkeypatch, _muse_tree)


def _next(reference, ids):
    return stock_forward(reference, ids).logits[0, -1]


def _verifier(model, width=len(PARENTS), **kwargs):
    kwargs.setdefault("prefill_chunk", 64)
    kwargs.setdefault("capacity_multiple", 512)
    return MuseTreeVerifier(model, width, **kwargs)


def test_tree_helpers():
    assert tree_depths(PARENTS) == [0, 1, 2, 3, 4, 5, 1, 2, 2, 3, 4, 3]
    assert tree_ancestors(PARENTS)[11] == [0, 6, 8, 11]
    with pytest.raises(ValueError):
        tree_depths([-1, 2, 0])
    path, bonus = accept_greedy(
        [7, 3, 4, 5, 6, 9, 8, 1, 2, 0, 0, 0], PARENTS, {0: 3, 1: 1, 7: 42}
    )
    assert (path, bonus) == ([0, 1, 7], 42)
    with pytest.raises(ValueError, match="duplicate"):
        accept_greedy([1, 2, 2], [-1, 0, 0], {0: 2})
    tokens, parents = pad_tree([9, 4], [-1, 0], 5)
    assert parents == [-1, 0, 0, 0, 0] and len(set(tokens[1:])) == 4


def test_best_first_tree_keeps_the_top_scoring_paths():
    ids = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    lp = [[-0.1, -1.0, -3.0], [-0.2, -2.0, -2.5], [-0.3, -0.4, -5.0]]
    tokens, parents = best_first_tree(7, ids, lp, budget=6)
    scores = {0: 0.0}
    for i in range(1, len(tokens)):
        depth = tree_depths(parents)[i]
        rank = ids[depth - 1].index(tokens[i])
        scores[i] = scores[parents[i]] + lp[depth - 1][rank]
    # Exhaustive check: the kept scores are the six best of all paths.
    every = [0.0]
    for d in range(1, 4):
        for ranks in itertools.product(range(3), repeat=d):
            every.append(sum(lp[k][r] for k, r in enumerate(ranks)))
    best = sorted(every, reverse=True)[:6]
    assert sorted(scores.values(), reverse=True) == pytest.approx(best)
    assert tokens[:4] == [7, 10, 20, 30]  # the greedy spine comes first
    tree_depths(parents)  # ancestor-closed


def test_layer_pattern_is_the_real_one(muse_pair):
    reference, _ = muse_pair
    assert text_pattern_ok(reference)


@pytest.mark.parametrize("prompt_len", [40, 150])
def test_tree_verify_matches_the_reference_on_every_path(muse_pair, prompt_len):
    reference, adapted = muse_pair
    g = torch.Generator().manual_seed(prompt_len)
    prompt = torch.randint(2, VOCAB - 2, (1, prompt_len), generator=g)
    verifier = _verifier(adapted)
    logits = verifier.prefill(prompt, max_new_tokens=64)[..., :VOCAB]
    ids = prompt[0].tolist()
    expected = _next(reference, ids)
    torch.testing.assert_close(logits[0, -1], expected, **TOL)
    tokens = [int(expected.argmax())] + torch.randint(
        2, VOCAB - 2, (len(PARENTS) - 1,), generator=g
    ).tolist()
    logits = verifier.verify(tokens, PARENTS)
    assert logits.shape[:2] == (1, len(PARENTS)) and logits.shape[-1] >= VOCAB
    logits = logits[..., :VOCAB]
    for node, path in enumerate(tree_ancestors(PARENTS)):
        want = _next(reference, ids + [tokens[p] for p in path])
        torch.testing.assert_close(logits[0, node], want, **TOL)


@pytest.mark.parametrize(
    "prompt_len, chunk", [(20, 64), (91, 64), (128, 64), (91, 128)]
)
def test_tree_prefill_matches_the_reference(muse_pair, prompt_len, chunk):
    reference, adapted = muse_pair
    prompt = torch.randint(
        2, VOCAB - 2, (1, prompt_len), generator=torch.Generator().manual_seed(1)
    )
    verifier = _verifier(adapted, prefill_chunk=chunk)
    got = verifier.prefill(prompt, max_new_tokens=32)[0, -1, :VOCAB]
    torch.testing.assert_close(got, _next(reference, prompt[0].tolist()), **TOL)
    assert verifier.state.position == prompt_len


@pytest.mark.parametrize("prompt_len, chunk", [(200, 64), (530, 512)])
def test_tree_prefill_follows_the_adapter_chunk_policy(muse_pair, prompt_len, chunk):
    """No explicit chunk: one 64-aligned chunk up to 512 tokens, 512-token
    chunks beyond (``hf_muse_glimmer.generate``'s default), both far past the
    window."""
    reference, adapted = muse_pair
    prompt = torch.randint(
        2, VOCAB - 2, (1, prompt_len), generator=torch.Generator().manual_seed(2)
    )
    verifier = _verifier(adapted, prefill_chunk=None)
    got = verifier.prefill(prompt, max_new_tokens=16)[0, -1, :VOCAB]
    padded = -(-prompt_len // chunk) * chunk
    assert verifier.state.prompt_rows == padded
    torch.testing.assert_close(got, _next(reference, prompt[0].tolist()), **TOL)


def test_commit_then_verify_continues_like_plain_decoding(muse_pair):
    """Commits off-spine paths for ~80 tokens: past the 48-token window, and
    (with a tight capacity) through a compaction."""
    reference, adapted = muse_pair
    g = torch.Generator().manual_seed(3)
    prompt = torch.randint(2, VOCAB - 2, (1, 50), generator=g)
    verifier = _verifier(adapted, capacity=256)
    verifier.prefill(prompt, max_new_tokens=100)
    history = prompt[0].tolist()
    anchor = int(_next(reference, history).argmax())
    for step in range(32):
        tokens = [anchor] + torch.randint(
            2, VOCAB - 2, (len(PARENTS) - 1,), generator=g
        ).tolist()
        logits = verifier.verify(tokens, PARENTS)[..., :VOCAB]
        path = [[0, 6, 8, 11], [0, 1, 2, 3], [0], [0, 1, 7]][step % 4]
        accepted = [tokens[p] for p in path]
        want = _next(reference, history + accepted)
        torch.testing.assert_close(logits[0, path[-1]], want, **TOL)
        verifier.commit(path)
        history += accepted
        anchor = int(want.argmax())
    assert len(history) - 50 > WINDOW
    assert verifier.state.compactions > 0


def test_taps_hold_the_tapped_layer_outputs(muse_pair):
    reference, adapted = muse_pair
    prompt = torch.randint(
        2, VOCAB - 2, (1, 30), generator=torch.Generator().manual_seed(9)
    )
    verifier = _verifier(adapted, tap_layers=(1, 4))
    verifier.prefill(prompt, max_new_tokens=64)
    states = stock_forward(reference, prompt, output_hidden_states=True).hidden_states
    feats = verifier.tap_features()
    assert feats.shape == (30, 2 * HIDDEN)
    torch.testing.assert_close(feats[:, :HIDDEN], states[2][0], **TOL)
    torch.testing.assert_close(feats[:, HIDDEN:], states[5][0], **TOL)
    tokens = list(range(5, 5 + len(PARENTS)))
    verifier.verify(tokens, PARENTS)
    feats = verifier.tap_features()
    for node, path in enumerate(tree_ancestors(PARENTS)):
        ids = prompt[0].tolist() + [tokens[p] for p in path]
        want = stock_forward(reference, ids, output_hidden_states=True).hidden_states
        torch.testing.assert_close(feats[node, :HIDDEN], want[2][0, -1], **TOL)


@pytest.mark.parametrize(
    "prompt_len, mode", [(30, None), (300, None), (1100, "anchored")]
)
def test_plain_generate_is_unaffected_by_a_tree_request(monkeypatch, prompt_len, mode):
    """``generate`` picks plain SDPA when the whole cache fits the window and
    the anchored sliding-window cache beyond it (here with 512-token prefill
    chunks); a tree request between two generations must change neither,
    and the tree prefill must match past the window too."""
    reference, adapted = prepared_pair(monkeypatch, _muse_tree, window=1024)
    g = torch.Generator().manual_seed(prompt_len)
    prompt = torch.randint(2, VOCAB - 2, (1, prompt_len), generator=g)
    new = 24
    with torch.no_grad():
        want = reference.generate(
            prompt, max_new_tokens=new, min_new_tokens=new, do_sample=False
        )
    first = hf_muse_glimmer.generate(adapted, prompt, max_new_tokens=new)
    assert getattr(adapted, "_spyre_swa_mode", None) == mode
    verifier = _verifier(adapted, prefill_chunk=None)
    logits = verifier.prefill(prompt, max_new_tokens=16)[0, -1, :VOCAB]
    torch.testing.assert_close(logits, _next(reference, prompt[0].tolist()), **TOL)
    tokens = [int(logits.argmax())] + list(range(5, 4 + len(PARENTS)))
    logits = verifier.verify(tokens, PARENTS)[..., :VOCAB]
    ids = prompt[0].tolist()
    for node, path in enumerate(tree_ancestors(PARENTS)):
        want_node = _next(reference, ids + [tokens[p] for p in path])
        torch.testing.assert_close(logits[0, node], want_node, **TOL)
    verifier.commit([0, 1])
    again = hf_muse_glimmer.generate(adapted, prompt, max_new_tokens=new)
    assert first.tolist() == want.tolist() == again.tolist()
