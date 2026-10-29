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

"""CPU coverage for the cached-context Muse DFlash drafter (``_muse_dflash``).

The oracle is ``transformers``' own ``MuseGlimmerAssistantModel`` driven as
``DFlashTokenCandidateGenerator`` drives it: a ``DFlashCache`` fed the
target's tapped hidden states step by step, the target's raw embedding for
the noise block, and the target's bare ``lm_head`` on block rows 1..B-1.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DFlashCache

from hf_adapters import _muse_dflash
from hf_adapters._muse_dflash import MuseDFlashDrafter
from hf_adapters._spec_tree import tree_depths
from tests._muse_helpers import (
    HIDDEN,
    MASK_ID,
    TAPS,
    VOCAB,
    WINDOW,
    prepared_pair,
    tiny_assistant,
)

TOL = dict(rtol=2e-4, atol=2e-4)


_TARGETS = {}


def setup_target(reference):
    return _TARGETS[id(reference)]


@pytest.fixture
def setup(monkeypatch):
    reference, adapted = prepared_pair(monkeypatch, _muse_dflash)
    _TARGETS[id(reference)] = adapted
    assistant = tiny_assistant()
    drafter = MuseDFlashDrafter(
        assistant, adapted, dtype=torch.float32, device="cpu", compile=False
    )
    return reference, assistant, drafter


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, len(TAPS) * HIDDEN, generator=g) * 3.0


class _HFDrafter:
    """``DFlashTokenCandidateGenerator``'s use of the assistant, step by step."""

    def __init__(self, assistant, reference):
        self.assistant = assistant
        self.embed = reference.get_input_embeddings().weight
        self.head = reference.lm_head
        self.block = assistant.config.block_size
        self.cache = DFlashCache(config=assistant.config)
        self.cache.activate_past_recording()
        self.first = True
        self.rows = 0

    @torch.no_grad()
    def draft(self, anchor, features):
        m = features.shape[0]
        self.cache.set_previous_accepted_tokens(m)
        if not self.first:
            self.cache.crop(-self.block)
        positions = torch.arange(self.rows, self.rows + m + self.block)[None]
        ids = torch.tensor([[anchor] + [MASK_ID] * (self.block - 1)])
        out = self.assistant(
            noise_embeds=F.embedding(ids, self.embed),
            context_hidden_states=features[None],
            position_ids=positions,
            attention_mask=torch.ones(1, self.rows + m + self.block, dtype=torch.long),
            past_key_values=self.cache,
        )
        self.first = False
        self.rows += m
        return self.head(out.last_hidden_state)[0]


@pytest.mark.parametrize("prompt_rows", [5, 70])
def test_draft_matches_the_hf_dflash_loop(setup, prompt_rows):
    """Prompt rows, then several steps of 1-8 new rows: every block's logits
    equal the stock assistant's, including once the context passes the window."""
    reference, assistant, drafter = setup
    hf = _HFDrafter(assistant, reference)
    drafter.reset(max_rows=256)
    prompt = _rows(prompt_rows, 0)
    drafter.append(prompt, chunk=16)
    new = None
    pending = prompt
    for step, m in enumerate([0, 3, 1, 8, 2, 5]):
        anchor = 11 + step
        if step:
            new = _rows(m, 10 + step) if m else None
            pending = new
        got = drafter.draft(anchor, new)
        want = hf.draft(anchor, pending if pending is not None else prompt[:0])
        torch.testing.assert_close(got, want, **TOL)
        pending = None
    assert drafter.rows > WINDOW or prompt_rows < WINDOW


def test_draft_with_no_new_rows_matches_too(setup):
    """A step that commits nothing new (root only, impossible in decoding but
    allowed by the API) rewrites the last row onto itself and keeps the result."""
    reference, assistant, drafter = setup
    hf = _HFDrafter(assistant, reference)
    drafter.reset(max_rows=64)
    prompt = _rows(12, 3)
    drafter.append(prompt)
    torch.testing.assert_close(drafter.draft(4), hf.draft(4, prompt), **TOL)
    torch.testing.assert_close(drafter.draft(9), hf.draft(9, prompt[:0]), **TOL)


def test_propose_builds_an_ancestor_closed_tree_from_the_block(setup):
    _, _, drafter = setup
    drafter.reset(max_rows=64)
    drafter.append(_rows(20, 4))
    tokens, parents = drafter.propose(anchor=3, budget=12)
    assert len(tokens) == 12 and tokens[0] == 3
    depths = tree_depths(parents)
    # The spine is the per-slot argmax of the same block.
    drafter.reset(max_rows=64)
    drafter.append(_rows(20, 4))
    greedy = drafter.draft(3)[1:].argmax(-1).tolist()
    spine = [i for i, d in enumerate(depths) if d and tokens[i] == greedy[d - 1]]
    assert len(spine) >= 2 and tokens[1] == greedy[0]
    assert all(0 <= t < VOCAB for t in tokens)
    seen = set()
    for t, p in zip(tokens[1:], parents[1:]):
        assert (p, t) not in seen
        seen.add((p, t))


def test_fp16_safe_norm_is_the_plain_rmsnorm():
    x = torch.randn(3, 64) * 5e4
    w = torch.rand(64)
    plain = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(_muse_dflash._rms(x, w, 1e-5), plain)


def test_fp8_drafter_with_lossless_codes_equals_the_hf_loop(monkeypatch, setup):
    """fp32 codes make W8A8 exact: the fused fp8 drafter graphs (fc blocks,
    q|k|v, o, gate|up, down) must reproduce the stock assistant."""
    from hf_adapters import _muse_fp8

    reference, assistant, _ = setup
    monkeypatch.setattr(_muse_fp8, "FP8", torch.float32)
    # The tiny assistant's widths are 128-aligned except the 384 intermediate.
    drafter = MuseDFlashDrafter(
        assistant,
        setup_target(reference),
        dtype=torch.float32,
        device="cpu",
        compile=False,
        fp8=True,
    )
    hf = _HFDrafter(assistant, reference)
    drafter.reset(max_rows=64)
    prompt = _rows(20, 5)
    drafter.append(prompt)
    torch.testing.assert_close(drafter.draft(3), hf.draft(3, prompt), **TOL)
    new = _rows(4, 6)
    torch.testing.assert_close(drafter.draft(8, new), hf.draft(8, new), **TOL)
    assert isinstance(drafter.layers[0]["qkv"], tuple)


def test_fp8_drafter_with_real_codes_tracks_the_16_bit_drafter(setup):
    reference, assistant, drafter16 = setup
    drafter8 = MuseDFlashDrafter(
        assistant,
        setup_target(reference),
        dtype=torch.float32,
        device="cpu",
        compile=False,
        fp8=True,
    )
    for d in (drafter16, drafter8):
        d.reset(max_rows=64)
        d.append(_rows(20, 7))
    a, b = drafter16.draft(3), drafter8.draft(3)
    assert torch.isfinite(b).all()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    assert cos.min() > 0.95, cos
