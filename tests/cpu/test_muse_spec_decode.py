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

"""CPU end-to-end: Muse tree speculative decoding reproduces plain greedy decoding."""

import pytest
import torch

from hf_adapters import _muse_dflash, _muse_tree
from hf_adapters.muse_spec_decode import TreeSpeculativeDecoder
from tests._muse_helpers import VOCAB, WINDOW, prepared_pair, tiny_assistant


@pytest.mark.parametrize("width, top_k", [(8, 2), (16, 4)])
def test_tree_speculative_decoding_equals_greedy(monkeypatch, width, top_k):
    reference, adapted = prepared_pair(monkeypatch, _muse_tree, _muse_dflash)
    prompt = torch.randint(
        2, VOCAB - 2, (1, 30), generator=torch.Generator().manual_seed(4)
    )
    new = WINDOW + 20  # decoding crosses the sliding window
    with torch.no_grad():
        greedy = reference.generate(
            prompt, max_new_tokens=new, min_new_tokens=new, do_sample=False
        )[0, prompt.shape[1] :]
    decoder = TreeSpeculativeDecoder(
        adapted,
        tiny_assistant(),
        width=width,
        top_k=top_k,
        prefill_chunk=64,
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    decoder.verifier.capacity_multiple = 512
    out = decoder.generate(prompt, max_new_tokens=new)
    assert out[0].tolist() == greedy.tolist()
    stats = decoder.stats
    assert sum(stats.accepted) >= new - 1 and stats.steps <= new - 1
    assert len(stats.verify_ms) == stats.steps


def test_an_oracle_drafter_accepts_whole_blocks(monkeypatch):
    """With the target's own continuation as the draft, each step emits the
    whole spine plus the bonus: the accept / commit / tap plumbing loses
    nothing."""
    reference, adapted = prepared_pair(monkeypatch, _muse_tree, _muse_dflash)
    prompt = torch.randint(
        2, VOCAB - 2, (1, 20), generator=torch.Generator().manual_seed(6)
    )
    new = 40
    with torch.no_grad():
        greedy = reference.generate(
            prompt, max_new_tokens=new + 16, min_new_tokens=new + 16, do_sample=False
        )[0, prompt.shape[1] :].tolist()
    decoder = TreeSpeculativeDecoder(
        adapted,
        tiny_assistant(),
        width=8,
        prefill_chunk=64,
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    real_draft = decoder.drafter.draft
    anchors = []

    def oracle(anchor, budget, new_features=None):
        real_draft(anchor, new_features)  # keep the drafter cache in step
        # The anchor is greedy[k]; the spine continues greedy from k + 1.
        k = len(anchors) and anchors[-1][0] + anchors[-1][1]
        anchors.append((k, 0))
        assert greedy[k] == anchor
        spine = greedy[k + 1 : k + budget]
        return [anchor] + spine, [-1] + list(range(len(spine)))

    def on_commit(path, _real=decoder.verifier.commit):
        k, _ = anchors[-1]
        anchors[-1] = (k, len(path))
        return _real(path)

    monkeypatch.setattr(decoder.drafter, "propose", oracle)
    monkeypatch.setattr(decoder.verifier, "commit", on_commit)
    out = decoder.generate(prompt, max_new_tokens=new)[0].tolist()
    assert out == greedy[:new]
    assert max(decoder.stats.accepted) == 8
    assert decoder.stats.steps <= -(-(new - 1) // 8) + 1
