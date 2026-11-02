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

"""CPU coverage for Gemma 4 MoE token-tree verify and accepted-path commit.

A tiny random Gemma 4 MoE goes through the real adapter preparation, then a
tree verify must agree with the stock Hugging Face model run on each
root-to-node path, and a commit must leave caches from which the next verify
continues exactly as plain decoding would.
"""

import pytest
import torch

from hf_adapters import hf_common, hf_gemma4_moe
from hf_adapters._gemma4_tree import (
    TreeVerifier,
    accept_greedy,
    sliding_tree_mask,
    tree_ancestors,
    tree_depths,
)
from tests._gemma4_tree_helpers import PARENTS, _reference_next, _unwrap


def test_tree_geometry_helpers():
    assert tree_depths(PARENTS) == [0, 1, 2, 3, 4, 5, 1, 2, 2, 3, 4, 3]
    assert tree_ancestors(PARENTS)[11] == [0, 6, 8, 11]
    with pytest.raises(ValueError):
        tree_depths([-1, 2, 0])
    path, bonus = accept_greedy(
        [7, 3, 4, 5, 6, 9, 8, 1, 2, 0, 0, 0], PARENTS[:12], {0: 3, 1: 1, 7: 42}
    )
    assert (path, bonus) == ([0, 1, 7], 42)
    with pytest.raises(ValueError, match="duplicate"):
        accept_greedy([1, 2, 2], [-1, 0, 0], {0: 2})


def test_sliding_mask_keeps_exactly_the_window():
    # Node depth 2 at write row 70 with a 64-wide window reads ring rows
    # [70 + 2 - 64 + 1, 70) = [9, 70) plus its ancestors.
    mask = sliding_tree_mask(
        [[0], [0, 1], [0, 1, 2]], [0, 1, 2], 70, 0, 128, 64, torch.float32
    )
    allowed = (mask[0, 0, 2] == 0).nonzero().flatten().tolist()
    assert allowed == list(range(9, 70)) + [70, 71, 72]


@pytest.mark.parametrize("prompt_len", [40, 150])
def test_tree_verify_matches_the_reference_on_every_path(
    gemma4_moe_models, eager_tree_graphs, prompt_len
):
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    generator = torch.Generator().manual_seed(prompt_len)
    prompt = torch.randint(0, 320, (1, prompt_len), generator=generator)
    verifier = TreeVerifier(adapted, len(PARENTS), prefill_chunk=64)
    logits = verifier.prefill(prompt, max_new_tokens=64)
    prompt_ids = prompt[0].tolist()
    expected = _reference_next(reference, prompt_ids)
    torch.testing.assert_close(logits[0, -1, :320], expected, rtol=1e-4, atol=1e-4)

    tokens = [int(expected.argmax())] + torch.randint(
        0, 320, (len(PARENTS) - 1,), generator=generator
    ).tolist()
    logits = verifier.verify(tokens, PARENTS)
    for node, path in enumerate(tree_ancestors(PARENTS)):
        want = _reference_next(reference, prompt_ids + [tokens[p] for p in path])
        torch.testing.assert_close(logits[0, node, :320], want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "prompt_len, chunk",
    [(20, 64), (91, 64), (128, 64), (91, 128)],
    ids=["short", "past-window-padded", "past-window-exact", "one-padded-chunk"],
)
def test_tree_prefill_matches_generate_at_padded_lengths(
    gemma4_moe_models, eager_tree_graphs, prompt_len, chunk
):
    """Left-padded prompts, shorter than a chunk or past the 64-row window, give
    finite last-row logits equal to the stock adapter ``generate`` and HF."""
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    prompt = torch.randint(
        0, 320, (1, prompt_len), generator=torch.Generator().manual_seed(prompt_len)
    )
    out = hf_common.generate(
        hf_gemma4_moe._run_forward,
        adapted,
        prompt,
        max_new_tokens=1,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
        prefill_backbone_fn=hf_gemma4_moe._run_backbone_forward,
    )
    stock = out.logits[0][0, :320]
    verifier = TreeVerifier(adapted, 16, prefill_chunk=chunk)
    got = verifier.prefill(prompt, max_new_tokens=32)[0, -1, :320]
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, stock, rtol=1e-4, atol=1e-4)
    expected = _reference_next(reference, prompt[0].tolist())
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    assert verifier.state.pad == -prompt_len % chunk


def test_commit_then_verify_continues_like_plain_decoding(
    gemma4_moe_models, eager_tree_graphs
):
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    generator = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, 320, (1, 50), generator=generator)
    # 24 steps commit ~70 tokens: past the 64-row window, and the ring rolls.
    verifier = TreeVerifier(adapted, len(PARENTS), prefill_chunk=64)
    verifier.prefill(prompt, max_new_tokens=128)
    history = prompt[0].tolist()
    anchor = int(_reference_next(reference, history).argmax())
    for step in range(24):
        tokens = [anchor] + torch.randint(
            0, 320, (len(PARENTS) - 1,), generator=generator
        ).tolist()
        logits = verifier.verify(tokens, PARENTS)
        # Accept a different path every step, including off-spine branches.
        path = [[0, 6, 8, 11], [0, 1, 2, 3], [0], [0, 1, 7]][step % 4]
        accepted = [tokens[p] for p in path]
        want = _reference_next(reference, history + accepted)
        torch.testing.assert_close(
            logits[0, path[-1], :320], want, rtol=1e-4, atol=1e-4
        )
        verifier.commit(path)
        history += accepted
        anchor = int(want.argmax())
    assert verifier.state.rolls > 0


def test_taps_hold_the_tapped_layer_outputs(gemma4_moe_models, eager_tree_graphs):
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    prompt = torch.randint(0, 320, (1, 30), generator=torch.Generator().manual_seed(9))
    verifier = TreeVerifier(adapted, len(PARENTS), tap_layers=(1,), prefill_chunk=64)
    verifier.prefill(prompt, max_new_tokens=64)
    with torch.no_grad():
        states = reference(prompt, output_hidden_states=True).hidden_states
    # Prefill taps: decoder layer 1's output = HF hidden_states[2].
    torch.testing.assert_close(verifier.state.taps[1], states[2], rtol=1e-4, atol=1e-4)
    tokens = list(range(5, 5 + len(PARENTS)))
    verifier.verify(tokens, PARENTS)
    tap = verifier.tap_buffers()[1]
    assert tap.shape == (1, len(PARENTS), 256)
    for node, path in enumerate(tree_ancestors(PARENTS)):
        ids = torch.tensor([prompt[0].tolist() + [tokens[p] for p in path]])
        with torch.no_grad():
            want = reference(ids, output_hidden_states=True).hidden_states[2][0, -1]
        torch.testing.assert_close(tap[0, node], want, rtol=1e-4, atol=1e-4)


def test_a_union_cap_covering_the_tree_union_gives_the_same_verify(
    gemma4_moe_models, eager_tree_graphs, monkeypatch
):
    # 16 experts, top-4: a cap of 16 covers any union, so the union form must
    # reproduce the per-token decode FFN through a whole verify.
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    prompt = torch.randint(0, 320, (1, 40), generator=torch.Generator().manual_seed(2))
    tokens = list(range(9, 9 + len(PARENTS)))
    plain = TreeVerifier(adapted, len(PARENTS), prefill_chunk=64)
    plain.prefill(prompt, max_new_tokens=64)
    expected = plain.verify(tokens, PARENTS)
    for block in hf_gemma4_moe._gemma4_backbone(adapted).layers:
        block._moe_union_cap = 16
        block.router.slot_identity = torch.eye(16)
        block._tree_regions = None  # rebuild as the A / P / B union regions
    union = TreeVerifier(adapted, len(PARENTS), prefill_chunk=64)
    union.prefill(prompt, max_new_tokens=64)
    torch.testing.assert_close(
        union.verify(tokens, PARENTS), expected, rtol=1e-5, atol=1e-5
    )
