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

"""CPU end-to-end: tree speculative decoding reproduces plain greedy decoding."""

import pytest
import torch
from transformers import Qwen3Config

from hf_adapters import _xpress_tree, gemma4_spec_decode, hf_xpress_qwen3
from hf_adapters.gemma4_spec_decode import TreeSpeculativeDecoder
from tests._gemma4_tree_helpers import _unwrap

VOCAB, HIDDEN = 320, 256


def _drafter(seed=0):
    config = Qwen3Config(
        architectures=["Qwen3XPressModel"],
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=32,
        use_sliding_window=True,
        max_window_layers=2,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        tie_word_embeddings=True,
        block_size=8,
        dflash_config={"mask_token_id": 4, "target_layer_ids": [0, 2]},
        xpress_rank=32,
        xpress_mlp_hidden=64,
        xpress_block_size=8,
        sample_from_anchor=False,
    )
    model = hf_xpress_qwen3.build_drafter(config)
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.05
    return model.float().eval()


@pytest.mark.parametrize("width", [8, 12])
def test_tree_speculative_decoding_equals_greedy(
    gemma4_moe_models, eager_tree_graphs, monkeypatch, width
):
    reference, adapted = gemma4_moe_models
    _unwrap(adapted)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    monkeypatch.setattr(gemma4_spec_decode.torch, "compile", lambda fn, **_: fn)
    prompt = torch.randint(
        0, VOCAB, (1, 30), generator=torch.Generator().manual_seed(4)
    )
    with torch.no_grad():
        greedy = reference.generate(
            prompt, max_new_tokens=40, do_sample=False, min_new_tokens=40
        )[0, prompt.shape[1] :]
    decoder = TreeSpeculativeDecoder(
        adapted,
        _drafter(),
        width=width,
        frontier=3,
        branching=3,
        depth=5,
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    out = decoder.generate(prompt, max_new_tokens=40)
    assert out[0].tolist() == greedy.tolist()
    stats = decoder.stats
    assert sum(stats.accepted) >= 39 and stats.steps <= 39
