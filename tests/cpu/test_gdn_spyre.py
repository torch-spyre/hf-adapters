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

"""The Spyre-form Gated DeltaNet layer (``gdn_spyre``) against stock HF in fp32.

A random Qwen3.5 linear-attention layer runs through
``hf_qwen3_5_decision._run_linear_attention_chunk`` (compiled graphs replaced by
their eager functions) in stick-sized chunks followed by single tokens, carrying
the convolution and recurrent states across calls, and must match the stock
decoder layer applied to the whole sequence at once.
"""

from typing import Any

import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from hf_adapters import hf_qwen3_5_decision
from hf_adapters.hf_common import BLOCK_SIZE

SINGLE_TOKENS = 3


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


@pytest.mark.parametrize("num_key_heads", [2, 4])
@torch.no_grad()
def test_spyre_linear_attention_matches_hf(
    monkeypatch: pytest.MonkeyPatch, num_key_heads: int
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.random.default_generator.manual_seed(794)
    config = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    reference = Qwen3_5DecoderLayer(config, layer_idx=0).eval()
    for name, parameter in reference.named_parameters():
        if name.endswith(("A_log", "dt_bias")) or "norm" in name:
            parameter.add_(0.1 * torch.randn_like(parameter))
    layer = Qwen3_5DecoderLayer(config, layer_idx=0).eval()
    layer.load_state_dict(reference.state_dict())
    block = hf_qwen3_5_decision._make_linear_attention_block(layer)
    constants = layer.linear_attn._spyre_gdn

    length = 2 * BLOCK_SIZE + SINGLE_TOKENS
    hidden_states = torch.randn(1, length, config.hidden_size)
    output = reference(hidden_states, position_embeddings=None, attention_mask=None)
    expected = output[0] if isinstance(output, tuple) else output

    conv_state = torch.zeros(1, constants.kernel_size - 1, constants.conv_dim)
    recurrent_state = torch.zeros(
        2 * constants.num_v_heads, constants.head_v_dim, constants.head_k_dim
    )
    padding_mask = torch.ones(1, length)
    pieces = [(start, BLOCK_SIZE) for start in (0, BLOCK_SIZE)]
    pieces += [(2 * BLOCK_SIZE + step, 1) for step in range(SINGLE_TOKENS)]
    outputs = []
    for start, size in pieces:
        output, conv_state, recurrent_state = (
            hf_qwen3_5_decision._run_linear_attention_chunk(
                block,
                layer.linear_attn,
                hidden_states[:, start : start + size],
                padding_mask[:, start : start + size],
                conv_state,
                recurrent_state,
            )
        )
        outputs.append(output)
    actual = torch.cat(outputs, dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
