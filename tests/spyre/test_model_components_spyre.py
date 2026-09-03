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

"""Focused Spyre tests for model and adapter components.

These tests exercise hardware-specific behavior without loading a complete model.
Run them explicitly on a Spyre pod::

    pytest -s -vvv tests/spyre/test_model_components_spyre.py
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import GraniteConfig
from transformers.models.granite.modeling_granite import (
    GraniteAttention,
    GraniteRotaryEmbedding,
)

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _move_to_spyre_with_layout,
    apply_rope_matmul,
    pad_lm_head,
    prepare_rope_and_heads,
)

pytestmark = pytest.mark.requires_spyre


class _Layer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GraniteAttention(config, layer_idx=0)


class _Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        # Granite 3.3 2B attention geometry. Keep only the modules needed by
        # prepare_rope_and_heads rather than constructing a complete causal LM.
        self.config = GraniteConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=49159,
            max_position_embeddings=131072,
            attention_multiplier=0.015625,
            rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
        )
        self.config.head_dim = 64
        self.layers = torch.nn.ModuleList([_Layer(self.config)])
        self.rotary_emb = GraniteRotaryEmbedding(self.config)
        self.to(dtype)


class _AttentionRunner(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, hidden_states, selected_freqs):
        return _attention_forward(self.attn, hidden_states, selected_freqs)


def _attention_forward(attn, hidden_states, selected_freqs):
    batch_size, seq_len, _ = hidden_states.shape
    q = (
        attn.q_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    k = (
        attn.k_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    v = (
        attn.v_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    q = apply_rope_matmul(q, selected_freqs)
    k = apply_rope_matmul(k, selected_freqs)
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        scale=attn.scaling,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    return attn.o_proj(attn_output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_module(dtype):
    """Exercise production-sized embedding transfer and gathers on Spyre.

    Verifies that torch-spyre's patched ``Module.to`` path accepts a large
    embedding table, selects a gather-compatible layout, and preserves CPU
    lookup results for the first, middle, and last vocabulary rows in both
    supported 16-bit dtypes.
    """

    # Gemma 4 12B text embedding geometry.
    vocab_size = 262144
    hidden_size = 3840

    torch.manual_seed(42)

    input_ids = torch.tensor([[0, vocab_size // 2, vocab_size - 1]], dtype=torch.int64)
    embedding = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    # Run the CPU reference before moving the module to Spyre.
    with torch.no_grad():
        cpu_output = embedding(input_ids)

    # Module.to("spyre") routes embeddings through load_model_to_spyre(), which
    # assigns the indirect-access layout required by the gather kernel.
    assert getattr(
        torch.nn.Module.to, "_spyre_patched", False
    ), "torch-spyre did not patch torch.nn.Module.to"
    embedding.to("spyre")

    with torch.no_grad():
        spyre_output = embedding(input_ids.to("spyre")).to("cpu")

    torch.testing.assert_close(spyre_output, cpu_output, rtol=0, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padded_lm_head(dtype):
    """Exercise smooth-stick padding of a production-sized LM head.

    Starts from an aligned 2374-stick vocabulary whose large prime factor can
    overflow Spyre's per-core EAR during bundling. Verifies that ``pad_lm_head``
    advances it to a safe stick count, that the resulting large projection
    compiles in both supported 16-bit dtypes, that original-vocabulary logits
    match the unpadded CPU projection, and that padded logits remain exactly
    zero.
    """

    vocab_size = 151936
    hidden_size = 4096

    torch.manual_seed(42)

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
    hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype)

    with torch.no_grad():
        cpu_output = model.lm_head(hidden_states)

    pad_lm_head(model)
    padded_vocab_size = model.lm_head.weight.shape[0]
    assert padded_vocab_size % BLOCK_SIZE == 0
    assert padded_vocab_size > vocab_size

    model.lm_head.to("spyre")
    compiled_lm_head = torch.compile(model.lm_head, dynamic=False)

    with torch.no_grad():
        spyre_output = compiled_lm_head(hidden_states.to("spyre")).to("cpu")

    torch.testing.assert_close(
        spyre_output[..., :vocab_size], cpu_output, rtol=0, atol=2e-2
    )
    torch.testing.assert_close(
        spyre_output[..., vocab_size:],
        torch.zeros_like(spyre_output[..., vocab_size:]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prepare_rope_and_heads_padded_attention_matches_cpu(dtype):
    """Exercise Granite head padding, RoPE preparation, and compiled attention.

    Uses Granite 3.3 2B geometry, whose 64-element heads must be padded to 128
    for Spyre's matmul-based RoPE. Verifies the projection and RoPE dimensions,
    preserved attention scaling, CPU equivalence of the padded adapter path to
    stock Hugging Face attention, production RoPE dtype propagation and cache
    prebuild during model transfer, and compiled Spyre output parity in both
    supported 16-bit dtypes.
    """

    torch.manual_seed(42)

    model = _Model(dtype).eval()
    attn = model.layers[0].self_attn
    hidden_states = torch.randn(1, BLOCK_SIZE, model.config.hidden_size, dtype=dtype)
    position_ids = torch.arange(BLOCK_SIZE).unsqueeze(0)

    with torch.no_grad():
        cos, sin = model.rotary_emb(hidden_states, position_ids)
        cpu_output, _ = attn(
            hidden_states,
            position_embeddings=(cos, sin),
        )

    freqs = torch.outer(position_ids[0].float(), model.rotary_emb.inv_freq)
    cpu_freqs = torch.stack(
        [torch.cos(freqs), -torch.sin(freqs), torch.sin(freqs), torch.cos(freqs)],
        dim=1,
    ).view(1, BLOCK_SIZE, 2, 2, model.config.head_dim // 2)
    cpu_freqs = cpu_freqs.to(dtype)

    original_scaling = attn.scaling
    prepare_rope_and_heads(model)

    padded_head_dim = 2 * BLOCK_SIZE
    assert model._spyre_head_dim == padded_head_dim
    assert attn.head_dim == padded_head_dim
    assert attn.scaling == original_scaling
    assert (
        attn.q_proj.out_features == model.config.num_attention_heads * padded_head_dim
    )
    assert (
        attn.k_proj.out_features == model.config.num_key_value_heads * padded_head_dim
    )
    assert (
        attn.v_proj.out_features == model.config.num_key_value_heads * padded_head_dim
    )
    assert attn.o_proj.in_features == model.config.num_attention_heads * padded_head_dim

    identity_padding = torch.zeros(
        1,
        BLOCK_SIZE,
        2,
        2,
        (padded_head_dim - model.config.head_dim) // 2,
        dtype=dtype,
    )
    identity_padding[:, :, 0, 0, :] = 1
    identity_padding[:, :, 1, 1, :] = 1
    padded_cpu_freqs = torch.cat([cpu_freqs, identity_padding], dim=-1)
    with torch.no_grad():
        padded_cpu_output = _attention_forward(attn, hidden_states, padded_cpu_freqs)
    torch.testing.assert_close(padded_cpu_output, cpu_output, rtol=0, atol=2e-3)

    _move_to_spyre_with_layout(model, dtype)
    spyre_freqs = model._spyre_rope(hidden_states, position_ids)
    assert spyre_freqs.shape[-1] == padded_head_dim // 2

    runner = _AttentionRunner(attn)
    compiled_runner = torch.compile(runner, dynamic=False)
    with torch.no_grad():
        spyre_output = compiled_runner(hidden_states.to("spyre"), spyre_freqs).to("cpu")

    torch.testing.assert_close(spyre_output, cpu_output, rtol=0, atol=2e-2)
