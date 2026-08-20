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
HuggingFace Transformers adapter for Granite 4.1 SWA models on Spyre.

GraniteSWAForCausalLM is identical to Granite 3.x except that layers alternate
between full attention and sliding-window attention (``layer_type`` attribute on
each ``GraniteSWADecoderLayer``). Sliding-window layers receive a local attention
mask that restricts each token to attend only within the ``sliding_window`` window;
full-attention layers delegate to ``make_standard_gqa_block`` (same as Granite 3.x).

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/tmp/models/granite-4.1-20b")
    tokenizer = AutoTokenizer.from_pretrained("/tmp/models/granite-4.1-20b")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    make_standard_gqa_block,
    pad_lm_head,
    prepare_rope_and_heads,
)
from hf_adapters.hf_granite import _run_backbone_forward, _run_forward  # noqa: F401


def _make_compiled_block(layer, sliding_window: int):
    """Compiled block for a GraniteSWA sliding-window attention layer.

    Builds a band mask: positions further than ``sliding_window`` steps back
    are set to ``-inf``.
    """
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    res_mult = layer.residual_multiplier
    v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )

        cache_len = key_cache.shape[2]
        q_len = q.shape[2]
        # Query row j occupies cache column block_base + j. Decode writes one
        # token per step, so block_base is simply the written slot (see
        # hf_gemma3/hf_gemma4 for the same band in eager code).
        #
        # Kept as a 0-d device tensor rather than read out with int(): unlike
        # Gemma 3/4, which build their band in eager code outside the compiled
        # block, this band is built *inside* it, and an int() here would sync a
        # scalar off the device mid-graph and re-specialize the binary per write
        # position — the very cost the tensor cache_index removed.
        block_base = cache_index[0]
        q_pos = (block_base + torch.arange(q_len, device=q.device)).unsqueeze(
            1
        )  # [q, 1]
        k_pos = torch.arange(cache_len, device=q.device).unsqueeze(0)  # [1, k]
        window_mask = (q_pos - k_pos) >= sliding_window  # [q, k]
        swa_mask = attn_mask.clone()
        swa_mask = swa_mask.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=swa_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out * res_mult

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h * res_mult

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a GraniteSWA model in-place."""

    sliding_window = model.config.sliding_window
    prepare_rope_and_heads(model)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        (
            _make_compiled_block(layer, sliding_window)
            if getattr(layer, "layer_type", "full_attention") == "sliding_attention"
            else make_standard_gqa_block(layer, True)
        )
        for layer in get_backbone(model).layers
    ]
