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
HuggingFace Transformers adapter for BERT-family encoder-only models on Spyre.

Supports models with BertConfig (e.g. BAAI/bge-base-en-v1.5,
sentence-transformers/all-MiniLM-L6-v2).

Key differences from decoder adapters:
- No RoPE: absolute learned position embeddings in the embedding table.
- No KV cache: bidirectional full attention, prefill-only.
- No lm_head: returns last_hidden_state directly.
- Block signature: (hidden_states, attn_mask) -> hidden_states (no cache args).
- Norm: nn.LayerNorm (post-attention, residual-then-LN), used as-is.
  ``encoder_backbone_forward`` clones each block's output between
  compiled-block calls — see that function for why.

Usage::

    from hf_adapters import AutoSpyreModel
    from hf_adapters.hf_common import encoder_backbone_forward, prefill_encoder
    from transformers import AutoTokenizer

    model = AutoSpyreModel.from_pretrained("BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    encoded = tokenizer(["hello", "world"], return_tensors="pt",
                        padding=True, padding_side="right")
    last_hidden_state, mask = prefill_encoder(
        encoder_backbone_forward, model,
        encoded["input_ids"], encoded["attention_mask"],
    )
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    encoder_backbone_forward,
    get_backbone,
    pad_attention_heads_simple,
)


def _make_compiled_encoder_block(layer):
    """Compiled block for BERT: bidirectional MHA + post-LN + FFN + post-LN.

    Block signature (forked from the decoder contract — no KV cache, no RoPE):

        block_forward(hidden_states, attn_mask) -> hidden_states

    Closes over the layer's weight modules so that torch.compile sees a
    static graph with no Python-level attribute lookups at inference time.

    Dropout layers are skipped — this adapter is eval-only.
    """
    attn_self = layer.attention.self
    # q / k / v are named "query", "key", "value" in BertSelfAttention
    q_proj = attn_self.query  # nn.Linear(H, H), bias=True
    k_proj = attn_self.key
    v_proj = attn_self.value
    num_heads = attn_self.num_attention_heads
    head_dim = attn_self.attention_head_size
    # When pad_attention_heads_simple was applied, head_dim is the padded
    # value but Q·K^T only sums over the original non-zero entries; we must
    # scale by 1/sqrt(orig). Otherwise SDPA's default 1/sqrt(head_dim) is
    # already correct.
    orig_head_dim = getattr(attn_self, "_spyre_orig_head_dim", head_dim)
    sdpa_scale = orig_head_dim**-0.5

    o_proj = layer.attention.output.dense
    attn_ln = layer.attention.output.LayerNorm

    ffn_in = layer.intermediate.dense
    act = layer.intermediate.intermediate_act_fn

    ffn_out = layer.output.dense
    out_ln = layer.output.LayerNorm

    def block_forward(hidden_states, attn_mask):
        bsz, seq_len, _ = hidden_states.shape

        # Self-attention
        q = (
            q_proj(hidden_states)
            .view(bsz, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            k_proj(hidden_states)
            .view(bsz, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            v_proj(hidden_states)
            .view(bsz, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )

        # No RoPE. No KV cache. Bidirectional: is_causal=False.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=sdpa_scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)

        # Post-attention: project + residual + LN  (BertSelfOutput pattern)
        attn_out = o_proj(attn_out)
        hidden_states = attn_ln(attn_out + hidden_states)

        # FFN: intermediate dense + activation + output dense + residual + LN
        ffn_h = act(ffn_in(hidden_states))
        ffn_h = ffn_out(ffn_h)
        hidden_states = out_ln(ffn_h + hidden_states)

        return hidden_states

    return torch.compile(block_forward, dynamic=False)


_run_backbone_forward = encoder_backbone_forward
_is_encoder_only = True


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a BERT-family encoder model in-place.

    Pads attention heads up to a Spyre stick boundary when ``head_dim`` is
    below ``BLOCK_SIZE`` (e.g. all-MiniLM-L6-v2 with head_dim=32 → 64). Then
    walks ``model.encoder.layer``, builds a compiled encoder block for each
    layer, and stores them on ``model._spyre_compiled_blocks``.
    """
    backbone = get_backbone(model)
    cfg = model.config
    orig_head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    stick_aligned_head_dim = (
        (orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    ) * BLOCK_SIZE
    if stick_aligned_head_dim > orig_head_dim:
        pad_attention_heads_simple(
            model,
            backbone.encoder.layer,
            orig_head_dim,
            stick_aligned_head_dim,
            cfg.num_attention_heads,
        )

    model._spyre_compiled_blocks = [
        _make_compiled_encoder_block(layer) for layer in backbone.encoder.layer
    ]
