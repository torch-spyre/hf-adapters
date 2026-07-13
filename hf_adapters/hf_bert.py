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
    last_hidden_state = prefill_encoder(
        encoder_backbone_forward, model,
        encoded["input_ids"], encoded["attention_mask"],
    )
"""

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    encoder_backbone_forward,
    get_backbone,
    make_encoder_block,
    pad_attention_heads_simple,
)


def _make_compiled_encoder_block(layer):
    """Resolve BERT's module layout and hand off to ``make_encoder_block``.

    BERT splits attention across ``attention.self`` (Q/K/V) and
    ``attention.output`` (O + post-attn LN); the compiled body itself is
    shared with MPNet via ``make_encoder_block``.
    """
    attn_self = layer.attention.self
    return make_encoder_block(
        attn_module=attn_self,
        q_proj=attn_self.query,
        k_proj=attn_self.key,
        v_proj=attn_self.value,
        o_proj=layer.attention.output.dense,
        attn_ln=layer.attention.output.LayerNorm,
        ffn_in=layer.intermediate.dense,
        act=layer.intermediate.intermediate_act_fn,
        ffn_out=layer.output.dense,
        out_ln=layer.output.LayerNorm,
        num_heads=attn_self.num_attention_heads,
        head_dim=attn_self.attention_head_size,
    )


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
