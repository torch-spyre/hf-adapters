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
HuggingFace Transformers adapter for XLM-RoBERTa encoder-only models on Spyre.

Supports models with RoBERTa/XLMRobertaConfig (e.g. BAAI/bge-m3,
intfloat/multilingual-e5-large, sentence-transformers/paraphrase-multilingual-*).

Structurally identical to BERT (same attention/FFN/post-LN module names) so the
encoder-block compile path from ``hf_bert`` is reused verbatim. Two semantic
differences from BERT must be honored in the embedding step:

- Position ids start at ``padding_idx + 1`` (fairseq-style), not 0. Real tokens
  get positions ``padding_idx+1 .. padding_idx+actual_len``; padding tokens get
  ``padding_idx``. ``prefill_encoder`` synthesizes 0-based position ids that are
  correct for BERT but wrong for XLM-R, so this adapter overrides
  ``_run_backbone_forward`` to recompute position ids from ``input_ids`` itself.
"""

from hf_adapters.hf_bert import _make_compiled_encoder_block
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    encoder_backbone_forward,
    fairseq_position_ids,
    get_backbone,
    pad_attention_heads_simple,
)


def _run_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """Encoder backbone forward with RoBERTa/XLM-R position ids."""
    emb = get_backbone(model).embeddings
    pos_ids = fairseq_position_ids(input_ids, emb.padding_idx)

    return encoder_backbone_forward(
        model, input_ids, attn_mask, pos_ids, token_type_ids
    )


_is_encoder_only = True
_is_reranker = True


def prepare_for_spyre(model):
    """Apply Spyre adaptations to an XLM-RoBERTa encoder model in-place.

    Same flow as BERT: stick-align ``head_dim`` if needed (most XLM-R models
    already have head_dim=64, e.g. bge-m3) and compile each encoder layer.
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

    if hasattr(model, "classifier"):
        model._spyre_cpu_submodules = ["classifier"]

    model._spyre_compiled_blocks = [
        _make_compiled_encoder_block(layer) for layer in backbone.encoder.layer
    ]
