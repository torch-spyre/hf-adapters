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
HuggingFace Transformers adapter for DistilBERT encoder-only models on Spyre.

Supports models with ``DistilBertConfig`` (e.g. distilbert/distilbert-base-uncased,
distilbert/distilbert-base-uncased-finetuned-sst-2-english).

Key differences from BERT (``hf_bert``) handled here:

- **Backbone path**: ``model.distilbert`` (not ``model.bert``). ``get_backbone``
  does not recognise this attribute, so ``_run_backbone_forward`` accesses the
  backbone directly.
- **Layer list**: ``backbone.transformer.layer`` (not ``backbone.encoder.layer``).
- **Attention module names**: ``layer.attention.{q_lin, k_lin, v_lin, out_lin}``
  (BERT uses ``attention.self.{query, key, value}`` + ``attention.output.dense``).
- **Post-attention LayerNorm**: ``layer.sa_layer_norm`` (not nested under
  ``attention.output``).
- **FFN modules**: ``layer.ffn.{lin1, activation, lin2}`` (BERT uses
  ``layer.intermediate.dense`` / ``layer.output.dense``).
- **Post-FFN LayerNorm**: ``layer.output_layer_norm`` (not ``layer.output.LayerNorm``).
- **No token_type_ids**: DistilBERT has no token-type embedding table; the
  ``token_type_ids`` argument from ``prefill_encoder`` is ignored.

Usage::

    from hf_adapters import AutoSpyreModelForSequenceClassification
    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoSpyreModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    encoded = tokenizer(["Hello, my dog is cute"], return_tensors="pt")
    scores = model(**encoded, return_dict=True).logits
    label = model.config.id2label[scores[0].argmax().item()]
    print(label)  # → POSITIVE
"""

import torch
import torch.nn as nn

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _pad_proj_input_simple,
    _pad_proj_output_simple,
    get_backbone,
    make_encoder_block,
)


def _make_compiled_encoder_block(layer):
    """Resolve DistilBERT's module layout and hand off to ``make_encoder_block``.

    DistilBERT's ``TransformerBlock`` has a flat attention submodule
    (``attention.{q_lin, k_lin, v_lin, out_lin}``) with ``sa_layer_norm`` beside
    it, and an FFN object (``ffn.{lin1, activation, lin2}``) with
    ``output_layer_norm`` beside it. The compiled body is shared with BERT/MPNet
    via ``make_encoder_block``.
    """
    attn = layer.attention
    return make_encoder_block(
        attn_module=attn,
        q_proj=attn.q_lin,
        k_proj=attn.k_lin,
        v_proj=attn.v_lin,
        o_proj=attn.out_lin,
        attn_ln=layer.sa_layer_norm,
        ffn_in=layer.ffn.lin1,
        act=layer.ffn.activation,
        ffn_out=layer.ffn.lin2,
        out_ln=layer.output_layer_norm,
        num_heads=attn.n_heads,
        head_dim=attn.attention_head_size,
    )


def _run_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """Encoder backbone forward for DistilBERT.

    Uses ``get_backbone`` so the function works both when ``model`` is a
    task-wrapper (``DistilBertForSequenceClassification``, etc., which carry a
    ``.distilbert`` sub-module) and when it is the bare backbone itself
    (``DistilBertModel``, returned by ``AutoModel``).

    ``token_type_ids`` is unused — DistilBERT has no token-type embedding table.
    The parameter is retained so ``prefill_encoder`` can dispatch through the
    same callable shape as BERT/XLM-R.
    """
    backbone = get_backbone(model)
    emb = backbone.embeddings
    h = emb.word_embeddings(input_ids) + emb.position_embeddings(position_ids)
    h = emb.LayerNorm(h)
    h = h.clone() if h.device.type == "spyre" else h
    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, attn_mask)
        if h.device.type == "spyre":
            h = h.clone()
    return h


_is_encoder_only = True


class _DistilBertClassifierHead(nn.Module):
    """Wraps DistilBERT's two-stage classification head into a single callable.

    ``prefill_sequence_classification`` calls ``model.classifier(last_hidden_state)`` where
    ``last_hidden_state`` is ``[B, L, H]``. XLM-RoBERTa's
    ``RobertaClassificationHead`` already does its own CLS slice internally.
    DistilBERT splits the head into ``pre_classifier`` (Linear + ReLU) and
    ``classifier`` (Linear) with CLS extraction happening in the model's forward
    method. This wrapper reunites them so the ``prefill_sequence_classification``
    call-site is unchanged.
    """

    def __init__(self, pre_classifier: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.pre_classifier = pre_classifier
        self.classifier = classifier

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":  # [B, L, H]
        pooled = hidden_states[:, 0]  # [B, H] — CLS token
        pooled = self.pre_classifier(pooled)
        pooled = torch.relu(pooled)
        result: torch.Tensor = self.classifier(pooled)  # [B, num_labels]
        return result


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a DistilBERT encoder model in-place.

    Pads attention heads up to a Spyre stick boundary when ``head_dim`` is below
    ``BLOCK_SIZE`` (DistilBERT-base has head_dim=64, already aligned, so padding
    is a no-op for the standard checkpoint). Then walks
    ``model.distilbert.transformer.layer``, builds a compiled encoder block for
    each layer, and stores them on ``model._spyre_compiled_blocks``.

    For ``DistilBertForSequenceClassification``, replaces ``model.classifier``
    with a ``_DistilBertClassifierHead`` that combines CLS extraction,
    ``pre_classifier``, and ``classifier`` into the single
    ``classifier(hidden_states)`` call-site that
    ``prefill_sequence_classification`` expects.
    """
    backbone = get_backbone(model)
    cfg = model.config
    orig_head_dim = cfg.dim // cfg.n_heads
    stick_aligned_head_dim = (
        (orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    ) * BLOCK_SIZE
    if stick_aligned_head_dim > orig_head_dim:
        padded = stick_aligned_head_dim
        n_heads = cfg.n_heads
        for layer in backbone.transformer.layer:
            attn = layer.attention
            attn.q_lin = _pad_proj_output_simple(
                attn.q_lin, n_heads, orig_head_dim, padded
            )
            attn.k_lin = _pad_proj_output_simple(
                attn.k_lin, n_heads, orig_head_dim, padded
            )
            attn.v_lin = _pad_proj_output_simple(
                attn.v_lin, n_heads, orig_head_dim, padded
            )
            attn.out_lin = _pad_proj_input_simple(
                attn.out_lin, n_heads, orig_head_dim, padded
            )
            attn.attention_head_size = padded
            attn.all_head_size = n_heads * padded
            attn._spyre_orig_head_dim = orig_head_dim
        model._spyre_head_dim = padded

    # For DistilBertForSequenceClassification: wrap the two-stage head so
    # ``prefill_sequence_classification(model.classifier(hidden_states))`` works.
    if hasattr(model, "pre_classifier") and hasattr(model, "classifier"):
        model.classifier = _DistilBertClassifierHead(
            model.pre_classifier, model.classifier
        )

    cpu_submodules = [
        name
        for name in (
            "classifier",
            "vocab_transform",
            "vocab_layer_norm",
            "vocab_projector",
            "qa_outputs",
        )
        if hasattr(model, name)
    ]
    if cpu_submodules:
        model._spyre_cpu_submodules = cpu_submodules

    model._spyre_compiled_blocks = [
        _make_compiled_encoder_block(layer) for layer in backbone.transformer.layer
    ]
