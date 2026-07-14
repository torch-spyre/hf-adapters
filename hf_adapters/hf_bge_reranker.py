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
HuggingFace Transformers adapter for XLM-RoBERTa cross-encoder rerankers on Spyre.

Supports ``XLMRobertaForSequenceClassification`` checkpoints used as cross-encoder
rerankers, e.g. ``BAAI/bge-reranker-v2-m3``.

Architecture
------------
``XLMRobertaForSequenceClassification`` wraps an XLM-RoBERTa encoder backbone
plus a two-layer classification head::

    XLMRobertaForSequenceClassification
      roberta:     XLMRobertaModel          ← encoder backbone (compiled blocks)
        embeddings
        encoder.layer[0..N-1]              ← compiled on Spyre
      classifier:  XLMRobertaClassificationHead
        dense:     nn.Linear(hidden, hidden)
        dropout:   nn.Dropout               ← run on CPU (not compiled)
        out_proj:  nn.Linear(hidden, 1)     ← raw relevance score

Relation to hf_xlm_roberta
---------------------------
The encoder backbone is structurally identical to ``BAAI/bge-m3`` (same
XLM-R architecture, same fairseq position-id convention, same post-LN
block layout). All of the encoder machinery is shared verbatim:

- ``_make_compiled_encoder_block`` from ``hf_bert`` (re-exported by
  ``hf_xlm_roberta``) builds each compiled BERT-style encoder block.
- ``fairseq_position_ids`` computes positions that start at
  ``padding_idx + 1``.
- ``pad_attention_heads_simple`` pads ``head_dim`` to a stick boundary when
  needed (``bge-reranker-v2-m3`` has ``head_dim=64``, already aligned).
- The ``h.clone()`` call after each compiled block and after the embedding
  ``LayerNorm`` is mandatory on Spyre: BERT post-LN ends on a broadcast
  against a 1-D weight/bias, leaving the Spyre tensor in a non-canonical
  layout that the next compiled block's matmul mis-reads.  ``clone()``
  forces a canonical-layout copy in eager Python (outside torch.compile)
  before the next block runs.

Spyre-specific workarounds for the classification head
------------------------------------------------------
The ``XLMRobertaClassificationHead`` has two operations that cannot run
inside a compiled graph on Spyre:

1. ``aten.slice`` for ``[CLS]`` extraction (``h[:, 0, :]``).
   The Spyre compiler does not lower ``aten.slice.Tensor`` inside compiled
   graphs (it falls back to CPU but only outside torch.compile).  Solution:
   move ``last_hidden_state`` to CPU first, then slice on CPU.

2. ``nn.Dropout`` calls ``torch.bernoulli``, which the Spyre Inductor
   backend rejects.  Even in eval mode the op is present in the graph.
   Solution: run the entire classification head *outside* torch.compile,
   on the device the head lives on (Spyre in production, CPU in tests).
   At eval, dropout is a no-op so the result is numerically identical.

Both issues are handled in ``prefill_reranker`` (``hf_common.py``), not here.
The compiled blocks end at ``last_hidden_state``; the head runs in plain eager.

Position IDs
------------
XLM-RoBERTa uses fairseq-style positions: real tokens occupy positions
``padding_idx + 1 .. padding_idx + actual_len``; padding tokens get
``padding_idx``.  ``prefill_encoder`` synthesises 0-based ids that are
correct for BERT but wrong for XLM-R.  ``_run_backbone_forward`` overrides
them using ``fairseq_position_ids``, mirroring ``hf_xlm_roberta``.
``fairseq_position_ids`` must run on CPU (``input_ids.ne(padding_idx)``
materialises a bool tensor; the Spyre Inductor backend rejects
``bool → int32`` conversions).

Usage
-----
::

    from hf_adapters import AutoSpyreModelForSequenceClassification
    from transformers import AutoTokenizer

    model = AutoSpyreModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3"
    )
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")

    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "London is the capital of England."),
    ]
    scores = model.rerank(tokenizer, pairs)
    # tensor([  5.23, -3.11])  — raw logits; sigmoid → probability

Or call the low-level driver directly::

    import torch
    from hf_adapters.hf_common import prefill_reranker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    encoded = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True)
    scores = prefill_reranker(
        _run_backbone_forward,
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    probs = torch.sigmoid(scores)
"""

from hf_adapters.hf_bert import _make_compiled_encoder_block
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    fairseq_position_ids,
    get_backbone,
    pad_attention_heads_simple,
)

# Reranker backbone uses fairseq-style position ids and the same compiled
# encoder blocks as the XLM-RoBERTa embedder.  The _is_encoder_only flag
# tells st_backend and the test harness that this adapter is prefill-only
# (no KV-cache, no decode loop).  _is_reranker distinguishes the reranker
# path (scores, not hidden states) from plain embedding models.
_is_encoder_only = True
_is_reranker = True


def _run_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """Encoder backbone forward with XLM-R fairseq position ids.

    Mirrors ``hf_xlm_roberta._run_backbone_forward`` exactly.  Called by
    ``prefill_reranker`` (via ``prefill_encoder``) to produce the full
    ``[B, padded_len, H]`` hidden state.  The classification head is applied
    separately in ``prefill_reranker`` to keep it outside torch.compile and
    off-Spyre (avoiding the ``aten.slice`` / ``torch.bernoulli`` limitations).

    The ``position_ids`` argument is ignored — XLM-R needs fairseq-style ids
    computed directly from ``input_ids``, not the 0-based ids that
    ``prefill_encoder`` synthesises.  The signature is kept compatible so
    ``prefill_encoder`` can dispatch via the standard callable shape.

    Spyre workarounds applied here:

    - ``fairseq_position_ids`` runs on CPU (bool/int conversion guard).
    - ``h.clone()`` after the embedding LayerNorm and after every compiled
      block fixes the post-broadcast layout corruption described in the module
      docstring.
    """
    backbone = get_backbone(model)
    emb = backbone.embeddings

    # Compute fairseq-style position ids on CPU (bool tensor guard).
    pos_ids = fairseq_position_ids(input_ids, emb.padding_idx)

    h = (
        emb.word_embeddings(input_ids)
        + emb.token_type_embeddings(token_type_ids)
        + emb.position_embeddings(pos_ids)
    )
    h = emb.LayerNorm(h)
    # Layout fixup: post-LayerNorm broadcast leaves Spyre tensor in a
    # non-canonical layout.  Clone outside torch.compile before first block.
    h = h.clone() if h.device.type == "spyre" else h

    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, attn_mask)
        # Layout fixup between blocks — same reason as above.
        if h.device.type == "spyre":
            h = h.clone()

    return h


def prepare_for_spyre(model):
    """Apply Spyre adaptations to an XLM-RoBERTa reranker model in-place.

    Applies the same encoder-backbone preparation as ``hf_xlm_roberta``:

    1. Stick-align ``head_dim`` with ``pad_attention_heads_simple`` if
       ``head_dim < BLOCK_SIZE``.  ``bge-reranker-v2-m3`` has ``head_dim=64``
       so no padding is needed; the path is there for future variants.
    2. Compile each encoder layer into a ``block_forward(hidden_states,
       attn_mask) -> hidden_states`` closure via ``_make_compiled_encoder_block``
       (shared with BERT and XLM-RoBERTa).

    The classification head (``model.classifier``) is left as-is — it runs in
    plain eager outside torch.compile (see ``prefill_reranker``).

    Args:
        model: ``XLMRobertaForSequenceClassification`` loaded on CPU in eval
            mode with ``requires_grad=False``.
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
    # The classification head has out_proj: Linear(hidden, 1).  Output dim=1
    # is not stick-aligned (must be a multiple of BLOCK_SIZE=64), so the Spyre
    # compiler cannot lower it.  Keep the entire head on CPU — prefill_reranker
    # already moves hidden states to CPU before calling it.
    model._spyre_cpu_submodules = ["classifier"]
