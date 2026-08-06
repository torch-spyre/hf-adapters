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
HuggingFace Transformers adapter for MPNet encoder-only models on Spyre.

Supports models with ``MPNetConfig`` (e.g. sentence-transformers/all-mpnet-base-v2,
sentence-transformers/multi-qa-mpnet-base-dot-v1, microsoft/mpnet-base).

Differences from BERT (``hf_bert``) honored here:

- Embeddings: word + position only (no ``token_type_embeddings``).
- Position ids are fairseq-style (``padding_idx + 1`` base), shared with
  XLM-RoBERTa — same recomputation from ``input_ids`` is used.
- Module names differ: ``layer.attention.attn.{q,k,v,o}`` instead of
  ``layer.attention.self.{query,key,value}`` + ``layer.attention.output.dense``;
  the post-attention LN lives at ``layer.attention.LayerNorm`` (not under an
  ``output`` submodule).
- Relative position bias: ``encoder.relative_attention_bias`` produces an
  additive ``[1, n_heads, L, L]`` tensor (T5-style log-bucket scheme) that
  must be added to attention scores in every layer. We fold it into the
  additive ``attn_mask`` once per forward — SDPA accepts the same shape — so
  the compiled block stays signature-compatible with the BERT path.
- Q·K^T uses ``1/sqrt(head_dim)`` scaling (no ``pad_attention_heads_simple``
  needed: ``all-mpnet-base-v2`` has ``head_dim=64`` already).
"""

import math

import torch

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    fairseq_position_ids,
    get_backbone,
    make_encoder_block,
)


def _make_compiled_encoder_block(layer):
    """Resolve MPNet's module layout and hand off to ``make_encoder_block``.

    MPNet groups attention under ``attention.attn.{q,k,v,o}`` with the post-
    attention LN at ``attention.LayerNorm`` (no ``output`` submodule). The
    relative position bias is folded into ``attn_mask`` by the caller (see
    ``_run_backbone_forward``), so the compiled body is identical to BERT's.
    """
    attn = layer.attention.attn
    return make_encoder_block(
        attn_module=attn,
        q_proj=attn.q,
        k_proj=attn.k,
        v_proj=attn.v,
        o_proj=attn.o,
        attn_ln=layer.attention.LayerNorm,
        ffn_in=layer.intermediate.dense,
        act=layer.intermediate.intermediate_act_fn,
        ffn_out=layer.output.dense,
        out_ln=layer.output.LayerNorm,
        num_heads=attn.num_attention_heads,
        head_dim=attn.attention_head_size,
    )


def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """T5-style log-bucket assignment, ported verbatim from ``MPNetEncoder``.

    Run on CPU — both the bias table indexing and the resulting
    ``[1, n_heads, L, L]`` tensor are precomputed once per forward and moved
    to Spyre before being added to the attention mask.
    """
    ret = torch.zeros_like(relative_position, dtype=torch.long)
    n = -relative_position

    half = num_buckets // 2
    ret += (n < 0).to(torch.long) * half
    n = torch.abs(n)

    max_exact = half // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (half - max_exact)
    ).to(torch.long)

    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, half - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


def _compute_position_bias(backbone, qlen: int, num_buckets: int = 32) -> torch.Tensor:
    """Build the additive position-bias tensor for a sequence of length ``qlen``.

    Mirrors ``MPNetEncoder.compute_position_bias`` but runs the bucket math
    AND the bias-table lookup on CPU, then moves the assembled
    ``[1, n_heads, qlen, qlen]`` tensor to ``DEVICE``. The lookup must stay
    on CPU: ``aten.embedding`` falls back to CPU on Spyre but returns a
    Spyre-resident tensor, and the subsequent ``permute/unsqueeze/contiguous``
    triggers a ``copy_from_d2d`` that the Spyre compiler can't lower. The
    bias table is tiny (``[num_buckets, n_heads]``) so indexing a CPU copy
    of the weights is negligible per forward.
    """
    context_position = torch.arange(qlen, dtype=torch.long)[:, None]
    memory_position = torch.arange(qlen, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = _relative_position_bucket(relative_position, num_buckets=num_buckets)

    bias_weight_cpu = backbone.encoder.relative_attention_bias.weight.detach().to("cpu")
    values: torch.Tensor = bias_weight_cpu[rp_bucket]  # [qlen, qlen, n_heads]
    bias = values.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, n_heads, q, q]
    return bias.to(DEVICE)


def _run_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """Encoder backbone forward for MPNet.

    Departs from ``encoder_backbone_forward`` in three places:

    - ``token_type_ids`` is unused — MPNet has no token-type table. The
      parameter is kept in the signature so ``prefill_encoder`` can dispatch
      via the same callable shape as BERT/XLM-R.
    - Position ids are recomputed fairseq-style from ``input_ids`` (the
      0-based ids ``prefill_encoder`` synthesizes are wrong for MPNet).
    - Relative position bias is computed once and added into ``attn_mask``.
      SDPA treats ``attn_mask`` as an additive bias, so folding the two
      means the compiled block needs no extra argument.
    """
    backbone = get_backbone(model)
    emb = backbone.embeddings

    pos_ids = fairseq_position_ids(input_ids, emb.padding_idx)

    h = emb.word_embeddings(input_ids) + emb.position_embeddings(pos_ids)
    h = emb.LayerNorm(h)
    h = h.clone() if h.device.type == "spyre" else h

    qlen = input_ids.shape[1]
    pos_bias = _compute_position_bias(backbone, qlen).to(attn_mask.dtype)
    attn_mask = attn_mask + pos_bias

    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, attn_mask)
        if h.device.type == "spyre":
            h = h.clone()
    return h


_is_encoder_only = True


def prepare_for_spyre(model):
    """Apply Spyre adaptations to an MPNet encoder model in-place.

    Walks ``model.encoder.layer``, builds a compiled encoder block for each
    layer, and stores them on ``model._spyre_compiled_blocks``.
    ``all-mpnet-base-v2`` has ``head_dim=64`` (already stick-aligned) so no
    head padding is needed; if a future MPNet variant ships a smaller
    ``head_dim``, add a pad pass here using ``_pad_proj_*`` from
    ``hf_common`` against the ``layer.attention.attn.{q,k,v,o}`` modules.
    """
    backbone = get_backbone(model)
    cfg = model.config
    orig_head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    assert orig_head_dim >= BLOCK_SIZE, (
        f"MPNet adapter assumes head_dim ({orig_head_dim}) >= BLOCK_SIZE "
        f"({BLOCK_SIZE}); add head padding for smaller variants."
    )

    cpu_submodules = [
        name for name in ("lm_head", "qa_outputs") if hasattr(model, name)
    ]
    if cpu_submodules:
        model._spyre_cpu_submodules = cpu_submodules

    model._spyre_compiled_blocks = [
        _make_compiled_encoder_block(layer) for layer in backbone.encoder.layer
    ]
