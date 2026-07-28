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
HuggingFace Transformers adapter for the Gemma4 **DSpark drafter** on Spyre.

A speculative-decoding *drafter* (``Gemma4DSparkModel``); see ``_dspark_common``
for the shared block-propose machinery and public ``_run_draft_block`` surface.

Gemma4's decoder layer differs enough from the shared 2-norm draft block that it
gets its own compiled block here (mirroring how the Gemma4 *target* adapter keeps
its own ``_make_compiled_block``):

- a four-LayerNorm sandwich (``input`` / ``post_attention`` / ``pre_feedforward``
  / ``post_feedforward``) with the post-attention norm applied to the attention
  output *before* the residual add;
- ``self.scaling = 1.0`` (SDPA scale is 1.0, not ``head_dim**-0.5``);
- per-head q/k RMSNorm;
- a per-layer ``layer_scalar`` multiply on the whole layer output;
- ``final_logit_softcapping`` applied by the drafter's own ``compute_logits``
  downstream (not here).

The concat-KV context handling, fixed ``ctx_pad``/``kv_pad`` widths, mask, RoPE
(``apply_rope_matmul``), and CPU embedding/markov are shared with
``_dspark_common``.

Usage: see ``hf_dspark_qwen3``.
"""

import torch
import torch.nn.functional as F

from hf_adapters._dspark_common import (
    _pad_markov_w2,
    build_ctx_block_mask,
    embed_noise_block,
)
from hf_adapters.hf_common import (
    DEVICE,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    pad_lm_head,
    patch_rmsnorm,
)

CTX_PAD = 56


def _make_gemma4_dspark_block(layer, *, kv_pad):
    """Compiled Gemma4 DSpark block: 4-norm sandwich + layer_scalar, concat-KV."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    head_dim = attn.head_dim
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    scaling = attn.scaling
    layer_scalar = float(layer.layer_scalar.item())

    def block_forward(hidden_states, target_hidden_states, selected_freqs, attn_mask):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, q_len, _ = h.shape
        ctx_len = target_hidden_states.shape[1]

        q = attn.q_proj(h).view(bsz, q_len, -1, head_dim)
        k_ctx = attn.k_proj(target_hidden_states)
        k_noise = attn.k_proj(h)
        v_ctx = attn.v_proj(target_hidden_states)
        v_noise = attn.v_proj(h)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)

        q = q_norm(q).transpose(1, 2)
        k = k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs[:, :, -q_len:])
        k = apply_rope_matmul(k, selected_freqs)

        pad = kv_pad - k.shape[-2]
        if pad > 0:
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scaling, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_out = attn.o_proj(attn_out)

        # Gemma4 sandwich: norm the attn output before the residual add.
        h = residual + post_attn_ln(attn_out)
        residual = h
        h = pre_ff_ln(h)
        h = mlp(h)
        h = post_ff_ln(h)
        h = residual + h
        return h * layer_scalar

    return torch.compile(block_forward, dynamic=False)


def _run_draft_block(
    model, draft_input_ids, target_hidden_states, selected_freqs, ctx_valid_len
):
    """Gemma4 DSpark block-propose forward (own block; shared ctx/mask/embed)."""
    spec = model._spyre_dspark
    ctx_pad, kv_pad, block_size = spec["ctx_pad"], spec["kv_pad"], spec["block_size"]
    ctx = model.hidden_norm(model.fc(target_hidden_states))
    h = embed_noise_block(model, draft_input_ids)
    q_len = h.shape[1]
    mask = build_ctx_block_mask(
        ctx_valid_len,
        ctx_pad,
        block_size,
        kv_pad,
        q_len,
        dtype=model.embed_tokens.weight.dtype,
    ).to(DEVICE)
    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, ctx, selected_freqs, mask)
    return model.norm(h)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to the Gemma4 DSpark drafter in-place."""
    from deepspec.modeling.dspark.gemma4.modeling import Gemma4RMSNorm

    block_size = int(model.block_size)
    kv_pad = ((CTX_PAD + block_size + 31) // 32) * 32
    model._spyre_rope = PrecomputedRotaryEmbedding(model.rotary_emb)

    patch_rmsnorm(Gemma4RMSNorm)
    pad_lm_head(model)
    _pad_markov_w2(model)
    model._spyre_dspark = {
        "ctx_pad": CTX_PAD,
        "kv_pad": kv_pad,
        "block_size": block_size,
    }
    model._spyre_compiled_blocks = [
        _make_gemma4_dspark_block(layer, kv_pad=kv_pad) for layer in model.layers
    ]
