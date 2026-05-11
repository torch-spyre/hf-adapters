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
HuggingFace Transformers adapter for Gemma models on Spyre.

Covers model_type ``gemma``: Gemma 2B, 7B, and other models that register
as ``gemma`` in HF Transformers.

Note: Gemma models are gated on HuggingFace and require authentication.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("google/gemma-7b")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    kv_cache_update,
)

# ---------------------------------------------------------------------------
# Gemma-specific RMSNorm patch
# ---------------------------------------------------------------------------


def _patch_gemma_rmsnorm(rmsnorm_cls):
    """Patch GemmaRMSNorm to stay in fp16 on Spyre.

    Gemma's RMSNorm uses (1 + weight) scaling instead of weight directly.
    """

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            eps = torch.ops.spyre.full(
                (1,),
                self.eps,
                hidden_states.device,
                torch.float16,
            )
            return (1.0 + self.weight) * (hidden_states * torch.rsqrt(variance + eps))
        else:
            xf = hidden_states.float()
            variance = (xf * xf).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + self.eps)
            return ((1.0 + self.weight.float()) * xf).to(hidden_states.dtype)

    rmsnorm_cls.forward = _forward_fp16


# ---------------------------------------------------------------------------
# Chunked LM head (vocab 256K exceeds Spyre per-core EAR limit)
# ---------------------------------------------------------------------------


def _chunk_lm_head(model, num_chunks=8):
    """Split LM head weight into N chunks along vocab dim."""
    w = model.lm_head.weight  # [vocab, hidden]
    vocab, hidden = w.shape
    chunk_size = (vocab + num_chunks - 1) // num_chunks

    STICK = 64
    chunks = nn.ModuleList()
    real_sizes = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, vocab)
        w_chunk = w[start:end].clone()
        sz = w_chunk.shape[0]
        real_sizes.append(sz)
        padded_sz = ((sz + STICK - 1) // STICK) * STICK
        if padded_sz != sz:
            w_chunk = F.pad(w_chunk, (0, 0, 0, padded_sz - sz))
        chunk = nn.Linear(hidden, padded_sz, bias=False)
        chunk.weight = nn.Parameter(w_chunk, requires_grad=False)
        chunks.append(chunk)

    model._spyre_lm_head_chunks = chunks
    model._spyre_lm_chunk_sizes = real_sizes


# ---------------------------------------------------------------------------
# Compiled block
# ---------------------------------------------------------------------------


def _make_compiled_block(layer):
    """Compiled block for Gemma: pre-norm, standard GQA, no multipliers."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            is_filling,
            token_index,
            cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Gemma forward: scaled embedding, blocks, norm, chunked LM head."""
    # GemmaTextScaledWordEmbedding already applies sqrt(hidden_size) multiplier
    h = model.model.embed_tokens(input_ids)

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = model.model.norm(h)

    # Chunked LM head: 256K vocab exceeds Spyre's per-core EAR limit.
    logits_parts = []
    for lm_chunk, real_sz in zip(
        model._spyre_lm_head_chunks, model._spyre_lm_chunk_sizes
    ):
        logits_parts.append(lm_chunk(h).to("cpu")[..., :real_sz])
    logits = torch.cat(logits_parts, dim=-1)
    return logits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Gemma model in-place."""
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    _patch_gemma_rmsnorm(GemmaRMSNorm)
    model._spyre_rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    _chunk_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in model.model.layers
    ]
