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
Shared utilities for HuggingFace Transformers adapters on Spyre.

Provides RoPE precomputation, RMSNorm patching, LM head padding, mask
construction, KV cache update helpers, and a model-agnostic generate loop.
Per-model adapters import from this module and provide only model-specific
compiled block functions.
"""

import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import factorint

DEVICE = "spyre"
BLOCK_SIZE = 64  # Spyre stick size at fp16 (128 bytes / 2 bytes per element)


class SpyreUnsupportedModelError(ValueError):
    """Architecture is supported, but this config can't run on Spyre."""


class SpyreNoAdapterError(ValueError):
    """No Spyre adapter is registered for this model's architecture."""


def assert_spyre_dimensions(config, model_name):
    """Reject configs whose ``hidden_size``/``intermediate_size`` is stick-misaligned.

    The Spyre compiler lays tensors out in ``BLOCK_SIZE``-element sticks.
    Matmuls over a dimension that is not a multiple of ``BLOCK_SIZE`` produce
    stick index expressions it can't lower (e.g. ``floor(d2/320)`` for a 312-wide
    dim), surfacing as a cryptic ``Unsupported stick expression`` deep in
    ``torch.compile``. This covers both sub-stick dims (e.g. ``hidden_size=8``)
    and misaligned ones (e.g. ``hidden_size=312``).

    ``head_dim`` is not checked — adapters auto-pad it to a stick boundary (see
    ``prepare_rope_and_heads`` / ``hf_bert.prepare_for_spyre``);
    ``hidden_size``/``intermediate_size`` can't be padded without changing the
    model's arithmetic. Real models clear this bar; it fires on tiny test
    fixtures (e.g. ``trl-internal-testing/tiny-*``, ``cointegrated/rubert-tiny2``).
    """
    dim_config = text_config(config)
    misaligned = [
        (f, v)
        for f in ("hidden_size", "intermediate_size")
        if (v := getattr(dim_config, f, None)) is not None and v % BLOCK_SIZE != 0
    ]
    if misaligned:
        details = ", ".join(f"{f}={v}" for f, v in misaligned)
        raise SpyreUnsupportedModelError(
            f"Model {model_name} has Spyre-incompatible dimensions: {details} "
            f"(not a multiple of one stick, {BLOCK_SIZE}). The Spyre compiler "
            f"cannot lower matmuls over stick-misaligned dimensions. Use a model "
            f"whose hidden_size and intermediate_size are both multiples of "
            f"{BLOCK_SIZE}."
        )


def get_backbone(model):
    """Return the transformer backbone of an HF model.

    Auto-loaded models come in two shapes:

    - ``AutoModelForCausalLM`` returns a wrapper (``Qwen3ForCausalLM``,
      ``LlamaForCausalLM``, ...) whose backbone lives at ``model.model``
      and which exposes ``model.lm_head``.
    - ``AutoModel`` returns the bare backbone (``Qwen3Model``,
      ``LlamaModel``, ...) — no ``.model`` attribute, no ``lm_head``.

    Adapter code reaches into the backbone to access ``embed_tokens``,
    ``layers``, ``norm``, ``rotary_emb``. This accessor resolves the right
    object regardless of how the model was loaded.

    Multimodal causal-LM wrappers (e.g. Gemma 4's
    ``Gemma4ForConditionalGeneration``) nest the text decoder one level deeper
    at ``model.model.language_model``; descend into ``.language_model`` when
    present so the text backbone is returned.

    GPT-2-family wrappers (``GPT2LMHeadModel``) keep the backbone at
    ``model.transformer`` rather than ``model.model``; fall back to it when
    ``.model`` is absent. The bare ``GPT2Model`` (``AutoModel``) has neither, so
    it is returned as-is — it already is the backbone.

    GPT-NeoX wrappers (``GPTNeoXForCausalLM``) keep the backbone at
    ``model.gpt_neox`` (with ``embed_in``/``layers``/``final_layer_norm``);
    fall back to it as well. The bare ``GPTNeoXModel`` is returned as-is.
    """
    if hasattr(model, "model"):
        inner = model.model
    elif hasattr(model, "transformer"):
        inner = model.transformer
    elif hasattr(model, "gpt_neox"):
        inner = model.gpt_neox
    elif hasattr(model, "roberta"):
        # XLMRobertaForSequenceClassification keeps the backbone at .roberta
        inner = model.roberta
    else:
        inner = model
    return getattr(inner, "language_model", inner)


def text_config(config):
    """Return the config carrying the text-decoder dims (``hidden_size``,
    ``head_dim``, ``num_hidden_layers``, ...).

    The config-level companion of ``get_backbone``: text-only configs expose
    these directly, while multimodal wrappers (Gemma 3/4, Granite Vision,
    Mistral3) nest them on a ``text_config`` sub-config. Returns whichever holds
    them, so the shared RoPE/KV/head helpers read dims.
    """
    return getattr(config, "text_config", None) or config


# ---------------------------------------------------------------------------
# RoPE: precompute rotation matrices on CPU (FMS approach)
# ---------------------------------------------------------------------------


class PrecomputedRotaryEmbedding(nn.Module):
    """Builds [S, 2, 2, D/2] rotation matrix cache on CPU.

    Returns ``selected_freqs`` [B, L, 2, 2, D/2] on Spyre, indexed by
    position_ids.  The companion ``apply_rope_matmul`` applies the rotation
    without any tensor slicing.

    When ``padded_head_dim`` is set (larger than the native rope dim),
    the rotation matrix is padded with identity entries so that
    ``apply_rope_matmul`` on the full padded head_dim passes through
    non-rotated dimensions unchanged.
    """

    def __init__(self, original_rope: nn.Module, padded_head_dim: Optional[int] = None):
        super().__init__()
        self.original = original_rope
        self.padded_head_dim = padded_head_dim
        self._freq_cache: Optional[torch.Tensor] = None
        self._cached_len = 0
        self._freq_dtype: torch.dtype = torch.float16

    def set_dtype(self, dtype: torch.dtype) -> None:
        """Switch the freq cache to a different dtype. Called when the model
        is moved to Spyre with a non-fp16 dtype (e.g. bf16 for Mistral)."""
        self._freq_dtype = dtype
        if self._freq_cache is not None:
            self._freq_cache = self._freq_cache.to(dtype)

    def _extend_cache(self, max_len: int):
        if max_len <= self._cached_len:
            return
        target_len = max(max_len, self._cached_len * 2, 2048)
        inv_freq = self.original.inv_freq.to("cpu").float()
        rope_half = inv_freq.shape[0]  # type: ignore[index]
        t = torch.arange(target_len, dtype=inv_freq.dtype)  # type: ignore[arg-type]
        freqs = torch.outer(
            t, inv_freq  # type: ignore[arg-type]
        ).float()  # [S, rope_half] # type: ignore[arg-type]
        scaling = getattr(self.original, "attention_scaling", 1.0)
        rot = torch.stack(
            [
                torch.cos(freqs) * scaling,
                -torch.sin(freqs) * scaling,
                torch.sin(freqs) * scaling,
                torch.cos(freqs) * scaling,
            ],
            dim=1,
        ).view(
            target_len, 2, 2, rope_half  # type: ignore[arg-type]
        )  # type: ignore[arg-type]

        if self.padded_head_dim is not None:
            padded_half = self.padded_head_dim // 2
            if padded_half > rope_half:
                pad_half = padded_half - rope_half
                ident = torch.zeros(target_len, 2, 2, pad_half)  # type: ignore[arg-type]
                ident[:, 0, 0, :] = 1.0
                ident[:, 1, 1, :] = 1.0
                rot = torch.cat([rot, ident], dim=-1)

        self._freq_cache = rot.contiguous().to(self._freq_dtype)
        self._cached_len = target_len

    def forward(self, hidden_states, position_ids):
        pos_cpu = position_ids.to("cpu")
        max_pos = int(pos_cpu.max().item()) + 1
        self._extend_cache(max_pos)
        selected = self._freq_cache[pos_cpu]  # [B, L, 2, 2, D/2]
        return selected.to(DEVICE)


def set_rope_dtype(model, dtype: torch.dtype) -> None:
    """Explicitly set the freq-cache dtype on a model's precomputed RoPE.

    ``model._spyre_rope`` is either a single ``PrecomputedRotaryEmbedding``
    (most adapters) or a ``dict`` of them keyed by layer type (Gemma 3/4, which
    use different RoPE per sliding/global layer). This applies ``set_dtype`` to
    whichever shape is present.
    """
    rope = getattr(model, "_spyre_rope", None)
    if rope is None:
        return
    ropes = rope.values() if isinstance(rope, dict) else [rope]
    for r in ropes:
        if hasattr(r, "set_dtype"):
            r.set_dtype(dtype)


def _rope_cache_len(model) -> int:
    """Length to pre-build the RoPE rotation cache to (>= any decode position).

    Prefer the model's declared context window; fall back to the
    ``PrecomputedRotaryEmbedding`` default (2048) when no config value is found.
    """
    cfg = model.config
    # Multimodal configs keep the text context window on a nested text_config;
    # check both (the outer config may carry a larger window than the text sub-config).
    objs = [cfg]
    if (inner := text_config(cfg)) is not cfg:
        objs.append(inner)
    candidates = []
    for obj in objs:
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and v > 0:
                candidates.append(v)
    return max(candidates) if candidates else 2048


def prebuild_rope_cache(model) -> None:
    """Build the precomputed RoPE rotation cache on CPU, before moving to Spyre.

    ``PrecomputedRotaryEmbedding`` builds its rotation-matrix cache lazily on the
    first ``forward``. On Spyre that first call happens *during* a compiled
    device forward, where the cache-construction ops (``arange``/``outer``/
    ``cos``/``sin``/``cat``) get pulled into the graph and silently corrupt the
    result — small GPT-NeoX/Pythia models produce wrong logits from the first
    token. Building the cache here, while still on CPU, keeps those ops out of
    the device graph; ``forward``'s ``_extend_cache`` then early-returns on Spyre
    and only indexes + transfers the prebuilt cache.

    Pre-extends to the model's full context length so ``forward`` never triggers
    an on-device rebuild mid-generation (which would re-introduce the bug).
    """
    rope = getattr(model, "_spyre_rope", None)
    if rope is None:
        return
    target_len = _rope_cache_len(model)
    ropes = rope.values() if isinstance(rope, dict) else [rope]
    for r in ropes:
        if hasattr(r, "_extend_cache"):
            r._extend_cache(target_len)


class InvFreqShim(nn.Module):
    """Minimal ``original_rope`` stand-in for ``PrecomputedRotaryEmbedding``.

    ``PrecomputedRotaryEmbedding`` reads ``.inv_freq`` and ``.attention_scaling``
    off its ``original`` module — the layout of stock HF's ``RotaryEmbedding``,
    which stores a single ``inv_freq`` buffer. Several models instead store one
    ``<layer_type>_inv_freq`` buffer (and, for Gemma, a matching
    ``<layer_type>_attention_scaling``) *per* layer type — sliding vs full
    attention with different theta. To build one ``PrecomputedRotaryEmbedding``
    per layer type, wrap each per-type ``inv_freq`` (+ scaling) in this shim.

    The ``inv_freq`` length equals ``head_dim / 2`` for that layer type;
    ``PrecomputedRotaryEmbedding`` derives the rotation-matrix width from it.

    ``attention_scaling`` defaults to ``1.0`` (the "default" RoPE type, e.g.
    ModernBERT, where no post-scaling is applied); pass the model's per-type
    scaling for RoPE types that use it.
    """

    def __init__(self, inv_freq, attention_scaling=1.0):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = attention_scaling


def apply_rope_matmul(x, selected_freqs):
    """Apply RoPE via matmul with [2,2,D/2] rotation matrix. No slicing.

    The stock-HF form ``x*cos + rotate_half(x)*sin`` slices along head_dim;
    that slice form COMPILES on Spyre but mis-lowers to wrong numbers in fp16
    (tracked upstream: torch-spyre/torch-spyre#3133). This matmul form avoids
    the head-dim slice entirely and lowers correctly.

    Args:
        x: [B, H, L, D] — query or key tensor
        selected_freqs: [B, L, 2, 2, D/2] — rotation matrices

    Returns: [B, H, L, D]
    """
    B, H, L, D = x.shape
    half = D // 2
    x_ = x.transpose(1, 2).reshape(B, L, H, 2, half)
    sf = selected_freqs[:, :, None, :, :, :]
    out = sf.mul(x_.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# KV cache update (inside compiled graph)
# ---------------------------------------------------------------------------


def kv_cache_update(
    k, v, key_cache, value_cache, is_filling, token_index, cache_position
):
    """Update pre-allocated KV cache via native slice assignment at cache_position.

    All args are tensors; is_filling/token_index/cache_position are Python
    scalars that torch.compile specializes on.
    """
    if is_filling:
        k_write = k[:, :, token_index : token_index + 1, :]
        v_write = v[:, :, token_index : token_index + 1, :]
    else:
        k_write = k
        v_write = v

    # Native slicing assignment, replacing the deprecated torch.ops.spyre.overwrite
    # cache_position is a Python int the compiler specializes on, so the slice
    # bound is a compile-time constant on both CPU and Spyre.
    # NOTE: on Spyre this still compiles one binary per distinct cache_position
    seq_len = k_write.shape[2]
    key_cache[:, :, cache_position : cache_position + seq_len, :] = k_write
    value_cache[:, :, cache_position : cache_position + seq_len, :] = v_write

    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def _pad_proj_output_simple(proj, n_heads, orig_head_dim, padded_head_dim):
    """End-pad each head of a [n_heads*head_dim, hidden] output projection.

    Used for V (no RoPE) and for Q/K/V on non-RoPE encoders.
    """
    w = proj.weight
    hidden = w.shape[1]
    new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
    for h in range(n_heads):
        s = h * orig_head_dim
        d = h * padded_head_dim
        new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
    new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_b[d : d + orig_head_dim] = proj.bias[s : s + orig_head_dim]
        new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def _pad_proj_input_simple(proj, n_heads, orig_head_dim, padded_head_dim):
    """End-pad each head along the input dim of an O-style projection.

    Shape goes from [hidden, n_heads*orig_head_dim] to
    [hidden, n_heads*padded_head_dim]. Bias is along the output dim and is
    unchanged.
    """
    w = proj.weight
    hidden = w.shape[0]
    new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
    for h in range(n_heads):
        s = h * orig_head_dim
        d = h * padded_head_dim
        new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
    new_proj = nn.Linear(n_heads * padded_head_dim, hidden, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(proj.bias.clone(), requires_grad=False)
    return new_proj


# ---------------------------------------------------------------------------
# Partial-RoPE alignment for split Q/K projections
#
# Shared by adapters whose model rotates only the first ``rope_dim`` dims of
# each head (partial RoPE) and/or needs the head padded to a Spyre stick.
# HF's ``rotate_half`` pairs (j, j + rope_dim//2) within the first ``rope_dim``
# dims; ``apply_rope_matmul`` pairs (j, j + head_dim//2) across the full head.
# These helpers reorder/pad the split projection weights (and biases, when
# present) so the two pairings agree. Used by hf_phi3 (bias-free) and
# hf_gpt_neox (bias-carrying); the bias branches no-op when ``proj.bias`` is None.
# ---------------------------------------------------------------------------


def rope_dim_permutation(head_dim, rope_dim):
    """Build index permutation aligning partial-RoPE pairing with apply_rope_matmul.

    HF rotate_half pairs (j, j+rope_dim//2) within the first rope_dim dims.
    apply_rope_matmul pairs (j, j+head_dim//2) across full head_dim.
    This reorders head_dim so both pairings agree.
    """
    rope_half = rope_dim // 2
    pass_half = (head_dim - rope_dim) // 2
    return torch.cat(
        [
            torch.arange(0, rope_half),
            torch.arange(rope_dim, rope_dim + pass_half),
            torch.arange(rope_half, rope_dim),
            torch.arange(rope_dim + pass_half, head_dim),
        ]
    )


def permute_proj_for_rope(proj, num_heads, head_dim, perm):
    """Permute a split Q/K projection's output dims within each head for RoPE.

    Permutes the weight, and the bias when present (e.g. GPT-NeoX carries an
    ``attention_bias`` on QKV; the bias must follow the same output-dim
    reordering as the weight or the rotary dims pick up the wrong offsets).
    Bias-free projections (Phi-3) skip the bias branch.
    """
    w = proj.weight.data.view(num_heads, head_dim, -1)
    proj.weight.data = w[:, perm, :].contiguous().view(num_heads * head_dim, -1)
    if proj.bias is not None:
        b = proj.bias.data.view(num_heads, head_dim)
        proj.bias.data = b[:, perm].contiguous().view(num_heads * head_dim)


def pad_qk_proj_for_rope(proj, n_heads, orig_head_dim, padded_head_dim):
    """Interleave-pad a split Q/K projection from orig_head_dim to padded_head_dim.

    Each head's data is placed so that ``apply_rope_matmul``'s
    ``[2, padded_head_dim//2]`` reshape sees the first rotary half in
    ``[0:orig_half]`` and the second in ``[padded_half:padded_half+orig_half]``;
    the gaps are zero and RoPE's identity-padding leaves them untouched. Zero
    dims contribute nothing to Q·K^T, so the result is numerically identical to
    the unpadded head. The bias (when present) is padded the same way.
    """
    orig_half = orig_head_dim // 2
    padded_half = padded_head_dim // 2
    w = proj.weight.data
    has_bias = proj.bias is not None
    hidden = w.shape[1]
    new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
    for h in range(n_heads):
        s = h * orig_head_dim
        d = h * padded_head_dim
        new_w[d : d + orig_half, :] = w[s : s + orig_half, :]
        new_w[d + padded_half : d + padded_half + orig_half, :] = w[
            s + orig_half : s + orig_head_dim, :
        ]
    new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=has_bias)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if has_bias:
        b = proj.bias.data
        new_b = torch.zeros(n_heads * padded_head_dim, dtype=b.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_b[d : d + orig_half] = b[s : s + orig_half]
            new_b[d + padded_half : d + padded_half + orig_half] = b[
                s + orig_half : s + orig_head_dim
            ]
        new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def pad_attention_heads(
    model, layers, orig_head_dim, padded_head_dim, num_heads, num_kv_heads
):
    """Zero-pad Q/K/V/O attention projections to a larger head_dim.

    Q and K use interleaved padding compatible with the RoPE [2, D/2]
    reshape: each half-group is padded separately so that
    ``apply_rope_matmul`` sees the original data in the correct
    positions and identity-rotates the zero-padded positions.

    V and O use simple end-padding per head (they don't pass through
    RoPE, so layout within a head doesn't matter).

    Note: only Q/K *need* padding for RoPE stick alignment.  V and O
    could use the original head_dim (PyTorch SDPA supports E != Ev),
    which would shrink the KV cache. This is blocked by the Spyre
    compiler — see https://github.com/torch-spyre/torch-spyre/issues/1739.
    Once that is resolved, remove V/O padding here and rely on the
    ``v_head_dim`` infrastructure already wired through the block
    forwards and cache creation.

    Args:
        model: HF model — stores padded dim as ``model._spyre_head_dim``.
        layers: Iterable of decoder layers (each must have ``self_attn``).
        orig_head_dim: Original head dimension.
        padded_head_dim: Target head dimension (must be > orig_head_dim).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
    """
    assert orig_head_dim % 2 == 0, f"head_dim must be even, got {orig_head_dim}"
    assert (
        padded_head_dim % 2 == 0
    ), f"padded head_dim must be even, got {padded_head_dim}"
    assert padded_head_dim > orig_head_dim, (
        f"padded_head_dim ({padded_head_dim}) must exceed "
        f"orig_head_dim ({orig_head_dim})"
    )
    assert padded_head_dim // 2 >= BLOCK_SIZE, (
        f"padded head_dim/2 ({padded_head_dim // 2}) must be >= "
        f"BLOCK_SIZE ({BLOCK_SIZE})"
    )

    orig_half = orig_head_dim // 2
    padded_half = padded_head_dim // 2

    def _pad_qk_rope(proj, n_heads):
        """Interleaved padding for Q/K: pad within each [2, D/2] group."""
        w = proj.weight
        hidden = w.shape[1]
        new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[d : d + orig_half, :] = w[s : s + orig_half, :]
            new_w[d + padded_half : d + padded_half + orig_half, :] = w[
                s + orig_half : s + orig_head_dim, :
            ]
        new_proj = nn.Linear(
            hidden, n_heads * padded_head_dim, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
            for h in range(n_heads):
                s = h * orig_head_dim
                d = h * padded_head_dim
                new_b[d : d + orig_half] = proj.bias[s : s + orig_half]
                new_b[d + padded_half : d + padded_half + orig_half] = proj.bias[
                    s + orig_half : s + orig_head_dim
                ]
            new_proj.bias = nn.Parameter(new_b, requires_grad=False)
        return new_proj

    for layer in layers:
        attn = layer.self_attn
        orig_scaling = attn.scaling
        attn.q_proj = _pad_qk_rope(attn.q_proj, num_heads)
        attn.k_proj = _pad_qk_rope(attn.k_proj, num_kv_heads)
        # V/O padding: needed until torch-spyre/torch-spyre#1739 is resolved.
        attn.v_proj = _pad_proj_output_simple(
            attn.v_proj, num_kv_heads, orig_head_dim, padded_head_dim
        )
        attn.o_proj = _pad_proj_input_simple(
            attn.o_proj, num_heads, orig_head_dim, padded_head_dim
        )
        attn.head_dim = padded_head_dim
        attn.scaling = orig_scaling

    model._spyre_head_dim = padded_head_dim


def pad_attention_heads_simple(
    model, layers, orig_head_dim, padded_head_dim, num_heads
):
    """Zero-pad Q/K/V/O attention projections for non-RoPE encoders (BERT).

    Counterpart of ``pad_attention_heads`` for encoders that do not run RoPE.
    All four projections use simple end-padding per head (no interleaved
    ``[2, D/2]`` layout). Used to lift ``head_dim`` to a Spyre stick boundary
    (one stick = ``BLOCK_SIZE`` elements at fp16) so SDPA / Q-K matmul lower
    cleanly — e.g. all-MiniLM-L6-v2 (head_dim=32 → 64).

    Updates ``BertSelfAttention`` attributes that the compiled block closes
    over: ``attention_head_size`` and ``all_head_size``. ``num_attention_heads``
    stays. The SDPA scale is auto-computed from the tensor's last dim, so no
    explicit scale fixup is needed.

    Args:
        model: HF model — stores padded dim as ``model._spyre_head_dim``.
        layers: Iterable of encoder layers (each must have ``attention.self``
            and ``attention.output.dense``).
        orig_head_dim: Original head dimension.
        padded_head_dim: Target head dimension (must be > orig_head_dim and
            >= BLOCK_SIZE).
        num_heads: Number of attention heads.
    """
    assert padded_head_dim > orig_head_dim, (
        f"padded_head_dim ({padded_head_dim}) must exceed "
        f"orig_head_dim ({orig_head_dim})"
    )
    assert (
        padded_head_dim >= BLOCK_SIZE
    ), f"padded_head_dim ({padded_head_dim}) must be >= BLOCK_SIZE ({BLOCK_SIZE})"

    for layer in layers:
        attn = layer.attention.self
        attn.query = _pad_proj_output_simple(
            attn.query, num_heads, orig_head_dim, padded_head_dim
        )
        attn.key = _pad_proj_output_simple(
            attn.key, num_heads, orig_head_dim, padded_head_dim
        )
        attn.value = _pad_proj_output_simple(
            attn.value, num_heads, orig_head_dim, padded_head_dim
        )
        layer.attention.output.dense = _pad_proj_input_simple(
            layer.attention.output.dense,
            num_heads,
            orig_head_dim,
            padded_head_dim,
        )
        attn.attention_head_size = padded_head_dim
        attn.all_head_size = num_heads * padded_head_dim
        # SDPA's default scale is 1/sqrt(D) on the *padded* dim, but Q·K^T
        # only sums over the original non-zero entries. Stash the original
        # so the compiled block can pass scale=1/sqrt(orig) explicitly.
        attn._spyre_orig_head_dim = orig_head_dim

    model._spyre_head_dim = padded_head_dim


def patch_rmsnorm(rmsnorm_cls):
    """Patch any RMSNorm class: variance reduction at input dtype on Spyre, fp32
    on CPU (to match stock HF).

    Args:
        rmsnorm_cls: The RMSNorm class to patch (e.g. GraniteRMSNorm, Qwen3RMSNorm).
    """

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            # Spyre path: variance reduction at input dtype (see the note above).
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            return self.weight * (
                hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            )
        else:
            # CPU path: use float32 for numerical stability (matches stock HF)
            xf = hidden_states.float()
            variance = (xf * xf).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * xf.to(hidden_states.dtype)

    rmsnorm_cls.forward = _forward_fp16


def patch_layernorm(*layernorms):
    """Patch ``nn.LayerNorm`` *instances* to a manual decomposition on Spyre.

    The fused ``F.layer_norm`` lowering can produce **NaN** on Spyre for some
    models (small-variance / near-constant rows).
    The Spyre path recomputes LayerNorm as explicit ops with the mean/variance
    reduction in **fp32** (matching CPU's hidden fp32 accumulate), normalizing
    back in the input dtype. CPU keeps stock ``F.layer_norm``.

    Patches the given **instances** (binds a new ``forward``), not the
    ``nn.LayerNorm`` class, so it never perturbs LayerNorms in other adapters.

    Args:
        *layernorms: ``nn.LayerNorm`` instances to patch (skips ``None``).
    """
    import types as _types

    def _forward(self, x):
        if x.device.type == "spyre":
            # Reduction in fp32 (matches CPU's hidden fp32 accumulate); the
            # normalize + affine run back in the input dtype.
            xf = x.float()
            mean = xf.mean(-1, keepdim=True)
            d = xf - mean
            variance = (d * d).mean(-1, keepdim=True)
            inv = torch.rsqrt(variance + self.eps).to(x.dtype)
            h = (x - mean.to(x.dtype)) * inv
            if self.weight is not None:
                h = h * self.weight
            if self.bias is not None:
                h = h + self.bias
            return h
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

    for ln in layernorms:
        if ln is None:
            continue
        assert isinstance(
            ln, torch.nn.LayerNorm
        ), f"patch_layernorm expects nn.LayerNorm instances, got {type(ln)}"
        ln.forward = _types.MethodType(_forward, ln)


_GELU_TANH_COEFF = math.sqrt(2.0 / math.pi)


def patch_new_gelu(gelu_cls):
    """Patch a tanh-approximation GELU class to avoid ``torch.pow`` on Spyre.

    The ``gelu_new`` / ``NewGELUActivation`` form
    ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))`` computes the ``x**3``
    term with ``torch.pow(x, 3.0)``, which the Spyre Inductor backend cannot
    lower (same class of issue as RMSNorm's ``pow(2)``). The Spyre path uses
    element-wise ``x * x * x`` instead; the CPU path keeps the original
    ``torch.pow`` form so adapter outputs stay bit-identical to stock HF on CPU.

    Shared by the learned-absolute-position decoders that use this activation
    (``hf_gpt2``, ``hf_gpt_neo``). ``tanh`` itself lowers fine on Spyre.

    Args:
        gelu_cls: the activation class to patch (e.g. ``NewGELUActivation``).
    """

    def _forward(self, input):
        if input.device.type == "spyre":
            inner = _GELU_TANH_COEFF * (input + 0.044715 * (input * input * input))
        else:
            inner = _GELU_TANH_COEFF * (input + 0.044715 * torch.pow(input, 3.0))
        return 0.5 * input * (1.0 + torch.tanh(inner))

    gelu_cls.forward = _forward


def _largest_prime_factor(n: int) -> int:
    """Largest prime factor of ``n`` (n >= 2). Used to bound the lm_head span."""
    return int(max(factorint(n)))


# Spyre per-core EAR (effective address range) limit for one tensor: 256 MB.
_EAR_LIMIT_BYTES = 256 * 1024 * 1024


def _get_lm_head(model):
    """Return the model's output-projection (LM head) module, or ``None``."""
    for name in ("lm_head", "embed_out"):
        head = getattr(model, name, None)
        if head is not None and hasattr(head, "weight"):
            return head
    return None


def pad_lm_head(model):
    """Pad the LM head vocab dim up to a stick boundary with a "smooth" stick count.

    The lm_head is a ``batchmatmul`` ``X[M,K] @ W[K,N]`` (K=hidden, N=padded_vocab)
    whose weight sticks on N: physical dims ``[N/BLOCK_SIZE sticks, K, BLOCK_SIZE]``.
    Work division splits the stick axis (``N/BLOCK_SIZE``) across cores. When that
    count has a large prime factor, the per-core residual can't be made small
    enough and its span overflows the 256 MB per-core EAR limit, aborting the
    bundler (``dxp_standalone --bundle`` SIGABRT in
    ``sdsc_fused_bmm_transpose_unsqueeze``).

    So we add sticks until the count is "smooth" enough that the worst-case
    residual fits: ``largest_prime_factor(sticks) * hidden * BLOCK_SIZE *
    dtype_bytes <= EAR_LIMIT``. The bound uses the largest prime factor as the
    smallest extent the splitter can be forced to leave on one core, and the full
    K (a core reuses the whole K of its N-partition under weight-stationary
    reuse). This is conservative — the exact split rule is inferred from the error,
    not read from compiler source — but it reproduces the observed spans and never
    under-pads. Examples: 152064 -> 2376 = 2**3*3**3*11 fits as-is; 151936 -> 2374
    = 2*1187 would leave residual 1187 (297 MB at hidden 2048), so bump to
    2375 = 5**3*19; primes (2377, 1601) are the worst case.

    Smooth counts are dense, so bumps are tiny (0-1 sticks for every current
    model). Keeps the single-kernel lm_head — no per-token decode cost, unlike
    ``chunk_lm_head`` (the fallback when even a smooth count can't fit).

    No-op when ``model`` has no ``lm_head`` (e.g. backbones loaded via
    ``AutoModel`` for embedding workloads).
    """
    head = _get_lm_head(model)
    if head is None:
        return
    w = head.weight
    vocab = w.shape[0]
    hidden = w.shape[1]
    dtype_bytes = w.element_size()
    # Max residual sticks whose per-core span fits the EAR limit.
    max_residual = _EAR_LIMIT_BYTES // (hidden * BLOCK_SIZE * dtype_bytes)
    sticks = (vocab + BLOCK_SIZE - 1) // BLOCK_SIZE
    while _largest_prime_factor(sticks) > max_residual:
        sticks += 1
    padded = sticks * BLOCK_SIZE
    if padded != vocab:
        head.weight = nn.Parameter(
            F.pad(w, (0, 0, 0, padded - vocab)), requires_grad=False
        )


def chunk_lm_head(model, num_chunks=8):
    """Split the LM head weight into N stick-padded chunks along the vocab dim.

    Fallback for when a single ``pad_lm_head`` head can't fit the 256 MB per-core
    EAR limit — i.e. no ``num_chunks``-free smooth stick count brings
    ``largest_prime_factor(sticks) * hidden * BLOCK_SIZE * dtype_bytes`` under the
    limit (only reachable at very large hidden). Splitting into N independent
    ``nn.Linear`` heads cuts each chunk's vocab extent by ~N, shrinking the
    per-core span at the cost of N kernels + N D2H copies + a CPU cat per token.

    Currently **unused**: every supported model (incl. 200K–262K vocab Phi-4 /
    Gemma) fits a single smooth-padded head, which is cheaper. Kept as the escape
    hatch for future models that don't. Callers must run the chunks and cat on
    CPU themselves (see ``model._spyre_lm_head_chunks`` /
    ``model._spyre_lm_chunk_sizes``).

    No-op when ``model`` has no ``lm_head``.
    """
    if not hasattr(model, "lm_head"):
        return
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


def split_fused_linear(w: torch.Tensor) -> tuple[nn.Linear, nn.Linear]:
    """Split a [2*out, in] fused weight into two [out, in] nn.Linear modules.

    Used by models with fused QKV (phi3) or fused gate+up MLP (granitemoehybrid).
    Each half of the weight becomes a separate linear layer with bias=False.
    """
    half = w.shape[0] // 2
    hidden = w.shape[1]

    def _mk(data):
        p = nn.Linear(hidden, half, bias=False)
        p.weight = nn.Parameter(data.clone(), requires_grad=False)
        return p

    return _mk(w[:half]), _mk(w[half:])


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------


def build_prefill_mask(
    batch_size,
    padded_len,
    max_cache_len,
    prompt_offsets,
    dtype=torch.float16,
):
    """Causal mask for prefill, masking left-padding and unused cache positions."""
    mask = torch.zeros((batch_size, 1, padded_len, max_cache_len), dtype=dtype)
    if isinstance(prompt_offsets, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, : prompt_offsets[b].item()] = -torch.inf
    else:
        mask[:, :, :, :prompt_offsets] = -torch.inf
    for i in range(padded_len):
        mask[:, :, i, i + 1 :] = -torch.inf
    return mask


def build_expansion_mask(
    batch_size,
    block_size,
    max_cache_len,
    used_cache_len,
    prompt_offsets,
    dtype=torch.float16,
):
    """Causal mask for an expansion decode step."""
    mask = torch.zeros((batch_size, 1, block_size, max_cache_len), dtype=dtype)
    if isinstance(prompt_offsets, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, : prompt_offsets[b].item()] = -torch.inf
    else:
        mask[:, :, :, :prompt_offsets] = -torch.inf
    for j in range(block_size):
        attend_up_to = used_cache_len - block_size + j + 1
        mask[:, :, j, attend_up_to:] = -torch.inf
    return mask


def build_prefill_mask_right_padded(
    batch_size,
    padded_len,
    actual_lengths,
    is_causal=True,
    dtype=torch.float16,
):
    """Prefill mask for right-padded sequences.

    Used by the embedding path. Sequences are right-padded: real tokens
    occupy positions ``0..actual_lengths[b]-1`` and trailing positions
    ``actual_lengths[b]..padded_len-1`` hold padding. Cache length equals
    ``padded_len`` since there is no decode budget — embeddings are
    prefill-only.

    When ``is_causal=True`` (default), token ``i`` attends to ``0..i``.
    When ``is_causal=False``, real tokens attend to every real token —
    used by embedding models with ``config.is_causal=False``.

    Compared to ``build_prefill_mask`` (left-padded, used by ``generate``):
      - Padding columns to mask are at the **end** of the row, not the start.
      - Output shape is ``[B, 1, padded_len, padded_len]``; there is no
        separate ``max_cache_len`` because no decode follows.

    Padding-position rows compute garbage; callers crop them away when
    returning ``[B, actual_length, H]``.
    """
    mask = torch.zeros((batch_size, 1, padded_len, padded_len), dtype=dtype)
    if is_causal:
        for i in range(padded_len):
            mask[:, :, i, i + 1 :] = -torch.inf
    if isinstance(actual_lengths, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, actual_lengths[b].item() :] = -torch.inf
    else:
        mask[:, :, :, actual_lengths:] = -torch.inf
    return mask


def add_sliding_window_band(mask, sliding_window, dtype=torch.float16):
    """Restrict an additive bidirectional mask to a local ``±sliding_window`` band.

    ModernBERT alternates global (full) attention layers with local
    (sliding-window) layers. The local layers let token ``i`` attend only to
    tokens ``j`` with ``|i - j| <= sliding_window`` — matching HF's
    ``create_bidirectional_sliding_window_mask`` (inclusive on both sides,
    using ``config.sliding_window`` directly).

    Takes the global additive mask ``[B, 1, L, L]`` (zeros for allowed pairs,
    ``-inf`` for padding) and returns a new mask with the same padding plus
    ``-inf`` on every off-band pair. The global padding is preserved by adding,
    so padded columns stay masked in the local layers too.

    Built on CPU (the band is a static ``[L, L]`` pattern) then left for the
    caller to move to ``DEVICE`` alongside the global mask.
    """
    padded_len = mask.shape[-1]
    idx = torch.arange(padded_len)
    off_band = (idx[:, None] - idx[None, :]).abs() > sliding_window  # [L, L]
    band = torch.zeros((padded_len, padded_len), dtype=dtype)
    band[off_band] = -torch.inf
    return mask + band[None, None, :, :]


def add_causal_sliding_window_band(mask, query_cache_coords, sliding_window):
    """Restrict an additive *causal* mask to a backward ``sliding_window`` band.

    Gemma 4's sliding ("local") attention layers are causal AND windowed: a
    query may attend to a key only when ``0 <= query_pos - key_pos <
    sliding_window`` (HF's ``create_sliding_window_causal_mask`` uses an
    exclusive lower bound — a window of ``sliding_window`` keys ending at the
    query). The global ("full") layers use the plain causal mask unchanged.

    This works in the **KV-cache coordinate system** used by ``generate`` and
    the test harness, where cache column ``c`` holds the token whose absolute
    position is ``c - prompt_offset`` and a query row's cache coordinate ``r``
    has absolute position ``r - prompt_offset``. The ``prompt_offset`` cancels
    in the difference, so ``query_pos - key_pos == r - c`` — the band is an
    index distance between the query's cache coordinate and the key column.
    This is why the caller passes cache coordinates, not absolute positions.

    Unlike ``add_sliding_window_band`` (symmetric ±window, for bidirectional
    encoders), the band here is one-sided (causal).

    Args:
        mask: additive causal mask ``[B, 1, Lq, Lk]`` (0 allowed, -inf masked);
            ``Lk`` is the cache length.
        query_cache_coords: ``[B, Lq]`` cache coordinate of each query row
            (column index the row's token occupies / will occupy in the cache).
        sliding_window: window size (number of keys, exclusive lower bound).

    Returns a new mask with the base padding/causality preserved plus -inf on
    every key outside ``(q - sliding_window, q]``. Same device/dtype as ``mask``.

    The band is computed on **CPU** (integer comparisons + a ``bool`` mask),
    then added to ``mask`` **on CPU**, and the combined mask is moved back to
    ``mask``'s original device. The comparisons must not run on Spyre: its
    Inductor backend rejects ``int64`` compare-to-constant and ``bool``
    intermediates. The *add* is also kept off-device because an on-device
    ``-inf + -inf`` has been observed to produce NaN on Spyre in bf16 (see the
    note at the return). Mirrors ``add_sliding_window_band``.
    """
    lk = mask.shape[-1]
    k_col = torch.arange(lk)[None, None, :]  # [1, 1, Lk] on CPU
    q_coord = query_cache_coords.to("cpu")[:, :, None].to(k_col.dtype)  # [B, Lq, 1]
    delta = q_coord - k_col  # [B, Lq, Lk]
    out_of_band = (delta < 0) | (delta >= sliding_window)  # CPU bool
    band = torch.zeros(out_of_band.shape, dtype=mask.dtype)  # CPU float
    band = band.masked_fill(out_of_band, -torch.inf)
    # Combine on CPU, then move the result to the input's device. Doing the
    # add on-device has been observed to NaN on Spyre in bf16 when the band's
    # -inf lands on an already -inf cell (-inf + -inf): the result poisons the
    # SDPA softmax, giving all-NaN Gemma 4 logits. fp16 happened not to hit this.
    # A finite sentinel (e.g. finfo.min) is not a reliable substitute here — it
    # can overflow once cells are summed. Combining on CPU avoids the issue; the
    # mask is tiny so the round-trip is cheap.
    orig_device = mask.device
    combined = mask.to("cpu") + band[:, None, :, :]
    return combined.to(orig_device)


# ---------------------------------------------------------------------------
# KV-cache allocation
# ---------------------------------------------------------------------------


def kv_cache_shapes(model):
    """Resolve the per-layer ``(num_kv_heads, head_dim, v_head_dim)`` KV shapes.

    Most models use one uniform shape across all layers, derived from
    ``num_key_value_heads`` and ``head_dim`` (with optional ``_spyre_head_dim`` /
    ``_spyre_v_head_dim`` overrides from head padding). Models whose layers
    differ — e.g. Gemma 4, where global ("full_attention") layers use a larger
    ``global_head_dim`` and a different KV-head count than the sliding layers —
    set ``model._spyre_kv_shapes`` to an explicit per-layer list. When present,
    that list wins and this returns it verbatim.

    Returns a list of length ``num_hidden_layers`` of
    ``(num_kv_heads, head_dim, v_head_dim)`` tuples.
    """
    explicit = getattr(model, "_spyre_kv_shapes", None)
    if explicit is not None:
        return list(explicit)

    cfg = text_config(model.config)
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(cfg, "head_dim", None)
        or cfg.hidden_size // cfg.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    return [(num_kv_heads, head_dim, v_head_dim) for _ in range(num_layers)]


def allocate_kv_caches(model, batch_size, max_cache_len, dtype, device=None):
    """Allocate zeroed per-layer key/value caches matching the model's shapes.

    Honors ``model._spyre_kv_shapes`` (see ``kv_cache_shapes``) so models with
    heterogeneous layer shapes get correctly-sized caches per layer. Returns
    ``(key_caches, value_caches)`` lists. ``device`` defaults to the module
    ``DEVICE`` resolved at call time (so the conftest CPU patch applies).
    """
    if device is None:
        device = DEVICE
    shapes = kv_cache_shapes(model)
    key_caches = [
        torch.zeros(batch_size, n_kv, max_cache_len, hd, dtype=dtype, device=device)
        for (n_kv, hd, _vhd) in shapes
    ]
    value_caches = [
        torch.zeros(batch_size, n_kv, max_cache_len, vhd, dtype=dtype, device=device)
        for (n_kv, _hd, vhd) in shapes
    ]
    return key_caches, value_caches


# ---------------------------------------------------------------------------
# Model-agnostic load + generate
# ---------------------------------------------------------------------------


def _patch_torch_empty():
    """Workaround for torch_spyre spyre_empty() not accepting size= kwarg.

    Upstream fix: https://github.com/torch-spyre/torch-spyre/issues/1729
    """
    _orig = torch.empty

    def _patched(*args, size=None, **kwargs):
        if size is not None:
            return _orig(size, **kwargs)
        return _orig(*args, **kwargs)

    if getattr(torch.empty, "_hf_adapters_patched", False):
        return
    torch.empty = _patched
    torch.empty._hf_adapters_patched = True


def _embedding_param_ids(model):
    """Data-pointers of weights that must keep the default (column-major) layout.

    Gather-only embedding weights (used via nn.Embedding, not matmul) must not
    receive a row-major SpyreTensorLayout. Returns the set of ``data_ptr()``
    values for all such weights.

    Covers:
    - Decoder-style backbones: ``backbone.embed_tokens``.
    - BERT-style backbones: ``backbone.embeddings.{word,position,token_type}_embeddings``.
    - GPT-2-style backbones: ``backbone.{wte,wpe}`` (token + learned-position
      tables — both gathered, never matmul'd).
    - GPT-NeoX backbones: ``backbone.embed_in`` (token gather table).
    """
    ids = set()
    backbone = get_backbone(model)

    # Decoder-style: single embed_tokens; GPT-NeoX: embed_in
    for name in ("embed_tokens", "embed_in"):
        embed = getattr(backbone, name, None)
        if embed is not None and hasattr(embed, "weight"):
            ids.add(embed.weight.data_ptr())

    # Encoder-style: embeddings submodule with multiple gather tables
    embeddings = getattr(backbone, "embeddings", None)
    if embeddings is not None:
        for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
            sub = getattr(embeddings, name, None)
            if sub is not None and hasattr(sub, "weight") and sub.weight.dim() == 2:
                ids.add(sub.weight.data_ptr())

    # GPT-2-style: word (wte) + learned-position (wpe) gather tables
    for name in ("wte", "wpe"):
        sub = getattr(backbone, name, None)
        if sub is not None and hasattr(sub, "weight") and sub.weight.dim() == 2:
            ids.add(sub.weight.data_ptr())

    return ids


def untie_embedding_and_lm_head(model):
    """If the token-embedding weight and the LM head weight share storage, clone
    the LM head's weight so each can take a different Spyre layout.
    """
    head = _get_lm_head(model)
    if head is None:
        return
    backbone = get_backbone(model)
    embed = (
        getattr(backbone, "embed_tokens", None)
        or getattr(backbone, "wte", None)
        or getattr(backbone, "embed_in", None)
    )
    if embed is None:
        return
    if embed.weight.data_ptr() == head.weight.data_ptr():
        head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Infer the floating-point dtype of a prepared model from its parameters.

    Used by KV-cache and mask allocators so they match the model dtype
    instead of hardcoding fp16. Falls back to fp16 for empty models.
    """
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16


def _move_to_spyre_with_layout(model, dtype):
    """Move all parameters and buffers to Spyre with row-major layout for 2D
    matmul weights, except embedding weights which keep the default layout.
    """
    # Propagate dtype to the precomputed RoPE module(s) so the freq cache
    # matches the chosen weight dtype (avoids fp16/bf16 mismatch in
    # apply_rope_matmul when dtype != fp16). Done before the CPU early-return so
    # both the CPU and Spyre paths get it.
    set_rope_dtype(model, dtype)

    # Build the RoPE rotation cache on CPU now, before any device move. If left
    # to its lazy first-forward build, the construction ops run inside the Spyre
    # graph and corrupt the result (see prebuild_rope_cache). Harmless on CPU.
    prebuild_rope_cache(model)

    if torch.device(DEVICE).type != "spyre":
        model.to(dtype=dtype)
        return

    # Prime torch-spyre autoload before importing torch_spyre._C or calling
    # torch.empty(..., device_layout=...). Calls with the spyre-only
    # device_layout kwarg fail kwarg validation before dispatch.
    torch.empty(1, device=DEVICE)

    from torch_spyre._C import SpyreTensorLayout  # type: ignore[import-not-found]

    skip_layout_ptrs = _embedding_param_ids(model)

    def _alloc_on_spyre(t: torch.Tensor) -> torch.Tensor:
        # The row-major [1, 0] dim_order describes a 2-D permutation, so it only
        # applies to 2-D matmul weights. 1-D tensors (norms, biases) and any
        # higher-rank weight (e.g. the 3-D/4-D Conv2d and position-embedding
        # tables in a multimodal checkpoint's vision/audio towers) keep the
        # default layout — forcing [1, 0] on them raises "Incompatible host_size
        # and dim_order". Embedding tables are gather-only and also skipped.
        if t.dim() == 2 and t.data_ptr() not in skip_layout_ptrs:
            stl = SpyreTensorLayout(t.shape, t.stride(), dtype, [1, 0])
        else:
            stl = None
        new: torch.Tensor = torch.empty(  # type: ignore[call-overload]
            t.shape,
            device=torch.device(DEVICE),
            device_layout=stl,
            dtype=dtype,
        )
        new.copy_(t.to(dtype))
        return new

    for name, param in list(model.named_parameters()):
        new = _alloc_on_spyre(param.data)
        module_path, _, attr = name.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        setattr(owner, attr, nn.Parameter(new, requires_grad=False))

    for name, buf in list(model.named_buffers()):
        new = _alloc_on_spyre(buf)
        module_path, _, attr = name.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        persistent = attr not in owner._non_persistent_buffers_set
        owner.register_buffer(attr, new, persistent=persistent)


def load_model_common(model_path, module, dtype=torch.float16, auto_model_cls=None):
    """Load an HF model.

    Args:
        model_path: HF model path or local directory.
        module: HF model module
        dtype: Weight dtype (default fp16).
        auto_model_cls: HF auto-model class to use (e.g. ``AutoModel``,
            ``AutoModelForCausalLM``). Defaults to ``AutoModel``.
    """
    if auto_model_cls is None:
        from transformers import AutoModel

        auto_model_cls = AutoModel

    if hasattr(module, "load_hf_model"):
        model = module.load_hf_model(model_path, dtype)
    else:
        model = auto_model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cpu",
        )

    model.eval()
    model.requires_grad_(False)
    return model


def move_model_to_spyre(model, module, dtype: torch.dtype) -> None:
    untie_embedding_and_lm_head(model)
    module.prepare_for_spyre(model)
    # print("Moving model to Spyre ...")
    _patch_torch_empty()
    _move_to_spyre_with_layout(model, dtype)
    for submod_name in getattr(model, "_spyre_cpu_submodules", []):
        model.get_submodule(submod_name).to("cpu")
    print("Model on Spyre ready.")


# ---------------------------------------------------------------------------
# Generation-parameter resolution
# ---------------------------------------------------------------------------

# Sentinel distinguishing "argument not passed" from an explicit ``None``.
_UNSET = object()


def _normalize_eos_ids(eos):
    """Normalize an EOS spec to a 1-D long tensor of ids, or ``None``.

    Accepts a scalar ``int`` (→ ``[int]``), a list/tuple of ints, an existing
    tensor, or ``None``. Multi-EOS models (Llama-3, Phi-4, Qwen) carry a list;
    older models carry a scalar. Returning a tensor lets the decode loop use
    ``torch.isin`` uniformly instead of ``==`` (which silently collapses to a
    scalar ``bool`` for lists).

    Mirrors stock HF, which does the same scalar/list/tensor → tensor
    normalization and ``torch.isin`` stop check in
    ``transformers.generation.stopping_criteria.EosTokenCriteria``.
    """
    if eos is None:
        return None
    if not isinstance(eos, torch.Tensor):
        if isinstance(eos, int):
            eos = [eos]
        eos = torch.tensor(eos)
    return eos


def _resolve_generation_params(model, tokenizer, overrides):
    """Resolve sampling + stop params via HF's ``_prepare_generation_config``.

    Precedence matches stock HF: ``explicit kwarg > model.generation_config >
    HF global defaults``. Parameters with ``None`` are dropped so HF
    fills them. EOS is normalized to a tensor.

    Returns a dict with keys ``do_sample, temperature, top_k, top_p`` plus
    ``eos_ids`` (a long tensor or ``None``).
    """
    eos_specified = "eos_token_id" in overrides
    explicit = {
        k: v for k, v in overrides.items() if k == "eos_token_id" or v is not None
    }
    cfg, _ = model._prepare_generation_config(None, **explicit)

    eos = cfg.eos_token_id
    # Fall back to the tokenizer only when EOS was unspecified — an explicit
    # eos_token_id=None means "disable EOS" and must not be re-enabled.
    if eos is None and not eos_specified:
        eos = getattr(tokenizer, "eos_token_id", None)

    return {
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "eos_ids": _normalize_eos_ids(eos),
    }


def pad_and_position(input_ids, actual_lengths):
    """Left block-pad ``input_ids`` to a BLOCK_SIZE multiple and build positions.

    Returns ``(padded_ids, padded_len, prompt_offsets, position_ids)``. Real
    tokens are right-aligned: each row's positions ``0..actual_len-1`` sit at
    padded indices ``prompt_offsets[b]..padded_len-1`` (the decode loop's
    convention). Shared by ``generate`` and the VLM adapters' prefill arms.
    """
    batch_size, prompt_length = input_ids.shape
    padded_len = math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
    block_pad = padded_len - prompt_length
    if block_pad > 0:
        pad = input_ids.new_zeros((batch_size, block_pad))
        input_ids = torch.cat([pad, input_ids], dim=1)

    prompt_offsets = padded_len - actual_lengths  # [B]
    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    for b in range(batch_size):
        n = actual_lengths[b].item()
        position_ids[b, prompt_offsets[b].item() :] = torch.arange(n)
    return input_ids, padded_len, prompt_offsets, position_ids


def select_next_token(next_logits, do_sample, temperature, top_k, top_p):
    """CPU token selection: greedy argmax, or temperature/top-k/top-p sampling.

    The top-p path mirrors HF's ``TopPLogitsWarper.__call__``.
    """
    if not do_sample:
        return torch.argmax(next_logits, dim=-1)  # [B]
    scaled = next_logits / temperature
    if top_k and top_k > 0:
        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1)
        scaled[scaled < v[:, -1:]] = -torch.inf
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -1:] = 0  # keep at least one token
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scaled = scaled.masked_fill(indices_to_remove, -torch.inf)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]


def decode_block_walk(result, num_generated, padded_len, eos_ids, tokenizer):
    """Per-sequence block-walk over generated slots → EOS-trimmed decoded strings.

    Generated tokens are contiguous within each BLOCK_SIZE block but blocks may
    be separated by unused slots; all sequences share the same block layout
    starting at ``padded_len``. Walks each sequence using its own
    ``num_generated`` count, trims at the first EOS, and decodes.
    """
    results = []
    for b in range(result.shape[0]):
        gen_ids_list = []
        block_start = padded_len
        remaining = num_generated[b].item()
        while remaining > 0:
            take = min(remaining, BLOCK_SIZE)
            for j in range(take):
                gen_ids_list.append(result[b, block_start + j].item())
            remaining -= take
            block_start += BLOCK_SIZE
        gen_ids = torch.tensor(gen_ids_list)
        if eos_ids is not None:
            eos_pos = torch.isin(gen_ids, eos_ids).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0].item()]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return results


def generate(
    run_forward_fn: Callable,
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    do_sample=None,
    temperature=None,
    top_k=None,
    top_p=None,
    eos_token_id=_UNSET,
    timing=False,
):
    """Model-agnostic generation with padded 64-block decode.

    When attached to a model via ``auto_spyre_model.py`` (which binds
    ``run_forward_fn`` to the adapter module's ``_run_forward``), the
    ``run_forward_fn`` parameter drops out of the public signature, so callers
    invoke it as::

        model.generate(tokenizer, ["Hello!"], max_new_tokens=32, **kwargs)

    Sampling and stop parameters follow stock-HF precedence:
    ``explicit kwarg > model.generation_config > HF global default``. Leaving a
    sampling knob at ``None`` (the default for ``do_sample``/``temperature``/
    ``top_k``/``top_p``) means "not specified" and defers to the model's
    ``generation_config``, then to HF defaults — so this matches
    ``model.generate()``. Pass a concrete value to force it regardless of
    config (e.g. ``do_sample=False`` for deterministic greedy on a model whose
    config bakes in sampling).

    ``max_new_tokens`` is REQUIRED and is not resolved from config: HF's
    default length goes through ``max_length`` (total prompt+new), which this
    decode loop does not implement. Callers must state the new-token budget.

    Args:
        run_forward_fn: ``fn(model, input_ids, position_ids, attn_mask,
            key_caches, value_caches, is_filling, token_index,
            cache_position) -> logits``
        model: Prepared HF model on Spyre (supplies ``generation_config``).
        tokenizer: HF tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Number of tokens to generate (required).
        do_sample: Sampling vs greedy.
        temperature: Sampling temperature.
        top_k: Top-k filtering (0/None disables).
        top_p: Nucleus (top-p) filtering (1.0 disables).
        eos_token_id: Override stop token(s); scalar or list. Omit to defer to
            config/tokenizer eos; pass ``None`` to disable EOS stopping (matches
            stock ``generate()``).
        timing: Print per-token latency.
    """
    overrides = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    # Include eos_token_id only when actually overridden, so an explicit None
    # (disable EOS) is distinguishable from "unspecified" (defer to config).
    if eos_token_id is not _UNSET:
        overrides["eos_token_id"] = eos_token_id
    params = _resolve_generation_params(model, tokenizer, overrides)
    do_sample = params["do_sample"]
    temperature = params["temperature"]
    top_k = params["top_k"]
    top_p = params["top_p"]
    eos_ids = params["eos_ids"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Force left-padding: with right-padding, shorter sequences end with
    # padding tokens, and logits[:, -1, :] would predict from a pad position.
    # Left-padding aligns all sequences to end at the same position.
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]

    # Per-sequence actual prompt length (excluding tokenizer left-padding)
    actual_prompt_lengths = attention_mask.sum(dim=1)  # [B]

    # Block-pad to a BLOCK_SIZE multiple; real tokens right-aligned (positions
    # 0..actual_len-1 at padded indices prompt_offsets[b]..padded_len-1).
    max_cache_len = (
        math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
        + math.ceil(max_new_tokens / BLOCK_SIZE) * BLOCK_SIZE
    )
    input_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
        input_ids, actual_prompt_lengths
    )

    # Initialize empty KV caches. Per-layer shapes come from the model
    # (``_spyre_kv_shapes``) for heterogeneous architectures like Gemma 4,
    # otherwise a single uniform shape derived from the config.
    # Match KV cache and mask dtype to the model's weight dtype.
    model_d_type = get_model_dtype(model)
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, model_d_type
    )

    # Decode state
    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None

    times_list = []
    finished = torch.zeros(batch_size, dtype=torch.bool)
    num_generated = torch.zeros(batch_size, dtype=torch.long)

    for i in range(max_new_tokens):
        t0 = time.time()

        if i == 0:
            # --- PREFILL ---
            prefill_mask = build_prefill_mask(
                batch_size,
                padded_len,
                max_cache_len,
                prompt_offsets,
                dtype=model_d_type,
            )
            logits = run_forward_fn(
                model,
                input_ids.to(DEVICE),
                position_ids.to(DEVICE),
                prefill_mask.to(DEVICE),
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=0,
            )
            logits_cpu = logits.to("cpu")
            next_logits = logits_cpu[:, -1, :]
            current_cache_len = padded_len
            # Initialize per-sequence decode_pos. After the first +BLOCK_SIZE
            # increment (before the first expansion forward), position 0 of
            # the block must equal actual_prompt_lengths[b] for that row.
            # So initial[b, j] = actual_prompt_lengths[b] + j - BLOCK_SIZE.
            decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
            for b in range(batch_size):
                actual_len = actual_prompt_lengths[b].item()
                for j in range(BLOCK_SIZE):
                    decode_pos[b, j] = actual_len + j - BLOCK_SIZE

        else:
            is_filling = tokens_in_block > 0
            next_input = result[:, -BLOCK_SIZE:].to(DEVICE)

            if is_filling:
                fill_pos = current_cache_len - BLOCK_SIZE + tokens_in_block
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),  # type: ignore[union-attr]
                    fill_mask_device,
                    key_caches,
                    value_caches,
                    is_filling=True,
                    token_index=tokens_in_block,
                    cache_position=fill_pos,
                )
                logits_cpu = logits.to("cpu")
                grab_idx = BLOCK_SIZE - tokens_in_block
                next_logits = logits_cpu[:, -grab_idx, :]

            else:
                current_cache_len += BLOCK_SIZE
                decode_pos = decode_pos + BLOCK_SIZE  # type: ignore[assignment, operator]
                exp_mask = build_expansion_mask(
                    batch_size,
                    BLOCK_SIZE,
                    max_cache_len,
                    current_cache_len,
                    prompt_offsets,
                    dtype=model_d_type,
                )
                logits = run_forward_fn(
                    model,
                    next_input,
                    decode_pos.to(DEVICE),  # type: ignore[union-attr]
                    exp_mask.to(DEVICE),
                    key_caches,
                    value_caches,
                    is_filling=False,
                    token_index=0,
                    cache_position=current_cache_len - BLOCK_SIZE,
                )
                logits_cpu = logits.to("cpu")
                next_logits = logits_cpu[:, -BLOCK_SIZE, :]
                fill_mask_device = exp_mask.to(DEVICE)

        # Token selection (CPU)
        next_tokens = select_next_token(
            next_logits, do_sample, temperature, top_k, top_p
        )

        if timing:
            times_list.append(time.time() - t0)

        # Place token in result (FMS logic)
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        if tokens_in_block == 0:
            # Just finished a block, pad for next block
            result = F.pad(result, (0, BLOCK_SIZE))
        grab_idx = (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        result[:, -grab_idx] = next_tokens  # [B]
        if eos_ids is not None:
            finished |= torch.isin(next_tokens, eos_ids)
        num_generated += (~finished).long()

        if finished.all():
            break

    # Timing
    if timing and times_list:
        print(f"\nFirst-token latency: {times_list[0]*1000:.3f} ms")
        if len(times_list) > 1:
            avg = sum(times_list[1:]) / len(times_list[1:])
            print(f"Avg next-token latency: {avg*1000:.3f} ms")
        print("Per-token: " + ", ".join(f"{t*1000:.1f}" for t in times_list) + " ms")

    return decode_block_walk(result, num_generated, padded_len, eos_ids, tokenizer)


# ---------------------------------------------------------------------------
# Standard GQA adapter helpers
# ---------------------------------------------------------------------------


def make_standard_gqa_block(layer):
    """Compiled block for standard GQA models (separate QKV, no multipliers).

    Shared by Llama, Qwen2, Mistral, and other standard GQA adapters.
    """
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)

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
        v = attn.v_proj(h).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

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


def make_decoder_block(
    *,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    attn_ln,
    ffn_in,
    act,
    ffn_out,
    ffn_ln,
    num_heads,
    head_dim,
    scale,
    pre_ln=True,
):
    """Compiled causal-decoder block for non-RoPE (learned-abs-pos) models.

    Shared by the non-RoPE decoder family — ``hf_gpt2`` today, OPT/BLOOM/MPT to
    come. These all have the standard-GQA block *shape* (pre/post-LN, KV cache,
    causal SDPA, residual + FFN tail) but differ from ``make_standard_gqa_block``
    in three ways, and from each other only in *where* their modules live — so,
    like ``make_encoder_block``, adapters resolve their own module layout and
    pass it in by keyword.

    Differences from ``make_standard_gqa_block``:
      - **no RoPE** — positions are learned absolute (added to the token
        embeddings in the backbone), so ``selected_freqs`` is accepted for
        signature parity with the generate/test harness and ignored;
      - **MHA** — kv heads == attention heads, so no ``enable_gqa=True``;
      - **explicit ``scale``** and a configurable ``pre_ln`` LN placement
        (``True`` = norm before each sublayer, as in GPT-2 / BLOOM / OPT≥1.3B;
        ``False`` = norm after, as in OPT-350m).

    Block signature matches the decoder harness::

        block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache, is_filling, token_index,
                      cache_position) -> (h, key_cache, value_cache)

    Dropout is skipped — these adapters are eval-only. ``act`` is the (possibly
    patched) activation module; the FFN is passed decomposed as
    ``ffn_out(act(ffn_in(x)))`` rather than as a single module, since some
    models (OPT) have no wrapping MLP module.
    """

    def block_forward(
        hidden_states,
        selected_freqs,  # unused — non-RoPE; kept for signature parity
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
    ):
        # --- attention sublayer ---
        residual = hidden_states
        h = attn_ln(hidden_states) if pre_ln else hidden_states

        bsz, seq_len, _ = h.shape
        q = q_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

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
            scale=scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)

        h = residual + attn_out
        if not pre_ln:
            h = attn_ln(h)

        # --- FFN sublayer ---
        residual = h
        f = ffn_ln(h) if pre_ln else h
        f = ffn_out(act(ffn_in(f)))
        h = residual + f
        if not pre_ln:
            h = ffn_ln(h)

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def standard_gqa_backbone_forward(
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
    """Standard GQA backbone: embedding, RoPE, compiled blocks, norm.

    Returns ``last_hidden_state`` (no ``lm_head``). Used directly by embedding
    callers; wrapped by ``standard_gqa_forward`` for causal-LM callers.
    """
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)

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

    h = backbone.norm(h)
    return h


def standard_gqa_forward(
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
    """Standard GQA causal-LM forward: backbone + LM head."""
    h = standard_gqa_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )
    return model.lm_head(h)


# ---------------------------------------------------------------------------
# Encoder-only forward (BERT-family: no RoPE, no KV cache)
# ---------------------------------------------------------------------------


def fairseq_position_ids(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """fairseq-style positions: real tokens start at ``padding_idx + 1``.

    Mirrors ``create_position_ids_from_input_ids`` in modeling_xlm_roberta /
    modeling_mpnet so the position_embeddings lookup matches stock HF exactly.
    Padding slots map to ``padding_idx``; the attention mask zeros them out
    later, so the embedding picked there is irrelevant.

    Computed on CPU even when ``input_ids`` lives on Spyre: the natural form
    ``input_ids.ne(padding_idx).int()`` materializes a bool tensor and the
    Spyre Inductor backend rejects ``bool → int32`` conversions. The CPU
    round-trip on a ``[B, L]`` int tensor is negligible.
    """
    ids_cpu = input_ids.to("cpu")
    mask = ids_cpu.ne(padding_idx).int()
    incremental = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return (incremental.long() + padding_idx).to(DEVICE)


def make_encoder_block(
    *,
    attn_module,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    attn_ln,
    ffn_in,
    act,
    ffn_out,
    out_ln,
    num_heads,
    head_dim,
):
    """Compile a bidirectional encoder block (MHA + post-LN + FFN + post-LN).

    Shared by ``hf_bert`` and ``hf_mpnet``. The two architectures differ only
    in *where* their projection / LN modules live (e.g. BERT's
    ``attention.self.query`` vs MPNet's ``attention.attn.q``); the compiled
    forward body is identical, so adapters resolve the modules and pass them
    in by keyword.

    Block signature (no KV cache, no RoPE):

        block_forward(hidden_states, attn_mask) -> hidden_states

    ``attn_module`` is the model's attention submodule (``attention.self`` for
    BERT-style, ``attention.attn`` for MPNet-style). It's read for
    ``_spyre_orig_head_dim`` — the marker ``pad_attention_heads_simple`` sets
    when it pads. When present, SDPA scales by ``1/sqrt(orig)`` instead of the
    default ``1/sqrt(padded)`` so the unpadded entries get the correct scale.

    Dropout is skipped — these adapters are eval-only.
    """
    sdpa_scale = getattr(attn_module, "_spyre_orig_head_dim", head_dim) ** -0.5

    def block_forward(hidden_states, attn_mask):
        bsz, seq_len, _ = hidden_states.shape

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

        attn_out = o_proj(attn_out)
        hidden_states = attn_ln(attn_out + hidden_states)

        ffn_h = act(ffn_in(hidden_states))
        ffn_h = ffn_out(ffn_h)
        hidden_states = out_ln(ffn_h + hidden_states)

        return hidden_states

    return torch.compile(block_forward, dynamic=False)


def make_vision_encoder_block(
    *,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    layer_norm1,
    layer_norm2,
    ffn_in,
    act,
    ffn_out,
    num_heads,
    head_dim,
    scale,
):
    """Compile a **pre-LN** bidirectional vision-transformer block (SigLIP/CLIP).

    Counterpart of ``make_encoder_block`` (which is post-LN, BERT-style). A
    SigLIP/CLIP encoder layer normalizes *before* each sub-block and adds the
    residual *after*:

        h = h + o_proj(SDPA(qkv(layer_norm1(h))))
        h = h + ffn_out(act(ffn_in(layer_norm2(h))))

    There is no attention mask and no KV cache — every patch attends to every
    patch (full bidirectional attention over a fixed-length patch sequence).

    Args:
        q_proj/k_proj/v_proj/o_proj: per-head attention projections
            (``self_attn.{q,k,v,out}_proj``). Already head-padded if needed.
        layer_norm1: pre-attention LayerNorm. layer_norm2: pre-MLP LayerNorm.
        ffn_in/act/ffn_out: MLP (``fc1`` → activation → ``fc2``).
        num_heads: attention head count. head_dim: per-head dim (post-padding).
        scale: SDPA scale (``1/sqrt(orig_head_dim)`` — pass the *original* dim
            when heads were zero-padded so padded entries don't change the
            softmax temperature).
    """

    def block_forward(hidden_states):
        bsz, seq_len, _ = hidden_states.shape

        residual = hidden_states
        h = layer_norm1(hidden_states)
        q = q_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        h = layer_norm2(hidden_states)
        h = ffn_out(act(ffn_in(h)))
        hidden_states = residual + h

        return hidden_states

    return torch.compile(block_forward, dynamic=False)


def encoder_backbone_forward(model, input_ids, attn_mask, position_ids, token_type_ids):
    """Encoder backbone forward: embedding table + LN + compiled encoder blocks.

    Used by BERT-style (no RoPE, no KV cache) encoder-only models. Counterpart
    of ``standard_gqa_backbone_forward`` for decoder models.

    Args:
        model: Prepared BertModel (or equivalent) on Spyre. Must have
            ``model._spyre_compiled_blocks`` set by ``prepare_for_spyre``.
        input_ids: ``[B, padded_len]`` token ids.
        attn_mask: ``[B, 1, padded_len, padded_len]`` additive fp16 mask built
            with ``is_causal=False`` (zeros for real-token pairs, -inf elsewhere).
        position_ids: ``[B, padded_len]`` position indices (0..actual_len-1 for
            real tokens, 0 for pad slots).
        token_type_ids: ``[B, padded_len]`` long tensor (all zeros for
            single-sentence embedding workloads).

    Returns:
        ``last_hidden_state`` ``[B, padded_len, H]``.

    Note: ``nn.LayerNorm`` runs as-is inside the compiled block on CPU. On
    Spyre the compiler should lower it via the ``spyre::layer_norm`` op; if not,
    a per-instance wrapper will be needed in a follow-up.
    """
    backbone = get_backbone(model)
    emb = backbone.embeddings
    h = (
        emb.word_embeddings(input_ids)
        + emb.position_embeddings(position_ids)
        + emb.token_type_embeddings(token_type_ids)
    )
    h = emb.LayerNorm(h)
    # Spyre layout workaround: BERT post-LN ends each block on a broadcast
    # against a 1D weight/bias. Spyre tensors produced this way read
    # correctly via ``.to("cpu")`` but are mis-read by subsequent on-device
    # ops, so the next compiled block's matmul sees garbage. ``.clone()``
    # in eager Python (outside torch.compile) allocates a fresh
    # canonical-layout tensor and copies through, fixing the handoff.
    h = h.clone() if h.device.type == "spyre" else h
    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h, attn_mask)
        if h.device.type == "spyre":
            h = h.clone()
    return h


def prepare_rope_and_heads(model):
    cfg = text_config(model.config)
    assert_spyre_dimensions(
        cfg, model_name=getattr(cfg, "name_or_path", "") or "<unknown>"
    )
    orig_head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )

    # RoPE reshape [B,L,H,2,D/2] requires D/2 >= BLOCK_SIZE.
    # Compute minimum stick-aligned head_dim: round up to next multiple of 2*BLOCK_SIZE.
    padded_head_dim = None
    stick_aligned_head_dim = (
        (orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)
    ) * (2 * BLOCK_SIZE)
    if stick_aligned_head_dim > orig_head_dim:
        padded_head_dim = stick_aligned_head_dim
        pad_attention_heads(
            model,
            get_backbone(model).layers,
            orig_head_dim,
            padded_head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )

    model._spyre_rope = PrecomputedRotaryEmbedding(
        get_backbone(model).rotary_emb,
        padded_head_dim=padded_head_dim,
    )


def prepare_standard_gqa(model, rmsnorm_cls):
    """Apply Spyre adaptations for standard GQA models in-place.

    Args:
        model: HF model (on CPU, eval mode, requires_grad=False).
        rmsnorm_cls: The model's RMSNorm class to patch.
    """
    prepare_rope_and_heads(model)
    patch_rmsnorm(rmsnorm_cls)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        make_standard_gqa_block(layer) for layer in get_backbone(model).layers
    ]


# ---------------------------------------------------------------------------
# Embedding-path prefill driver
# ---------------------------------------------------------------------------


def prefill_embed(
    run_backbone_forward_fn: Callable,
    model,
    input_ids,
    attention_mask,
):
    """One-shot prefill that returns ``last_hidden_state``.

    Counterpart of ``generate``'s prefill arm, specialized for embedding
    workloads. Differences:

    - **Right-padded** instead of left-padded. ST tokenizes with right-pad
      and pooling layers depend on ``attention_mask`` to find real tokens;
      flipping to left-pad would silently break pooling.
    - **No decode follow-up.** Caches are sized to ``padded_len`` only and
      thrown away after this call. The compiled block still writes to them
      (we can't change its signature), but we don't reuse them.
    - **No ``lm_head``.** Returns ``[B, L, H]`` hidden states directly,
      so callers (pooling, normalize) operate on the backbone output.

    Device split: all the auxiliary tensors this driver derives from the inputs
    (padding, ``position_ids``, ``attention_mask``, throwaway KV caches) are
    built on **CPU**, then moved to Spyre right before the backbone call. Only
    the backbone matmuls run on Spyre; the returned hidden state is on Spyre.

    Args:
        run_backbone_forward_fn: ``fn(model, input_ids, position_ids,
            attn_mask, key_caches, value_caches, is_filling, token_index,
            cache_position) -> [B, padded_len, H]``. Pass the adapter's
            ``_run_backbone_forward`` (or ``standard_gqa_backbone_forward``).
        model: Prepared HF backbone on Spyre (loaded via ``AutoModel``).
        input_ids: ``[B, L]`` token ids on CPU. Right-padded to ``L`` by
            the tokenizer; this function further pads to a ``BLOCK_SIZE``
            multiple.
        attention_mask: ``[B, L]`` mask on CPU; ``1`` for real tokens,
            ``0`` for tokenizer pads.

    Returns:
        ``last_hidden_state``: ``[B, L, H]`` on Spyre, cropped back to the input
        ``L``. Caller moves to CPU as needed.
    """
    bsz, seq_len = input_ids.shape

    # Pad to BLOCK_SIZE multiple on the right
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    pad_amount = padded_len - seq_len
    if pad_amount > 0:
        input_ids = F.pad(input_ids, (0, pad_amount), value=0)

    # Per-sequence real-token count (excludes both tokenizer pad and block pad)
    actual_lengths = attention_mask.sum(dim=1)  # [B]

    # Position ids: real tokens get 0..actual_len-1, pads get 0 (masked out
    # in attention so the rotation applied at those positions doesn't matter)
    position_ids = torch.zeros((bsz, padded_len), dtype=torch.long)
    for b in range(bsz):
        actual = actual_lengths[b].item()
        position_ids[b, :actual] = torch.arange(actual)

    # Match KV cache and mask dtype to the model's weight dtype so the
    # compiled block sees a consistent dtype across q/k/v_proj outputs and
    # the SDPA inputs.
    model_d_type = get_model_dtype(model)

    # Causal right-padded mask, or bidirectional for some embedders
    is_causal = getattr(model.config, "is_causal", True) and not getattr(
        model.config, "use_bidirectional_attention", False
    )
    mask = build_prefill_mask_right_padded(
        bsz,
        padded_len,
        actual_lengths,
        is_causal=is_causal,
        dtype=model_d_type,
    )

    # Throwaway KV caches sized to padded_len (no decode budget)
    cfg = text_config(model.config)
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(cfg, "head_dim", None)
        or cfg.hidden_size // cfg.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    key_caches = [
        torch.zeros(
            bsz,
            num_kv_heads,
            padded_len,
            head_dim,
            dtype=model_d_type,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.zeros(
            bsz,
            num_kv_heads,
            padded_len,
            v_head_dim,
            dtype=model_d_type,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]

    h = run_backbone_forward_fn(
        model,
        input_ids.to(DEVICE),
        position_ids.to(DEVICE),
        mask.to(DEVICE),
        key_caches,
        value_caches,
        is_filling=False,
        token_index=0,
        cache_position=0,
    )

    # Crop the block-pad back off; tokenizer pad stays so pooling can mask it.
    # h is on Spyre here.
    h = h[:, :seq_len, :]
    return h


# ---------------------------------------------------------------------------
# Encoder-only prefill driver (BERT-family)
# ---------------------------------------------------------------------------


def prefill_encoder(
    run_encoder_forward_fn: Callable,
    model,
    input_ids,
    attention_mask,
    token_type_ids=None,
):
    """One-shot prefill for encoder-only (BERT-style) embedding models.

    Counterpart of ``prefill_embed`` for models with no KV cache and no RoPE.
    (``prefill_embed`` reads `model.config.num_key_value_heads`` and
    allocates KV caches, which ``BertConfig`` and similar encoder configs do not provide.

    Device split: same as ``prefill_embed``. The auxiliary tensors
    (padding, ``position_ids``, ``token_type_ids``, ``attn_mask``) are built on
    **CPU**, then moved to Spyre right before the backbone call. Only the
    backbone matmuls run on Spyre; the returned hidden state is on Spyre.

    Args:
        run_encoder_forward_fn: ``fn(model, input_ids, attn_mask, position_ids,
            token_type_ids) -> [B, padded_len, H]``. Pass the adapter's
            ``_run_backbone_forward`` (i.e. ``encoder_backbone_forward``).
        model: Prepared encoder backbone on Spyre (loaded via ``AutoModel``).
        input_ids: ``[B, L]`` token ids on CPU. Right-padded by the tokenizer.
        attention_mask: ``[B, L]`` mask on CPU; 1 for real tokens, 0 for pad.
        token_type_ids: Optional ``[B, L]`` on CPU. Defaults to all-zeros when
            None (correct for single-sentence embedding workloads).

    Returns:
        ``last_hidden_state``: ``[B, L, H]`` on Spyre, cropped back to the input
        ``L``. Caller moves to CPU as needed.
    """
    bsz, seq_len = input_ids.shape

    # Pad to BLOCK_SIZE multiple on the right
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    pad_amount = padded_len - seq_len
    if pad_amount > 0:
        input_ids = F.pad(input_ids, (0, pad_amount), value=0)

    # Per-sequence real-token count
    actual_lengths = attention_mask.sum(dim=1)  # [B]

    # Position ids: 0..actual_len-1 for real tokens, 0 for pads
    position_ids = torch.zeros((bsz, padded_len), dtype=torch.long)
    for b in range(bsz):
        actual = actual_lengths[b].item()
        position_ids[b, :actual] = torch.arange(actual)

    # Token type ids: zero tensor if not provided
    if token_type_ids is None:
        tt_ids = torch.zeros((bsz, padded_len), dtype=torch.long)
    else:
        tt_pad = padded_len - token_type_ids.shape[1]
        tt_ids = (
            F.pad(token_type_ids, (0, tt_pad), value=0)
            if tt_pad > 0
            else token_type_ids
        )

    # Bidirectional mask: real tokens attend to all other real tokens
    mask = build_prefill_mask_right_padded(
        bsz, padded_len, actual_lengths, is_causal=False
    )

    h = run_encoder_forward_fn(
        model,
        input_ids.to(DEVICE),
        mask.to(DEVICE),
        position_ids.to(DEVICE),
        tt_ids.to(DEVICE),
    )

    # Crop the block-pad back off. h is on Spyre here.
    h = h[:, :seq_len, :]

    return h


# ---------------------------------------------------------------------------
# Cross-encoder reranker prefill driver (XLM-RoBERTa / BGE reranker family)
# ---------------------------------------------------------------------------


def prefill_reranker(
    run_encoder_forward_fn: Callable,
    model,
    input_ids,
    attention_mask,
    token_type_ids=None,
):
    """One-shot prefill for cross-encoder reranker models.

    Runs the encoder backbone on Spyre via ``prefill_encoder``,
    then applies the classification head (``model.classifier``) to produce a
    scalar relevance score per query-document pair.

    The classification head is run outside torch.compile on CPU to avoid:
    - ``torch.bernoulli`` (Dropout) which the Spyre backend cannot lower.
    - ``aten.slice`` for CLS extraction which does not lower on Spyre.
    - ``out_proj: Linear(hidden, 1)`` whose output dim=1 is not stick-aligned.

    Args:
        run_encoder_forward_fn: ``fn(model, input_ids, attn_mask, position_ids,
            token_type_ids) -> [B, padded_len, H]``.
        model: Prepared ``XLMRobertaForSequenceClassification`` on Spyre.
        input_ids: ``[B, L]`` token ids on CPU.
        attention_mask: ``[B, L]`` mask on CPU.
        token_type_ids: Optional ``[B, L]``. Defaults to all-zeros when None.

    Returns:
        ``scores``: ``[B]`` float32 tensor on CPU — raw logits.
    """
    last_hidden_state = prefill_encoder(
        run_encoder_forward_fn,
        model,
        input_ids,
        attention_mask,
        token_type_ids=token_type_ids,
    )

    # Pass the full [B, L, H] hidden state to the classifier; .to(cls_device)
    # moves it off Spyre to avoid aten.slice. The classification head does its
    # own [:, 0, :] CLS extraction internally.

    # Run the classification head on the same device it lives on
    # (CPU — kept off Spyre via _spyre_cpu_submodules in prepare_for_spyre).
    classifier = model.classifier
    cls_device = next(classifier.parameters()).device
    scores = classifier(last_hidden_state.to(cls_device))  # [B, 1]

    return scores[:, 0].to("cpu")  # [B] raw logits on CPU


# ---------------------------------------------------------------------------
# Vision-encoder prefill driver
# ---------------------------------------------------------------------------


def vision_backbone_forward(model, patch_embeds, output_hidden_states=False):
    """Vision-tower backbone forward: compiled pre-LN blocks + post-LN.

    The patch embedding (Conv2d + position-embedding add) is done *before* this
    call by ``prefill_vision`` — it produces ``patch_embeds`` ``[B, P, H]``,
    which this function feeds through the compiled encoder blocks and the
    tower's final LayerNorm.

    Mirrors ``encoder_backbone_forward``'s Spyre layout workaround: a pre-LN /
    LayerNorm-terminated block can leave a Spyre tensor in a layout the next
    block's matmul mis-reads, so we ``.clone()`` between on-device blocks to
    force a canonical-layout copy.

    Args:
        model: prepared vision tower; must have ``model._spyre_compiled_blocks``
            and ``model._spyre_post_layernorm`` set by ``prepare_for_spyre``.
        patch_embeds: ``[B, P, H]`` patch sequence (already on DEVICE).
        output_hidden_states: when True, also return the per-layer hidden states
            (HF convention: ``[patch_embeds, out_layer0, ..., out_layerN-1]`` —
            N+1 entries, NOT post-LN'd). VLMs select an intermediate layer
            (e.g. ``vision_feature_layer=-2``) from this tuple.

    Returns:
        ``last_hidden_state`` ``[B, P, H]`` (post-LN), or
        ``(last_hidden_state, hidden_states_tuple)`` when ``output_hidden_states``.
    """
    h = patch_embeds
    h = h.clone() if h.device.type == "spyre" else h
    hidden_states = [h] if output_hidden_states else None
    for compiled_block in model._spyre_compiled_blocks:
        h = compiled_block(h)
        if h.device.type == "spyre":
            h = h.clone()
        if output_hidden_states:
            hidden_states.append(h)
    h = model._spyre_post_layernorm(h)
    if h.device.type == "spyre":
        h = h.clone()
    if output_hidden_states:
        return h, tuple(hidden_states)
    return h


def prefill_vision(
    run_vision_forward_fn: Callable, model, pixel_values, output_hidden_states=False
):
    """One-shot vision-tower prefill: pixels -> patch-sequence hidden states.

    The vision tower has no padding, no attention mask, and a fixed patch count
    (``(image_size / patch_size) ** 2``), so this is simpler than
    ``prefill_encoder``: the patch sequence is already a clean BLOCK_SIZE
    multiple (e.g. SigLIP 384/16 → 24² = 576 = 9·64) and every patch is real.

    The patch embedding (Conv2d) runs on CPU and the result is moved to DEVICE
    — Conv2d is a tiny fraction of tower FLOPs and ``nn.Conv2d`` does not
    currently lower on Spyre. The position embedding (``nn.Embedding``) is
    added on CPU for the same reason; both match the embedding-lookup
    CPU-fallback pattern used by the decoder adapters.

    Args:
        run_vision_forward_fn: ``fn(model, patch_embeds) -> [B, P, H]`` — pass
            the adapter's ``_run_backbone_forward`` (``vision_backbone_forward``).
        model: prepared vision tower. Must expose ``model._spyre_patch_embed``
            (callable pixels -> ``[B, P, H]`` on CPU, set by ``prepare_for_spyre``).
        pixel_values: ``[B, C, image_size, image_size]`` (e.g. ``[B, 3, 384, 384]``).
            A multi-tile ``[B, T, C, H, W]`` input must be flattened to
            ``[B*T, C, H, W]`` by the caller.

    Returns:
        ``last_hidden_state`` ``[B, P, H]``, or
        ``(last_hidden_state, hidden_states_tuple)`` when ``output_hidden_states``.
    """
    patch_embeds = model._spyre_patch_embed(pixel_values)  # CPU, [B, P, H]
    return run_vision_forward_fn(
        model, patch_embeds.to(DEVICE), output_hidden_states=output_hidden_states
    )
