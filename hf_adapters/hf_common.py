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

DEVICE = "spyre"
BLOCK_SIZE = 64  # Spyre stick size at fp16 (128 bytes / 2 bytes per element)


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
    """
    return model.model if hasattr(model, "model") else model


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

        self._freq_cache = rot.contiguous().to(torch.float16)
        self._cached_len = target_len

    def forward(self, hidden_states, position_ids):
        pos_cpu = position_ids.to("cpu")
        max_pos = int(pos_cpu.max().item()) + 1
        self._extend_cache(max_pos)
        selected = self._freq_cache[pos_cpu]  # [B, L, 2, 2, D/2]
        return selected.to(DEVICE)


def apply_rope_matmul(x, selected_freqs):
    """Apply RoPE via matmul with [2,2,D/2] rotation matrix. No slicing.

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
    """Update pre-allocated KV cache via overwrite at cache_position.

    All args are tensors; is_filling/token_index/cache_position are Python
    scalars that torch.compile specializes on.
    """
    if is_filling:
        k_write = k[:, :, token_index : token_index + 1, :]
        v_write = v[:, :, token_index : token_index + 1, :]
    else:
        k_write = k
        v_write = v

    if key_cache.device.type == "spyre":
        torch.ops.spyre.overwrite(
            input=k_write,
            output=key_cache,
            dims=[2],
            offsets=[cache_position],
        )
        torch.ops.spyre.overwrite(
            input=v_write,
            output=value_cache,
            dims=[2],
            offsets=[cache_position],
        )
    else:
        seq_len = k_write.shape[2]
        key_cache[:, :, cache_position : cache_position + seq_len, :] = k_write
        value_cache[:, :, cache_position : cache_position + seq_len, :] = v_write

    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


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

    # V/O padding: needed until torch-spyre/torch-spyre#1739 is resolved.
    def _pad_v_simple(proj, n_heads):
        """Simple end-padding per head for V."""
        w = proj.weight
        hidden = w.shape[1]
        new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
        new_proj = nn.Linear(
            hidden, n_heads * padded_head_dim, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
            for h in range(n_heads):
                s = h * orig_head_dim
                d = h * padded_head_dim
                new_b[d : d + orig_head_dim] = proj.bias[s : s + orig_head_dim]
            new_proj.bias = nn.Parameter(new_b, requires_grad=False)
        return new_proj

    def _pad_o(proj, n_heads):
        """Simple end-padding along input dim for O."""
        w = proj.weight
        hidden = w.shape[0]
        new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
        for h in range(n_heads):
            s = h * orig_head_dim
            d = h * padded_head_dim
            new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
        new_proj = nn.Linear(
            n_heads * padded_head_dim, hidden, bias=proj.bias is not None
        )
        new_proj.weight = nn.Parameter(new_w, requires_grad=False)
        if proj.bias is not None:
            new_proj.bias = nn.Parameter(proj.bias.clone(), requires_grad=False)
        return new_proj

    for layer in layers:
        attn = layer.self_attn
        orig_scaling = attn.scaling
        attn.q_proj = _pad_qk_rope(attn.q_proj, num_heads)
        attn.k_proj = _pad_qk_rope(attn.k_proj, num_kv_heads)
        attn.v_proj = _pad_v_simple(attn.v_proj, num_kv_heads)
        attn.o_proj = _pad_o(attn.o_proj, num_heads)
        attn.head_dim = padded_head_dim
        attn.scaling = orig_scaling

    model._spyre_head_dim = padded_head_dim


def patch_rmsnorm(rmsnorm_cls):
    """Patch any RMSNorm class to stay in fp16 (Spyre has no dtype conversion).

    Args:
        rmsnorm_cls: The RMSNorm class to patch (e.g. GraniteRMSNorm, Qwen3RMSNorm).
    """

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            # Spyre path: no dtype conversion, stay in fp16
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


def pad_lm_head(model):
    """Pad LM head vocab dim to stick-aligned size (+64 for work division).

    No-op when ``model`` has no ``lm_head`` (e.g. backbones loaded via
    ``AutoModel`` for embedding workloads).
    """
    if not hasattr(model, "lm_head"):
        return
    w = model.lm_head.weight
    vocab = w.shape[0]
    padded = ((vocab + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) + BLOCK_SIZE
    if padded != vocab:
        model.lm_head.weight = nn.Parameter(
            F.pad(w, (0, 0, 0, padded - vocab)), requires_grad=False
        )


def chunk_lm_head(model, num_chunks=8):
    """Split LM head weight into N chunks along vocab dim.

    Large vocab (200K+) exceeds Spyre's per-core 256 MB EAR limit.
    We replace the single lm_head with N smaller nn.Linear modules.
    Each chunk processes vocab_size/N output dims.

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


def build_prefill_mask(batch_size, padded_len, max_cache_len, prompt_offsets):
    """Causal mask for prefill, masking left-padding and unused cache positions."""
    mask = torch.zeros((batch_size, 1, padded_len, max_cache_len), dtype=torch.float16)
    if isinstance(prompt_offsets, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, : prompt_offsets[b].item()] = -torch.inf
    else:
        mask[:, :, :, :prompt_offsets] = -torch.inf
    for i in range(padded_len):
        mask[:, :, i, i + 1 :] = -torch.inf
    return mask


def build_expansion_mask(
    batch_size, block_size, max_cache_len, used_cache_len, prompt_offsets
):
    """Causal mask for an expansion decode step."""
    mask = torch.zeros((batch_size, 1, block_size, max_cache_len), dtype=torch.float16)
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
    batch_size, padded_len, actual_lengths, is_causal=True
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
    mask = torch.zeros((batch_size, 1, padded_len, padded_len), dtype=torch.float16)
    if is_causal:
        for i in range(padded_len):
            mask[:, :, i, i + 1 :] = -torch.inf
    if isinstance(actual_lengths, torch.Tensor):
        for b in range(batch_size):
            mask[b, :, :, actual_lengths[b].item() :] = -torch.inf
    else:
        mask[:, :, :, actual_lengths:] = -torch.inf
    return mask


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

    The token embedding is used as a gather, not a matmul, so it should not
    get a row-major SpyreTensorLayout. Returns the set of ``data_ptr()`` values
    for the embedding weight(s) we want to leave alone.
    """
    ids = set()
    backbone = get_backbone(model)
    embed = getattr(backbone, "embed_tokens", None)
    if embed is not None and hasattr(embed, "weight"):
        ids.add(embed.weight.data_ptr())
    return ids


def _untie_embedding_and_lm_head(model):
    """If ``embed_tokens.weight`` and ``lm_head.weight`` share storage, clone the
    LM head's weight so each can take a different Spyre layout.
    """
    if not hasattr(model, "lm_head"):
        return
    backbone = get_backbone(model)
    embed = getattr(backbone, "embed_tokens", None)
    if embed is None:
        return
    if embed.weight.data_ptr() == model.lm_head.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(
            model.lm_head.weight.detach().clone(), requires_grad=False
        )
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False


def _move_to_spyre_with_layout(model, dtype):
    """Move all parameters and buffers to Spyre with row-major layout for 2D
    matmul weights, except embedding weights which keep the default layout.
    """
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
        if t.dim() > 1 and t.data_ptr() not in skip_layout_ptrs:
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


def load_model_common(model_path, prepare_fn, dtype=torch.float16, auto_model_cls=None):
    """Load an HF model, apply Spyre adaptations, move to device.

    Args:
        model_path: HF model path or local directory.
        prepare_fn: Model-specific ``prepare_for_spyre(model)`` callable.
        dtype: Weight dtype (default fp16).
        auto_model_cls: HF auto-model class to use (e.g. ``AutoModel``,
            ``AutoModelForCausalLM``). Defaults to ``AutoModel``.
    """
    if auto_model_cls is None:
        from transformers import AutoModel

        auto_model_cls = AutoModel

    _patch_torch_empty()
    print(f"Loading model from {model_path} ...")
    model = auto_model_cls.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)
    _untie_embedding_and_lm_head(model)
    prepare_fn(model)
    print("Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, dtype)
    print("Model ready.")
    return model


def generate(
    run_forward_fn: Callable,
    model,
    tokenizer,
    prompts,
    max_new_tokens=128,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    timing=False,
):
    """Model-agnostic generation with padded 64-block decode.

    Args:
        run_forward_fn: ``fn(model, input_ids, position_ids, attn_mask,
            key_caches, value_caches, is_filling, token_index,
            cache_position) -> logits``
        model: Prepared HF model on Spyre.
        tokenizer: HF tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Max tokens to generate.
        do_sample: Sampling vs greedy.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        timing: Print per-token latency.
    """
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

    # Pad further to BLOCK_SIZE multiple (uniform left-pad for all sequences)
    padded_len = math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
    block_pad_offset = padded_len - prompt_length
    max_cache_len = padded_len + math.ceil(max_new_tokens / BLOCK_SIZE) * BLOCK_SIZE
    if block_pad_offset > 0:
        pad = input_ids.new_zeros((batch_size, block_pad_offset))
        input_ids = torch.cat([pad, input_ids], dim=1)

    # Per-sequence total left-padding (tokenizer pad + block-alignment pad)
    prompt_offsets = padded_len - actual_prompt_lengths  # [B]

    # Position IDs: real tokens at the END of the padded sequence.
    # Each sequence's real tokens span positions 0..actual_len-1, placed at
    # padded indices prompt_offsets[b]..padded_len-1.
    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    for b in range(batch_size):
        actual_len = actual_prompt_lengths[b].item()
        offset = prompt_offsets[b].item()
        position_ids[b, offset:] = torch.arange(actual_len)

    # Initialize empty KV caches
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    key_caches = [
        torch.zeros(
            batch_size,
            num_kv_heads,
            max_cache_len,
            head_dim,
            dtype=torch.float16,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.zeros(
            batch_size,
            num_kv_heads,
            max_cache_len,
            v_head_dim,
            dtype=torch.float16,
            device=DEVICE,
        )
        for _ in range(num_layers)
    ]

    # Decode state
    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None

    times_list = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    finished = torch.zeros(batch_size, dtype=torch.bool)
    num_generated = torch.zeros(batch_size, dtype=torch.long)

    for i in range(max_new_tokens):
        t0 = time.time()

        if i == 0:
            # --- PREFILL ---
            prefill_mask = build_prefill_mask(
                batch_size, padded_len, max_cache_len, prompt_offsets
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
        if do_sample:
            scaled = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1)
                scaled[scaled < v[:, -1:]] = -torch.inf
            probs = F.softmax(scaled, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        else:
            next_tokens = torch.argmax(next_logits, dim=-1)  # [B]

        if timing:
            times_list.append(time.time() - t0)

        # Place token in result (FMS logic)
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        if tokens_in_block == 0:
            # Just finished a block, pad for next block
            result = F.pad(result, (0, BLOCK_SIZE))
        grab_idx = (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        result[:, -grab_idx] = next_tokens  # [B]
        if eos_token_id is not None:
            finished |= next_tokens == eos_token_id
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

    # Decode text — walk the block structure per sequence using each
    # sequence's own num_generated count. Within a sequence, generated
    # tokens are contiguous within each BLOCK_SIZE block but blocks may
    # be separated by unused slots; all sequences share the same block
    # layout starting at padded_len.
    results = []
    for b in range(batch_size):
        gen_ids_list = []
        block_start = padded_len
        remaining = num_generated[b].item()
        while remaining > 0:
            take = min(remaining, BLOCK_SIZE)
            for j in range(take):  # type: ignore[arg-type]
                gen_ids_list.append(result[b, block_start + j].item())
            remaining -= take
            block_start += BLOCK_SIZE
        gen_ids = torch.tensor(gen_ids_list)
        if eos_token_id is not None:
            eos_pos = (gen_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0].item()]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return results


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


def prepare_rope_and_heads(model):
    cfg = model.config
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
        Tuple ``(last_hidden_state, attention_mask)``:
          - ``last_hidden_state``: ``[B, L, H]`` on Spyre, cropped back to
            the input ``L``. Caller moves to CPU as needed.
          - ``attention_mask``: the input mask, unchanged. Returned for
            convenience so the caller can pipe ``(h, mask)`` straight
            into a pooling layer.
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

    # Causal right-padded mask (or bidirectional for models with
    # ``config.is_causal=False``)
    is_causal = getattr(model.config, "is_causal", True)
    mask = build_prefill_mask_right_padded(
        bsz, padded_len, actual_lengths, is_causal=is_causal
    )

    # Throwaway KV caches sized to padded_len (no decode budget)
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    key_caches = [
        torch.zeros(
            bsz,
            num_kv_heads,
            padded_len,
            head_dim,
            dtype=torch.float16,
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
            dtype=torch.float16,
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

    # Crop the block-pad back off; tokenizer pad stays so pooling can mask it
    h = h[:, :seq_len, :]
    return h, attention_mask
