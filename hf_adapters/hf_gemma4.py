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
HuggingFace Transformers adapter for Gemma 4 causal-LM models on Spyre.

Targets the **12B / 31B dense** variants and the **E2B / E4B E-variants**
(``model_type`` ``gemma4_text`` / ``gemma4``); the E-variant per-layer
embeddings and KV-sharing are handled below (see "E-variant support"). Gemma 4
departs from the standard GQA decoder (``hf_qwen2`` etc.)
in several ways, so it gets a custom compiled block rather than reusing
``make_standard_gqa_block``:

- **Local / global alternating attention.** ``config.layer_types`` mixes
  ``sliding_attention`` (4 of every 5 layers) with ``full_attention``. Each
  type carries its own RoPE *and its own head_dim* (sliding ``head_dim``,
  global ``global_head_dim``), so the two layer types use different KV-cache
  shapes — see ``model._spyre_kv_shapes`` and ``hf_common.allocate_kv_caches``.
- **Partial rotary on global layers.** The global RoPE is "proportional" with
  ``partial_rotary_factor=0.25`` (theta 1e6); HF builds an ``inv_freq`` of
  length ``global_head_dim/2`` whose tail is zeros, so those dims rotate by
  angle 0 (identity). The existing ``PrecomputedRotaryEmbedding`` /
  ``apply_rope_matmul`` handle that unchanged — no special casing needed.
  Sliding layers use full rotary (theta 1e4) over ``head_dim``.
- **Q / K / V RMSNorm.** Per-head RMSNorm on Q and K (scaled) and on V
  (``with_scale=False``), applied before RoPE. The norms run in the compiled
  block.
- **K == V on global layers** (12B ``attention_k_eq_v=true``). Global layers
  have no ``v_proj``; V is the *raw* ``k_proj`` output (pre-k_norm, pre-RoPE)
  reshaped to the KV-head layout, then passed through ``v_norm`` (matching
  stock HF, which applies ``v_norm`` to the aliased value tensor). Sliding
  layers keep a separate V.
- **Embedding scaling.** ``embed_tokens`` multiplies by ``sqrt(hidden_size)``
  (``Gemma4TextScaledWordEmbedding``); this is part of the loaded module and
  runs as-is.
- **"Sandwich" norms.** Four norms per layer: ``input_layernorm`` (pre-attn),
  ``post_attention_layernorm`` (applied to the *attn output* before the
  residual add), ``pre_feedforward_layernorm`` (pre-MLP), and
  ``post_feedforward_layernorm`` (applied to the *MLP output* before the
  residual add). Unlike the 2-norm pre-norm of standard GQA.
- **Per-layer scalar.** Each decoder layer multiplies its output by a learned
  ``layer_scalar`` buffer (init 1.0).
- **Unscaled attention.** ``Gemma4TextAttention.scaling == 1.0`` — Q·Kᵀ is NOT
  divided by ``sqrt(head_dim)``. SDPA is called with ``scale=1.0``.
- **Large vocab + logit softcap.** 262K vocab → chunked LM head (like
  ``hf_phi3``); ``final_logit_softcapping`` (30.0) applies a
  ``cap * tanh(logits / cap)`` after the head.

E-variant support (E2B / E4B): the two E-variant features are handled here:

- **Per-Layer Embeddings (PLE).** E-variants inject a per-layer residual after
  the MLP: ``embed_tokens_per_layer`` + a projected/normed context term, gated
  and added back per layer (``_compute_per_layer_inputs`` + ``_ple_tail``,
  mirroring stock ``get_per_layer_inputs`` / ``project_per_layer_inputs`` and
  the decoder tail). Gated off (``has_ple=False``) for the dense 12B/31B
  variants, which carry no PLE submodules.
- **KV-sharing across layers.** The trailing ``num_kv_shared_layers`` layers
  reuse the KV cache of the nearest preceding non-shared layer of the same
  ``layer_type`` (stock ``store_full_length_kv`` semantics), so they run a lean
  Q-only block (no k/v proj, no cache write) against that producer's cache — see
  ``_shared_producer_map`` and ``Gemma4SharedBlock``.

The MoE 26B-A4B variant is handled by the sibling ``hf_gemma4_moe`` adapter,
which reuses this module's attention-side setup and forward driver.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("google/gemma-4-12B-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import (
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    add_causal_sliding_window_band,
    apply_rope_matmul,
    get_backbone,
    kv_cache_update,
    optional_spyre_config_patch,
    pad_lm_head,
    text_config,
)
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    allocate_swa_caches,
    anchored_step,
    compact_sliding_buffers,
    fit_attention_mask,
    physical_cache_capacity,
    rebind_shared_caches,
    roll_sliding_buffers,
    sliding_window_attention,
    valid_start_for,
)


def _gemma4_backbone(model):
    """Return the Gemma 4 text decoder backbone.

    ``AutoModelForCausalLM`` loads the *as-published* multimodal model
    (``Gemma4ForConditionalGeneration`` / ``Gemma4UnifiedForConditionalGeneration``)
    whose text decoder is nested at ``model.model.language_model`` (a
    ``Gemma4TextModel`` / ``Gemma4UnifiedTextModel`` with ``layers``,
    ``embed_tokens``, ``norm``, ``rotary_emb``). The shared ``get_backbone``
    descends into ``.language_model`` for exactly this case; this wrapper names
    the intent. The ``lm_head`` stays at the top level (``model.lm_head``),
    matching where ``pad_lm_head`` looks.
    """
    return get_backbone(model)


def _gemma4_rms_norm(hidden_states, weight, eps):
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = (x * torch.rsqrt(variance + eps)).to(hidden_states.dtype)
    return normed if weight is None else normed * weight


_compiled_gemma4_rms_norm = torch.compile(_gemma4_rms_norm, dynamic=False)


def _compute_per_layer_inputs(model, inputs_embeds, input_ids):
    """Compute the combined Per-Layer Embeddings tensor, or None if the model
    has no PLE. Mirrors stock ``Gemma4TextModel.get_per_layer_inputs`` +
    ``project_per_layer_inputs`` (transformers ``modeling_gemma4``).

    Returns ``[B, S, num_hidden_layers, hidden_size_per_layer_input]``.
    """
    backbone = _gemma4_backbone(model)
    if not getattr(backbone, "hidden_size_per_layer_input", 0):
        return None

    cfg = text_config(model.config)
    ple_dim = cfg.hidden_size_per_layer_input
    n_layers = cfg.num_hidden_layers

    # Token-identity component (scaled embedding already applies sqrt(ple_dim)).
    token_identity = backbone.embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape, n_layers, ple_dim
    )
    # Context component: project the main embeds, scale, reshape, RMSNorm.
    context = backbone.per_layer_model_projection(inputs_embeds)
    context = context * backbone.per_layer_model_projection_scale
    context = context.reshape(*inputs_embeds.shape[:-1], n_layers, ple_dim)
    context = backbone.per_layer_projection_norm(context)

    return (context + token_identity) * backbone.per_layer_input_scale


def _ple_tail(block, h, per_layer_input):
    """Gemma 4 PLE per-layer residual injection (stock modeling_gemma4 tail).

    ``block`` is the registered decoder block carrying the per-layer PLE
    submodules (``per_layer_input_gate`` / ``per_layer_projection`` /
    ``post_per_layer_input_norm``), captured in ``__init__`` when ``has_ple``.
    """
    residual = h
    x = block.per_layer_input_gate(h)
    x = F.gelu(x, approximate="tanh")
    x = x * per_layer_input
    x = block.per_layer_projection(x)
    x = block.post_per_layer_input_norm(x)
    return residual + x


def _offset_zero_per_layer_input(per_layer_inputs, layer_index):
    """Copy one PLE layer slice into fresh, offset-zero storage.

    This does not copy a KV cache. ``contiguous()`` is insufficient here: a
    decode slice has shape ``[B, 1, ple_dim]``, so PyTorch considers it
    contiguous even though it retains the parent tensor's nonzero storage
    offset. ``clone()`` always materializes the small PLE slice and therefore
    also covers the singleton decode shape.
    """
    return per_layer_inputs[:, :, layer_index, :].clone()


def _shared_producer_map(cfg):
    """For each layer, the producer layer index whose KV a shared layer reuses.

    Shared layers are the last ``num_kv_shared_layers`` layers; each reuses the
    KV of the nearest preceding NON-shared layer of the same ``layer_type``
    (stock Gemma4 ``store_full_length_kv`` semantics). Non-shared layers map to
    ``None``. Returns ``producer_of``.
    """
    n = cfg.num_hidden_layers
    n_shared = getattr(cfg, "num_kv_shared_layers", 0)
    first = n - n_shared
    layer_types = cfg.layer_types
    producer_of = [None] * n
    if n_shared <= 0:
        return producer_of
    # Last non-shared layer index per type (only layers before `first` produce).
    last_by_type = {}
    for i in range(first):
        last_by_type[layer_types[i]] = i
    for i in range(first, n):
        p = last_by_type.get(layer_types[i])
        assert p is not None, (
            f"Gemma 4 KV-share: shared layer {i} (type {layer_types[i]}) has no "
            "same-type producer before the KV-share boundary."
        )
        producer_of[i] = p
    return producer_of


def _query_row_mask(h, attn_mask):
    """Return a multiplier that is zero for fully masked query rows.

    Under ``hf_common.generate``'s block-padded prefill, a short prompt is
    left-padded to a ``BLOCK_SIZE`` multiple. The leading pad query rows attend
    to *no* key (their whole mask row contains only the finite mask fill value),
    so their output is discarded
    downstream — but they still flow through the decoder and write K/V into the
    padded cache slots. On Gemma 4 that is fatal: the ``<pad>`` (id 0) embedding
    RMSNorms to a large-magnitude activation whose MLP ``gate*up`` product
    overflows **fp16** to ``+inf`` (real rows peak two orders of magnitude
    lower), and the sandwich ``post_feedforward_layernorm`` then computes
    ``inf * rsqrt(inf) = inf * 0 = NaN``. That NaN is written as K/V at the pad
    cache positions; because ``NaN + (-inf) = NaN`` (an additive mask cannot
    suppress a NaN score), the *next* layer's real query rows pick it up through
    SDPA and every real-token logit goes NaN — ``generate`` then emits all
    ``<pad>``. (A mask fix cannot help: the poison is a NaN *key*, not a masking
    error.)

    The mask builder uses a finite negative sentinel rather than ``-inf``.
    Consequently SDPA can produce a nonzero result for an invalid query even
    when its input embedding is zero. The caller therefore applies this mask
    before the first block and after every block, keeping invalid rows (and the
    K/V they write in the following layer) neutral throughout the decoder.
    Valid query rows always contain at least one zero-valued, attendable entry.

    The fully-masked test is derived on **CPU** (a boolean reduction) and only
    the resulting float multiplier is moved to ``h``'s device, mirroring
    ``add_causal_sliding_window_band`` — Spyre's compiled backend rejects
    on-device boolean reductions. This runs in the eager block driver, outside
    any compiled region, so it is static and Spyre-safe.
    """
    # attn_mask: [B, 1, S, cache_len]. Allowed entries are exactly zero;
    # disallowed entries use the finite value returned by _mask_fill_value.
    am = attn_mask.to("cpu")
    live_rows = (am == 0).any(dim=-1).any(dim=1).to(h.dtype)  # [B, S]
    return live_rows.to(h.device)[:, :, None]


def _patch_gemma4_rmsnorm(rmsnorm_cls):
    """Patch Gemma 4 RMSNorm for Spyre."""

    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            weight = self.weight if self.with_scale else None
            # When a parent region is already being traced, expose the RMSNorm
            # operations directly to that graph. A standalone eager module call
            # needs the whole upcast/reduction/downcast chain compiled together
            # so the DL16<->fp32 staggered element arrangement is preserved.
            if torch.compiler.is_compiling():
                return _gemma4_rms_norm(hidden_states, weight, self.eps)
            return _compiled_gemma4_rms_norm(hidden_states, weight, self.eps)
        # CPU path: fp32 for numerical parity with stock HF.
        xf = hidden_states.float()
        variance = (xf * xf).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            xf = xf * self.weight.float()
        return xf.type_as(hidden_states)

    rmsnorm_cls.forward = _forward_fp16


class Gemma4Attention(nn.Module):
    """Attention executed by the dense Gemma 4 Spyre adapter path.

    On global ``attention_k_eq_v`` layers (``is_kv_eq_v=True``) there is no
    ``v_proj``: V is the raw ``k_proj`` output (before k_norm and RoPE) put
    through ``v_norm``, mirroring stock HF.
    """

    def __init__(
        self,
        attn,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        is_sliding=False,
        window_size=None,
        swa_mode=None,
        is_causal=True,
    ):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj  # None when is_kv_eq_v
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.v_norm = attn.v_norm
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_kv_eq_v = is_kv_eq_v
        self.scaling = attn.scaling  # 1.0 for Gemma 4
        self.is_sliding = is_sliding
        self.window_size = window_size
        # None keeps the band-masked SDPA path. "phase1" reads the window out of
        # the full-length cache as a correctness harness; anchored mode uses the
        # compact production cache.
        self.swa_mode = swa_mode
        # This is a static access-plan guarantee, not an alternate source of
        # mask semantics. False supports mixed masks such as causal text with
        # bidirectional vision blocks by making the kernel scan the full cache.
        self.is_causal = is_causal
        self._use_compiled_rms_norm = False

    def _rms_norm(self, hidden_states, norm):
        if self._use_compiled_rms_norm:
            weight = norm.weight if norm.with_scale else None
            return _compiled_gemma4_rms_norm(hidden_states, weight, norm.eps)
        return norm(hidden_states)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
    ):
        bsz, seq_len, _ = hidden_states.shape
        # Q/K/V projections viewed as [B, L, n_heads, head_dim]; norms are
        # applied per-head (last dim = head_dim) before the transpose.
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_q_heads, self.head_dim
        )
        k_lin = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        if self.is_kv_eq_v:
            # V reuses the raw k_proj output (pre-k_norm, pre-RoPE) but still
            # passes through v_norm: stock HF aliases value_states = key_states
            # *before* k_norm/RoPE, then applies self.v_norm(value_states)
            # unconditionally (modeling_gemma4 Gemma4TextAttention.forward). The
            # norm exists on these layers even though v_proj is None.
            v = self._rms_norm(k_lin, self.v_norm).transpose(1, 2)
        else:
            v = self.v_proj(hidden_states).view(
                bsz, seq_len, self.num_kv_heads, self.head_dim
            )
            v = self._rms_norm(v, self.v_norm).transpose(1, 2)

        q = self._rms_norm(q, self.q_norm).transpose(1, 2)
        k = self._rms_norm(k_lin, self.k_norm).transpose(1, 2)
        # Materialize the transpose returned by RoPE before the cache scatter.
        # A view here can make index_copy_ consume the wrong physical layout.
        q = apply_rope_matmul(q, selected_freqs).contiguous()
        k = apply_rope_matmul(k, selected_freqs).contiguous()

        # The 64-row roll that keeps this buffer compact is NOT here: it is an eager
        # driver step (roll_compact_buffer) run before this compiled block on shift
        # steps. An in-place roll on this same cache self-aliases — the device fused
        # its index_select read into its index_copy_ write and clobbered live rows.
        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )
        if self.is_sliding and self.swa_mode is not None:
            # The mask carries the window, cache position, and padding as tensor
            # data for both prefill and decode.
            attn_out = sliding_window_attention(
                q,
                key_cache,
                value_cache,
                attn_mask,
                window_size=self.window_size,
                is_causal=self.is_causal,
                scale=self.scaling,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                key_cache,
                value_cache,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.scaling,
                enable_gqa=True,
            )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), key_cache, value_cache


class Gemma4Block(nn.Module):
    """Registered dense Gemma 4 decoder block used by the Spyre adapter."""

    def __init__(
        self,
        layer,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        has_ple,
        is_sliding=False,
        window_size=None,
        swa_mode=None,
        is_causal=True,
    ):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
            is_sliding=is_sliding,
            window_size=window_size,
            swa_mode=swa_mode,
            is_causal=is_causal,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        # E-variant per-layer-embedding submodules (absent on dense 12B/31B).
        self.has_ple = has_ple
        if has_ple:
            self.per_layer_input_gate = layer.per_layer_input_gate
            self.per_layer_projection = layer.per_layer_projection
            self.post_per_layer_input_norm = layer.post_per_layer_input_norm
        self.train(layer.training)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
        per_layer_input=None,
        query_row_mask=None,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            h,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        # Sandwich: norm the attention output BEFORE adding the residual.
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        if self.has_ple:
            h = _ple_tail(self, h, per_layer_input)
        h = h * layer_scalar
        if query_row_mask is not None:
            h = h * query_row_mask
        return h, key_cache, value_cache


class Gemma4SharedBlock(nn.Module):
    """KV-sharing Gemma 4 decoder block (E-variant trailing layers).

    Runs a lean Q-only attention against a *producer* layer's KV cache: no
    k/v projection, no k_norm/v_norm, no RoPE-on-K, no cache update. The
    producer's cache is passed in by the driver (``_run_blocks_over_embeds``
    selects ``key_caches[producer_of[i]]``), so this block returns only the
    updated hidden state (never a cache tuple) and needs no ``cache_index`` —
    it never writes.
    """

    def __init__(
        self,
        layer,
        num_q_heads,
        head_dim,
        has_ple,
        is_sliding=False,
        window_size=None,
        swa_mode=None,
        is_causal=True,
    ):
        super().__init__()
        attn = layer.self_attn
        self.q_proj = attn.q_proj
        self.q_norm = attn.q_norm
        self.o_proj = attn.o_proj
        self.scaling = attn.scaling  # 1.0 for Gemma 4
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.is_sliding = is_sliding
        self.window_size = window_size
        self.swa_mode = swa_mode
        self.is_causal = is_causal
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        self.has_ple = has_ple
        if has_ple:
            self.per_layer_input_gate = layer.per_layer_input_gate
            self.per_layer_projection = layer.per_layer_projection
            self.post_per_layer_input_norm = layer.post_per_layer_input_norm
        self.train(layer.training)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        layer_scalar,
        per_layer_input=None,
        query_row_mask=None,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        bsz, seq_len, _ = h.shape
        q = self.q_proj(h).view(bsz, seq_len, self.num_q_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        q = apply_rope_matmul(q, selected_freqs).contiguous()
        if self.is_sliding and self.swa_mode is not None:
            attn_out = sliding_window_attention(
                q,
                key_cache,
                value_cache,
                attn_mask,
                window_size=self.window_size,
                is_causal=self.is_causal,
                scale=self.scaling,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                key_cache,
                value_cache,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.scaling,
                enable_gqa=True,
            )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = self.o_proj(attn_out)
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        if self.has_ple:
            h = _ple_tail(self, h, per_layer_input)
        h = h * layer_scalar
        if query_row_mask is not None:
            h = h * query_row_mask
        return h


def prepare_gemma4_blocks(
    layers,
    num_q_heads_per_layer,
    kv_shapes,
    is_kv_eq_v_per_layer,
    producer_of,
    has_ple,
    layer_types,
    window_size,
    swa_mode,
    swa_is_causal,
):
    """Replace Gemma 4 decoder layers with registered blocks and compile them.

    ``producer_of[i]`` is ``None`` for a normal (KV-writing) layer and an int
    for a KV-sharing layer, in which case a lean ``Gemma4SharedBlock`` is built
    instead of a full ``Gemma4Block``.
    """
    blocks = []
    for i, layer in enumerate(list(layers)):
        if producer_of[i] is None:
            block = Gemma4Block(
                layer,
                num_q_heads_per_layer[i],
                kv_shapes[i][0],
                kv_shapes[i][1],
                is_kv_eq_v_per_layer[i],
                has_ple,
                is_sliding=layer_types[i] == "sliding_attention",
                window_size=window_size,
                swa_mode=swa_mode,
                is_causal=swa_is_causal,
            )
        else:
            block = Gemma4SharedBlock(
                layer,
                num_q_heads_per_layer[i],
                kv_shapes[i][1],
                has_ple,
                is_sliding=layer_types[i] == "sliding_attention",
                window_size=window_size,
                swa_mode=swa_mode,
                is_causal=swa_is_causal,
            )
        layers[i] = block
        blocks.append(torch.compile(block, dynamic=False))
    return blocks


def _build_layer_masks(
    model,
    attn_mask,
    seq_len,
    batch_size,
    block_base,
):
    """Build the text-only per-layer-type mask dict {full_attention, sliding_attention}.

    ``attn_mask`` is the base causal mask the caller built (column index = cache
    slot). Global ("full_attention") layers use it as-is (plain causal). Sliding
    ("sliding_attention") layers intersect it with a causal sliding-window band
    using each query row's cache coordinate ``block_base + j`` (see
    ``add_causal_sliding_window_band``).

    ``block_base`` is the cache column the first query row occupies — the first
    entry of the block's ``cache_index`` (``int(cache_index[0])``).

    This is the text-decoder mask policy. The unified VLM adapter
    (``hf_gemma4_mm``) needs a bidirectional vision overlay on sliding layers,
    so it builds its own mask dict and passes it to
    ``_run_blocks_over_embeds(..., masks=...)`` rather than calling this.
    """
    cfg = text_config(model.config)
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(
        batch_size, seq_len
    )
    sliding_mask = add_causal_sliding_window_band(
        attn_mask, query_coords, cfg.sliding_window
    )
    return {"full_attention": attn_mask, "sliding_attention": sliding_mask}


@optional_spyre_config_patch({"frontend_pool_allocation": True})
def _run_blocks_over_embeds(
    model,
    h,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    masks=None,
    per_layer_inputs=None,
    query_row_mask=None,
):
    """Run the compiled Gemma 4 decoder blocks over precomputed embeddings.

    Shared by the text-only causal LM (``_run_backbone_forward``) and the VLM
    adapter (``hf_gemma4_mm``, which drives the decoder from image-scattered
    ``inputs_embeds``). Builds per-type RoPE freqs, then runs the blocks under a
    per-layer-type mask dict and applies the final norm.

    ``masks`` (optional ``{layer_type: mask}``) lets a caller supply its own
    per-type masks — the VLM passes its upstream-compatible causal-global and
    blockwise sliding masks. When ``None``, the text-only causal + sliding masks are built from
    ``attn_mask`` via ``_build_layer_masks`` (``attn_mask`` is ignored when
    ``masks`` is given).

    ``per_layer_inputs`` (optional ``[B, S, num_hidden_layers, ple_dim]``) is
    the combined PLE tensor from ``_compute_per_layer_inputs``, or ``None`` for
    non-PLE models (dense 12B/31B and the 12B VLM) — in which case
    ``per_layer_input=None`` is passed to each block, never read since
    ``has_ple=False`` gates the PLE tail off. A ``None`` (not a zero-length
    tensor) keeps the "no zero-length tensors on Spyre" rule on the shared path.

    ``query_row_mask`` is the optional ``[B, S, 1]`` validity multiplier for
    block-padded text prefill. Each compiled block applies it to its output
    because a finite all-masked attention row is not guaranteed to produce
    zero. Keeping this operation inside the compiled graph also preserves the
    expected layout at the boundary between blocks.

    Dense/E layers dispatch on ``model._spyre_producer_of[i]``: ``None`` runs
    the full block (writes its own KV cache, returns a 3-tuple); an int runs the
    shared block against that producer layer's cache (returns 1 tensor; the
    shared layer's own cache entry is never written). Dedicated MoE blocks keep
    their original seven-argument call contract.
    """
    swa_mode = getattr(model, "_spyre_swa_mode", None)
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    # Per-layer-type RoPE freqs (sliding theta vs global proportional theta).
    freqs = {
        layer_type: rope(h, position_ids)
        for layer_type, rope in model._spyre_rope.items()
    }

    bsz, seq_len = h.shape[0], h.shape[1]
    # The scalar read syncs from the device; deliberately not optimized, and now
    # needed by the op path as well as by the band. Once per step, not per layer.
    block_base = int(cache_index[0]) if masks is None or swa_mode else 0

    if masks is None:
        masks = _build_layer_masks(model, attn_mask, seq_len, bsz, block_base)

    if seq_len > 1 and block_base == 0:
        # A prefill call starts a new generation: drop any state a previous
        # generate() on this model left behind.
        model._spyre_swa_state = None
    state = getattr(model, "_spyre_swa_state", None)

    if swa_mode == "anchored" and state is None and seq_len == 1:
        # Compact at the first decode call, after every prefill chunk has written
        # into the original cache list. This also works when common.generate used
        # prefix views for chunked prefill: decode receives the owning tensors,
        # so replacing entries here updates the list consumed by later steps.
        prompt_len = getattr(model, "_spyre_padded_prompt_len", block_base)
        state = SlidingWindowCache.after_prefill(
            cfg.sliding_window, prompt_len, valid_start_for(model, bsz)
        )
        compact_sliding_buffers(
            cfg.layer_types,
            key_caches,
            value_caches,
            state,
            prompt_len,
            producer_of=model._spyre_producer_of,
        )
        model._spyre_swa_state = state

    if swa_mode and state is not None:
        # Anchored decode: geometry has one fixed tensor signature; mask contents
        # and the per-step cache write position both remain runtime values.
        step = anchored_step(state, cache_index.device, h.dtype)
        if step.do_shift:
            # Roll every sliding layer's compact buffer down one stick BEFORE its
            # compiled block writes this token — eager, into fresh allocations, the
            # same proven path as compaction. Kept out of the graph on purpose: an
            # in-place roll on the cache self-aliases and the device fuses the read
            # into the write (see roll_compact_buffer).
            roll_sliding_buffers(
                cfg.layer_types,
                key_caches,
                value_caches,
                producer_of=model._spyre_producer_of,
            )
        masks["sliding_attention"] = step.attention_mask
        sliding_index = step.cache_index
    elif swa_mode:
        sliding_index = cache_index
    else:
        sliding_index = cache_index

    backbone_layers = backbone.layers
    producer_of = model._spyre_producer_of
    is_moe = bool(getattr(cfg, "enable_moe_block", False))
    sliding_masks_by_capacity = {}
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        is_sliding = lt == "sliding_attention"
        # Each PLE slice needs fresh offset-0 storage at the graph boundary. A
        # singleton decode slice reports contiguous despite retaining the
        # parent's nonzero storage offset, so contiguous() is not sufficient;
        # clone() is. Inductor otherwise drops the graph input's storage_offset
        # and later layers read layer 0's PLE values. This copies only the small
        # PLE slice, not any KV cache.
        pli = (
            _offset_zero_per_layer_input(per_layer_inputs, i)
            if per_layer_inputs is not None
            else None
        )
        p = producer_of[i]
        selected_mask = masks[lt]
        if is_sliding and swa_mode:
            cache = key_caches[i] if p is None else key_caches[p]
            capacity = physical_cache_capacity(cache)
            if capacity not in sliding_masks_by_capacity:
                sliding_masks_by_capacity[capacity] = fit_attention_mask(
                    selected_mask, capacity
                )
            selected_mask = sliding_masks_by_capacity[capacity]
        # Pass the per-layer scalar as a tensor read fresh from the registered,
        # device-moved block — NOT as a Python float — so Dynamo guards on tensor
        # metadata instead of recompiling for each distinct learned value.
        if is_moe:
            # MoE keeps its split compiled-region call contract, while its
            # attention region uses the same sliding-window op and compact cache.
            h, key_caches[i], value_caches[i] = compiled_block(
                h,
                freqs[lt],
                selected_mask,
                key_caches[i],
                value_caches[i],
                sliding_index if is_sliding else cache_index,
                backbone_layers[i].layer_scalar,
            )
        elif p is None:
            h, key_caches[i], value_caches[i] = compiled_block(
                h,
                freqs[lt],
                selected_mask,
                key_caches[i],
                value_caches[i],
                sliding_index if is_sliding else cache_index,
                backbone_layers[i].layer_scalar,
                pli,
                query_row_mask,
            )
        else:
            # KV-sharing layer: read the producer's cache, write nothing.
            h = compiled_block(
                h,
                freqs[lt],
                selected_mask,
                key_caches[p],
                value_caches[p],
                backbone_layers[i].layer_scalar,
                pli,
                query_row_mask,
            )

    # A compiled producer may return a fresh tensor object even when it preserves
    # the underlying cache allocation. Keep the otherwise-unused consumer entries
    # pointed at the current producer buffers so they cannot retain stale storage.
    rebind_shared_caches(key_caches, value_caches, producer_of)

    if swa_mode == "anchored" and state is not None and seq_len == 1:
        state.advance()

    norm = backbone.norm
    weight = norm.weight if norm.with_scale else None
    h = _compiled_gemma4_rms_norm(h, weight, norm.eps)
    return h


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Gemma 4 backbone: scaled embedding, per-type RoPE + masks, blocks, norm.

    Text-only path: embed the ids (scaled word embedding), neutralize left-pad
    rows for dense/E models, compute the PLE tensor (``None`` for non-PLE
    models), then delegate to ``_run_blocks_over_embeds`` (no blockwise vision
    band).
    """
    backbone = _gemma4_backbone(model)
    h = backbone.embed_tokens(input_ids)
    # Only prefill can contain fully-masked left-pad query rows. Avoid the CPU
    # mask transfer and reduction on every single-token decode step.
    query_row_mask = None
    cfg = text_config(model.config)
    if h.shape[1] > 1 and not getattr(cfg, "enable_moe_block", False):
        query_row_mask = _query_row_mask(h, attn_mask)
        h = h * query_row_mask
    per_layer_inputs = _compute_per_layer_inputs(model, h, input_ids)
    return _run_blocks_over_embeds(
        model,
        h,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
        per_layer_inputs=per_layer_inputs,
        query_row_mask=query_row_mask,
    )


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Gemma 4 causal-LM forward: backbone + LM head + logit softcap."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )

    logits = model.lm_head(h)

    cap = text_config(model.config).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


def _setup_gemma4_text_decoder(model, *, allow_moe=False):
    """Shared attention-side Spyre prep for the Gemma 4 text decoder (in-place).

    Factored out of ``prepare_text_decoder_for_spyre`` so the MoE adapter
    (``hf_gemma4_moe``) can reuse the identical RMSNorm patch, per-type RoPE,
    per-layer KV-cache shapes, and LM-head padding without duplicating them or
    inheriting the dense path's MoE assert / dense-block compile. This helper
    does everything EXCEPT build ``model._spyre_compiled_blocks`` — the caller
    owns that (dense vs. MoE blocks differ).

    Steps:
      1. Record the supported E-variant PLE/KV-sharing features. The MoE gate is
         caller-controlled via ``allow_moe`` (the dense path forbids MoE; the
         MoE path requires it and asserts that separately).
      2. Patch ``Gemma4RMSNorm`` for the fp16 Spyre path.
      3. Build one ``PrecomputedRotaryEmbedding`` per layer type.
      4. Record per-layer KV-cache shapes (sliding vs global differ).
      5. Chunk the LM head for the large vocab.

    Returns ``(num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer)`` — the
    per-layer geometry the caller needs to compile its blocks.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    # E-variant features (handled): PLE flag + KV-share producer map.
    model._spyre_has_ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
    model._spyre_producer_of = _shared_producer_map(cfg)

    if allow_moe:
        assert (
            not model._spyre_has_ple
        ), "Gemma 4 MoE adapter does not support per-layer embeddings (PLE)."
        assert not getattr(
            cfg, "num_kv_shared_layers", 0
        ), "Gemma 4 MoE adapter does not support KV-sharing across layers."
    else:
        assert not getattr(cfg, "enable_moe_block", False), (
            "Gemma 4 dense adapter does not support MoE blocks "
            "(enable_moe_block=True); use hf_gemma4_moe."
        )

    # Patch whichever concrete RMSNorm class this model uses. The norm module
    # closest to a decoder layer's input_layernorm is representative.
    rmsnorm_cls = type(backbone.layers[0].input_layernorm)
    _patch_gemma4_rmsnorm(rmsnorm_cls)

    attention_k_eq_v = getattr(cfg, "attention_k_eq_v", False)
    layer_configs = cfg.per_layer_config
    num_q_heads_per_layer = []
    kv_shapes = []
    is_kv_eq_v_per_layer = []
    for i, (layer_type, layer_cfg) in enumerate(zip(cfg.layer_types, layer_configs)):
        num_q_heads_per_layer.append(layer_cfg.num_attention_heads)
        head_dim = layer_cfg.head_dim
        assert head_dim % 2 == 0 and head_dim // 2 >= 64, (
            f"Gemma 4 layer {i} head_dim={head_dim}: head_dim/2 must be >= 64 "
            "(one Spyre stick). A padded variant is not implemented for this adapter."
        )
        num_kv_heads = layer_cfg.num_key_value_heads
        kv_shapes.append((num_kv_heads, head_dim, head_dim))
        is_kv_eq_v_per_layer.append(attention_k_eq_v and layer_type == "full_attention")
    model._spyre_kv_shapes = kv_shapes

    # One PrecomputedRotaryEmbedding per layer type, reading the model's
    # per-type inv_freq + attention_scaling buffers via a shim. No padding:
    # the global proportional RoPE already encodes its NoPE tail as zero freqs.
    rope = backbone.rotary_emb
    model._spyre_rope = {}
    for layer_type in set(cfg.layer_types):
        inv_freq = getattr(rope, f"{layer_type}_inv_freq")
        scaling = getattr(rope, f"{layer_type}_attention_scaling")
        model._spyre_rope[layer_type] = PrecomputedRotaryEmbedding(
            InvFreqShim(inv_freq, scaling)
        )

    # LM head: smooth-padded to a stick-aligned vocab whose per-core span fits
    # the 256 MB EAR limit (see hf_common.pad_lm_head).
    pad_lm_head(model)

    return num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer


def prepare_text_decoder_for_spyre(model):
    """Prepare ONLY the Gemma 4 **dense** text decoder for Spyre (in-place).

    Runs the shared attention-side setup (``_setup_gemma4_text_decoder``:
    feature dispatch, RMSNorm patch, per-type RoPE, KV shapes, LM-head padding)
    then compiles a dense ``Gemma4Block`` per decoder layer. The MoE
    adapter (``hf_gemma4_moe``) calls the same seam with ``allow_moe=True`` and
    compiles its own MoE blocks instead.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)
    num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer = _setup_gemma4_text_decoder(
        model, allow_moe=False
    )

    # The sliding-window op path is the default; set model._spyre_swa_mode = None
    # before prepare to fall back to band-masked SDPA, or "phase1" for the
    # full-cache correctness harness.
    if not hasattr(model, "_spyre_swa_mode"):
        model._spyre_swa_mode = "anchored"
    if not hasattr(model, "_spyre_swa_is_causal"):
        model._spyre_swa_is_causal = True
    if model._spyre_swa_mode == "anchored":
        model._spyre_cache_allocator = allocate_swa_caches

    model._spyre_compiled_blocks = prepare_gemma4_blocks(
        backbone.layers,
        num_q_heads_per_layer,
        kv_shapes,
        is_kv_eq_v_per_layer,
        model._spyre_producer_of,
        model._spyre_has_ple,
        cfg.layer_types,
        cfg.sliding_window,
        getattr(model, "_spyre_swa_mode", None),
        model._spyre_swa_is_causal,
    )


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Gemma 4 causal-LM model in-place.

    Handles both the dense 12B/31B variants and the E2B/E4B E-variants
    (per-layer embeddings + KV-sharing); see ``prepare_text_decoder_for_spyre``.
    """
    prepare_text_decoder_for_spyre(model)
