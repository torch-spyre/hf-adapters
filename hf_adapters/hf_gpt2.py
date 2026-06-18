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
HuggingFace Transformers adapter for GPT-2 models on Spyre.

Covers model_type ``gpt2``: GPT-2 (124M/medium/large/xl), DistilGPT-2,
Cerebras-GPT, and other GPT-2-architecture fine-tunes.

Unlike the RoPE decoders, GPT-2 uses:
  - **learned absolute** position embeddings (``wpe``), added to the token
    embeddings — no RoPE matmul at all (so ``head_dim/2 >= 64`` does not apply);
  - **LayerNorm** (weight + bias), pre-norm;
  - a **fused ``c_attn`` ``Conv1D``** for QKV and ``Conv1D`` everywhere a
    decoder would use ``nn.Linear`` (``Conv1D`` is ``y = x @ W + b`` with
    ``W`` shaped ``[in, out]``, i.e. the transpose of ``nn.Linear``);
  - a GELU MLP (``c_fc`` → ``gelu_new`` → ``c_proj``);
  - the backbone at ``model.transformer`` (not ``model.model``).

``prepare_for_spyre`` rewrites every ``Conv1D`` into an ``nn.Linear`` (so the
Spyre row-major matmul layout applies) and splits the fused ``c_attn`` into
separate q/k/v linears, then compiles one block per layer via the shared
``make_decoder_block`` (the non-RoPE causal-decoder factory, also used by the
OPT/BLOOM/MPT family).

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import math

import torch
import torch.nn as nn

from hf_adapters.hf_common import (
    assert_spyre_dimensions,
    get_backbone,
    make_decoder_block,
    pad_lm_head,
)

_GELU_COEFF = math.sqrt(2.0 / math.pi)


def _patch_gpt2_gelu(gelu_cls):
    """Patch ``NewGELUActivation`` to avoid ``torch.pow`` on Spyre.

    GPT-2's MLP uses the tanh-approximation GELU
    ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))``. Stock HF computes the
    ``x**3`` term with ``torch.pow(x, 3.0)``, which the Spyre Inductor backend
    cannot lower (``Unsupported: ... PointwiseOp(op='mul', ...)`` on the
    ``[1, 64, 3072]`` MLP intermediate — same class of issue as RMSNorm's
    ``pow(2)``). Element-wise multiply is native, so the Spyre path uses
    ``x * x * x``. ``tanh`` itself lowers fine (Gemma softcapping uses it).

    Like ``patch_rmsnorm``, the CPU path keeps the original ``torch.pow`` form so
    adapter outputs stay bit-identical to stock HF on CPU; only the on-device
    path takes the multiply rewrite (within fp16 noise, ~1e-3).
    """

    def _forward(self, input):
        if input.device.type == "spyre":
            inner = _GELU_COEFF * (input + 0.044715 * (input * input * input))
        else:
            inner = _GELU_COEFF * (input + 0.044715 * torch.pow(input, 3.0))
        return 0.5 * input * (1.0 + torch.tanh(inner))

    gelu_cls.forward = _forward


def _conv1d_to_linear(conv):
    """Convert an HF ``Conv1D`` to an equivalent ``nn.Linear``.

    ``Conv1D`` computes ``y = x @ W + b`` with ``W`` of shape ``[in, out]`` and
    ``b`` of shape ``[out]``. ``nn.Linear`` computes ``y = x @ W.T + b`` with
    ``W`` of shape ``[out, in]``, so the weight is the transpose. Spyre's
    row-major matmul layout (see ``_move_to_spyre_with_layout``) targets 2-D
    ``nn.Linear`` weights, so we convert before moving to device.
    """
    in_features, out_features = conv.weight.shape
    linear = nn.Linear(in_features, out_features, bias=conv.bias is not None)
    linear.weight = nn.Parameter(conv.weight.t().contiguous(), requires_grad=False)
    if conv.bias is not None:
        linear.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
    return linear


def _split_c_attn(c_attn, embed_dim):
    """Split the fused ``c_attn`` ``Conv1D`` ``[H, 3H]`` into q/k/v ``nn.Linear``.

    HF computes ``self.c_attn(x).split(embed_dim, dim=2)`` to obtain query, key,
    value (in that order) along the output dim. The ``Conv1D`` weight is
    ``[H, 3H]`` (in, out) and the bias is ``[3H]``; we slice the output dim into
    three ``[H, H]`` blocks and build a ``nn.Linear`` from each (transposing to
    ``[out, in]``).
    """
    w = c_attn.weight  # [H, 3H]
    b = c_attn.bias  # [3H]

    def _mk(start):
        end = start + embed_dim
        linear = nn.Linear(embed_dim, embed_dim, bias=True)
        linear.weight = nn.Parameter(
            w[:, start:end].t().contiguous(), requires_grad=False
        )
        linear.bias = nn.Parameter(b[start:end].detach().clone(), requires_grad=False)
        return linear

    return _mk(0), _mk(embed_dim), _mk(2 * embed_dim)


def _make_compiled_block(layer):
    """Resolve GPT-2's module layout and hand off to ``make_decoder_block``.

    GPT-2 is pre-norm MHA with learned absolute positions (no RoPE), so the
    compiled body is the shared non-RoPE decoder block; only the module names
    are GPT-2-specific (``attn``/``ln_1``/``ln_2``, fused-then-split q/k/v,
    ``c_proj`` for output, ``mlp.{c_fc,act,c_proj}`` for the FFN). The FFN is
    passed decomposed; ``mlp(h)``'s trailing dropout is identity in eval, so
    ``c_proj(act(c_fc(h)))`` is equivalent. ``act`` is the gelu instance already
    patched by ``_patch_gpt2_gelu``. ``scale == head_dim**-0.5 == attn.scaling``
    (GPT-2's ``scale_attn_weights`` default).
    """
    attn = layer.attn
    mlp = layer.mlp
    return make_decoder_block(
        q_proj=attn.q_proj,
        k_proj=attn.k_proj,
        v_proj=attn.v_proj,
        o_proj=attn.c_proj,
        attn_ln=layer.ln_1,
        ffn_in=mlp.c_fc,
        act=mlp.act,
        ffn_out=mlp.c_proj,
        ffn_ln=layer.ln_2,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        scale=attn.head_dim**-0.5,
        pre_ln=True,
    )


def _run_backbone_forward(
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
    """GPT-2 backbone: token + learned position embeddings, compiled blocks, ln_f.

    Returns ``last_hidden_state`` (no ``lm_head``). ``position_ids`` index the
    learned ``wpe`` table directly — exactly the indices the generate/test
    harness already supplies.
    """
    bb = get_backbone(model)
    h = bb.wte(input_ids) + bb.wpe(position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            None,  # selected_freqs — no RoPE
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = bb.ln_f(h)
    return h


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
    """GPT-2 causal-LM forward: backbone + LM head.

    Crops the logits back to the true vocab size: ``pad_lm_head`` extends the
    head's output dim to a smooth stick count with zero-weight rows, which
    produce logit 0. With GPT-2's fp16 logits those padding rows can outrank
    every real token at some positions and be argmax-selected — an
    out-of-vocab id that then crashes the next ``wte`` lookup. Slicing here
    drops the padding columns so token selection only ever sees real vocab.
    """
    h = _run_backbone_forward(
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
    logits = model.lm_head(h)
    return logits[..., : model.config.vocab_size]


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a GPT-2 model in-place.

    Rewrites every ``Conv1D`` to ``nn.Linear`` (splitting the fused ``c_attn``
    into q/k/v), pads the LM head vocab to a smooth stick count, and compiles
    one block per layer. No RoPE module is created — GPT-2 uses learned absolute
    positions.
    """
    cfg = model.config
    assert_spyre_dimensions(cfg, model_name=getattr(cfg, "name_or_path", "") or "gpt2")

    bb = get_backbone(model)
    embed_dim = cfg.n_embd

    # Patch the MLP activation (gelu_new / NewGELUActivation) to drop torch.pow
    # on Spyre. Patch the actual instantiated class so we follow whichever GELU
    # variant the config selected (gpt2 default is gelu_new).
    _patch_gpt2_gelu(type(bb.h[0].mlp.act))

    for layer in bb.h:
        attn = layer.attn
        attn.q_proj, attn.k_proj, attn.v_proj = _split_c_attn(attn.c_attn, embed_dim)
        attn.c_proj = _conv1d_to_linear(attn.c_proj)
        layer.mlp.c_fc = _conv1d_to_linear(layer.mlp.c_fc)
        layer.mlp.c_proj = _conv1d_to_linear(layer.mlp.c_proj)

    pad_lm_head(model)

    # GPT-2 is MHA (kv heads == attention heads) and its config uses n_head /
    # n_embd rather than the standard num_key_value_heads / head_dim that
    # kv_cache_shapes reads. Set the per-layer KV shapes explicitly so the
    # cache allocator (allocate_kv_caches / prefill) bypasses those lookups.
    num_heads = cfg.n_head
    head_dim = embed_dim // num_heads
    model._spyre_kv_shapes = [
        (num_heads, head_dim, head_dim) for _ in range(cfg.n_layer)
    ]

    model._spyre_compiled_blocks = [_make_compiled_block(layer) for layer in bb.h]
