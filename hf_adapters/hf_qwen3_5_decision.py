# Copyright 2026 The Torch-Spyre Authors.
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

"""Adapter for Transformers decision models on a Qwen3.5 backbone.

``AutoSpyreModelForDecision`` loads the ``XxxForDecision`` model (e.g.
``vllm-sr/d3-nano``, a ``JevForDecision`` on Qwen3.5 2B) and scores through its
own ``decide``; the backbone forward becomes one right-padded readout prefill
(``prefill_embed``) of this adapter's ``_run_backbone_forward``.

A readout returns the hidden state of every prompt token, and checkpoints such as
d3 read it bidirectionally, so the full-attention layers attend over the whole
prompt at once. The Gated DeltaNet layers step through the prompt one
stick-sized chunk at a time, with their convolution, gates and delta rule on
Spyre (:mod:`hf_adapters._qwen3_5_gdn`) and their recurrent state resident on the
card. The linear decision head runs on Spyre as well; the vision tower, unused by
text requests, stays on CPU. Generation is not part of this adapter: causal
Qwen3.5 language models keep using ``hf_qwen3_5``.
"""

import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters import _qwen3_5_gdn, hf_common
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    SpyreUnsupportedModelError,
    embed_text_tokens,
    get_backbone,
    prebuild_rope_cache,
    text_config,
)
from hf_adapters.hf_qwen3_5 import (
    _allocate_caches as _allocate_attention_caches,
)
from hf_adapters.hf_qwen3_5 import (
    _dense_mlp,
    _make_attention_block,
    _padding_mask,
    _prepare_attention_projections,
    _rms_norm,
    _run_full_attention_layer,
    _setup_qwen3_5_text_decoder,
)


def _check_linear_attention_supported(linear_attn):
    """Fail fast on DeltaNet layers the Spyre path cannot represent faithfully."""
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d"):
        if getattr(linear_attn, name).bias is not None:
            raise SpyreUnsupportedModelError(
                f"Qwen3.5 decision adapter: linear_attn.{name} has a bias, which "
                "the Spyre DeltaNet path does not apply"
            )
    for name in ("head_k_dim", "head_v_dim"):
        dim = getattr(linear_attn, name)
        if dim % BLOCK_SIZE:
            raise SpyreUnsupportedModelError(
                f"Qwen3.5 decision adapter: {name}={dim} is not a multiple of "
                f"the {BLOCK_SIZE}-element Spyre stick"
            )


def _make_linear_attention_block(layer):
    """Compile one Gated DeltaNet layer whose core runs on Spyre.

    The q/k/v/z projections are fused into one linear and the two scalar gates
    into one stick-padded ``[b | 0 | a | 0]`` linear; the original projections and
    the convolution module are released so their weights are not carried onto
    the accelerator twice.
    """
    linear_attn = layer.linear_attn
    _check_linear_attention_supported(linear_attn)
    input_norm = layer.input_layernorm
    post_attention_norm = layer.post_attention_layernorm
    mlp = layer.mlp
    linear_attn._spyre_gdn = _qwen3_5_gdn.prepare_gdn_constants(
        linear_attn, hf_common.DEVICE
    )
    linear_attn._spyre_gdn_selectors = {}
    qkvz_weight = torch.cat(
        (linear_attn.in_proj_qkv.weight.detach(), linear_attn.in_proj_z.weight.detach())
    )
    ba_weight = _qwen3_5_gdn.pad_ba_weight(
        linear_attn.in_proj_b.weight.detach(), linear_attn.in_proj_a.weight.detach()
    )
    linear_attn._spyre_in_proj_qkvz = nn.Linear(
        qkvz_weight.shape[1], qkvz_weight.shape[0], bias=False, dtype=qkvz_weight.dtype
    )
    linear_attn._spyre_in_proj_qkvz.weight = nn.Parameter(
        qkvz_weight, requires_grad=False
    )
    linear_attn._spyre_in_proj_ba = nn.Linear(
        ba_weight.shape[1], ba_weight.shape[0], bias=False, dtype=ba_weight.dtype
    )
    linear_attn._spyre_in_proj_ba.weight = nn.Parameter(ba_weight, requires_grad=False)
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d"):
        setattr(linear_attn, name, nn.Identity())

    def project(hidden_states, padding_mask):
        h = _rms_norm(hidden_states, input_norm) * padding_mask[:, :, None]
        core_dtype = _qwen3_5_gdn.core_dtype(h.dtype)
        return (
            linear_attn._spyre_in_proj_qkvz(h).to(core_dtype),
            linear_attn._spyre_in_proj_ba(h).to(core_dtype),
        )

    def finish_projection(residual, core_output):
        return residual + linear_attn.out_proj(core_output.to(residual.dtype))

    def ffn_forward(hidden_states):
        residual = hidden_states
        h = _rms_norm(hidden_states, post_attention_norm)
        return residual + _dense_mlp(h, mlp)

    return (
        torch.compile(project, dynamic=False),
        # One code object for every layer: same shapes, per-layer tensor inputs.
        torch.compile(_qwen3_5_gdn.gdn_token_graph, dynamic=False, fullgraph=True),
        torch.compile(finish_projection, dynamic=False),
        torch.compile(ffn_forward, dynamic=False),
    )


def _token_selectors(linear_attn, sequence_length, dtype, device):
    """The cached one-hot row selectors of a ``sequence_length``-token chunk."""
    cache = linear_attn._spyre_gdn_selectors
    key = (sequence_length, dtype, str(device))
    if key not in cache:
        cache[key] = _qwen3_5_gdn.token_selectors(sequence_length, dtype, device)
    return cache[key]


def _run_linear_attention_chunk(
    compiled_block, linear_attn, hidden_states, padding_mask, conv_state, ssm_state
):
    """Run one Gated DeltaNet layer over a chunk, one token per Spyre graph call.

    Returns the layer output and the convolution and recurrent states after the
    chunk, which seed the next one.
    """
    project, token_step, finish_projection, ffn = compiled_block
    constants = linear_attn._spyre_gdn
    batch_size, sequence_length = hidden_states.shape[:2]
    qkvz, ba = project(hidden_states, padding_mask)
    core_output = torch.zeros(
        batch_size,
        sequence_length,
        constants.value_dim,
        dtype=qkvz.dtype,
        device=hidden_states.device,
    )
    for selector in _token_selectors(
        linear_attn, sequence_length, qkvz.dtype, qkvz.device
    ):
        core_output, conv_state, ssm_state = token_step(
            qkvz,
            ba,
            selector,
            core_output,
            conv_state,
            ssm_state,
            *constants.tensors,
            **constants.static(),
        )
    hidden_states = ffn(finish_projection(hidden_states, core_output))
    return hidden_states, conv_state, ssm_state


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Readout backbone: whole-prompt attention, chunked DeltaNet layers on Spyre."""
    backbone = get_backbone(model)
    hidden_states = embed_text_tokens(model, input_ids)
    selected_freqs = model._spyre_rope(hidden_states, position_ids)
    padding_mask = _padding_mask(attn_mask, hidden_states.shape[1], cache_index)
    decode = hidden_states.shape[1] == 1

    for index, layer_type in enumerate(text_config(model.config).layer_types):
        compiled_block = model._spyre_compiled_blocks[index]
        if layer_type == "full_attention":
            hidden_states = _run_full_attention_layer(
                compiled_block,
                hidden_states,
                selected_freqs,
                attn_mask,
                key_caches,
                value_caches,
                index,
                cache_index,
                decode=decode,
            )
            continue
        linear_attn = backbone.layers[index].linear_attn
        conv_state, ssm_state = key_caches[index], value_caches[index]
        chunk_outputs = []
        for start in range(0, hidden_states.shape[1], BLOCK_SIZE):
            output, conv_state, ssm_state = _run_linear_attention_chunk(
                compiled_block,
                linear_attn,
                hidden_states[:, start : start + BLOCK_SIZE],
                padding_mask[:, start : start + BLOCK_SIZE],
                conv_state,
                ssm_state,
            )
            chunk_outputs.append(output)
        hidden_states = torch.cat(chunk_outputs, dim=1)

    return model._spyre_compiled_norm(hidden_states)


def _allocate_caches(model, batch_size, max_cache_len, dtype, device):
    """KV caches for the attention layers; Spyre-resident DeltaNet states."""
    cfg = text_config(model.config)
    key_caches, value_caches = _allocate_attention_caches(
        model, batch_size, max_cache_len, dtype, device
    )
    for index, layer in enumerate(get_backbone(model).layers):
        if cfg.layer_types[index] == "full_attention":
            continue
        constants = layer.linear_attn._spyre_gdn
        key_caches[index] = torch.zeros(
            batch_size,
            constants.kernel_size - 1,
            constants.conv_dim,
            dtype=_qwen3_5_gdn.core_dtype(dtype),
            device=device,
        )
        # The fp32 recurrent state as fp16 (hi, lo) rows.
        value_caches[index] = torch.zeros(
            2 * batch_size * constants.num_v_heads,
            constants.head_v_dim,
            constants.head_k_dim,
            dtype=torch.float16,
            device=device,
        )
    return key_caches, value_caches


def _prepare_head_for_spyre(head) -> None:
    """Run a linear decision head on Spyre in place, keeping its call contract.

    As with the LM head, the output rows are zero-padded to a whole stick and
    cropped on the host. The decision model gathers the rows it reads (the marks)
    on CPU; they are padded to a stick-multiple row count for one fixed-shape
    graph, and only the head output returns to the host.
    """
    out_features = head.out_features
    padding = -out_features % BLOCK_SIZE
    head.weight = nn.Parameter(
        F.pad(head.weight.detach(), (0, 0, 0, padding)), requires_grad=False
    )
    if head.bias is not None:
        head.bias = nn.Parameter(
            F.pad(head.bias.detach(), (0, padding)), requires_grad=False
        )

    def project(hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, head.weight, head.bias)

    head._spyre_compiled_forward = torch.compile(project, dynamic=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        leading = hidden_states.shape[:-1]
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        num_rows = rows.shape[0]
        padded_rows = math.ceil(num_rows / BLOCK_SIZE) * BLOCK_SIZE
        rows = F.pad(
            rows.to("cpu", self.weight.dtype), (0, 0, 0, padded_rows - num_rows)
        )
        output: torch.Tensor = self._spyre_compiled_forward(
            rows.to(hf_common.DEVICE)
        ).to("cpu")
        return output[:num_rows, :out_features].reshape(*leading, out_features)

    head.forward = MethodType(forward, head)  # type: ignore[method-assign]


def prepare_for_spyre(model):
    """Prepare a Qwen3.5 decision model for Spyre in place."""
    backbone = model.base_model
    text_decoder = get_backbone(model)
    cfg, decoder, rope_permutation = _setup_qwen3_5_text_decoder(text_decoder)
    text_decoder._spyre_cache_allocator = _allocate_caches
    compiled_blocks = []
    for layer_type, layer in zip(cfg.layer_types, decoder.layers):
        if layer_type == "full_attention":
            query_projection, gate_projection = _prepare_attention_projections(
                text_decoder, layer, cfg.head_dim, rope_permutation
            )
            compiled_blocks.append(
                _make_attention_block(
                    layer, query_projection, gate_projection, cfg.head_dim
                )
            )
        else:
            compiled_blocks.append(_make_linear_attention_block(layer))
    text_decoder._spyre_compiled_blocks = compiled_blocks

    for head in model.children():
        if head is backbone:
            continue
        if not isinstance(head, nn.Linear):
            raise SpyreUnsupportedModelError(
                f"{type(model).__name__} has a decision head that is not a linear layer"
            )
        _prepare_head_for_spyre(head)

    model._spyre_cpu_submodules = [
        f"{model.base_model_prefix}.{name}"
        for name, child in backbone.named_children()
        if child is not text_decoder
    ]
    # The device move prepares ``model._spyre_rope``; the text decoder's RoPE is
    # built here to the decoder's own context window, which the decision config
    # nests one level deeper than the move looks.
    prebuild_rope_cache(text_decoder)
    model._spyre_rope = text_decoder._spyre_rope
