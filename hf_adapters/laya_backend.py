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

"""Adapter for Laya typed-decision models.

Laya wraps a stock ModernBERT encoder in a custom decision model. The encoder
and two-layer decision transformer run on Spyre; type embedding, option scoring,
and action scoring remain on CPU. The upstream ``laya`` package is optional and
supplies the model class and checkpoint construction logic.
"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

import hf_adapters.hf_common as hf_common
from hf_adapters import hf_modernbert
from hf_adapters.hf_common import BLOCK_SIZE, get_model_dtype

_MIN_LAYA_VERSION = "0.3.28"
_MAX_LAYA_VERSION = "0.4"
_SUPPORTED_MODEL_IDS = {
    "convaiinnovations/laya",
    "convaiinnovations/laya-typed-decisions",
    "convaiinnovations/laya-multilingual",
}


def _require_laya():
    try:
        installed = version("laya")
    except PackageNotFoundError as exc:
        raise ImportError(
            "Laya support requires the optional 'laya' package. "
            f"Install it with `pip install 'laya>={_MIN_LAYA_VERSION}'`."
        ) from exc

    from packaging.version import Version

    if not (
        Version(_MIN_LAYA_VERSION) <= Version(installed) < Version(_MAX_LAYA_VERSION)
    ):
        raise ImportError(
            f"Laya {installed} is installed, but hf-adapters requires "
            f"laya>={_MIN_LAYA_VERSION},<{_MAX_LAYA_VERSION}."
        )


def _resolve_model_dir(
    model_path, *, subfolder=None, include_tokenizer=False, token=None, revision=None
):
    model_path = os.fspath(model_path)
    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download
    from laya.revisions import resolve_revision

    revision = resolve_revision(model_path, revision)
    prefix = f"{subfolder}/" if subfolder else ""
    allow_patterns = [
        "config.json",
        prefix + "rl_agent_config.json",
        prefix + "model.safetensors",
        prefix + "encoder/*",
    ]
    if include_tokenizer:
        allow_patterns.append(prefix + "tokenizer/*")
    kwargs = {
        "token": token or os.environ.get("HF_TOKEN") or None,
        "allow_patterns": allow_patterns,
    }
    if revision:
        kwargs["revision"] = revision
    return snapshot_download(model_path, **kwargs)


def _validate_checkpoint(model_dir, subfolder=None):
    root_config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(root_config_path):
        raise FileNotFoundError(
            "Laya support currently targets the root convaiinnovations/laya checkpoint"
        )
    with open(root_config_path) as f:
        root_config = json.load(f)
    if root_config.get("model_type") != "laya" or "LayaTypedDecisions" not in (
        root_config.get("architectures") or []
    ):
        raise ValueError("Expected a LayaTypedDecisions checkpoint bundle")
    checkpoint_dir = os.path.join(model_dir, subfolder) if subfolder else model_dir
    for name in ("rl_agent_config.json", "model.safetensors", "encoder", "tokenizer"):
        if not os.path.exists(os.path.join(checkpoint_dir, name)):
            raise FileNotFoundError(
                f"Laya checkpoint does not contain the required {name!r} artifact"
            )


def load(
    model_id_or_path="convaiinnovations/laya",
    dtype=None,
    token=None,
    subfolder=None,
    revision=None,
    expected_sha256=None,
    lang_temperatures=None,
    hooks=None,
    on_predict_start=None,
    on_predict_end=None,
    hooks_raise=True,
    hooks_concurrent=True,
    hooks_timeout=None,
    calibration=None,
):
    """Load a supported Laya Agent with its ModernBERT encoder on Spyre."""
    _require_laya()
    if subfolder not in (None, "typed-decisions", "multilingual"):
        raise ValueError(f"Unsupported Laya checkpoint subfolder: {subfolder!r}")
    if (
        not os.path.isdir(model_id_or_path)
        and os.fspath(model_id_or_path) not in _SUPPORTED_MODEL_IDS
    ):
        raise ValueError(f"Unsupported Laya checkpoint: {model_id_or_path!r}")
    model_dir = _resolve_model_dir(
        model_id_or_path,
        subfolder=subfolder,
        include_tokenizer=True,
        token=token,
        revision=revision,
    )
    _validate_checkpoint(model_dir, subfolder)

    from laya import load as load_laya

    from hf_adapters.auto_spyre_model import dtype_for_model_path

    agent = load_laya(
        model_dir,
        device="cpu",
        token=token,
        subfolder=subfolder,
        expected_sha256=expected_sha256,
        lang_temperatures=lang_temperatures,
        hooks=hooks,
        on_predict_start=on_predict_start,
        on_predict_end=on_predict_end,
        hooks_raise=hooks_raise,
        hooks_concurrent=hooks_concurrent,
        hooks_timeout=hooks_timeout,
        calibration=calibration,
        backend="eager",
    )
    if dtype is None:
        dtype = dtype_for_model_path(model_id_or_path, target_device=hf_common.DEVICE)
    hf_common.move_model_to_spyre(agent.model, sys.modules[__name__], dtype)
    return agent


def _pad_explicit_masks(masks, pad_amount, *, batch_size=None, sequence_length=None):
    padded = {}
    for layer_type, mask in masks.items():
        mask = mask.to("cpu")
        if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-2] != mask.shape[-1]:
            raise ValueError(
                "Laya attention masks must have shape [batch, 1, sequence, sequence]"
            )
        if mask.dtype != torch.bool:
            raise ValueError("Laya explicit attention masks must be boolean")
        if batch_size is not None and mask.shape[0] != batch_size:
            raise ValueError("Laya attention-mask batch size must match input_ids")
        if sequence_length is not None and mask.shape[-1] != sequence_length:
            raise ValueError("Laya attention-mask sequence size must match input_ids")
        if pad_amount:
            mask = F.pad(mask, (0, pad_amount, 0, pad_amount), value=False)
        padded[layer_type] = mask.to(hf_common.DEVICE)
    return padded


def _encoder_forward(
    encoder,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    **kwargs,
):
    if kwargs.get("inputs_embeds") is not None:
        raise ValueError(
            "Laya on Spyre requires input_ids; inputs_embeds is unsupported"
        )
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have non-empty [batch, sequence] shape")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to("cpu")
    bsz, seq_len = input_ids.shape
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    pad_amount = padded_len - seq_len
    if pad_amount:
        input_ids = F.pad(input_ids, (0, pad_amount), value=0)

    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long).expand(bsz, -1)
    else:
        position_ids = position_ids.to("cpu")
        if position_ids.shape != (bsz, seq_len):
            raise ValueError("position_ids must match input_ids shape")
    if pad_amount:
        position_ids = F.pad(position_ids, (0, pad_amount), value=0)

    dtype = get_model_dtype(encoder)
    if isinstance(attention_mask, dict):
        required_masks = set(encoder.config.layer_types)
        missing_masks = required_masks - attention_mask.keys()
        if missing_masks:
            raise ValueError(
                f"Laya attention masks are missing {sorted(missing_masks)}"
            )
        masks = _pad_explicit_masks(
            attention_mask,
            pad_amount,
            batch_size=bsz,
            sequence_length=seq_len,
        )
    else:
        attention_mask = attention_mask.to("cpu")
        if attention_mask.shape != (bsz, seq_len):
            raise ValueError("attention_mask must match input_ids shape")
        if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
            raise ValueError("Laya inputs must be right-padded")
        lengths = attention_mask.sum(dim=1)
        masks = hf_common.build_prefill_mask_right_padded(
            bsz,
            padded_len,
            lengths,
            is_causal=False,
            dtype=dtype,
        ).to(hf_common.DEVICE)

    hidden = hf_modernbert._run_backbone_forward(
        encoder,
        input_ids.to(hf_common.DEVICE),
        masks,
        position_ids.to(hf_common.DEVICE),
        None,
    )
    hidden = hidden[:, :seq_len].to("cpu")
    return BaseModelOutput(last_hidden_state=hidden)


def _split_decision_qkv(attn):
    """Replace packed MultiheadAttention QKV parameters with separate linears."""
    embed_dim = attn.embed_dim
    weights = attn.in_proj_weight.detach().split(embed_dim, dim=0)
    biases = (
        (None, None, None)
        if attn.in_proj_bias is None
        else attn.in_proj_bias.detach().split(embed_dim, dim=0)
    )

    projections = []
    for weight, bias in zip(weights, biases):
        projection = nn.Linear(
            embed_dim,
            embed_dim,
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        projection.weight = nn.Parameter(weight.clone(), requires_grad=False)
        if bias is not None:
            projection.bias = nn.Parameter(bias.clone(), requires_grad=False)
        projections.append(projection)

    attn._spyre_q_proj, attn._spyre_k_proj, attn._spyre_v_proj = projections

    def remove_split_projections(module, state_dict, prefix, local_metadata):
        del module, local_metadata
        for name in ("q", "k", "v"):
            state_dict.pop(f"{prefix}_spyre_{name}_proj.weight", None)
            state_dict.pop(f"{prefix}_spyre_{name}_proj.bias", None)

    attn.register_state_dict_post_hook(remove_split_projections)


def _make_compiled_decision_block(layer):
    """Compile one pre-norm Laya decision-transformer layer."""
    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    scale = head_dim**-0.5

    def block_forward(hidden_states, attn_mask):
        bsz, seq_len, _ = hidden_states.shape
        residual = hidden_states
        h = layer.norm1(hidden_states)
        q = (
            attn._spyre_q_proj(h)
            .reshape(bsz, seq_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            attn._spyre_k_proj(h)
            .reshape(bsz, seq_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            attn._spyre_v_proj(h)
            .reshape(bsz, seq_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(bsz, seq_len, -1)
        hidden_states = residual + attn.out_proj(attn_out)

        residual = hidden_states
        h = layer.norm2(hidden_states)
        hidden_states = residual + layer.linear2(layer.activation(layer.linear1(h)))
        return hidden_states

    return torch.compile(block_forward, dynamic=False)


def _decision_keep_mask(key_padding_mask, padded_len):
    """Convert a key-padding mask to an SDPA keep mask, including block padding."""
    key_padding_mask = key_padding_mask.to("cpu")
    if key_padding_mask.ndim != 2 or key_padding_mask.dtype != torch.bool:
        raise ValueError(
            "Laya decision key-padding mask must be boolean [batch, sequence]"
        )
    if key_padding_mask.shape[1] == 0 or padded_len < key_padding_mask.shape[1]:
        raise ValueError("Invalid Laya decision sequence length")
    if torch.any(key_padding_mask[:, :-1] & ~key_padding_mask[:, 1:]):
        raise ValueError("Laya decision inputs must be right-padded")

    keep = ~key_padding_mask
    if padded_len > keep.shape[1]:
        keep = F.pad(keep, (0, padded_len - keep.shape[1]), value=False)
    return keep[:, None, None, :].expand(-1, 1, padded_len, -1)


def _decision_layer_forward(
    layer,
    hidden_states,
    src_mask=None,
    src_key_padding_mask=None,
    is_causal=False,
):
    if layer.training or torch.is_grad_enabled():
        raise RuntimeError("Laya decision layers on Spyre support inference only")
    if src_mask is not None or is_causal:
        raise ValueError(
            "Laya decision layers support only bidirectional padding masks"
        )
    if src_key_padding_mask is None:
        raise ValueError("Laya decision layers require a key-padding mask")

    logical_len = src_key_padding_mask.shape[1]
    if hidden_states.shape[1] == logical_len:
        padded_len = math.ceil(logical_len / BLOCK_SIZE) * BLOCK_SIZE
        if padded_len > logical_len:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padded_len - logical_len))
        hidden_states = hidden_states.to(hf_common.DEVICE).clone()
    else:
        padded_len = hidden_states.shape[1]

    attn_mask = _decision_keep_mask(src_key_padding_mask, padded_len).to(
        hf_common.DEVICE
    )
    hidden_states = layer._spyre_compiled_blocks[0](hidden_states, attn_mask).clone()
    if layer._spyre_is_last:
        hidden_states = hidden_states[:, :logical_len].to("cpu")
    return hidden_states


def _decision_forward(
    model,
    input_ids,
    attention_mask,
    marker_pos,
    marker_mask,
    qtype,
    detach_encoder=False,
    position_ids=None,
    option_ids=None,
):
    if position_ids is not None and option_ids is None:
        option_ids = torch.zeros_like(position_ids)

    return model._spyre_original_forward(
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        detach_encoder=detach_encoder,
        position_ids=position_ids,
        option_ids=option_ids,
    )


def prepare_for_spyre(model):
    """Prepare Laya's nested ModernBERT encoder and preserve its CPU decision heads."""
    if hasattr(model, "_spyre_original_forward"):
        raise RuntimeError("Laya model is already prepared for Spyre")
    if not isinstance(model.encoder, ModernBertModel):
        raise TypeError(
            "Laya support currently requires model.encoder to be ModernBertModel, got "
            f"{type(model.encoder).__name__}"
        )

    hf_common.assert_spyre_dimensions(model.encoder.config, model_name="Laya encoder")
    hf_modernbert.prepare_for_spyre(model.encoder)
    model._spyre_rope = model.encoder._spyre_rope
    model.config = model.encoder.config
    model.encoder.forward = MethodType(_encoder_forward, model.encoder)

    if model.head is not None:
        layers = list(model.head.layers)
        for layer in layers:
            if layer.self_attn.head_dim != BLOCK_SIZE:
                raise ValueError(
                    "Laya decision attention requires a 64-element head dimension, got "
                    f"{layer.self_attn.head_dim}"
                )
            _split_decision_qkv(layer.self_attn)
        for index, layer in enumerate(layers):
            layer._spyre_compiled_blocks = [_make_compiled_decision_block(layer)]
            layer._spyre_is_last = index == len(layers) - 1
            layer.forward = MethodType(_decision_layer_forward, layer)

    model._spyre_original_forward = model.forward
    model.forward = MethodType(_decision_forward, model)
    model._spyre_cpu_submodules = [
        name
        for name in ("type_emb", "scorer", "act_head")
        if getattr(model, name, None) is not None
    ]


_is_encoder_only = True
