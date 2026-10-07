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
runs on Spyre through :mod:`hf_modernbert`; the decision transformer and scoring
heads remain on CPU. The upstream ``laya`` package is optional and supplies the
model class and checkpoint construction logic.
"""

from __future__ import annotations

import json
import math
import os
from importlib.metadata import PackageNotFoundError, version
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

import hf_adapters.hf_common as hf_common
from hf_adapters import hf_modernbert
from hf_adapters.hf_common import BLOCK_SIZE, get_model_dtype

_MIN_LAYA_VERSION = "0.3.28"
_MAX_LAYA_VERSION = "0.4"
_SUPPORTED_MODEL_TYPE = "modernbert"


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


def _resolve_model_dir(model_path):
    model_path = os.fspath(model_path)
    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download
    from laya.revisions import resolve_revision

    revision = resolve_revision(model_path, None)
    kwargs = {
        "token": os.environ.get("HF_TOKEN") or None,
        "allow_patterns": [
            "config.json",
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/*",
        ],
    }
    if revision:
        kwargs["revision"] = revision
    return snapshot_download(model_path, **kwargs)


def load_hf_model(model_path, dtype, trust_remote_code=None):
    """Construct the upstream Laya model and strictly load its root checkpoint."""
    del trust_remote_code

    _require_laya()

    from laya.agent import _verify_compatibility
    from laya.common import build_model, uses_parallel_layout

    model_dir = _resolve_model_dir(model_path)

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
        raise ValueError(
            "Laya support currently targets a root LayaTypedDecisions checkpoint"
        )

    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    encoder_dir = os.path.join(model_dir, "encoder")
    if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"{model_path!r} is not a Laya checkpoint: expected "
            "rl_agent_config.json and model.safetensors"
        )
    if not os.path.isdir(encoder_dir):
        raise FileNotFoundError(
            f"{model_path!r} does not contain the required encoder/ subfolder"
        )

    with open(cfg_path) as f:
        cfg = json.load(f)
    uses_parallel_layout(cfg)

    with open(os.path.join(encoder_dir, "config.json")) as f:
        encoder_cfg = json.load(f)
    if encoder_cfg.get("model_type") != _SUPPORTED_MODEL_TYPE:
        raise ValueError(
            "Laya support currently requires a ModernBERT encoder, got "
            f"{encoder_cfg.get('model_type')!r}"
        )

    from safetensors.torch import load_file

    model = build_model(cfg, encoder_dir=encoder_dir, pretrained=False)
    weights = load_file(weights_path)
    _verify_compatibility(model, cfg, weights, os.fspath(model_path))
    model.load_state_dict(weights, strict=True)
    model.to(device="cpu", dtype=dtype)
    model._spyre_laya_config = cfg
    return model


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


def _decision_forward(model, *args, **kwargs):
    if kwargs.get("position_ids") is not None and kwargs.get("option_ids") is None:
        kwargs["option_ids"] = torch.zeros_like(kwargs["position_ids"])
    return model._spyre_original_forward(*args, **kwargs)


def prepare_for_spyre(model):
    """Prepare Laya's nested ModernBERT encoder and preserve its CPU decision heads."""
    if not isinstance(model.encoder, ModernBertModel):
        raise TypeError(
            "Laya support currently requires model.encoder to be ModernBertModel, got "
            f"{type(model.encoder).__name__}"
        )

    hf_common.assert_spyre_dimensions(model.encoder.config, model_name="Laya encoder")
    hf_modernbert.prepare_for_spyre(model.encoder)
    model.encoder.forward = MethodType(_encoder_forward, model.encoder)

    model._spyre_original_forward = model.forward
    model.forward = MethodType(_decision_forward, model)
    model._spyre_cpu_submodules = [
        name
        for name in ("head", "type_emb", "scorer", "act_head")
        if getattr(model, name, None) is not None
    ]


_is_encoder_only = True
