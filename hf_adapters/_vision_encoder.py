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

"""Shared dual-encoder transformer preparation and execution on Spyre."""

from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextConfig,
    CLIPVisionConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPEncoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    make_vision_encoder_block,
    pad_attention_heads_linear,
    pad_encoder_mlp,
)

Encoder = CLIPEncoder | SiglipEncoder
EncoderConfig = (
    CLIPTextConfig | CLIPVisionConfig | SiglipTextConfig | SiglipVisionConfig
)
CompiledBlock = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


def prepare_encoder(
    encoder: Encoder, config: EncoderConfig
) -> tuple[list[CompiledBlock], int]:
    """Pad heads/MLPs and compile pre-LN blocks with the original SDPA scale."""
    num_heads = config.num_attention_heads
    original_head_dim: int = getattr(config, "head_dim", None) or (
        config.hidden_size // num_heads
    )
    head_dim = ((original_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    layers: list[CLIPEncoderLayer | SiglipEncoderLayer] = []
    for layer in encoder.layers:
        if not isinstance(layer, (CLIPEncoderLayer, SiglipEncoderLayer)):
            raise TypeError(f"Unsupported encoder layer: {type(layer).__name__}")
        layers.append(layer)

    if head_dim > original_head_dim:
        pad_attention_heads_linear(
            encoder,
            (layer.self_attn for layer in layers),
            original_head_dim,
            head_dim,
            num_heads,
        )
    intermediate_size = config.intermediate_size
    padded_intermediate = (
        (intermediate_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    ) * BLOCK_SIZE
    if padded_intermediate > intermediate_size:
        pad_encoder_mlp(layers, intermediate_size, padded_intermediate)

    blocks: list[CompiledBlock] = [
        make_vision_encoder_block(
            q_proj=layer.self_attn.q_proj,
            k_proj=layer.self_attn.k_proj,
            v_proj=layer.self_attn.v_proj,
            o_proj=layer.self_attn.out_proj,
            layer_norm1=layer.layer_norm1,
            layer_norm2=layer.layer_norm2,
            ffn_in=layer.mlp.fc1,
            act=layer.mlp.activation_fn,
            ffn_out=layer.mlp.fc2,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=original_head_dim**-0.5,
        )
        for layer in layers
    ]
    return blocks, head_dim


def patch_encoder(encoder: Encoder, blocks: list[CompiledBlock]) -> None:
    """Install a padded block loop, retaining HF's explicit/implicit causal mask."""

    def encoder_forward(
        self: Encoder,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        if kwargs.get("output_attentions", False):
            raise ValueError("Spyre encoder blocks do not return attention weights")
        if kwargs.get("output_hidden_states", False):
            raise ValueError(
                "Spyre encoder blocks do not return intermediate hidden states"
            )
        seq_len = inputs_embeds.shape[1]
        mask = attention_mask.cpu() if attention_mask is not None else None
        # HF's SDPA path can omit a causal text mask and rely on is_causal.
        # The compiled block uses is_causal=False, so materialize it here.
        if mask is None and kwargs.get("is_causal", False):
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool).tril()[None, None]
        hidden = inputs_embeds.to(DEVICE)
        pad_len = (-seq_len) % BLOCK_SIZE
        if pad_len:
            hidden = F.pad(hidden, (0, 0, 0, pad_len))
            if mask is None:
                mask = torch.ones((1, 1, 1, seq_len), dtype=torch.bool)
            # Added queries may attend to real keys, avoiding all-masked rows.
            # Added keys must remain inaccessible to every query.
            if mask.shape[-2] == seq_len:
                allowed: bool | float = True if mask.dtype == torch.bool else 0.0
                mask = F.pad(mask, (0, 0, 0, pad_len), value=allowed)
            blocked: bool | float = False if mask.dtype == torch.bool else float("-inf")
            mask = F.pad(mask, (0, pad_len), value=blocked)
        if mask is not None:
            mask = mask.to(DEVICE)
        hidden = hidden.clone()
        for block in blocks:
            hidden = block(hidden, mask)
            if hidden.device.type == "spyre":
                hidden = hidden.clone()
        # Crop on CPU, where HF also performs pooling and final normalization.
        output = BaseModelOutput()
        output["last_hidden_state"] = hidden.cpu()[:, :seq_len]
        return output

    setattr(encoder, "_spyre_compiled_blocks", blocks)
    setattr(encoder, "forward", MethodType(encoder_forward, encoder))


def patch_cpu_inputs(module: torch.nn.Module, input_names: tuple[str, ...]) -> None:
    """Keep named tensor inputs on CPU before HF embeddings/mask construction.

    Names follow the original forward's positional argument order. Other
    arguments, including interpolation and output flags, pass through unchanged.
    """
    original_forward = module.forward

    @wraps(type(module).forward)
    def forward(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        cpu_args = tuple(
            (
                value.cpu()
                if index < len(input_names) and isinstance(value, torch.Tensor)
                else value
            )
            for index, value in enumerate(args)
        )
        cpu_kwargs = {
            name: (
                value.cpu()
                if name in input_names and isinstance(value, torch.Tensor)
                else value
            )
            for name, value in kwargs.items()
        }
        return original_forward(*cpu_args, **cpu_kwargs)

    module.forward = MethodType(forward, module)
