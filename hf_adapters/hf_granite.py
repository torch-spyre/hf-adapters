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
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

import torch

from hf_adapters.hf_common import (
    _SDPA_MAX_SEQUENCE_TILE_SIZE,
    PaddedGatedMLP,
    embed_text_tokens,
    get_backbone,
    prepare_lm_head_for_spyre,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
    run_lm_head,
    text_config,
)


def _can_pad_mlp(mlp):
    """Only widen the standard inference MLP with local TP channel semantics."""
    from transformers.activations import SiLUActivation
    from transformers.models.granite.modeling_granite import GraniteMLP

    if (
        type(mlp) is not GraniteMLP
        or mlp.training
        or type(mlp.act_fn) not in (torch.nn.SiLU, SiLUActivation)
        or "forward" in mlp.__dict__
        or mlp._forward_pre_hooks
        or mlp._forward_hooks
        or mlp._backward_hooks
        or "forward" in mlp.act_fn.__dict__
        or mlp.act_fn._forward_pre_hooks
        or mlp.act_fn._forward_hooks
    ):
        return False
    for name, role in (
        ("gate_proj", "colwise"),
        ("up_proj", "colwise"),
        ("down_proj", "rowwise"),
    ):
        proj = getattr(mlp, name)
        if (
            type(proj) is not torch.nn.Linear
            or proj.weight.dtype not in (torch.float16, torch.bfloat16)
            or "forward" in proj.__dict__
        ):
            return False
        # Unhooked projections are local. Hooked projections need the exact
        # supported Spyre style, with no extra hooks that could change channels.
        if proj._forward_pre_hooks or proj._forward_hooks:
            if (
                getattr(proj, "_spyre_mlp_width_role", None) != role
                or len(proj._forward_pre_hooks) != 1
                or len(proj._forward_hooks) != 1
                or getattr(proj, "_spyre_mlp_width_hooks", None)
                != (
                    tuple(proj._forward_pre_hooks.items()),
                    tuple(proj._forward_hooks.items()),
                )
            ):
                return False
    width = mlp.down_proj.weight.shape[1]
    return (
        mlp.gate_proj.weight.shape[0] == mlp.up_proj.weight.shape[0] == width
        and width % 256 != 0
    )


def _can_force_mlp(mlp, config):
    """Use only the checked Granite 8B, four-rank, FP16 weight shapes."""
    if (
        getattr(config, "model_type", None) != "granite"
        or config.hidden_size != 4096
        or config.intermediate_size != 12800
        or config.num_hidden_layers != 40
    ):
        return False
    projections = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    mesh = getattr(mlp.gate_proj, "_spyre_mlp_width_mesh", None)
    if mesh is None or mesh.ndim != 1 or mesh.size() != 4:
        return False
    return all(
        proj.weight.dtype == torch.float16
        and proj.bias is None
        and tuple(proj.weight.shape) == shape
        and getattr(proj, "_spyre_mlp_width_mesh", None) is mesh
        for proj, shape in zip(projections, ((3200, 4096), (3200, 4096), (4096, 3200)))
    )


class _GraniteTP4PrefillMLP(PaddedGatedMLP):
    """Keep the checked prefill work split across the complete MLP chain."""

    def _use_forced_division(self, x):
        return (
            not self.training
            and x.device.type == "spyre"
            and x.dtype == torch.float16
            and tuple(x.shape) == (1, 512, 4096)
        )

    def forward(self, x):
        if not self._use_forced_division(x):
            return super().forward(x)

        from torch_spyre._inductor.propagate_hints import spyre_hint

        with spyre_hint(
            named_dims=[
                "granite_mlp_batch",
                "granite_mlp_rows",
                "granite_mlp_channels",
            ],
            work_div={"granite_mlp_rows": 8, "granite_mlp_channels": 4},
        ):
            product = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Keep the 3200 real channels and let the compiler place the input.
        with spyre_hint(
            named_dims=["granite_mlp_batch", "granite_mlp_rows", "granite_mlp_hidden"],
            work_div={"granite_mlp_rows": 8, "granite_mlp_hidden": 4},
        ):
            return self.down_proj(product[..., : self.logical_width])


def _prepare_mlp(mlp, config):
    if not _can_pad_mlp(mlp):
        return mlp
    cls = _GraniteTP4PrefillMLP if _can_force_mlp(mlp, config) else PaddedGatedMLP
    return cls(mlp, 256)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm."""
    h = embed_text_tokens(model, input_ids)

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    h = model._spyre_compiled_norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    return run_lm_head(model, h)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    prepare_rope_and_heads(model)
    logits_scaling = text_config(model.config).logits_scaling
    prepare_lm_head_for_spyre(
        model, logits_processor=lambda logits: logits / logits_scaling
    )
    backbone = get_backbone(model)
    # Four fp16/bf16 output sticks balance the 32-core prefill projection.
    # Apply after TP loading: each rank preserves its original channel prefix.
    for layer in backbone.layers:
        layer.mlp = _prepare_mlp(layer.mlp, text_config(model.config))
    model._spyre_compiled_blocks = prepare_standard_gqa_blocks(backbone.layers, True)
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
    model._spyre_prefill_chunk_size = _SDPA_MAX_SEQUENCE_TILE_SIZE
