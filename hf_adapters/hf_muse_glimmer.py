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

"""Text-generation adapter for Muse Glimmer's multimodal checkpoints.

Reuses Gemma 3's sandwich-norm blocks and sliding/global cache driver. Muse
Glimmer normalizes embeddings and Q/K without learned scales, multiplies Q,
gates attention before the output projection, and disables RoPE on NoPE
layers. The stock vision encoder and projector are dropped for this causal-LM
path; image inputs require a separate multimodal adapter.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from transformers import (
    AutoModelForImageTextToText,
    MuseGlimmerTextConfig,
    PreTrainedModel,
)

from hf_adapters import hf_gemma3
from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    SpyreUnsupportedModelError,
    _projection_head_count,
    get_backbone,
    load_model_common,
    prepare_lm_head_for_spyre,
    text_config,
)
from hf_adapters.swa_attention import allocate_swa_caches

if TYPE_CHECKING:
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerTextAttention,
        MuseGlimmerTextModel,
    )

_run_backbone_forward = hf_gemma3._run_backbone_forward
_run_forward = hf_gemma3._run_forward


def load_hf_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    tp_plan: dict[str, str] | str | None = None,
    trust_remote_code: bool | None = None,
) -> PreTrainedModel:
    """Load the stock conditional model, retaining only its text path."""
    model: PreTrainedModel = load_model_common(
        model_path,
        module=None,
        dtype=dtype,
        auto_model_cls=AutoModelForImageTextToText,
        tp_plan=tp_plan,
        trust_remote_code=trust_remote_code,
    )
    del model.model.vision_tower
    del model.model.vision_adapter
    del model.model.vision_projection
    del model.model.perception_emb_norm
    return model


def prepare_for_spyre(model: PreTrainedModel) -> None:
    """Prepare Muse Glimmer's text backbone using the shared Gemma 3 path."""
    backbone: MuseGlimmerTextModel = get_backbone(model)
    cfg: MuseGlimmerTextConfig = text_config(model.config)
    assert cfg.layer_types is not None and cfg.layer_rope_theta is not None
    head_dim: int = cfg.head_dim
    if head_dim % 2 or head_dim // 2 < 64:
        raise SpyreUnsupportedModelError(
            f"Muse Glimmer head_dim={head_dim} requires an even head dimension "
            "of at least 128 for Spyre RoPE."
        )

    # Muse's centered norm is the same (1 + weight) RMSNorm as Gemma 3.
    hf_gemma3._patch_gemma_rmsnorm(
        type(backbone.layers[0].get_submodule("input_layernorm"))
    )

    # HF uses a single rotary embedding and switches it off for NoPE layers.
    # Both attention types share that table; each block decides whether to use it.
    rope = PrecomputedRotaryEmbedding(backbone.rotary_emb)
    setattr(model, "_spyre_rope", dict.fromkeys(cfg.layer_types, rope))

    # Compile the weightless embedding norm and final standard RMSNorm so their
    # fp32 reductions are lowered inside compiled regions on Spyre.
    setattr(
        backbone.embed_tokens,
        "embed_norm",
        torch.compile(backbone.embed_tokens.embed_norm, dynamic=False),
    )
    setattr(backbone, "norm", torch.compile(backbone.norm, dynamic=False))
    setattr(model, "_spyre_compiled_norm", backbone.norm)

    output_multiplier: float = cfg.output_multiplier
    cap: float = cfg.final_logit_softcapping

    def process_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = logits * output_multiplier
        return torch.tanh(logits / cap) * cap

    prepare_lm_head_for_spyre(model, logits_processor=process_logits)

    swa_mode: str | None = getattr(model, "_spyre_swa_mode", "anchored")
    setattr(model, "_spyre_swa_mode", swa_mode)
    if swa_mode == "anchored":
        setattr(model, "_spyre_cache_allocator", allocate_swa_caches)

    kv_shapes: list[tuple[int, int, int]] = []
    blocks: list[Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    for i, layer in enumerate(backbone.layers):
        attn = cast("MuseGlimmerTextAttention", layer.get_submodule("self_attn"))
        num_q_heads: int = _projection_head_count(
            attn.q_proj, cfg.num_attention_heads, head_dim
        )
        num_kv_heads: int = _projection_head_count(
            attn.k_proj, cfg.num_key_value_heads, head_dim
        )
        kv_shapes.append((num_kv_heads, head_dim, head_dim))
        blocks.append(
            hf_gemma3._make_compiled_block(
                layer,
                num_q_heads,
                num_kv_heads,
                head_dim,
                is_sliding=cfg.layer_types[i] == "sliding_attention",
                window_size=cfg.sliding_window,
                swa_mode=swa_mode,
                query_norm=attn.qk_norm,
                key_norm=attn.qk_norm,
                query_scale=attn.qk_scale_factor,
                output_gate=attn.gate_proj,
                use_rope=bool(cfg.layer_rope_theta[i]),
            )
        )
    setattr(model, "_spyre_kv_shapes", kv_shapes)
    setattr(model, "_spyre_compiled_blocks", blocks)
