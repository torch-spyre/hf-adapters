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
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import (
    AutoModelForImageTextToText,
    GenerationConfig,
    MuseGlimmerTextConfig,
    PreTrainedModel,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from hf_adapters import hf_gemma3
from hf_adapters.hf_common import (
    NormalizedGenerationInputs,
    PrecomputedRotaryEmbedding,
    SpyreUnsupportedModelError,
    _projection_head_count,
    allocate_kv_cache_tensor,
    get_backbone,
    kv_cache_shapes,
    load_model_common,
    normalize_generation_inputs,
    prepare_lm_head_for_spyre,
    text_config,
    untie_embedding_and_lm_head,
)
from hf_adapters.hf_common import (
    generate as generate_common,
)
from hf_adapters.swa_attention import allocate_swa_caches

if TYPE_CHECKING:
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerTextAttention,
        MuseGlimmerTextModel,
    )

_run_backbone_forward = hf_gemma3._run_backbone_forward
_run_forward = hf_gemma3._run_forward

_PREFILL_CHUNK_SIZE = 512


def _embed_tokens(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Match layer zero's input layout to the decoder's linear output layout.

    Muse's fp32 norm arithmetic leaves decoder outputs in [batch, hidden-stick,
    sequence, stick] order. The generic embedding layout uses a different order
    and would compile a separate first-layer graph for every prefill shape.
    """
    backbone: MuseGlimmerTextModel = get_backbone(model)
    hidden_states: torch.Tensor = backbone.embed_tokens(
        input_ids.to(backbone.embed_tokens.weight.device)
    )
    if hidden_states.device.type != "spyre":
        return hidden_states
    from torch_spyre._C import SpyreTensorLayout

    target = SpyreTensorLayout(
        list(hidden_states.shape),
        list(hidden_states.stride()),
        hidden_states.dtype,
        [1, 0, 2],
    )
    result: torch.Tensor = hidden_states.to(device_layout=target)  # type: ignore[call-overload]
    return result


def generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    generation_config: GenerationConfig | None = None,
    **kwargs: Any,
) -> torch.Tensor | GenerateDecoderOnlyOutput:
    """Bound long-prompt compilation while preserving short-prompt shapes.

    Explicit chunk kwargs (including None) take precedence over generation
    config and this default. Use normalized prompt length so caller padding
    cannot turn a short request into a 512-token prefill.
    """
    cfg: GenerationConfig = (
        generation_config if generation_config is not None else model.generation_config
    )
    if (
        "prefill_chunk_size" not in kwargs
        and getattr(cfg, "prefill_chunk_size", None) is None
    ):
        normalized: NormalizedGenerationInputs = normalize_generation_inputs(
            input_ids, attention_mask
        )
        if normalized.padded_len > _PREFILL_CHUNK_SIZE:
            kwargs["prefill_chunk_size"] = _PREFILL_CHUNK_SIZE
    return cast(
        torch.Tensor | GenerateDecoderOnlyOutput,
        generate_common(
            _run_forward,
            model,
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            prefill_backbone_fn=_run_backbone_forward,
            **kwargs,
        ),
    )


def _allocate_caches(
    model: PreTrainedModel,
    batch_size: int,
    max_cache_len: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Select the decoder and cache together for each generation request."""
    cfg: MuseGlimmerTextConfig = text_config(model.config)
    assert cfg.sliding_window is not None
    if max_cache_len > cfg.sliding_window:
        setattr(model, "_spyre_swa_mode", "anchored")
        setattr(model, "_spyre_compiled_blocks", model._spyre_muse_swa_blocks)
        return cast(
            tuple[list[torch.Tensor], list[torch.Tensor]],
            allocate_swa_caches(model, batch_size, max_cache_len, dtype, device),
        )

    # The complete generation fits in the window. Plain SDPA avoids allocating
    # a full sliding window for each layer.
    setattr(model, "_spyre_swa_mode", None)
    setattr(model, "_spyre_compiled_blocks", model._spyre_muse_full_blocks)
    shapes: list[tuple[int, int, int]] = kv_cache_shapes(model)
    keys: list[torch.Tensor] = [
        allocate_kv_cache_tensor(batch_size, heads, max_cache_len, dim, dtype, device)
        for heads, dim, _ in shapes
    ]
    values: list[torch.Tensor] = [
        allocate_kv_cache_tensor(batch_size, heads, max_cache_len, dim, dtype, device)
        for heads, _, dim in shapes
    ]
    return keys, values


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

    # Keep stock Muse norms: their fp32 accumulation and (1 + weight) scaling
    # are lowered inside compiled blocks without a class-wide Gemma patch.

    # The stock multimodal wrapper embeds image/video placeholder IDs as token
    # zero even without pixels. Copy those input rows once rather than remapping
    # IDs across host/device on every decode. Keep the output head unchanged.
    if backbone.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
        untie_embedding_and_lm_head(model)
    with torch.no_grad():
        for token_id in (model.config.image_token_id, model.config.video_token_id):
            backbone.embed_tokens.weight[token_id].copy_(
                backbone.embed_tokens.weight[0]
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
    setattr(model, "_spyre_embed_tokens", _embed_tokens)

    output_multiplier: float = cfg.output_multiplier
    cap: float = cfg.final_logit_softcapping

    def process_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = logits * output_multiplier
        return torch.tanh(logits / cap) * cap

    prepare_lm_head_for_spyre(model, logits_processor=process_logits)

    swa_mode: str | None = getattr(model, "_spyre_swa_mode", "anchored")
    setattr(model, "_spyre_swa_mode", swa_mode)
    if swa_mode == "anchored":
        setattr(model, "_spyre_cache_allocator", _allocate_caches)

    kv_shapes: list[tuple[int, int, int]] = []
    blocks: list[Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    full_blocks: list[
        Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []
    for i, layer in enumerate(backbone.layers):
        attn = cast("MuseGlimmerTextAttention", layer.get_submodule("self_attn"))
        num_q_heads: int = _projection_head_count(
            attn.q_proj, cfg.num_attention_heads, head_dim
        )
        num_kv_heads: int = _projection_head_count(
            attn.k_proj, cfg.num_key_value_heads, head_dim
        )
        kv_shapes.append((num_kv_heads, head_dim, head_dim))
        make_block = partial(
            hf_gemma3._make_compiled_block,
            layer,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_sliding=cfg.layer_types[i] == "sliding_attention",
            window_size=cfg.sliding_window,
            query_norm=attn.qk_norm,
            key_norm=attn.qk_norm,
            query_scale=attn.qk_scale_factor,
            output_gate=attn.gate_proj,
            use_rope=bool(cfg.layer_rope_theta[i]),
        )
        full_block: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            make_block(swa_mode=None)
        )
        full_blocks.append(full_block)
        blocks.append(
            make_block(swa_mode=swa_mode)
            if swa_mode is not None and cfg.layer_types[i] == "sliding_attention"
            else full_block
        )
    setattr(model, "_spyre_kv_shapes", kv_shapes)
    setattr(model, "_spyre_compiled_blocks", blocks)
    setattr(model, "_spyre_muse_swa_blocks", blocks)
    setattr(model, "_spyre_muse_full_blocks", full_blocks)
