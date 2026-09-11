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

"""HuggingFace Transformers adapter for OPT causal language models on Spyre."""

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    SpyreUnsupportedModelError,
    get_backbone,
    make_decoder_block,
    pad_attention_heads_linear,
    pad_lm_head,
    patch_new_gelu,
)


def _make_compiled_block(layer):
    attn = layer.self_attn
    return make_decoder_block(
        q_proj=attn.q_proj,
        k_proj=attn.k_proj,
        v_proj=attn.v_proj,
        o_proj=attn.out_proj,
        attn_ln=layer.self_attn_layer_norm,
        ffn_in=layer.fc1,
        act=layer.activation_fn,
        ffn_out=layer.fc2,
        ffn_ln=layer.final_layer_norm,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        scale=1.0,
        pre_ln=layer.do_layer_norm_before,
        query_scale=attn.scaling,
    )


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    decoder = get_backbone(model).decoder
    h = decoder.embed_tokens(input_ids)
    if decoder.project_in is not None:
        h = decoder.project_in(h)
    h = h + decoder.embed_positions(None, position_ids=position_ids).to(h.device)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            None,
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    if decoder.final_layer_norm is not None:
        h = decoder.final_layer_norm(h)
    if decoder.project_out is not None:
        h = decoder.project_out(h)
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
    return logits[..., : model.config.vocab_size]


def prepare_for_spyre(model):
    cfg = model.config
    model_name = getattr(cfg, "name_or_path", "") or "opt"
    if cfg.ffn_dim % BLOCK_SIZE != 0:
        raise SpyreUnsupportedModelError(
            f"Model {model_name} has Spyre-incompatible dimensions: "
            f"ffn_dim={cfg.ffn_dim} (not a multiple of one stick, {BLOCK_SIZE})."
        )

    decoder = get_backbone(model).decoder
    layers = decoder.layers

    if cfg.activation_function == "gelu_new":
        patch_new_gelu(type(layers[0].activation_fn))

    num_heads = cfg.num_attention_heads
    orig_head_dim = cfg.hidden_size // num_heads
    padded_head_dim = ((orig_head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_head_dim > orig_head_dim:
        pad_attention_heads_linear(
            model,
            (layer.self_attn for layer in layers),
            orig_head_dim,
            padded_head_dim,
            num_heads,
        )

    pad_lm_head(model)
    model._spyre_kv_shapes = [
        (num_heads, padded_head_dim, padded_head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    model._spyre_compiled_blocks = [_make_compiled_block(layer) for layer in layers]
