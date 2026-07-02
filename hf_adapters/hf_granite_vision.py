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
HuggingFace Transformers adapter for the text backbone of Granite Vision models.

Loads the stock Granite Vision VLM (no trust_remote_code) and runs *text-only*
causal inference over its Granite text decoder. The SigLIP vision tower and the
projection layers are loaded but unused — for the full image→text pipeline see
``hf_granite_vision_mm``.

The text decoder is architecturally identical to Granite 3.3, so the compiled
block and backbone forward are reused from ``hf_granite``. The only differences
from the bare-Granite causal adapter are checkpoint-shape details that the
shared helpers already handle (``get_backbone`` descends to
``model.model.language_model``; ``text_config`` reads the nested text dims):
this adapter just patches the VLM's own ``Granite4VisionTextRMSNorm`` and reads
``logits_scaling`` from the nested ``text_config``.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "ibm-granite/granite-vision-4.1-4b")
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-vision-4.1-4b")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch

from hf_adapters.hf_common import (
    get_backbone,
    load_model_common,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    text_config,
)
from hf_adapters.hf_granite import _make_compiled_block, _run_backbone_forward


def load_hf_model(model_path, dtype=torch.float16):
    """Load the stock Granite Vision VLM (text-only reference for the harness).

    Returns the ``Granite4VisionForConditionalGeneration`` with the vision tower
    and projectors dropped; its text-only ``forward(input_ids=...)`` (no
    ``pixel_values``) gives the causal-LM reference logits used by the CPU
    accuracy test, and the dropped modules aren't dragged onto Spyre by the
    layout move in ``load_model``.
    """
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="cpu"
    )
    # Drop the SigLIP vision tower and the deepstack/spatial projectors — text-only
    # inference (the full image→text pipeline lives in hf_granite_vision_mm).
    if hasattr(model, "model"):
        mm = model.model
        for attr in ("vision_tower", "layerwise_projectors", "spatial_projectors"):
            if hasattr(mm, attr):
                delattr(mm, attr)
    model.eval()
    model.requires_grad_(False)
    return model


def prepare_for_spyre(model):
    """Apply Spyre adaptations to the Granite Vision text decoder in-place.

    Mirrors ``hf_granite.prepare_for_spyre`` against the VLM's nested text
    backbone: the shared RoPE/head prep and LM-head padding descend via
    ``get_backbone``/``text_config`` on their own, so the only VLM-specific step
    is patching the decoder's ``Granite4VisionTextRMSNorm`` (not Granite 3.3's
    ``GraniteRMSNorm``). The vision tower is left untouched — this is the
    text-only path.
    """
    from transformers.models.granite4_vision.modeling_granite4_vision import (
        Granite4VisionTextRMSNorm,
    )

    prepare_rope_and_heads(model)
    patch_rmsnorm(Granite4VisionTextRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in get_backbone(model).layers
    ]


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
    """Granite Vision text causal-LM forward: backbone + head / scaling.

    Identical to ``hf_granite._run_forward`` except ``logits_scaling`` lives on
    the nested ``text_config`` rather than the top-level VLM config.
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
    return logits / text_config(model.config).logits_scaling


def load_model(model_path, dtype=torch.float16):
    """Load Granite Vision text backbone for Spyre."""
    from transformers import AutoModelForImageTextToText

    return load_model_common(
        model_path,
        prepare_for_spyre,
        dtype,
        auto_model_cls=AutoModelForImageTextToText,
    )
