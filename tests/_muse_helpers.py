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

"""A tiny random Muse Glimmer target + DFlash assistant for the CPU lane.

The target keeps the real layer pattern (``[sliding x 3, full NoPE]``
repeated) and a short sliding window, so a few hundred tokens cross it; the
assistant reads taps from the same layers as the 52-layer original reads
from (every 12th, starting at 1), scaled down.
"""

import copy

import torch
from transformers import (
    MuseGlimmerAssistantConfig,
    MuseGlimmerAssistantModel,
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
)

from hf_adapters import hf_common, hf_muse_glimmer

VOCAB, HIDDEN, LAYERS, WINDOW = 320, 256, 8, 48
TAPS = [1, 4, 7]
MASK_ID = 5


def eager(monkeypatch, *modules):
    """Make ``torch.compile`` the identity in ``modules`` (CPU, same math)."""
    for module in (hf_muse_glimmer, *modules):
        monkeypatch.setattr(module.torch, "compile", lambda fn=None, **_: fn)


def tiny_target(seed=0, heads=2, kv_heads=1, intermediate=384, window=WINDOW):
    torch.random.default_generator.manual_seed(seed)
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": intermediate,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": 128,
            "max_position_embeddings": 4096,
            "sliding_window": window,
            "rope_parameters": {"rope_type": "default", "rope_theta": 5e5},
            "bos_token_id": 1,
            "eos_token_id": None,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "patch_temporal": 2,
            "merge_size": 2,
            "layer_types": ["window_attention", "full_attention"],
        },
        image_token_id=VOCAB - 2,
        video_token_id=VOCAB - 1,
        out_hidden_size=512,
        projector_hidden_size=128,
    )
    model = MuseGlimmerForConditionalGeneration(config).eval()
    # Random norm weights, so the (1 + w) / w conventions are exercised.
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                offset = 1.0 if "language_model.norm" in name else 0.0
                p.copy_(torch.randn(p.shape, generator=g) * 0.2 + offset)
    return model


def text_pattern_ok(model):
    cfg = model.config.text_config
    return (
        cfg.layer_types
        == ["sliding_attention"] * 3
        + ["full_attention"]
        + ["sliding_attention"] * 3
        + ["full_attention"]
        and [bool(t) for t in cfg.layer_rope_theta]
        == [
            True,
            True,
            True,
            False,
        ]
        * 2
    )


def tiny_assistant(seed=0, heads=2, kv_heads=1):
    torch.random.default_generator.manual_seed(seed + 7)
    config = MuseGlimmerAssistantConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,  # 128-aligned, as fp8 drafter linears need
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=128,
        sliding_window=WINDOW,
        rope_parameters={"rope_type": "default", "rope_theta": 5e5},
        block_size=8,
        mask_token_id=MASK_ID,
        target_layer_ids=TAPS,
    )
    model = MuseGlimmerAssistantModel(config).eval()
    g = torch.Generator().manual_seed(seed + 8)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1:
                p.copy_(torch.randn(p.shape, generator=g) * 0.06)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2 + 1.0)
    return model


def text_only(model):
    """What ``hf_muse_glimmer.load_hf_model`` keeps: the text path only."""
    for name in (
        "vision_tower",
        "vision_adapter",
        "vision_projection",
        "perception_emb_norm",
    ):
        delattr(model.model, name)
    return model


def prepared_pair(
    monkeypatch,
    *modules,
    seed=0,
    parts="",
    heads=2,
    kv_heads=1,
    intermediate=384,
    window=WINDOW,
):
    """``(stock reference, hf_muse_glimmer-prepared text-only copy)``."""
    monkeypatch.setenv("SPYRE_MUSE_FP8", parts)
    eager(monkeypatch, *modules)
    reference = tiny_target(seed, heads, kv_heads, intermediate, window)
    adapted = text_only(copy.deepcopy(reference))
    hf_muse_glimmer.prepare_for_spyre(adapted)
    hf_common.set_rope_dtype(adapted, torch.float32)
    return reference, adapted


def stock_forward(reference, ids, **kwargs):
    """Stock HF forward on ``ids`` (no cache)."""
    with torch.no_grad():
        return reference(torch.as_tensor(ids).reshape(1, -1), use_cache=False, **kwargs)
