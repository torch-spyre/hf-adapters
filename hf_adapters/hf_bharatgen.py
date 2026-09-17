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
HuggingFace Transformers adapter for BharatGen Param models on Spyre.

Covers ``model_type`` ``parambharatgen`` (architecture
``ParamBharatGenForCausalLM``), e.g. ``bharatgenai/Param-1-5B``.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    # Pass trust_remote_code=True for BharatGen Model to be loaded.
    model = AutoSpyreModelForCausalLM.from_pretrained("bharatgenai/Param-1-5B", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("bharatgenai/Param-1-5B")
    encoded = tokenizer(["Hello!"], return_tensors="pt", padding=True)
    sequences = model.generate(**encoded, max_new_tokens=32)
    print(tokenizer.batch_decode(sequences[:, encoded["input_ids"].shape[1] :]))
"""

import torch

from hf_adapters.hf_common import (
    prepare_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def load_hf_model(model_path, dtype=torch.float16, trust_remote_code=None):
    """Load a BharatGen Param checkpoint as a native ``ParamBharatGenForCausalLM``.

    Param ships custom modeling code, so the caller must opt in with
    ``trust_remote_code=True`` (threaded from ``AutoSpyreModelForCausalLM.from_pretrained``
    through ``hf_common.load_model_common``); loading without it raises the same
    "requires custom code" error stock HF would. This loader also defuses the
    repo's buggy ``rope_scaling`` config default before instantiation.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    param_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    # config_parambharatgen defaults rope_scaling to a dict whose keys the
    # modeling code's _init_rope mis-reads (config key ``rope_type`` vs modeling
    # key ``type``) -> KeyError at construction. This checkpoint applies no
    # scaling (rope_theta only), so clear it and take the unscaled RoPE branch.
    param_config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=param_config,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    return model


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Param model in-place.

    Param-specific fixups before handing off to the shared standard-GQA path.
    They exist because Param's remote code predates modern Transformers/Llama
    conventions the shared path assumes:

    * **Rebuild ``inv_freq``.** Param registers the rotary ``inv_freq`` as a
      ``persistent=False`` buffer, so it is absent from the checkpoint and does
      not survive ``from_pretrained(dtype=...)`` — it comes back zeroed (fp16) or
      as uninitialized junk (fp32), which poisons RoPE (NaN logits on CPU, wrong
      tokens on Spyre). Recompute it from the closed form
      ``1 / rope_theta ** (arange(0, head_dim, 2) / head_dim)`` on every layer.
    * **Hoist RoPE to the backbone.** Param's remote code predates model-level
      RoPE: it builds a ``ParamBharatGenRotaryEmbedding`` per attention module
      (``layers[i].self_attn.rotary_emb``) with no ``model.rotary_emb``, whereas
      the shared ``prepare_rope_and_heads`` reads ``get_backbone(model).rotary_emb``.
      Every layer's rotary module is identical (same ``head_dim``/``rope_theta``),
      and the shared path only consumes its ``inv_freq``, so we expose one of them
      on the backbone. head_dim is 128 (>= one stick after /2), so no padding.
    * **Materialize ``self_attn.scaling``.** Param's remote attention scales
      scores inline as ``... / math.sqrt(self.head_dim)`` and never stores a
      ``scaling`` attribute, whereas the shared compiled block reads
      ``self_attn.scaling`` (SDPA ``scale=``). Set it to the equivalent
      ``head_dim ** -0.5`` on each layer.
    """
    from hf_adapters.hf_common import get_backbone

    backbone = get_backbone(model)

    for layer in backbone.layers:
        attn = layer.self_attn
        # Rebuild the non-persistent inv_freq clobbered by the dtype-cast load.
        rope = attn.rotary_emb
        rope_dim = rope.inv_freq.shape[0] * 2  # == head_dim
        inv_freq = 1.0 / (
            rope.base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        rope.inv_freq = inv_freq.to(rope.inv_freq.device)
        if not hasattr(attn, "scaling"):
            attn.scaling = attn.head_dim**-0.5

    if not hasattr(backbone, "rotary_emb"):
        backbone.rotary_emb = backbone.layers[0].self_attn.rotary_emb

    prepare_standard_gqa(model)
