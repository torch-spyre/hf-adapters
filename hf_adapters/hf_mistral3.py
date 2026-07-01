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

"""HuggingFace Transformers adapter for the text backbone of Mistral-3-family models.

Loads a Mistral-3-family multimodal checkpoint (no trust_remote_code) and runs
*text-only* causal inference over its decoder. The vision encoder and projector
are dropped — for the full image→text pipeline this adapter is not used.

Covers two text-backbone variants, both shipped as
``Mistral3ForConditionalGeneration`` and distinguished only by their
``text_config.model_type``:

- ``"mistral"`` — e.g. ``mistralai/Mistral-Small-3.2-24B-Instruct-2506``
- ``"ministral3"`` — e.g. ``mistralai/Ministral-3-14B-Instruct-2512`` (blocked-FP8,
  dequantized on load)

Both decoders are architecturally identical to Mistral 7B (standard GQA), so the
compiled block, forward, and prepare logic are reused from ``hf_mistral``;
``get_backbone``/``prepare_standard_gqa`` handle the nested VLM shape and the
RMSNorm class is auto-detected per variant.

Also supports the standalone causal-LM variant (``MinistralConfig``):

- ``"ministral"`` — e.g. ``mistralai/Ministral-8B-Instruct-2410``

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Ministral-3-14B-Instruct-2512")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Ministral-3-14B-Instruct-2512")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    _move_to_spyre_with_layout,
    _untie_embedding_and_lm_head,
    generate,
    get_backbone,
    prepare_standard_gqa,
)
from hf_adapters.hf_mistral import (
    _run_backbone_forward,  # noqa: F401  re-exported as adapter module API
    _run_forward,  # noqa: F401  re-exported as adapter module API
)


def _load_hf_model_ministral8b(model_path, dtype):
    """Load a standalone Ministral-8B causal-LM (``MinistralConfig``).

    ``mistralai/Ministral-8B-Instruct-2410`` ships as a plain causal-LM with
    no multimodal wrapper — load directly via ``AutoModelForCausalLM``.

    The tokenizer for this checkpoint (``LlamaTokenizer``) uses a GPT-2-style
    byte-level BPE fallback vocabulary where non-printable bytes are encoded as
    Unicode characters in the range U+0100–U+0142 (e.g. space → Ġ U+0120,
    newline → Ċ U+010A). ``tokenizer.decode()`` returns these characters
    verbatim, so the ``_generate`` hook post-processes the decoded strings
    through ``_fix_ministral_decode`` to restore the original bytes.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _fix_ministral_decode(text: str) -> str:
    """Translate GPT-2 byte-level BPE escape characters back to real bytes.

    The GPT-2 encoding assigns printable ASCII bytes (0x21–0x7E) to themselves
    and maps the remaining 162 non-printable bytes to codepoints starting at
    U+0100 in the order they are encountered when iterating 0–255.  This
    function inverts that mapping so generated text is returned as normal UTF-8.
    """
    printable = set(range(ord("!"), ord("~") + 1))  # 0x21–0x7E
    table: dict = {}
    extra_cp = 256
    for byte_val in range(256):
        if byte_val in printable:
            pass  # codepoint == byte_val — no remapping needed
        else:
            cp = extra_cp
            extra_cp += 1
            if cp != byte_val:
                table[cp] = chr(byte_val)
    return text.translate(str.maketrans(table))


def load_hf_model(model_path, dtype):
    """Load a Mistral-3-family text decoder from its stock multimodal checkpoint.

    Both variants (Mistral-Small-3.2 with a ``mistral`` text backbone, and
    Ministral-3 with a ``ministral3`` one) ship as
    ``Mistral3ForConditionalGeneration``, so the full VLM is loaded directly —
    no shard key-remap. ``get_backbone`` descends through
    ``model.model.language_model`` to the text decoder and the top-level
    ``lm_head`` is used as-is (the Gemma4 adapter pattern); the text-only
    ``forward(input_ids=...)`` provides the causal-LM reference for the harness.

    ``from_pretrained`` also dequantizes automatically: the Ministral-3-14B
    checkpoint is blocked-FP8 (weight + weight_scale_inv per projection), which a
    raw key-remap would load with the wrong shapes — here it comes back as
    fp16/bf16. The vision tower and projector are dropped to free memory since
    this is text-only inference.

    For the standalone ``MinistralConfig`` variant (Ministral-8B), dispatches to
    ``_load_hf_model_ministral8b`` instead.
    """
    from transformers import AutoConfig
    from transformers.models.ministral.configuration_ministral import MinistralConfig
    from transformers.models.mistral3.modeling_mistral3 import (
        Mistral3ForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(model_path)
    if isinstance(cfg, MinistralConfig):
        return _load_hf_model_ministral8b(model_path, dtype)

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    )
    # Drop the vision tower and multi-modal projector — text-only inference.
    if hasattr(model, "model"):
        mm = model.model
        if hasattr(mm, "vision_tower"):
            del mm.vision_tower
        if hasattr(mm, "multi_modal_projector"):
            del mm.multi_modal_projector
    model.eval()
    model.requires_grad_(False)
    return model


def load_model(model_path, dtype):
    """Load a Mistral-3-family text decoder and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    # FP8 checkpoints (e.g. Ministral-3-14B) are dequantized to bf16 by
    # transformers regardless of the requested dtype. Use the model's actual
    # dtype for _move_to_spyre_with_layout so every parameter is cast
    # consistently and no bf16/fp16 mismatch arises.
    actual_dtype = next(model.parameters()).dtype
    _untie_embedding_and_lm_head(model)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, actual_dtype)
    print("Model ready.")
    return model


def _generate(model, tokenizer, prompts, **kwargs):
    """Module-level generate hook checked by ``AutoSpyreModelForCausalLM``.

    For ``MinistralConfig`` models (Ministral-8B) the tokenizer returns GPT-2
    byte-level BPE escape characters (e.g. Ġ for space, Ċ for newline) in the
    decoded text.  Post-process the results through ``_fix_ministral_decode`` to
    restore the original bytes.  All other variants decode cleanly and are
    returned as-is.
    """
    from transformers.models.ministral.configuration_ministral import MinistralConfig

    results = generate(_run_forward, model, tokenizer, prompts, **kwargs)
    if isinstance(model.config, MinistralConfig):
        return [_fix_ministral_decode(s) for s in results]
    return results


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Mistral-3-family model in-place."""
    from transformers.models.ministral.modeling_ministral import MinistralRMSNorm
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    # Decide the correct RMSNorm class in one place by inspecting the first
    # decoder layer's norm — Ministral3 uses Ministral3RMSNorm, Mistral-Small
    # uses MistralRMSNorm, Ministral-8B uses MinistralRMSNorm.
    first_norm = get_backbone(model).layers[0].input_layernorm
    if isinstance(first_norm, MistralRMSNorm):
        rmsnorm_cls = MistralRMSNorm
    elif isinstance(first_norm, MinistralRMSNorm):
        rmsnorm_cls = MinistralRMSNorm
    else:
        rmsnorm_cls = Ministral3RMSNorm

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "max_position_embeddings",
            "vocab_size",
        ):
            if hasattr(text_cfg, name) and not hasattr(cfg, name):
                setattr(cfg, name, getattr(text_cfg, name))

    prepare_standard_gqa(model, rmsnorm_cls)
