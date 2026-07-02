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


def _patch_ministral_tokenizer(tokenizer):
    """Patch tokenizer.decode / batch_decode to fix GPT-2 byte-level BPE escapes.

    The Ministral-8B tokenizer (``LlamaTokenizer``) returns Unicode characters
    such as Ġ (U+0120, space) and Ċ (U+010A, newline) verbatim from decode()
    instead of converting them back to the real bytes.  Patching the instance
    methods ensures both our ``generate`` path and any external decode call
    (e.g. the CPU accuracy test's HF-reference comparison) see clean text.

    Idempotent — a second call on an already-patched tokenizer is a no-op.
    """
    if getattr(tokenizer, "_ministral_decode_patched", False):
        return

    _orig_decode = tokenizer.decode
    _orig_batch_decode = tokenizer.batch_decode

    def _decode(token_ids, **kw):
        return _fix_ministral_decode(_orig_decode(token_ids, **kw))

    def _batch_decode(sequences, **kw):
        return [_fix_ministral_decode(s) for s in _orig_batch_decode(sequences, **kw)]

    tokenizer.decode = _decode
    tokenizer.batch_decode = _batch_decode
    tokenizer._ministral_decode_patched = True


def _generate(model, tokenizer, prompts, **kwargs):
    """Module-level generate hook checked by ``AutoSpyreModelForCausalLM``.

    For ``MinistralConfig`` models (Ministral-8B) the tokenizer returns GPT-2
    byte-level BPE escape characters (e.g. Ġ for space, Ċ for newline) in the
    decoded text.  Patch the tokenizer instance so that all decode calls —
    including any HF-reference comparison made by the test harness — see clean
    text.  All other variants decode cleanly and are returned as-is.
    """
    from transformers.models.ministral.configuration_ministral import MinistralConfig

    if isinstance(model.config, MinistralConfig):
        _patch_ministral_tokenizer(tokenizer)
    return generate(_run_forward, model, tokenizer, prompts, **kwargs)



def _patch_embed_tokens_for_spyre(model):
    """Wrap embed_tokens to keep its weight on CPU and move embeddings to Spyre.

    ``_move_to_spyre_with_layout`` iterates ``model.named_parameters()`` and
    moves every parameter — including ``embed_tokens.weight`` — to Spyre.
    This torch-spyre version's ``aten.embedding.default`` fallback then crashes
    because ``input_ids`` arrives on CPU (the Spyre int64→int32 monkey-patch
    returns a CPU tensor).

    Fix: replace ``backbone.embed_tokens`` with a plain ``nn.Module`` that
    stores the embedding weight as a non-parameter Python attribute (so it is
    invisible to ``named_parameters()`` and stays on CPU), performs the gather
    on CPU, then moves the float result to Spyre by copying to the device of
    the first attention projection.

    Called from ``prepare_for_spyre`` only for ``MinistralConfig`` models,
    before ``_move_to_spyre_with_layout`` runs.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = get_backbone(model)
    orig_embed = backbone.embed_tokens  # nn.Embedding, weight on CPU here

    # Extract the raw weight tensor and detach it from nn.Embedding's
    # parameter tracking.  _ref_weight is a plain Tensor attribute — not a
    # nn.Parameter — so named_parameters() will not find it and
    # _move_to_spyre_with_layout will not move it to Spyre.
    ref_weight = orig_embed.weight.detach()

    class _EmbedCPU(nn.Module):
        """Embedding whose weight stays on CPU; output is moved to Spyre."""

        def __init__(self):
            super().__init__()
            # Store as a plain attribute, not nn.Parameter, to hide it from
            # named_parameters() and thus from _move_to_spyre_with_layout.
            object.__setattr__(self, "_w", ref_weight)

        def forward(self, input_ids):
            w = object.__getattribute__(self, "_w")  # always CPU
            h = F.embedding(input_ids.cpu(), w)
            # Move to the device of the first Q-projection weight (Spyre after
            # _move_to_spyre_with_layout, CPU during CPU tests).
            target = backbone.layers[0].self_attn.q_proj.weight.device
            return h.to(target)

    backbone.embed_tokens = _EmbedCPU()


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Mistral-3-family model in-place."""
    from transformers.models.ministral.configuration_ministral import MinistralConfig
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

    # For MinistralConfig: patch embed_tokens before _move_to_spyre_with_layout
    # so its weight is kept on CPU and the embedding result is moved to Spyre.
    if isinstance(model.config, MinistralConfig):
        _patch_embed_tokens_for_spyre(model)

    prepare_standard_gqa(model, rmsnorm_cls)
