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

"""HuggingFace Transformers adapter for standalone Ministral causal-LM models.

Supports ``MinistralConfig`` (``model_type: ministral``) checkpoints that ship
as plain causal-LMs with no multimodal wrapper, e.g.:

- ``mistralai/Ministral-8B-Instruct-2410``

The decoder architecture is identical to Mistral 7B (standard GQA), so the
compiled block, forward, and prepare logic are reused from ``hf_mistral``.

Two model-specific issues are handled here:

1. **Tokenizer byte-escape fix** — ``LlamaTokenizer`` uses a GPT-2-style
   byte-level BPE fallback vocabulary where non-printable bytes are encoded as
   Unicode characters (e.g. space → Ġ U+0120, newline → Ċ U+010A).
   ``tokenizer.decode()`` returns these characters verbatim; ``_generate``
   patches the tokenizer instance so all decode calls see clean text.

2. **Spyre embed_tokens device fix** — ``_move_to_spyre_with_layout`` moves
   every ``nn.Parameter`` to Spyre, including ``embed_tokens.weight``. The
   Spyre int64→int32 monkey-patch returns ``input_ids`` on CPU, which causes a
   device mismatch at ``aten.embedding.default``. ``_patch_embed_tokens_for_spyre``
   replaces ``embed_tokens`` with ``_EmbedCPU``, a module that keeps its weight
   as a plain Python attribute (invisible to ``named_parameters()``), gathers on
   CPU, then moves the result to the Spyre device.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

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


def load_hf_model(model_path, dtype):
    """Load a standalone Ministral causal-LM (``MinistralConfig``).

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


def load_model(model_path, dtype):
    """Load a Ministral causal-LM and prepare it for Spyre."""
    model = load_hf_model(model_path, dtype)
    actual_dtype = next(model.parameters()).dtype
    _untie_embedding_and_lm_head(model)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, actual_dtype)
    print("Model ready.")
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

    Patches the tokenizer instance to fix GPT-2 byte-level BPE escape
    characters (e.g. Ġ for space, Ċ for newline) returned verbatim by
    ``LlamaTokenizer.decode()``, then delegates to ``hf_common.generate``.
    """
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

    Called from ``prepare_for_spyre`` before ``_move_to_spyre_with_layout``.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = get_backbone(model)
    orig_embed = backbone.embed_tokens  # nn.Embedding, weight on CPU here

    # Extract the raw weight tensor and detach it from nn.Embedding's
    # parameter tracking.  ref_weight is a plain Tensor attribute — not a
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
    """Apply Spyre adaptations to a Ministral causal-LM in-place."""
    from transformers.models.ministral.modeling_ministral import MinistralRMSNorm

    # Patch embed_tokens before _move_to_spyre_with_layout so its weight is
    # kept on CPU and the embedding result is moved to Spyre.
    _patch_embed_tokens_for_spyre(model)

    prepare_standard_gqa(model, MinistralRMSNorm)
