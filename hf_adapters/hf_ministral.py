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

Supports ``MinistralConfig`` (``model_type: ministral``) checkpoints, e.g.
``mistralai/Ministral-8B-Instruct-2410``.

The decoder is architecturally identical to Mistral 7B (standard GQA); compiled
block, forward, and prepare logic are reused from ``hf_mistral``.

Two model-specific workarounds are applied:

- ``LlamaTokenizer.decode()`` returns GPT-2 byte-level BPE escape characters
  verbatim (e.g. Ġ for space).  ``_generate`` patches the tokenizer instance
  to translate them back to real bytes before returning output.
- ``_move_to_spyre_with_layout`` moves ``embed_tokens.weight`` to Spyre, but
  the Spyre int64→int32 monkey-patch returns ``input_ids`` on CPU, causing a
  device mismatch.  ``_patch_embed_tokens_for_spyre`` keeps the weight on CPU
  as a non-parameter attribute and moves the float output to Spyre.

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
    generate,
    get_backbone,
    move_to_spyre_with_layout,
    prepare_standard_gqa,
    untie_embedding_and_lm_head,
)
from hf_adapters.hf_mistral import (
    _run_backbone_forward,  # noqa: F401  re-exported as adapter module API
    _run_forward,  # noqa: F401  re-exported as adapter module API
)


def load_hf_model(model_path, dtype):
    """Load a standalone Ministral causal-LM via ``AutoModelForCausalLM``."""
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
    untie_embedding_and_lm_head(model)
    prepare_for_spyre(model)
    print("Moving model to Spyre ...")
    move_to_spyre_with_layout(model, actual_dtype)
    print("Model ready.")
    return model


def _fix_ministral_decode(text: str) -> str:
    """Invert the GPT-2 byte-level BPE escape mapping back to real UTF-8 bytes."""
    printable = set(range(ord("!"), ord("~") + 1))
    table: dict = {}
    extra_cp = 256
    for byte_val in range(256):
        if byte_val not in printable:
            cp = extra_cp
            extra_cp += 1
            if cp != byte_val:
                table[cp] = chr(byte_val)
    return text.translate(str.maketrans(table))


def _patch_ministral_tokenizer(tokenizer):
    """Patch tokenizer decode methods to fix GPT-2 byte-level BPE escapes.

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
    """Module-level generate hook: patch tokenizer then delegate to hf_common.generate."""
    _patch_ministral_tokenizer(tokenizer)
    return generate(_run_forward, model, tokenizer, prompts, **kwargs)


def _patch_embed_tokens_for_spyre(model):
    """Replace embed_tokens with a CPU-resident wrapper to avoid Spyre device mismatch.

    Stores the embedding weight as a plain Python attribute (not ``nn.Parameter``)
    so ``_move_to_spyre_with_layout`` does not move it to Spyre.  The gather runs
    on CPU; the float output is moved to the device of the first Q-projection.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = get_backbone(model)
    ref_weight = backbone.embed_tokens.weight.detach()

    class _EmbedCPU(nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "_w", ref_weight)

        def forward(self, input_ids):
            w = object.__getattribute__(self, "_w")
            h = F.embedding(input_ids.cpu(), w)
            target = backbone.layers[0].self_attn.q_proj.weight.device
            return h.to(target)

    backbone.embed_tokens = _EmbedCPU()


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Ministral causal-LM in-place."""
    from transformers.models.ministral.modeling_ministral import MinistralRMSNorm

    _patch_embed_tokens_for_spyre(model)
    prepare_standard_gqa(model, MinistralRMSNorm)
