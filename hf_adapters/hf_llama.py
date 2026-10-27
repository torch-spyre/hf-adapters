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
HuggingFace Transformers adapter for Llama models on Spyre.

Covers model_type ``llama``: Llama 1/2/3, Code Llama, Yi, and other
models that register as ``llama`` in HF Transformers.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    prepare_standard_gqa,
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward

#: Opt in to ``load_model_common(direct_to_device=True)``. ``prepare_for_spyre``
#: here only rewrites blocks and reads parameter shapes, and the RoPE cache is
#: built from the non-persistent ``inv_freq`` buffer, which never travels
#: through the checkpoint loader and so is still on the host when
#: ``prebuild_rope_cache`` runs. Covered by
#: ``tests/spyre/test_direct_load_compare_spyre.py``.
SPYRE_SUPPORTS_DIRECT_LOAD = True

#: ``pad_lm_head`` rebuilds the head weight with ``F.pad`` during
#: ``prepare_for_spyre``. On an already-transferred weight that pad would run as
#: an eager device op and the padded result would no longer carry the
#: ``dim_order=[1, 0]`` layout chosen for it, so the head is padded on the host
#: and transferred afterwards by ``model.to(device)``.
#:
#: On a checkpoint with ``tie_word_embeddings=True`` this also holds back
#: ``model.embed_tokens.weight``, because it *is* the head weight -- see
#: ``spyre_direct_load._tied_to_host``. Without that, untying would read the
#: head back off the device, which fp16 does not survive intact. The embedding
#: table is usually among the largest tensors in the model, so measure the
#: saving on a tied checkpoint rather than assuming it matches an untied one.
SPYRE_DIRECT_HOST_MODULES = ("lm_head",)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Llama model in-place."""
    prepare_standard_gqa(model)
