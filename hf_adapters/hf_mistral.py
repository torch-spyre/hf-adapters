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
HuggingFace Transformers adapter for Mistral models on Spyre.

Covers model_type ``mistral``: Mistral 7B v0.2 and v0.3.
Same architecture as Llama with GQA.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    prepare_standard_gqa,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Mistral model in-place."""
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    prepare_standard_gqa(model, MistralRMSNorm)
