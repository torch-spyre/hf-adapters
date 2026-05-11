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

    from hf_adapters.hf_llama import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch

from hf_adapters.hf_common import (
    generate as _generate,
)
from hf_adapters.hf_common import (
    load_model_common,
    prepare_standard_gqa,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Llama model in-place."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    prepare_standard_gqa(model, LlamaRMSNorm)


def load_model(model_path, dtype=torch.float16):
    """Load Llama model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Llama on Spyre."""
    return _generate(standard_gqa_forward, model, tokenizer, prompts, **kwargs)
