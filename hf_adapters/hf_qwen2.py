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
HuggingFace Transformers adapter for Qwen2 models on Spyre.

Covers model_type ``qwen2``: Qwen 1.5 (0.5B-7B), Qwen 2 (0.5B-7B),
Qwen 2.5 (0.5B-7B).  Like Qwen3 but without per-head Q/K RMSNorm.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.hf_common import (
    prepare_standard_gqa,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Qwen2 model in-place."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    prepare_standard_gqa(model, Qwen2RMSNorm)
