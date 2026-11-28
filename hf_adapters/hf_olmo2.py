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
HuggingFace Transformers adapter for OLMo2 models on Spyre.

Covers model_type ``olmo2``: OLMo 2 7B.
Post-norm architecture (norm after attention/MLP, before residual add)
with Q/K RMSNorm applied to flattened projections before reshape.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

from transformers import PreTrainedModel

from hf_adapters._olmo_common import prepare_olmo_common
from hf_adapters.hf_common import (
    standard_gqa_backbone_forward,
    standard_gqa_forward,
)

_run_forward = standard_gqa_forward
_run_backbone_forward = standard_gqa_backbone_forward


def prepare_for_spyre(model: PreTrainedModel) -> None:
    """Apply Spyre adaptations to OLMo2 model in-place."""
    prepare_olmo_common(model, "OLMo 2")
