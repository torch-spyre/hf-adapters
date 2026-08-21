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

import torch  # noqa: F401  (should happen before first torch_spyre import)

try:
    from torch_spyre._inductor import (  # type: ignore[import-not-found]
        config as spyre_config,
    )

    # Bundle-scoped HBM pool planning currently corrupts
    # outputs of multiple models.
    setattr(spyre_config, "hbm_pool_planning", False)
except ImportError:
    pass

from hf_adapters.auto_spyre_model import (
    AutoSpyreModel,
    AutoSpyreModelForCausalLM,
    AutoSpyreModelForImageTextToText,
    AutoSpyreModelForMaskedLM,
    AutoSpyreModelForQuestionAnswering,
    AutoSpyreModelForSequenceClassification,
    AutoSpyreModelForTokenClassification,
)

__all__ = [
    "AutoSpyreModel",
    "AutoSpyreModelForCausalLM",
    "AutoSpyreModelForImageTextToText",
    "AutoSpyreModelForMaskedLM",
    "AutoSpyreModelForQuestionAnswering",
    "AutoSpyreModelForSequenceClassification",
    "AutoSpyreModelForTokenClassification",
]
