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
Offline regression test for DistilBERT backbone resolution.

A bare ``DistilBertModel`` (what ``AutoModel`` returns for DistilBERT embedding
checkpoints) has a ``.transformer`` attribute that is its layer stack, not a
backbone. ``hf_common.get_backbone`` used to search wrapper attributes by name,
matched ``transformer`` (meant for GPT-2), resolved the layer stack as the
backbone, and ``prepare_for_spyre`` raised
``AttributeError: 'Transformer' object has no attribute 'transformer'``. It now
resolves the backbone through ``base_model_prefix``.

Uses a tiny randomly initialised config, so nothing is downloaded.
"""

import pytest
import torch
from torch import nn
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertModel,
    PreTrainedModel,
)

from hf_adapters import hf_common, hf_distilbert
from tests.cpu.conftest import _unwrap_compiled_blocks, min_cosine

_TINY_CONFIG: dict[str, int] = {
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "n_heads": 2,
}


def _tiny_model(model_cls: type[PreTrainedModel]) -> PreTrainedModel:
    torch.manual_seed(0)
    model: PreTrainedModel = model_cls(DistilBertConfig(**_TINY_CONFIG))
    return model.eval()


@pytest.mark.parametrize(
    ("model_cls", "backbone_attr"),
    [(DistilBertModel, None), (DistilBertForSequenceClassification, "distilbert")],
    ids=["bare", "task_wrapper"],
)
def test_backbone_is_the_distilbert_model(
    model_cls: type[PreTrainedModel], backbone_attr: str | None
) -> None:
    model: PreTrainedModel = _tiny_model(model_cls)
    expected: nn.Module = (
        model if backbone_attr is None else getattr(model, backbone_attr)
    )

    assert hf_common.get_backbone(model) is expected


def test_bare_model_prefill_matches_hf() -> None:
    model: PreTrainedModel = _tiny_model(DistilBertModel)
    input_ids: torch.Tensor = torch.tensor([[5, 17, 42, 9, 3], [8, 21, 4, 0, 0]])
    attention_mask: torch.Tensor = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.no_grad():
        reference: torch.Tensor = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    hf_distilbert.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)
    with torch.no_grad():
        hidden: torch.Tensor = hf_common.prefill_encoder(
            hf_distilbert._run_backbone_forward, model, input_ids, attention_mask
        )

    assert hidden.shape == reference.shape
    assert min_cosine(hidden, reference, attention_mask) > 0.999
