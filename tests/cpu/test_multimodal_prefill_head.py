# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise every multimodal prefill hook with a real vocabulary projection."""

from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from hf_adapters import auto_spyre_model, hf_common


def _adapter_id(adapter: ModuleType) -> str:
    return adapter.__name__


VLM_ADAPTERS: list[ModuleType] = sorted(
    set(auto_spyre_model.IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING.values()),
    key=_adapter_id,
)


@pytest.mark.parametrize("adapter", VLM_ADAPTERS, ids=_adapter_id)
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_multimodal_prefill_projects_only_requested_rows(
    monkeypatch: pytest.MonkeyPatch, adapter: ModuleType, batch_size: int
) -> None:
    torch.manual_seed(5)
    ids = torch.arange(1, 9).expand(batch_size, -1).clone()
    hidden = torch.randn(batch_size, ids.shape[1], 64)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(64, 129, bias=False)
    model.config = SimpleNamespace(
        enable_moe_block=True,
        use_bidirectional_attention=None,
        text_config=SimpleNamespace(vocab_size=129),
    )
    model._spyre_has_ple = False
    backbone_rows: list[int] = []
    head_rows: list[int] = []

    def backbone(
        model: torch.nn.Module,
        inputs_embeds: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        backbone_rows.append(inputs_embeds.shape[1])
        return inputs_embeds * 2

    def head(value: torch.Tensor) -> torch.Tensor:
        head_rows.append(value.shape[1])
        # Keep the prepared head's postprocessing in the exercised path.
        return torch.tanh(model.lm_head(value) / 2) * 2

    model._spyre_lm_head_forward = head
    kwargs: dict[str, Any] = dict(
        model=model,
        input_ids=ids,
        position_ids=ids - 1,
        attention_mask=torch.zeros(batch_size, 1, 8, 8),
        key_caches=[],
        value_caches=[],
        cache_index=torch.arange(8),
        prefill_context=None,
        pixel_values=None,
    )
    if adapter.__name__.endswith("hf_gemma4_mm"):
        monkeypatch.setattr(adapter, "_image_features", Mock(return_value=None))
        monkeypatch.setattr(
            adapter, "_embed_and_scatter", Mock(return_value=(hidden, None, ids))
        )
        monkeypatch.setattr(adapter.hf_gemma4, "_run_blocks_over_embeds", backbone)
        kwargs.update(image_position_ids=None, mm_token_type_ids=torch.zeros_like(ids))
    elif adapter.__name__.endswith("hf_muse_glimmer_mm"):
        monkeypatch.setattr(
            adapter, "_prepare_prefill_inputs", Mock(return_value=hidden)
        )
        monkeypatch.setattr(adapter, "_run_backbone_from_embeds", backbone)
        model.config.image_token_id = 127
        model.config.video_token_id = 128
        kwargs["image_grid_thw"] = None
    else:
        monkeypatch.setattr(adapter, "embed_text_tokens", Mock(return_value=hidden))
        mask_shape = (
            (batch_size, 8, 1)
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else (batch_size, 8)
        )
        monkeypatch.setattr(
            adapter,
            "_vision_mask",
            Mock(return_value=torch.zeros(mask_shape, dtype=torch.bool)),
        )
        feature_fn = (
            "_deepstack_features"
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else "_image_features"
        )
        monkeypatch.setattr(adapter, feature_fn, Mock(return_value=None))
        monkeypatch.setattr(adapter, "_run_text_backbone", backbone)
        kwargs["image_sizes"] = None

    reference = adapter._prefill_forward(**kwargs)
    actual = adapter._prefill_forward(**kwargs, logits_to_keep=1)
    assert reference.shape == (batch_size, 8, 129)
    assert actual.shape == (batch_size, 1, 129)
    torch.testing.assert_close(actual, reference[:, -1:, :])
    assert backbone_rows == [8, 8]
    assert head_rows == [8, 1]


@pytest.mark.parametrize("adapter", VLM_ADAPTERS, ids=_adapter_id)
def test_multimodal_generate_requests_last_token(
    monkeypatch: pytest.MonkeyPatch, adapter: ModuleType
) -> None:
    model = SimpleNamespace()
    monkeypatch.setattr(
        auto_spyre_model, "resolve_adapter_module", Mock(return_value=adapter)
    )
    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(Mock(return_value=model)),
    )
    expected = object()

    def generate(
        forward: Any, actual_model: Any, ids: torch.Tensor, **kwargs: Any
    ) -> object:
        assert actual_model is model
        prefill: Any = kwargs["prefill_fn"]
        assert prefill.func is adapter._prefill_forward
        assert prefill.keywords["logits_to_keep"] == 1
        assert "prefill_inputs_fn" not in kwargs
        assert "prefill_backbone_fn" not in kwargs
        return expected

    monkeypatch.setattr(hf_common, "generate", generate)
    loaded = auto_spyre_model.AutoSpyreModelForImageTextToText.from_pretrained(
        "tiny-vlm"
    )
    processor_inputs = {name: None for name in adapter._GENERATION_INPUT_NAMES}
    assert loaded.generate(torch.tensor([[1]]), **processor_inputs) is expected
