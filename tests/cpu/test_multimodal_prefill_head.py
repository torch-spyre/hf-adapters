# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise every multimodal prefill hook with a real vocabulary projection."""

from types import SimpleNamespace

import pytest
import torch

from hf_adapters import auto_spyre_model, hf_common

VLM_ADAPTERS = sorted(
    set(auto_spyre_model.IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING.values()),
    key=lambda module: module.__name__,
)


@pytest.mark.parametrize("adapter", VLM_ADAPTERS, ids=lambda module: module.__name__)
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_multimodal_prefill_projects_only_requested_rows(
    monkeypatch, adapter, batch_size
):
    torch.manual_seed(5)
    ids = torch.arange(1, 9).expand(batch_size, -1).clone()
    hidden = torch.randn(batch_size, ids.shape[1], 64)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(64, 129, bias=False)
    model.config = SimpleNamespace(
        enable_moe_block=True, use_bidirectional_attention=None
    )
    model._spyre_has_ple = False
    backbone_rows, head_rows = [], []

    def backbone(model, inputs_embeds, *args, **kwargs):
        backbone_rows.append(inputs_embeds.shape[1])
        return inputs_embeds * 2

    def head(value):
        head_rows.append(value.shape[1])
        # Keep the prepared head's postprocessing in the exercised path.
        return torch.tanh(model.lm_head(value) / 2) * 2

    model._spyre_lm_head_forward = head
    kwargs = dict(
        model=model,
        input_ids=ids,
        position_ids=ids - 1,
        attention_mask=torch.zeros(batch_size, 1, 8, 8),
        key_caches=[],
        value_caches=[],
        cache_index=torch.arange(8),
        pixel_values=None,
    )
    if adapter.__name__.endswith("hf_gemma4_mm"):
        monkeypatch.setattr(adapter, "_image_features", lambda *args: None)
        monkeypatch.setattr(
            adapter, "_embed_and_scatter", lambda *args: (hidden, None, ids)
        )
        monkeypatch.setattr(adapter.hf_gemma4, "_run_blocks_over_embeds", backbone)
        kwargs.update(image_position_ids=None, mm_token_type_ids=torch.zeros_like(ids))
    else:
        monkeypatch.setattr(adapter, "_embed_text", lambda *args: hidden)
        mask_shape = (
            (batch_size, 8, 1)
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else (batch_size, 8)
        )
        monkeypatch.setattr(
            adapter,
            "_vision_mask",
            lambda *args: torch.zeros(mask_shape, dtype=torch.bool),
        )
        feature_fn = (
            "_deepstack_features"
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else "_image_features"
        )
        monkeypatch.setattr(adapter, feature_fn, lambda *args: None)
        monkeypatch.setattr(adapter, "_run_text_backbone", backbone)
        kwargs["image_sizes"] = None

    reference = adapter._prefill_forward(**kwargs)
    actual = adapter._prefill_forward(**kwargs, logits_to_keep=1)
    assert reference.shape == (batch_size, 8, 129)
    assert actual.shape == (batch_size, 1, 129)
    torch.testing.assert_close(actual, reference[:, -1:, :])
    assert backbone_rows == [8, 8]
    assert head_rows == [8, 1]


@pytest.mark.parametrize("adapter", VLM_ADAPTERS, ids=lambda module: module.__name__)
def test_multimodal_generate_requests_last_token(monkeypatch, adapter):
    model = SimpleNamespace()
    monkeypatch.setattr(
        auto_spyre_model, "resolve_adapter_module", lambda *args, **kwargs: adapter
    )
    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    expected = object()

    def generate(forward, actual_model, ids, **kwargs):
        assert actual_model is model
        prefill = kwargs["prefill_fn"]
        assert prefill.func is adapter._prefill_forward
        assert prefill.keywords["logits_to_keep"] == 1
        return expected

    monkeypatch.setattr(hf_common, "generate", generate)
    loaded = auto_spyre_model.AutoSpyreModelForImageTextToText.from_pretrained(
        "tiny-vlm"
    )
    processor_inputs = {name: None for name in adapter._GENERATION_INPUT_NAMES}
    assert loaded.generate(torch.tensor([[1]]), **processor_inputs) is expected
