# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Selective CPU input transfer must remain unchanged for existing adapters."""

from typing import Any

import pytest
import torch

from hf_adapters.hf_common import patch_cpu_inputs


def test_named_cpu_inputs_preserve_other_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Frontend(torch.nn.Module):
        def forward(
            self: "Frontend", *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

    def simulated_cpu_copy(self: torch.Tensor) -> torch.Tensor:
        assert self.device.type == "meta"
        return torch.zeros(self.shape, dtype=self.dtype)

    model = Frontend()
    patch_cpu_inputs(model, ("input_ids",))
    monkeypatch.setattr(torch.Tensor, "cpu", simulated_cpu_copy)
    tokens = torch.empty((1, 3), dtype=torch.int64, device="meta")
    args, kwargs = model(
        tokens, tokens, input_ids=tokens, token_type_ids=tokens, flag=True
    )
    assert args[0].device.type == "cpu"
    assert kwargs["input_ids"].device.type == "cpu"
    assert args[1] is tokens
    assert kwargs["token_type_ids"] is tokens
    assert kwargs["flag"] is True
