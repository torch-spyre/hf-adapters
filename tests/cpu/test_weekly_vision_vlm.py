"""Check the callable VLM verifier without loading checkpoint weights."""

from __future__ import annotations

import gc
import weakref
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tests._vision_helpers import VLMReference
from tests.spyre import test_e2e_smoke_vision_spyre as smoke
from tests.spyre import test_vlm_e2e_spyre as vlm


class _WeakModel(SimpleNamespace):
    __slots__ = ("__weakref__",)


@pytest.mark.parametrize(
    "corruption", [None, "nan", "missing_step", "missing_vocab", "drift", "empty_text"]
)
def test_callable_verifier_checks_logits_before_smoke_and_reuses_model(
    monkeypatch: pytest.MonkeyPatch, corruption: str | None
) -> None:
    reference_logits = [
        torch.arange(1, 9).float() for _ in range(vlm.NUM_COMPARE_STEPS)
    ]
    actual_logits = [logits.clone() for logits in reference_logits]
    if corruption == "nan":
        actual_logits[0][0] = float("nan")
    elif corruption == "missing_step":
        actual_logits.pop()
    elif corruption == "missing_vocab":
        actual_logits[0] = actual_logits[0][:-1]
    elif corruption == "drift":
        actual_logits[0] *= -1
    reference = VLMReference(
        logits=reference_logits, token_ids=[7] * vlm.NUM_COMPARE_STEPS, text="caption"
    )

    def decode(ids: Any, **kwargs: Any) -> str:
        return "" if corruption == "empty_text" else "caption"

    def generate(**kwargs: Any) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5]])

    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=decode))
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2),
        "pixel_values": torch.zeros(1, 3, 4, 4),
    }
    model = _WeakModel(generate=generate)
    monkeypatch.setattr(
        vlm, "resolve_adapter_module", lambda *a, **kw: ModuleType("adapter")
    )
    monkeypatch.setattr(vlm, "dtype_for_model_path", lambda *a, **kw: torch.float32)
    monkeypatch.setattr(vlm, "resolve_model_revision", lambda *a, **kw: "immutable")
    monkeypatch.setattr(vlm, "build_vlm_batch", lambda *a, **kw: (processor, batch))
    monkeypatch.setattr(
        vlm, "get_or_create_vlm_reference", lambda **kw: (reference, "hit")
    )
    monkeypatch.setattr(
        vlm, "_adapter_teacher_forced_steps", lambda *a, **kw: actual_logits
    )
    monkeypatch.setattr(
        vlm.AutoSpyreModelForImageTextToText, "from_pretrained", lambda **kw: model
    )
    smoke_models: list[Any] = []

    def run_smoke(model_path: str, **kwargs: Any) -> dict[str, str]:
        smoke_models.append(kwargs["model"])
        return {"status": "PASS"}

    monkeypatch.setattr(smoke, "run_vision_smoke_test", run_smoke)
    reference_completed: list[bool] = []
    if corruption is None:
        vlm.run_vlm_compare(
            "org/model",
            include_smoke=True,
            on_reference=lambda: reference_completed.append(True),
        )
        assert smoke_models == [model]
    else:
        with pytest.raises(AssertionError) as retained_failure:
            vlm.run_vlm_compare(
                "org/model",
                include_smoke=True,
                on_reference=lambda: reference_completed.append(True),
            )
        assert not smoke_models
        # Keep the failure traceback alive, as pytest does between checkpoints.
        assert retained_failure.value.__traceback__ is not None
        model_ref = weakref.ref(model)
        del model
        gc.collect()
        assert model_ref() is None, "failed verifier retained its loaded model"
    assert reference_completed == [True]
