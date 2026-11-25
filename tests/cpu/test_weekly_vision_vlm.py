"""Check the callable VLM verifier without loading checkpoint weights."""

from __future__ import annotations

import gc
import traceback
import weakref
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tests._vision_helpers import VLMReference
from tests.spyre import test_e2e_smoke_vision_spyre as smoke
from tests.spyre import test_vlm_e2e_spyre as vlm
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.weekly_sub_process import eval_model


class _WeakModel(SimpleNamespace):
    __slots__ = ("__weakref__",)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "nan",
        "missing_step",
        "missing_vocab",
        "drift",
        "empty_text",
        "inner_steps",
        "inner_generate",
        "inner_smoke",
        "inner_load",
        "inner_chained_steps",
        "cpu_load",
        "cpu_forward",
        "cpu_processor",
    ],
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
    cache_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def decode(ids: Any, **kwargs: Any) -> str:
        return "" if corruption == "empty_text" else "caption"

    def generate(**kwargs: Any) -> torch.Tensor:
        if corruption == "inner_generate":
            loaded_model = model
            cache = torch.ones(4)
            cache_refs.append(weakref.ref(cache))
            assert loaded_model is not None
            raise RuntimeError("generate failed")
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

    def build_batch(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, torch.Tensor]]:
        if corruption == "cpu_processor":
            raise ValueError("processor failed")
        return processor, batch

    def get_reference(**kwargs: Any) -> tuple[VLMReference, str]:
        if corruption in ("cpu_load", "cpu_forward"):
            return kwargs["compute"](), "saved"
        return reference, "hit"

    def load_reference(**kwargs: Any) -> _WeakModel:
        if corruption == "cpu_load":
            raise ValueError("checkpoint failed")
        return model

    def stock_reference(**kwargs: Any) -> VLMReference:
        if corruption == "cpu_forward":
            raise ValueError("stock forward failed")
        return reference

    monkeypatch.setattr(vlm, "build_vlm_batch", build_batch)
    monkeypatch.setattr(vlm, "get_or_create_vlm_reference", get_reference)
    monkeypatch.setattr(vlm, "load_ref_model", load_reference)
    monkeypatch.setattr(vlm, "stock_vlm_reference", stock_reference)

    def lower_step(model: Any) -> None:
        cache = torch.ones(4)
        cache_refs.append(weakref.ref(cache))
        assert model is not None
        raise ValueError("lowering failed")

    def steps(
        adapter: ModuleType,
        model: Any,
        batch: dict[str, torch.Tensor],
        forced_tokens: list[int],
    ) -> list[torch.Tensor]:
        if corruption in ("inner_steps", "inner_chained_steps"):
            key_cache = torch.ones(4)
            value_cache = torch.ones(4)
            cache_refs.extend((weakref.ref(key_cache), weakref.ref(value_cache)))
            assert model is not None
            if corruption == "inner_chained_steps":
                try:
                    lower_step(model)
                except ValueError as cause:
                    raise RuntimeError("compile failed") from cause
            raise RuntimeError("decode failed")
        return actual_logits

    def load_model(**kwargs: Any) -> _WeakModel:
        loaded_model = model
        if corruption == "inner_load":
            cache = torch.ones(4)
            cache_refs.append(weakref.ref(cache))
            raise RuntimeError("prepare failed")
        return loaded_model

    monkeypatch.setattr(vlm, "_adapter_teacher_forced_steps", steps)
    monkeypatch.setattr(
        vlm.AutoSpyreModelForImageTextToText, "from_pretrained", load_model
    )
    smoke_models: list[Any] = []

    def run_smoke(model_path: str, **kwargs: Any) -> dict[str, str]:
        if corruption == "inner_smoke":
            loaded_model = kwargs["model"]
            cache = torch.ones(4)
            cache_refs.append(weakref.ref(cache))
            assert loaded_model is not None
            raise RuntimeError("smoke failed")
        smoke_models.append(kwargs["model"])
        return {"status": "PASS"}

    monkeypatch.setattr(smoke, "run_vision_smoke_test", run_smoke)
    reference_completed: list[bool] = []
    if corruption in ("cpu_load", "cpu_forward", "cpu_processor"):
        metrics = eval_model("org/model", ModuleType("adapter"), ModelType.VISION)
        assert metrics["load"] is False and metrics["correct"] is False
        assert metrics["failure_category"] == (
            "cpu_load_failed" if corruption == "cpu_load" else "cpu_generate_failed"
        )
        assert not smoke_models and not reference_completed
        return
    if corruption is None:
        vlm.run_vlm_compare(
            "org/model",
            include_smoke=True,
            on_reference=lambda: reference_completed.append(True),
        )
        assert smoke_models == [model]
    else:
        expected_error: type[Exception] = (
            RuntimeError if corruption.startswith("inner_") else AssertionError
        )
        with pytest.raises(expected_error) as retained_failure:
            vlm.run_vlm_compare(
                "org/model",
                include_smoke=True,
                on_reference=lambda: reference_completed.append(True),
            )
        assert not smoke_models
        # Keep the failure traceback alive, as pytest does between checkpoints.
        assert retained_failure.value.__traceback__ is not None
        formatted = "".join(traceback.format_exception(retained_failure.value))
        assert expected_error.__name__ in formatted
        if corruption == "inner_chained_steps":
            assert "lowering failed" in formatted and "compile failed" in formatted
        model_ref = weakref.ref(model)
        del model
        gc.collect()
        assert model_ref() is None, "failed verifier retained its loaded model"
        assert all(
            cache_ref() is None for cache_ref in cache_refs
        ), "callee retained KV caches"
    assert reference_completed == [True]
