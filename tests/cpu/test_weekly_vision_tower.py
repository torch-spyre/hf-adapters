"""Exercise the weekly tower loader and feature checks on small stock models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    Gemma4VisionConfig,
    Gemma4VisionModel,
    PixtralVisionConfig,
    PixtralVisionModel,
    PreTrainedModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from tests.spyre.weekly_generation import vision_tower


def _tiny_tower(kind: str) -> PreTrainedModel:
    if kind == "siglip":
        return SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=16,
                patch_size=4,
            )
        ).eval()
    if kind == "pixtral":
        return PixtralVisionModel(
            PixtralVisionConfig(
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=16,
                patch_size=4,
            )
        ).eval()
    return Gemma4VisionModel(
        Gemma4VisionConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            patch_size=2,
            pooling_kernel_size=2,
            position_embedding_size=32,
        )
    ).eval()


@pytest.mark.parametrize("kind", ["siglip", "pixtral", "gemma4"])
@pytest.mark.parametrize("sharded", [False, True])
def test_stock_tower_loader_extracts_all_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, sharded: bool
) -> None:
    tower = _tiny_tower(kind)
    state = {
        f"model.vision_tower.{name}": value
        for name, value in tower.state_dict().items()
    }
    filename = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
    save_file(state, tmp_path / filename)
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: filename for name in state}})
        )

    revisions: list[str] = []

    def download(model_path: str, name: str, *, revision: str) -> str:
        revisions.append(revision)
        path = tmp_path / name
        if not path.exists():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError("missing index")
        return str(path)

    monkeypatch.setattr(vision_tower, "hf_hub_download", download)
    monkeypatch.setattr(
        vision_tower.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            vision_config=tower.config, _commit_hash="a" * 40
        ),
    )
    actual = vision_tower.load_stock_tower("org/test", torch.float32)
    assert len(revisions) == 2 and set(revisions) == {"a" * 40}
    for name, expected in tower.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected, rtol=0, atol=0)
    inputs = vision_tower.tower_inputs(actual)
    with torch.no_grad():
        reference = tower(**inputs).last_hidden_state
        result = actual(**inputs).last_hidden_state
    vision_tower.compare_features(reference, result)


@pytest.mark.parametrize("kind", ["siglip", "pixtral", "gemma4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weekly_tower_pipeline_matches_stock_on_cpu(
    monkeypatch: pytest.MonkeyPatch, kind: str, dtype: torch.dtype
) -> None:
    tower = _tiny_tower(kind).to(dtype=dtype)
    names = {
        "siglip": "hf_siglip_vision",
        "pixtral": "hf_pixtral_vision",
        "gemma4": "hf_gemma4_vision",
    }
    from importlib import import_module

    adapter = import_module(f"hf_adapters.{names[kind]}")

    def eager(function: Any, **kwargs: Any) -> Any:
        return function

    def dtype_policy(
        model_path: str, *, target_device: str, trust_remote_code: bool
    ) -> torch.dtype:
        assert target_device == "spyre" and trust_remote_code is False
        return dtype

    loaded_dtypes: list[torch.dtype] = []

    def load_stock(model_path: str, dtype: torch.dtype) -> PreTrainedModel:
        loaded_dtypes.append(dtype)
        return tower

    monkeypatch.setattr(torch, "compile", eager)
    monkeypatch.setattr(vision_tower, "DEVICE", "cpu")
    monkeypatch.setattr(vision_tower, "resolve_tower_adapter", lambda path: adapter)
    monkeypatch.setattr(vision_tower, "dtype_for_model_path", dtype_policy)
    monkeypatch.setattr(vision_tower, "load_stock_tower", load_stock)
    reference_completed: list[bool] = []
    vision_tower.run_tower_compare(
        "org/test", on_reference=lambda: reference_completed.append(True)
    )
    assert reference_completed == [True]
    assert loaded_dtypes == [dtype]


@pytest.mark.parametrize("revision", [None, "main"])
def test_stock_tower_requires_an_immutable_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str | None
) -> None:
    monkeypatch.setattr(
        vision_tower.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(_commit_hash=revision),
    )
    with pytest.raises(ValueError, match="immutable checkpoint revision"):
        vision_tower.load_stock_tower("org/test", torch.float32)


@pytest.mark.parametrize("adapter_name", ["hf_example.py", "hf_example"])
def test_tower_adapter_suffix_is_optional(
    monkeypatch: pytest.MonkeyPatch, adapter_name: str
) -> None:
    from types import ModuleType

    adapter = ModuleType("hf_adapters.hf_example")
    imports: list[str] = []

    def import_adapter(name: str) -> ModuleType:
        imports.append(name)
        return adapter

    monkeypatch.setattr(
        vision_tower,
        "VISION_MODELS",
        {"example": {"path": "org/test", "kind": "tower", "adapter": adapter_name}},
    )
    monkeypatch.setattr(vision_tower.importlib, "import_module", import_adapter)
    assert vision_tower.resolve_tower_adapter("org/test") is adapter
    assert imports == ["hf_adapters.hf_example"]


@pytest.mark.parametrize("corruption", ["nan", "inf", "shape", "scale", "direction"])
def test_feature_check_rejects_corrupt_outputs(corruption: str) -> None:
    reference = torch.randn(1, 16, 128)
    actual = reference.clone()
    if corruption in ("nan", "inf"):
        actual[0, 0, 0] = float(corruption)
    elif corruption == "shape":
        actual = actual[:, :-1]
    elif corruption == "scale":
        actual *= 2
    else:
        actual *= -1
    with pytest.raises(AssertionError):
        vision_tower.compare_features(reference, actual)
