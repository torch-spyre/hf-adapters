# Copyright 2026 The Torch-Spyre Authors.
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

"""CPU tests for ``scripts/convert_openclip_to_hf.py`` on a tiny synthetic OpenCLIP model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import CLIPModel

open_clip = pytest.importorskip("open_clip")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "convert_openclip_to_hf.py"

# Head width 16 (not open_clip's default 64) so a wrong head count changes outputs.
TEXT_CFG = {
    "context_length": 8,
    "vocab_size": 64,
    "width": 32,
    "heads": 2,
    "layers": 3,
}
VISION_CFG = {"image_size": 32, "patch_size": 16, "width": 32, "layers": 2}
VISION_HEAD_WIDTH = 16


def _load_converter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_openclip_to_hf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_converter()


def _make_openclip() -> torch.nn.Module:
    torch.manual_seed(0)
    model = open_clip.CLIP(
        embed_dim=16,
        vision_cfg={**VISION_CFG, "head_width": VISION_HEAD_WIDTH},
        text_cfg=TEXT_CFG,
    )
    return model.eval()


def _write_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    prefix: str = "",
    with_config: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    state = {f"{prefix}{k}": v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(directory / "open_clip_model.safetensors"))
    if with_config:
        cfg = {
            "model_cfg": {
                "embed_dim": 16,
                "vision_cfg": {**VISION_CFG, "head_width": VISION_HEAD_WIDTH},
                "text_cfg": TEXT_CFG,
            }
        }
        (directory / "open_clip_config.json").write_text(json.dumps(cfg))
    return directory


def _convert(src: Path, out: Path, **kwargs: object) -> None:
    converter.convert(src, out, processor_source=None, **kwargs)


@pytest.fixture(scope="module")
def openclip_model() -> torch.nn.Module:
    return _make_openclip()


@pytest.mark.parametrize("prefix", ["", "model."])
def test_converted_model_matches_openclip(
    tmp_path: Path, openclip_model: torch.nn.Module, prefix: str
) -> None:
    _write_checkpoint(tmp_path / "src", openclip_model, prefix=prefix)
    _convert(tmp_path / "src", tmp_path / "out")
    hf = CLIPModel.from_pretrained(tmp_path / "out", dtype=torch.float32).eval()

    ids = torch.randint(1, 60, (3, 8))
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        # OpenCLIP's text tower is causal and pools at the argmax token id, as is HF's
        # when eos_token_id is the highest id, so put it last.
        ids[:, -1] = 63
        torch.testing.assert_close(
            hf.get_text_features(input_ids=ids).pooler_output,
            openclip_model.encode_text(ids),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            hf.get_image_features(pixel_values=pixels).pooler_output,
            openclip_model.encode_image(pixels),
            atol=1e-5,
            rtol=1e-5,
        )


def test_output_is_self_contained(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    _write_checkpoint(tmp_path / "src", openclip_model)
    _convert(tmp_path / "src", tmp_path / "out")
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["architectures"] == ["CLIPModel"]
    assert config["text_config"]["num_hidden_layers"] == 3
    assert config["vision_config"]["num_hidden_layers"] == 2
    assert config["text_config"]["num_attention_heads"] == 2
    assert config["vision_config"]["num_attention_heads"] == 2
    assert config["text_config"]["hidden_act"] == "gelu"
    with safe_open(str(tmp_path / "out" / "model.safetensors"), "pt") as f:
        assert f.metadata() == {"format": "pt"}


def test_layer_counts_are_inferred_not_defaulted(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    """A 3-layer text tower must not be truncated to a default of 12 or 2."""
    _write_checkpoint(tmp_path / "src", openclip_model)
    _convert(tmp_path / "src", tmp_path / "out")
    state = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert "text_model.encoder.layers.2.self_attn.q_proj.weight" in state
    assert "text_model.encoder.layers.3.self_attn.q_proj.weight" not in state


def test_head_count_comes_from_open_clip_config(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    _write_checkpoint(tmp_path / "src", openclip_model)
    _convert(tmp_path / "src", tmp_path / "out")
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    # hidden 32 / head_width 16 = 2 heads; the old hidden // 64 would give 0.
    assert config["vision_config"]["num_attention_heads"] == 2


def test_head_count_cli_override(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    _write_checkpoint(tmp_path / "src", openclip_model, with_config=False)
    _convert(tmp_path / "src", tmp_path / "out", text_heads=2, vision_heads=2)
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["text_config"]["num_attention_heads"] == 2


def test_indivisible_head_count_is_an_error(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    _write_checkpoint(tmp_path / "src", openclip_model, with_config=False)
    with pytest.raises(converter.ConversionError, match="not divisible"):
        _convert(tmp_path / "src", tmp_path / "out", text_heads=5, vision_heads=2)


def test_missing_key_names_the_key(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    src = _write_checkpoint(tmp_path / "src", openclip_model)
    path = src / "open_clip_model.safetensors"
    state = load_file(str(path))
    del state["transformer.resblocks.1.attn.in_proj_weight"]
    save_file(state, str(path))
    with pytest.raises(
        converter.ConversionError, match="transformer.resblocks.1.attn.in_proj_weight"
    ):
        _convert(src, tmp_path / "out")


def test_missing_layer_is_an_error(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    src = _write_checkpoint(tmp_path / "src", openclip_model)
    path = src / "open_clip_model.safetensors"
    state = {
        k: v
        for k, v in load_file(str(path)).items()
        if "resblocks.1." not in k or "visual" in k
    }
    save_file(state, str(path))
    with pytest.raises(converter.ConversionError, match="missing layers"):
        _convert(src, tmp_path / "out")


def test_unconverted_tensors_fail_instead_of_warn(
    tmp_path: Path, openclip_model: torch.nn.Module
) -> None:
    src = _write_checkpoint(tmp_path / "src", openclip_model)
    path = src / "open_clip_model.safetensors"
    state = load_file(str(path))
    state["something.unexpected"] = torch.zeros(2)
    save_file(state, str(path))
    with pytest.raises(converter.ConversionError, match="something.unexpected"):
        _convert(src, tmp_path / "out")


def test_not_an_openclip_checkpoint(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_file({"text_model.foo": torch.zeros(1)}, str(src / "model.safetensors"))
    with pytest.raises(converter.ConversionError, match="not an OpenCLIP"):
        _convert(src, tmp_path / "out")
