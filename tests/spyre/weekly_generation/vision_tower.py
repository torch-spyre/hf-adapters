"""Stock-HF versus Spyre feature comparison for registered standalone towers."""

import importlib
import json
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from transformers import (
    AutoConfig,
    Gemma4VisionModel,
    PixtralVisionModel,
    PreTrainedModel,
    SiglipVisionModel,
)

from hf_adapters.hf_common import DEVICE
from tests.model_registry import VISION_MODELS

MIN_COSINE = 0.99
MAX_RELATIVE_L2 = 0.2


def resolve_tower_adapter(model_path: str) -> ModuleType:
    """Resolve the tower role explicitly, rather than its checkpoint's text adapter."""
    for info in VISION_MODELS.values():
        if info["path"] == model_path and info.get("kind") == "tower":
            return importlib.import_module(f"hf_adapters.{info['adapter'][:-3]}")
    raise ValueError(f"No registered vision tower for {model_path}")


def load_stock_tower(model_path: str, dtype: torch.dtype) -> PreTrainedModel:
    """Materialize only vision weights, with strict stock-model state validation.

    Select tensors from checkpoint shards with safe_open instead of loading the
    full language decoder or all tensors in a shard into host memory.
    """
    config: Any = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    classes: dict[str, type[PreTrainedModel]] = {
        "siglip_vision_model": SiglipVisionModel,
        "pixtral": PixtralVisionModel,
        "gemma4_vision": Gemma4VisionModel,
    }
    tower_cls = classes[config.vision_config.model_type]
    # Keep non-persistent RoPE buffers on CPU; they are absent from the state dict.
    with init_empty_weights(include_buffers=False):
        tower = tower_cls(config.vision_config)
    expected: set[str] = set(tower.state_dict())
    try:
        index_path = hf_hub_download(model_path, "model.safetensors.index.json")
    except EntryNotFoundError:
        shards: list[str] = ["model.safetensors"]
    else:
        with open(index_path, encoding="utf-8") as stream:
            index: dict[str, Any] = json.load(stream)
        shards = sorted(
            {
                shard
                for key, shard in index["weight_map"].items()
                if "vision_tower." in key
            }
        )

    state: dict[str, torch.Tensor] = {}
    prefixes = (
        "model.vision_tower.vision_model.",
        "model.vision_tower.",
        "vision_tower.vision_model.",
        "vision_tower.",
    )
    for shard in shards:
        with safe_open(hf_hub_download(model_path, shard), framework="pt") as tensors:
            for key in tensors.keys():
                for prefix in prefixes:
                    name = key.removeprefix(prefix)
                    if name != key and name in expected:
                        state[name] = tensors.get_tensor(key)
                        break
    tower.load_state_dict(state, strict=True, assign=True)
    tower.to(dtype=dtype)
    tower.eval()
    tower.requires_grad_(False)
    return tower


def tower_inputs(tower: PreTrainedModel) -> dict[str, Any]:
    """Seeded canonical inputs appropriate to each tower's stock forward interface."""
    config: Any = tower.config
    generator = torch.Generator().manual_seed(771)
    dtype: torch.dtype = next(tower.parameters()).dtype
    if config.model_type == "gemma4_vision":
        side: int = config.pooling_kernel_size * 8
        coords = torch.cartesian_prod(torch.arange(side), torch.arange(side))
        return {
            "pixel_values": torch.randn(
                1,
                side * side,
                3 * config.patch_size**2,
                generator=generator,
                dtype=dtype,
            ),
            "pixel_position_ids": coords.unsqueeze(0),
        }
    side = config.image_size
    if config.model_type == "pixtral":
        side = min(side, 384) // config.patch_size * config.patch_size
    inputs: dict[str, Any] = {
        "pixel_values": torch.randn(1, 3, side, side, generator=generator, dtype=dtype)
    }
    if config.model_type == "pixtral":
        inputs["image_sizes"] = [(side, side)]
    return inputs


def compare_features(reference: torch.Tensor, actual: torch.Tensor) -> None:
    """Reject missing patches, NaN/Inf, direction drift, and large magnitude drift."""
    actual = actual.detach().cpu().float()
    reference = reference.detach().cpu().float()
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    assert actual.numel() > 0, "empty vision features"
    assert torch.isfinite(reference).all().item(), "stock tower produced NaN/Inf"
    assert torch.isfinite(actual).all().item(), "Spyre tower produced NaN/Inf"
    cosine = F.cosine_similarity(reference, actual, dim=-1).min().item()
    relative_l2 = (
        (actual - reference).norm() / reference.norm().clamp_min(1e-12)
    ).item()
    print(f"Tower features: min cosine={cosine:.6f}, relative L2={relative_l2:.6f}")
    assert cosine >= MIN_COSINE, f"tower cosine {cosine} < {MIN_COSINE}"
    assert relative_l2 <= MAX_RELATIVE_L2, f"tower relative L2 {relative_l2}"


def run_tower_compare(
    model_path: str, *, on_reference: Callable[[], None] | None = None
) -> None:
    """Run stock forward before preparation, then the compiled tower on Spyre."""
    adapter = resolve_tower_adapter(model_path)
    tower = load_stock_tower(model_path, torch.float16)
    inputs = tower_inputs(tower)
    with torch.no_grad():
        reference: torch.Tensor = (
            tower(**inputs).last_hidden_state.detach().float().cpu().clone()
        )
    assert torch.isfinite(reference).all().item(), "stock tower produced NaN/Inf"
    if on_reference is not None:
        on_reference()

    with torch.no_grad():
        if tower.config.model_type == "gemma4_vision":
            # Gemma's position lookup and spatial pooling stay on CPU, as in the VLM.
            model: Any = SimpleNamespace(model=SimpleNamespace(vision_tower=tower))
            adapter.prepare_for_spyre(model)
            torch.nn.Module.to(tower.encoder, DEVICE)
            actual: torch.Tensor = adapter.prefill_vision_tower(
                model, inputs["pixel_values"], inputs["pixel_position_ids"]
            )
        else:
            adapter.prepare_for_spyre(tower)
            torch.nn.Module.to(tower, DEVICE)
            actual = adapter.prefill_vision_tower(tower, **inputs)
    compare_features(reference, actual)
