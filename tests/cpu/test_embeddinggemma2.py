# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline parity with stock Transformers, including interleaved media."""

from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    EmbeddingGemma2Config,
    EmbeddingGemma2Model,
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextModel,
    Gemma4AudioConfig,
    Gemma4VisionConfig,
)

from hf_adapters import hf_embeddinggemma2
from hf_adapters.hf_common import SpyreUnsupportedModelError
from tests._multimodal_embedding_helpers import forbid_model_device_moves


def _eager(function: Any, **kwargs: Any) -> Any:
    return function


def _text_config() -> EmbeddingGemma2TextConfig:
    return EmbeddingGemma2TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        global_head_dim=256,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=64,
        embedding_dim=192,
        sliding_window=2,
        layer_types=["sliding_attention", "full_attention"],
    )


def _vision_config() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=32,
        standardize=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_text_parity(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, use_embeds: bool
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    model = EmbeddingGemma2TextModel(_text_config()).eval().to(dtype)
    # Nontrivial norm scales expose the fp32 multiply required by this model.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.5, 1.5)
    ids = torch.randint(3, 100, (2, 9))
    mask = torch.tensor([[1] * 7 + [0, 0], [0, 0] + [1] * 7])
    positions = torch.arange(9)[None].expand(2, -1) + torch.tensor([[0], [3]])
    with torch.inference_mode():
        inputs: dict[str, torch.Tensor] = {
            "attention_mask": mask,
            "position_ids": positions,
        }
        inputs["inputs_embeds" if use_embeds else "input_ids"] = (
            model.embed_tokens(ids) if use_embeds else ids
        )
        expected = model(**inputs).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
        tuple_output = model(**inputs, return_dict=False)
    assert actual.shape == (2, 9, 192)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(tuple_output[0], actual)
    # Padding queries may attend to their own diagonal to keep softmax finite;
    # their discarded outputs need not match HF's safe-SDPA padding outputs.
    actual, expected = actual[mask.bool()], expected[mask.bool()]
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual.float(), expected.float(), dim=-1
        )
        assert cosine.min() > 0.999


def _plain_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    is_causal: bool,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Expose all-masked rows instead of relying on CPU SDPA's safe softmax."""
    assert not is_causal
    if enable_gqa:
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    scores = query @ key.transpose(-2, -1) * scale + attn_mask
    return scores.softmax(dim=-1) @ value


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_mixed_lengths_with_unsafe_softmax(
    monkeypatch: pytest.MonkeyPatch, padding_side: str
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    config = _text_config()
    config.sliding_window = 512
    model = EmbeddingGemma2TextModel(config).eval()
    ids = torch.randint(3, 100, (2, 1200))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1] = False
    mask[1, :10] = padding_side == "right"
    mask[1, -10:] = padding_side == "left"
    positions = torch.arange(ids.shape[1])[None].expand_as(ids)
    with torch.inference_mode():
        # Reference forwards must precede adapter preparation.
        expected: list[torch.Tensor] = [
            model(
                ids[row, mask[row]][None],
                position_ids=positions[row, mask[row]][None],
            ).last_hidden_state[0]
            for row in range(ids.shape[0])
        ]
        monkeypatch.setattr(F, "scaled_dot_product_attention", _plain_softmax_attention)
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual: torch.Tensor = model(
            ids, attention_mask=mask, position_ids=positions
        ).last_hidden_state
    assert torch.isfinite(actual).all()
    for row, reference in enumerate(expected):
        torch.testing.assert_close(
            actual[row, mask[row]], reference, atol=3e-5, rtol=3e-5
        )


@pytest.mark.parametrize("tower_name", ["text", "vision"])
@pytest.mark.parametrize("config_return_dict", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"return_dict": None}, id="none"),
        pytest.param({"return_dict": True}, id="true"),
        pytest.param({"return_dict": False}, id="false"),
    ],
)
def test_return_dict_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tower_name: str,
    config_return_dict: bool,
    kwargs: dict[str, bool | None],
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    model = EmbeddingGemma2Model(
        EmbeddingGemma2Config(
            text_config=_text_config(),
            vision_config=_vision_config(),
            audio_config=None,
        )
    ).eval()
    tower: torch.nn.Module
    inputs: dict[str, torch.Tensor]
    if tower_name == "text":
        tower = model.language_model
        inputs = {"input_ids": torch.tensor([[3, 4, 5]])}
    else:
        tower = model.vision_tower
        inputs = {
            "pixel_values": torch.randn(1, 16, 12),
            "pixel_position_ids": torch.cartesian_prod(
                torch.arange(4), torch.arange(4)
            )[None],
        }
    tower.config.return_dict = config_return_dict
    hf_embeddinggemma2.prepare_for_spyre(model)
    with torch.inference_mode():
        result = tower(**inputs, **kwargs)
    requested = kwargs.get("return_dict")
    expected_dict = config_return_dict if requested is None else requested
    assert isinstance(result, tuple) is not expected_dict
    hidden: torch.Tensor = result.last_hidden_state if expected_dict else result[0]
    assert torch.isfinite(hidden).all()


@pytest.mark.parametrize("backend_enabled", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_sentence_transformers_device_opt_in(
    monkeypatch: pytest.MonkeyPatch, backend_enabled: bool, opt_in: bool
) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    monkeypatch.setattr(torch, "compile", _eager)
    backbone = EmbeddingGemma2TextModel(_text_config()).eval()
    model = SentenceTransformer(modules=[backbone], device="cpu")
    setattr(model, "_spyre_backend", backend_enabled)
    if opt_in:
        hf_embeddinggemma2.prepare_for_spyre(backbone)
    # Simulate mixed CPU/accelerator placement without requiring hardware.
    projection: torch.nn.Linear = backbone.layers[0].self_attn.q_proj
    projection.to("meta")

    def forbid_parameter_scan(
        *args: Any, **kwargs: Any
    ) -> Iterator[torch.nn.Parameter]:
        raise AssertionError("Device selection must not scan ST parameters")

    monkeypatch.setattr(model, "parameters", forbid_parameter_scan)
    assert model.device == torch.device("meta" if backend_enabled and opt_in else "cpu")
    # The override follows current placement rather than caching a stale device.
    projection.to_empty(device="cpu")
    assert model.device == torch.device("cpu")


class _PlacementFrontend(torch.nn.Module):
    """Exercise real ST encode routing with simulated accelerator weights."""

    def __init__(
        self: "_PlacementFrontend", backbone: EmbeddingGemma2TextModel
    ) -> None:
        super().__init__()
        self.auto_model = backbone

    def tokenize(
        self: "_PlacementFrontend", texts: list[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

    def forward(
        self: "_PlacementFrontend", features: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # Meta stands in for Spyre in this placement test; numerical parity is
        # tested separately. The real adapter also returns embeddings on CPU.
        return {"sentence_embedding": torch.ones((features["input_ids"].shape[0], 4))}


@pytest.mark.parametrize(
    "resident,device",
    [
        ("meta", None),
        ("meta", "meta"),
        ("meta", torch.device("meta")),
        ("cpu", "cpu:0"),
    ],
)
def test_repeated_encode_preserves_mixed_placement(
    monkeypatch: pytest.MonkeyPatch, resident: str, device: str | torch.device | None
) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    monkeypatch.setattr(torch, "compile", _eager)
    backbone = EmbeddingGemma2TextModel(_text_config()).eval()
    hf_embeddinggemma2.prepare_for_spyre(backbone)
    model = SentenceTransformer(
        modules=[_PlacementFrontend(backbone)], backend="spyre", device="cpu"
    )
    backbone.layers.to(resident)
    with forbid_model_device_moves(model):
        for _ in range(2):
            output: torch.Tensor = model.encode(
                ["hello", "world"], device=device, convert_to_tensor=True
            )
            torch.testing.assert_close(output, torch.ones((2, 4)))
            assert backbone.embed_tokens.weight.device.type == "cpu"
            assert backbone.layers[0].self_attn.q_proj.weight.device.type == resident


def test_init_preserves_cpu_helper_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    monkeypatch.setattr(torch, "compile", _eager)
    backbone = EmbeddingGemma2TextModel(_text_config()).eval()
    hf_embeddinggemma2.prepare_for_spyre(backbone)
    buffers = torch.nn.Module()
    buffers.register_buffer("mask", torch.tensor([True, False]), persistent=False)
    buffers.register_buffer("offset", torch.tensor([16777217], dtype=torch.int64))
    buffers.register_buffer("count", torch.tensor([3], dtype=torch.int32))
    buffers.register_buffer("scale", torch.tensor([0.1234567]), persistent=False)
    backbone.embed_tokens.add_module("buffers_to_preserve", buffers)
    expected: dict[str, torch.Tensor] = {
        name: buffer.clone() for name, buffer in backbone.embed_tokens.named_buffers()
    }
    head = torch.nn.Linear(4, 4)
    head.register_buffer("scale", torch.tensor(0.1234567))
    model = SentenceTransformer(
        modules=[_PlacementFrontend(backbone), head], backend="spyre", device="meta"
    )
    # Meta represents the accelerator: no CPU helper may pass through it.
    model.to("meta", dtype=torch.bfloat16)
    assert backbone.layers[0].self_attn.q_proj.weight.device.type == "meta"
    assert backbone.embed_tokens.weight.device.type == "cpu"
    assert backbone.embed_tokens.weight.dtype == torch.bfloat16
    assert head.weight.device.type == "cpu"
    assert head.weight.dtype == torch.bfloat16
    torch.testing.assert_close(head.scale, torch.tensor(0.1234567), rtol=0, atol=0)
    for name, buffer in backbone.embed_tokens.named_buffers():
        assert buffer.device.type == "cpu"
        torch.testing.assert_close(buffer, expected[name], rtol=0, atol=0)
    assert "mask" not in buffers.state_dict()
    assert "scale" not in buffers.state_dict()


def test_shared_load_preserves_cpu_helper_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hf_adapters.hf_common as common

    monkeypatch.setattr(torch, "compile", _eager)
    model = EmbeddingGemma2TextModel(_text_config()).eval()
    expected = model.embed_tokens.embed_scale.clone()

    def move_to_meta(model: torch.nn.Module, dtype: torch.dtype) -> None:
        model.to(device="meta", dtype=dtype)

    monkeypatch.setattr(common, "_move_to_spyre_with_layout", move_to_meta)
    common.move_model_to_spyre(model, hf_embeddinggemma2, torch.bfloat16)
    assert model.embed_tokens.weight.device.type == "cpu"
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.embed_tokens.embed_scale, expected, rtol=0, atol=0)


@pytest.mark.parametrize("composite", [False, True])
def test_local_bf16_config_resolves_to_bf16(tmp_path: Path, composite: bool) -> None:
    from hf_adapters.auto_spyre_model import dtype_for_model_path

    text_config = _text_config()
    text_config.dtype = torch.bfloat16
    config: EmbeddingGemma2TextConfig | EmbeddingGemma2Config = text_config
    if composite:
        config = EmbeddingGemma2Config(
            text_config=text_config,
            vision_config=None,
            audio_config=None,
            dtype=torch.bfloat16,
        )
    config.save_pretrained(tmp_path)
    assert dtype_for_model_path(tmp_path, "spyre") == torch.bfloat16


def test_local_checkpoint_load_honors_dtype_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hf_adapters.auto_spyre_model import AutoSpyreModel

    monkeypatch.setattr(torch, "compile", _eager)
    model = EmbeddingGemma2TextModel(_text_config()).to(torch.bfloat16).eval()
    model.save_pretrained(tmp_path)
    default = AutoSpyreModel.from_pretrained(tmp_path)
    explicit = AutoSpyreModel.from_pretrained(tmp_path, dtype=torch.float32)
    assert default.dtype == torch.bfloat16
    assert explicit.dtype == torch.float32
    with torch.inference_mode():
        output: torch.Tensor = default(torch.tensor([[3, 4, 5]])).last_hidden_state
    assert torch.isfinite(output).all()
    with pytest.raises(SpyreUnsupportedModelError, match="float16 overflows"):
        AutoSpyreModel.from_pretrained(tmp_path, dtype=torch.float16)


@pytest.mark.parametrize("device", ["cpu", "meta:1", "cuda:0"])
def test_encode_rejects_conflicting_device(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    monkeypatch.setattr(torch, "compile", _eager)
    backbone = EmbeddingGemma2TextModel(_text_config()).eval()
    hf_embeddinggemma2.prepare_for_spyre(backbone)
    model = SentenceTransformer(
        modules=[_PlacementFrontend(backbone)], backend="spyre", device="cpu"
    )
    backbone.layers.to("meta")
    with (
        forbid_model_device_moves(model),
        pytest.raises(ValueError, match="cannot relocate"),
    ):
        model.encode(["hello"], device=device)


@pytest.mark.parametrize("backend_enabled,opt_in", [(False, True), (True, False)])
def test_inference_placement_without_opt_in_uses_st(
    monkeypatch: pytest.MonkeyPatch, backend_enabled: bool, opt_in: bool
) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    monkeypatch.setattr(torch, "compile", _eager)
    backbone = EmbeddingGemma2TextModel(_text_config()).eval()
    if opt_in:
        hf_embeddinggemma2.prepare_for_spyre(backbone)
    model = SentenceTransformer(modules=[_PlacementFrontend(backbone)], device="cpu")
    setattr(model, "_spyre_backend", backend_enabled)
    # Stock ST must still honor relocation for models outside the opt-in.
    assert model._resolve_inference_device("meta") == torch.device("meta")
    assert all(parameter.device.type == "meta" for parameter in model.parameters())


@pytest.mark.parametrize(
    "method_name",
    ["forward", "get_image_features", "get_video_features", "get_audio_features"],
)
def test_cpu_frontend_moves_unknown_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    model = EmbeddingGemma2Model(
        EmbeddingGemma2Config(
            text_config=_text_config(), vision_config=None, audio_config=None
        )
    ).eval()

    def capture(
        self: EmbeddingGemma2Model, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    def simulated_cpu_copy(self: torch.Tensor) -> torch.Tensor:
        assert self.device.type == "meta"
        return torch.zeros(self.shape, dtype=self.dtype)

    monkeypatch.setattr(model, method_name, MethodType(capture, model))
    hf_embeddinggemma2.prepare_for_spyre(model)
    monkeypatch.setattr(torch.Tensor, "cpu", simulated_cpu_copy)
    tokens = torch.empty((1, 3), dtype=torch.int64, device="meta")
    args, kwargs = getattr(model, method_name)(
        tokens,
        tokens,
        token_type_ids=tokens,
        mm_token_type_ids=tokens,
        return_dict=None,
    )
    for tensor in (*args, kwargs["token_type_ids"], kwargs["mm_token_type_ids"]):
        assert tensor.device.type == "cpu"
    assert kwargs["return_dict"] is None


@pytest.mark.parametrize("modality", ["image", "video"])
def test_interleaved_vision_parity(
    monkeypatch: pytest.MonkeyPatch, modality: str
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    config = EmbeddingGemma2Config(
        text_config=_text_config(),
        vision_config=_vision_config(),
        audio_config=None,
        image_token_id=125,
        video_token_id=126,
        audio_token_id=127,
    )
    model = EmbeddingGemma2Model(config).eval()
    coords = torch.cartesian_prod(torch.arange(4), torch.arange(4))[None]
    pixels = torch.randn(1, 16, 12)
    token = config.image_token_id if modality == "image" else config.video_token_id
    inputs: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([[3, token, token, token, token, 4, 5]])
    }
    if modality == "image":
        inputs.update(pixel_values=pixels, image_position_ids=coords)
    else:
        inputs.update(
            pixel_values_videos=pixels,
            video_position_ids=coords,
            num_frames_per_video=torch.tensor([1]),
        )
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert "vision_tower.patch_embedder" in model._spyre_cpu_submodules


def test_reject_fp16() -> None:
    model = EmbeddingGemma2TextModel(_text_config()).half()
    with pytest.raises(SpyreUnsupportedModelError, match="float16 overflows"):
        hf_embeddinggemma2.prepare_for_spyre(model)


@pytest.mark.parametrize("rope_type", ["dynamic", "longrope"])
def test_reject_sequence_length_dependent_rope(rope_type: str) -> None:
    config: EmbeddingGemma2TextConfig = _text_config()
    rope: dict[str, object] = {"rope_type": rope_type, "rope_theta": 1000000.0}
    if rope_type == "dynamic":
        rope["factor"] = 2.0
    else:
        half: int = config.per_layer_config["full_attention"].head_dim // 2
        rope.update(
            short_factor=[1.0] * half,
            long_factor=[2.0] * half,
            original_max_position_embeddings=config.max_position_embeddings // 2,
        )
    config.rope_parameters["full_attention"] = rope
    model = EmbeddingGemma2TextModel(config).to(torch.bfloat16)
    with pytest.raises(SpyreUnsupportedModelError, match=f"full_attention={rope_type}"):
        hf_embeddinggemma2.prepare_for_spyre(model)


def test_audio_cpu_tower_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    audio = Gemma4AudioConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        output_proj_dims=64,
        subsampling_conv_channels=[128, 4],
    )
    model = EmbeddingGemma2Model(
        EmbeddingGemma2Config(
            text_config=_text_config(),
            audio_config=audio,
            vision_config=None,
            image_token_id=125,
            video_token_id=126,
            audio_token_id=127,
        )
    ).eval()
    features = torch.randn(1, 64, 128)
    feature_mask = torch.ones(1, 64, dtype=torch.bool)
    with torch.inference_mode():
        audio_output = model.get_audio_features(features, feature_mask)
        count = int(audio_output.attention_mask.sum())
        ids = torch.tensor([[3] + [127] * count + [4]])
        expected = model(
            input_ids=ids, input_features=features, input_features_mask=feature_mask
        ).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(
            input_ids=ids, input_features=features, input_features_mask=feature_mask
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert "audio_tower" in model._spyre_cpu_submodules


def test_fullgraph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_fn = torch.compile

    def trace(function: Any, **kwargs: Any) -> Any:
        return compile_fn(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", trace)
    model = EmbeddingGemma2TextModel(_text_config()).eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        expected = model(ids).last_hidden_state
        hf_embeddinggemma2.prepare_for_spyre(model)
        actual = model(ids).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.slow
@pytest.mark.model_harness("multimodal_embedding")
@pytest.mark.parametrize("model_path", ["google/embeddinggemma-2"])
def test_sentence_transformers_checkpoint(
    monkeypatch: pytest.MonkeyPatch, model_path: str
) -> None:
    """Public ST loading/processing/pooling against the full checkpoint."""
    import gc

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401
    from tests._multimodal_embedding_helpers import TEXT_PROMPTS, bounded_cpu_threads

    monkeypatch.setattr(torch, "compile", _eager)
    image = Image.new("RGB", (64, 64), color=(150, 80, 35))
    with bounded_cpu_threads(), torch.inference_mode():
        model = SentenceTransformer(
            model_path, device="cpu", model_kwargs={"dtype": torch.bfloat16}
        )
        expected = model.encode(
            TEXT_PROMPTS + [{"image": image}], convert_to_tensor=True
        )
        del model
        gc.collect()
        model = SentenceTransformer(model_path, backend="spyre", device="cpu")
        actual = model.encode(TEXT_PROMPTS + [{"image": image}], convert_to_tensor=True)
    assert actual.shape == (4, 768)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(actual.float(), expected.float())
    assert cosine.min() > 0.99
