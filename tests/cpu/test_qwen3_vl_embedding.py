# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline Qwen3-VL text, packed vision, and DeepStack parity with stock HF."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLTextConfig,
    Qwen3VLTextModel,
    Qwen3VLVisionConfig,
)

from hf_adapters import hf_qwen3, hf_qwen3_vl_embedding
from hf_adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    SpyreUnsupportedFeatureError,
    SpyreUnsupportedModelError,
)


def _eager(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return function


def _config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
            use_cache=False,
        ),
        vision_config=Qwen3VLVisionConfig(
            depth=3,
            hidden_size=128,
            intermediate_size=256,
            num_heads=2,
            out_hidden_size=128,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 2],
        ),
        image_token_id=126,
        video_token_id=125,
        vision_start_token_id=124,
        vision_end_token_id=123,
    )


def _assert_parity(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    assert torch.isfinite(actual).all()
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    else:
        assert (
            F.cosine_similarity(actual.float(), expected.float(), dim=-1).min() > 0.999
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("position_axes", [2, 3, 4])
def test_text_parity(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    padding_side: str,
    position_axes: int,
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(42)
    model = Qwen3VLTextModel(_config().text_config).eval().to(dtype)
    ids = torch.randint(3, 100, (2, 11))
    mask = torch.ones_like(ids)
    if padding_side == "left":
        mask[1, :4] = 0
    else:
        mask[1, -4:] = 0
    positions = torch.arange(11)[None].expand(2, -1)
    if position_axes > 2:
        positions = positions[None].repeat(position_axes, 1, 1)
        positions[-1] += 5
        positions[-2] += 2
    inputs: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": positions,
        "use_cache": False,
    }
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        hf_qwen3_vl_embedding.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
        tuple_output = model(**inputs, return_dict=False)
    _assert_parity(actual[mask.bool()], expected[mask.bool()], dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(tuple_output[0], actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_vision_parity(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(7)
    model = Qwen3VLModel(_config()).eval().to(dtype)
    # Two differently shaped images plus two video frames; each frame must
    # attend only within its own segment, including after stick padding.
    grid = torch.tensor([[1, 2, 4], [2, 4, 2]])
    pixels = torch.randn(24, 24, dtype=dtype)
    with torch.no_grad():
        for layer in model.visual.blocks:
            layer.attn.qkv.bias.uniform_(-0.1, 0.1)
    with torch.inference_mode():
        expected = model.visual(pixels, grid, return_dict=True)
        hf_qwen3_vl_embedding.prepare_for_spyre(model)
        actual = model.visual(pixels, grid, return_dict=True)
        repeated = model.visual(pixels, grid, return_dict=False)
    _assert_parity(actual.last_hidden_state, expected.last_hidden_state, dtype)
    _assert_parity(actual.pooler_output, expected.pooler_output, dtype)
    assert len(actual.deepstack_features) == 2
    for actual_feature, expected_feature in zip(
        actual.deepstack_features, expected.deepstack_features
    ):
        _assert_parity(actual_feature, expected_feature, dtype)
    torch.testing.assert_close(repeated[1], actual.pooler_output)
    assert model.visual.blocks[0].attn.head_dim == 128
    assert model.visual.blocks[0].attn.scaling == 64**-0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("modality", ["image", "video", "mixed"])
def test_multimodal_deepstack_parity(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, modality: str
) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(8)
    model = Qwen3VLModel(_config()).eval().to(dtype)
    ids = torch.tensor([[3, 124, 126, 126, 123, 5, 124, 125, 123, 7]])
    types = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 2, 0, 0]])
    inputs: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": types,
        "use_cache": False,
    }
    if modality in ("image", "mixed"):
        inputs.update(
            pixel_values=torch.randn(8, 24, dtype=dtype),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )
    else:
        ids[0, 2:4] = 9
        types[0, 2:4] = 0
    if modality in ("video", "mixed"):
        inputs.update(
            pixel_values_videos=torch.randn(4, 24, dtype=dtype),
            video_grid_thw=torch.tensor([[1, 2, 2]]),
        )
    else:
        ids[0, 7] = 9
        types[0, 7] = 0
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        hf_qwen3_vl_embedding.prepare_for_spyre(model)
        actual = model(**inputs).last_hidden_state
        # A subsequent text-only request must ignore stale multimodal rope_deltas.
        text_ids = torch.tensor([[3, 7, 11, 9]])
        text_actual = model(input_ids=text_ids, use_cache=False).last_hidden_state
    _assert_parity(actual, expected, dtype)
    assert text_actual.shape == (1, 4, 128)
    assert torch.isfinite(text_actual).all()


@pytest.mark.parametrize("mode", ["text", "vision"])
def test_fullgraph_capture(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    def capture(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        return torch_compile(function, backend="eager", dynamic=False, fullgraph=True)

    torch_compile = torch.compile
    monkeypatch.setattr(torch, "compile", capture)
    model = Qwen3VLModel(_config()).eval()
    with torch.inference_mode():
        hf_qwen3_vl_embedding.prepare_for_spyre(model)
        if mode == "text":
            output = model(
                input_ids=torch.tensor([[3, 9, 7]]), use_cache=False
            ).last_hidden_state
        else:
            output = model.visual(
                torch.randn(8, 24), torch.tensor([[1, 2, 4]]), return_dict=True
            ).pooler_output
    assert torch.isfinite(output).all()


def test_generation_cache_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new cache-free branch must preserve Qwen3 generation-cache writes."""
    monkeypatch.setattr(torch, "compile", _eager)
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
        )
    ).eval()
    hidden = torch.randn(1, 3, 128)
    rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    rope.set_dtype(torch.float32)
    frequencies = rope(hidden, torch.arange(3)[None])
    block = hf_qwen3._make_compiled_block(model.model.layers[0])
    mask = torch.zeros(1, 1, 3, 3).masked_fill_(
        ~torch.ones(3, 3, dtype=torch.bool).tril(), -torch.inf
    )
    with torch.inference_mode():
        no_cache, keys, values = block(hidden, frequencies, mask, None, None, None)
        key_cache = torch.zeros_like(keys)
        value_cache = torch.zeros_like(values)
        cached, stored_keys, stored_values = block(
            hidden, frequencies, mask, key_cache, value_cache, torch.arange(3)
        )
    torch.testing.assert_close(cached, no_cache)
    torch.testing.assert_close(stored_keys, keys)
    torch.testing.assert_close(stored_values, values)
    assert stored_keys is key_cache


def test_unsupported_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", _eager)
    model = Qwen3VLTextModel(_config().text_config).eval()
    hf_qwen3_vl_embedding.prepare_for_spyre(model)
    with pytest.raises(SpyreUnsupportedFeatureError, match="cache-free"):
        model(input_ids=torch.tensor([[3]]), use_cache=True)
    with pytest.raises(SpyreUnsupportedFeatureError, match="intermediate"):
        model(input_ids=torch.tensor([[3]]), output_hidden_states=True)
    with pytest.raises(SpyreUnsupportedFeatureError, match="2D"):
        model(input_ids=torch.tensor([[3]]), attention_mask=torch.zeros(1, 1, 1, 1))
    invalid = _config().text_config
    invalid.rope_parameters = {
        "rope_type": "dynamic",
        "rope_theta": 500000,
        "factor": 2,
        "mrope_section": [24, 20, 20],
    }
    with pytest.raises(SpyreUnsupportedModelError, match="dynamic"):
        hf_qwen3_vl_embedding.prepare_for_spyre(Qwen3VLTextModel(invalid))


@pytest.mark.slow
@pytest.mark.model_harness("multimodal_embedding")
@pytest.mark.parametrize("model_path", ["Qwen/Qwen3-VL-Embedding-2B"])
def test_sentence_transformers_checkpoint(
    monkeypatch: pytest.MonkeyPatch, model_path: str
) -> None:
    """Exercise public ST media processing, last-token pooling, and normalization."""
    import gc

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401
    from tests._multimodal_embedding_helpers import (
        TEXT_PROMPTS,
        bounded_cpu_threads,
        forbid_model_device_moves,
    )

    monkeypatch.setattr(torch, "compile", _eager)
    image = Image.new("RGB", (64, 64), color=(150, 80, 35))
    inputs: list[Any] = TEXT_PROMPTS + [
        image,
        {"text": "A colorful picture.", "image": image},
    ]
    with bounded_cpu_threads(), torch.inference_mode():
        model = SentenceTransformer(
            model_path, device="cpu", model_kwargs={"dtype": torch.bfloat16}
        )
        expected = model.encode(inputs, convert_to_tensor=True)
        del model
        gc.collect()
        model = SentenceTransformer(model_path, backend="spyre", device="cpu")
        assert model[0].auto_model.dtype == torch.bfloat16
        with forbid_model_device_moves(model):
            actual = model.encode(inputs, convert_to_tensor=True)
            repeated = model.encode(inputs[:1], convert_to_tensor=True)
    assert actual.shape == (5, 2048)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(repeated).all()
    assert F.cosine_similarity(actual.float(), expected.float()).min() > 0.99
    torch.testing.assert_close(
        actual.float().norm(dim=-1), torch.ones(5), atol=0.005, rtol=0
    )


def test_cpu_helpers_preserved_during_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every declared CPU helper must avoid even a temporary accelerator move."""
    from itertools import chain

    import hf_adapters.hf_common as common

    monkeypatch.setattr(torch, "compile", _eager)
    model = Qwen3VLModel(_config()).eval()
    expected: dict[str, torch.Tensor] = {
        name: buffer.clone() for name, buffer in model.named_buffers()
    }

    def move_to_meta(model: torch.nn.Module, dtype: torch.dtype) -> None:
        model.to(device="meta", dtype=dtype)

    monkeypatch.setattr(common, "_move_to_spyre_with_layout", move_to_meta)
    common.move_model_to_spyre(model, hf_qwen3_vl_embedding, torch.bfloat16)
    assert model.language_model.layers[0].self_attn.q_proj.weight.device.type == "meta"
    assert model.visual.blocks[0].attn.proj.weight.device.type == "meta"
    for path in model._spyre_cpu_submodules:
        helper: torch.nn.Module = model.get_submodule(path)
        assert all(
            tensor.device.type == "cpu"
            for tensor in chain(helper.parameters(), helper.buffers())
        )
        for name, buffer in helper.named_buffers():
            torch.testing.assert_close(
                buffer, expected[f"{path}.{name}"], rtol=0, atol=0
            )
