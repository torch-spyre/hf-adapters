"""Stock HF parity for EmbeddingGemma 2's heterogeneous bidirectional layers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("transformers.models.embedding_gemma2")
from transformers import (  # noqa: E402
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextModel,
)

from hf_adapters import hf_embeddinggemma2  # noqa: E402


@pytest.mark.parametrize("seq_len", [7, 65])
def test_embeddinggemma2_text_parity(
    seq_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(17)
    config = EmbeddingGemma2TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        hidden_size_per_layer_input=128,
        embedding_dim=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        sliding_window=4,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        per_layer_config={"01": {"head_dim": 256, "num_key_value_heads": 1}},
    )
    model = EmbeddingGemma2TextModel(config).eval()
    input_ids = torch.randint(3, 128, (2, seq_len))
    mask = torch.ones_like(input_ids)
    mask[1, -3:] = 0
    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
    monkeypatch.setattr(hf_embeddinggemma2, "DEVICE", "cpu")

    def identity_compile(fn: Any, **kwargs: Any) -> Any:
        return fn

    monkeypatch.setattr(torch, "compile", identity_compile)
    hf_embeddinggemma2.prepare_for_spyre(model)
    with torch.no_grad():
        actual = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
    assert actual.shape == (2, seq_len, 64)
    torch.testing.assert_close(
        actual[mask.bool()], expected[mask.bool()], atol=2e-5, rtol=2e-5
    )


def test_embeddinggemma2_rejects_fp16() -> None:
    config = EmbeddingGemma2TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        hidden_size_per_layer_input=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=["full_attention"],
    )
    model = EmbeddingGemma2TextModel(config).half()
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        hf_embeddinggemma2.prepare_for_spyre(model)
