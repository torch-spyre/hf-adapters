# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare hybrid attention's CPU recurrence with stock Transformers."""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as dense
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

from hf_adapters.hf_qwen3_5 import (
    _causal_conv_decode,
    _causal_conv_prefill,
    _delta_recurrence,
    _linear_normalize_inputs,
    _prepare_linear_attention_constants,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 64])
def test_causal_convolution_matches_hf(dtype: torch.dtype, tokens: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(772)
        layer = dense.Qwen3_5GatedDeltaNet(
            dense.Qwen3_5TextConfig(
                hidden_size=256,
                linear_num_key_heads=1,
                linear_num_value_heads=1,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
            ),
            layer_idx=0,
        ).to(dtype)
        channels = layer.conv1d.weight.shape[0]
        x = torch.randn(2, channels, tokens, dtype=dtype) * 0.1
        state = torch.randn(2, channels, 64, dtype=dtype) * 0.1
    with torch.no_grad():
        expected = F.silu(
            F.conv1d(
                torch.cat((state[..., -(layer.conv_kernel_size - 1) :], x), dim=-1),
                layer.conv1d.weight,
                groups=channels,
            )
        )
        _prepare_linear_attention_constants(layer)
        if tokens == 1:
            actual, actual_state = _causal_conv_decode(
                x,
                state,
                layer._spyre_conv_taps_cpu,
                layer._spyre_conv_decode_matrices_cpu,
            )
        else:
            actual, actual_state = _causal_conv_prefill(
                x,
                state,
                layer._spyre_conv_taps,
                layer._spyre_conv_shift_matrices,
            )
    expected_state = torch.cat((state, x), dim=-1)[..., -64:]
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 4])
def test_delta_recurrence_matches_hf(
    family: str, dtype: torch.dtype, tokens: int
) -> None:
    # Keep the fixture independent of other seeded checkpoint regressions.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(772)
        config_cls, layer_cls, reference = (
            (
                dense.Qwen3_5TextConfig,
                dense.Qwen3_5GatedDeltaNet,
                dense.torch_recurrent_gated_delta_rule,
            )
            if family == "dense"
            else (
                moe.Qwen3_5MoeTextConfig,
                moe.Qwen3_5MoeGatedDeltaNet,
                moe.torch_recurrent_gated_delta_rule,
            )
        )
        config = config_cls(
            hidden_size=256,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
        layer = layer_cls(config, layer_idx=0)
        query = torch.randn(2, tokens, 256, dtype=dtype) * 0.1
        key = torch.randn_like(query) * 0.1
        value = torch.randn(2, tokens, 512, dtype=dtype) * 0.1
        a = torch.randn(2, tokens, 4, dtype=dtype)
        beta = torch.randn_like(a).sigmoid()
        a_log = torch.randn(4)
        dt_bias = torch.randn(4)
        state = torch.randn(2, 4, 128, 128) * 0.01

    expected, expected_state = reference(
        query.view(2, tokens, 2, 128).repeat_interleave(2, dim=2),
        key.view(2, tokens, 2, 128).repeat_interleave(2, dim=2),
        value.view(2, tokens, 4, 128),
        g=-a_log.exp() * F.softplus(a.float() + dt_bias),
        beta=beta,
        initial_state=state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    normalized_query, normalized_key, shaped_value = _linear_normalize_inputs(
        layer, query, key, value
    )
    actual, actual_state = _delta_recurrence(
        normalized_query,
        normalized_key,
        shaped_value,
        a,
        beta,
        state.clone(),
        a_log,
        dt_bias,
    )
    assert expected_state is not None
    # The cache is fp32 even for low-precision models; checking it catches
    # drift that can flip a later MoE routing decision near the top-k boundary.
    torch.testing.assert_close(actual_state, expected_state, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
