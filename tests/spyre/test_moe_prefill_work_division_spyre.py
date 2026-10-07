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

"""Exercise the real prefill driver and its compiled expert loop for #618."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import fresh_inductor_cache
from torch_spyre._inductor.spyre_kernel import _iter_op_specs
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

from hf_adapters import hf_qwen3_5_moe
from hf_adapters.hf_common import moe_prefill_all_experts
from hf_adapters.hf_gemma4_moe import Gemma4MoEBlock
from hf_adapters.hf_olmoe import OlmoeMoEBlock

pytestmark = pytest.mark.requires_spyre


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    ("block_type", "activation"),
    [(Gemma4MoEBlock, "gelu_tanh"), (OlmoeMoEBlock, "silu")],
)
def test_prefill_driver_preserves_token_work_division(
    monkeypatch, batch, block_type, activation
):
    # Keep Gemma 4 26B's expert geometry: smaller matrix shapes can choose all
    # cores even without a hint and would miss the performance regression.
    experts, tokens, hidden, intermediate = 128, 512, 2816, 704
    torch.manual_seed(0)
    x = torch.randn(batch, tokens // batch, hidden, dtype=torch.float16) * 0.1
    gate = torch.randn(experts, hidden, intermediate, dtype=torch.float16) * 0.01
    up = torch.randn_like(gate) * 0.01
    down = torch.randn(experts, intermediate, hidden, dtype=torch.float16) * 0.01
    selected = torch.rand(tokens, experts).topk(8, dim=-1).indices
    routes = torch.zeros(tokens, experts).scatter(-1, selected, 0.125).half()

    # FP32 reference keeps the check independent of device accumulation order.
    xf = x.float().reshape(tokens, hidden)
    expected = torch.zeros_like(xf)
    for expert in range(experts):
        gate_out = xf @ gate[expert].float()
        activated = (
            F.gelu(gate_out, approximate="tanh")
            if activation == "gelu_tanh"
            else F.silu(gate_out)
        )
        activated *= xf @ up[expert].float()
        expected += (activated @ down[expert].float()) * routes[:, expert, None]

    weights = SimpleNamespace(
        gate_proj=dma_moe_expert_weight_to_spyre(gate),
        up_proj=dma_moe_expert_weight_to_spyre(up),
        down_proj=dma_moe_expert_weight_to_spyre(down),
    )
    device_routes = routes.unsqueeze(-1).to("spyre")
    device_x = x.to("spyre")

    def ffn(hidden_states, layer_scalar=None):
        result = moe_prefill_all_experts(
            hidden_states.reshape(tokens, hidden),
            device_routes,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            activation,
        )
        return result.reshape_as(hidden_states)

    def attention(hidden_states, freqs, mask, key_cache, value_cache, index):
        return hidden_states, key_cache, value_cache

    # Use the production forward method, including its eager/compiled boundary.
    # Attention is independent of the FFN work division and needs no checkpoint.
    driver = SimpleNamespace(
        experts=weights,
        _compiled_prefill_attn=attention,
        _compiled_prefill_ffn=torch.compile(ffn, dynamic=False, fullgraph=True),
    )
    emitted = []
    real_sdsc = SpyreAsyncCompile.sdsc

    def capture(self, name, specs, *args, **kwargs):
        emitted.extend(_iter_op_specs(specs))
        return real_sdsc(self, name, specs, *args, **kwargs)

    monkeypatch.setattr(SpyreAsyncCompile, "sdsc", capture)
    call_args = (driver, device_x, None, None, None, None, None)
    if block_type is Gemma4MoEBlock:
        call_args += (None,)
    torch._dynamo.reset()
    with fresh_inductor_cache(), torch.no_grad():
        actual, _, _ = block_type.forward(*call_args)
        torch.testing.assert_close(
            actual.cpu().float(), expected.reshape_as(x), atol=2e-5, rtol=2e-2
        )
        # The cache-hit path must also run correctly after annotations are reset.
        repeated, _, _ = block_type.forward(*call_args)
        torch.testing.assert_close(repeated.cpu(), actual.cpu(), atol=0, rtol=0)

    matmuls = [op for op in emitted if op.op == "batchmatmul"]
    assert len(matmuls) == 3
    for op in matmuls:
        dimensions = list(op.iteration_space.values())
        # The backend can retain B and S as separate axes (B=2, S=256,
        # splits 2x16). Require all 32 cores to partition aggregate tokens.
        token_dimensions = [
            (size, split)
            for size, split in dimensions
            if size != 1 and size in (batch, tokens // batch, tokens)
        ]
        assert math.prod(size for size, _ in token_dimensions) == tokens, dimensions
        assert math.prod(split for _, split in token_dimensions) == 32, dimensions
        assert math.prod(split for _, split in dimensions) == 32, dimensions


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("attention_type", ["full_attention", "linear_attention"])
def test_qwen_prefill_driver_preserves_token_work_division(
    monkeypatch, batch, attention_type
):
    # Qwen3.5-35B-A3B geometry also leaves most cores idle without input names.
    experts, tokens, hidden, intermediate = 256, 512, 2048, 512
    torch.manual_seed(0)
    x = torch.randn(batch, tokens // batch, hidden, dtype=torch.float16) * 0.1
    gate = torch.randn(experts, hidden, intermediate, dtype=torch.float16) * 0.01
    up = torch.randn_like(gate) * 0.01
    down = torch.randn(experts, intermediate, hidden, dtype=torch.float16) * 0.01
    selected = torch.rand(tokens, experts).topk(8, dim=-1).indices
    routes = torch.zeros(tokens, experts).scatter(-1, selected, 0.125).half()

    xf = x.float().reshape(tokens, hidden)
    expected = torch.zeros_like(xf)
    for expert in range(experts):
        activated = F.silu(xf @ gate[expert].float()) * (xf @ up[expert].float())
        expected += (activated @ down[expert].float()) * routes[:, expert, None]

    weights = SimpleNamespace(
        gate_proj=dma_moe_expert_weight_to_spyre(gate),
        up_proj=dma_moe_expert_weight_to_spyre(up),
        down_proj=dma_moe_expert_weight_to_spyre(down),
    )
    device_x = x.reshape(tokens, hidden).to("spyre")
    device_routes = routes.to("spyre")
    residual = x.to("spyre")
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=weights),
        self_attn=None,
        input_layernorm=None,
        post_attention_layernorm=None,
    )

    # Keep both production prefill dispatchers and their compiled routed-expert
    # function. Stub unrelated stages at their existing compilation boundaries;
    # normalization supplies the flattened [T, H] input used by real Qwen blocks.
    stage_stubs = {
        "finish_attention": lambda residual, *args: residual,
        "normalize_ffn_input": lambda hidden_states: device_x,
        "finish_ffn": lambda hidden_states, x, routed: routed,
    }
    real_compile = torch.compile

    def compile_routed_only(function, **kwargs):
        if function.__name__ == "prefill_routed_experts":
            return real_compile(function, **kwargs)
        return stage_stubs.get(function.__name__, function)

    monkeypatch.setattr(
        hf_qwen3_5_moe, "_moe_prefill_routing", lambda *args: device_routes
    )
    monkeypatch.setattr(
        hf_qwen3_5_moe, "_make_dense_linear_attention_block", lambda layer: (None,) * 16
    )
    with monkeypatch.context() as construction:
        construction.setattr(torch, "compile", compile_routed_only)
        if attention_type == "full_attention":
            block = hf_qwen3_5_moe._make_attention_block(layer, None, None, 64, 8, 64)
            prefill, call_args = block[6], (residual, None, None)
        else:
            block = hf_qwen3_5_moe._make_linear_attention_block(layer, 8, 64)
            prefill, call_args = block[14], (residual,)

    emitted = []
    real_sdsc = SpyreAsyncCompile.sdsc

    def capture(self, name, specs, *args, **kwargs):
        emitted.extend(_iter_op_specs(specs))
        return real_sdsc(self, name, specs, *args, **kwargs)

    monkeypatch.setattr(SpyreAsyncCompile, "sdsc", capture)
    torch._dynamo.reset()
    with fresh_inductor_cache(), torch.no_grad():
        actual = prefill(*call_args)
        torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-5, rtol=2e-2)
        repeated = prefill(*call_args)
        torch.testing.assert_close(repeated.cpu(), actual.cpu(), atol=0, rtol=0)

    matmuls = [op for op in emitted if op.op == "batchmatmul"]
    assert len(matmuls) == 3
    for op in matmuls:
        # T and M are both 512, so identify token axes by their presence only
        # in the left operand, rather than by their extent or generated name.
        input_axes = [
            {
                symbol
                for coord in arg.device_coordinates
                for symbol in coord.free_symbols
            }
            for arg in op.args
            if arg.is_input
        ]
        token_dimensions = [
            op.iteration_space[symbol] for symbol in input_axes[0] - input_axes[1]
        ]
        assert math.prod(size for size, _ in token_dimensions) == tokens
        assert math.prod(split for _, split in token_dimensions) == 32
        assert math.prod(split for _, split in op.iteration_space.values()) == 32
