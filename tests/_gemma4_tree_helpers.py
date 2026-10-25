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

"""A tiny random Gemma 4 MoE through the real adapter preparation, on CPU.

Shared by the tree-verify and speculative-decoding CPU tests.
"""

import copy
import sys
import types

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from hf_adapters import _gemma4_tree, hf_gemma4_moe
from hf_adapters.hf_common import move_model_to_spyre

WINDOW = 64
# A 12-node tree: a spine 0-1-2-3-4-5 plus side branches.
PARENTS = [-1, 0, 1, 2, 3, 4, 0, 1, 6, 2, 3, 8]


def _prepare_on_cpu(model):
    """Run the adapter preparation with the two torch_spyre helpers it imports."""
    stub = types.ModuleType("torch_spyre")
    c = types.ModuleType("torch_spyre._C")
    c.get_elem_in_stick = lambda dtype: 64
    utils = types.ModuleType("torch_spyre.model_utils")
    utils.dma_moe_per_expert_scale_to_spyre = lambda scale: scale
    stub._C, stub.model_utils = c, utils
    names = ("torch_spyre", "torch_spyre._C", "torch_spyre.model_utils")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(dict(zip(names, (stub, c, utils))))
    try:
        move_model_to_spyre(model, hf_gemma4_moe, torch.float32)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _config():
    return Gemma4TextConfig(
        vocab_size=320,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=128,
        global_head_dim=128,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=WINDOW,
        enable_moe_block=True,
        num_experts=16,
        top_k_experts=4,
        moe_intermediate_size=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
    )


def _register_cpu_keep_by_index():
    """The MoE prefill router's ``spyre::keep_by_index``, for a CPU-only host.

    torch-spyre registers this op (with a CPU kernel) on import; here the
    same semantics come from a scatter.
    """
    if hasattr(torch.ops, "spyre") and hasattr(torch.ops.spyre, "keep_by_index"):
        return
    library = torch.library.Library("spyre", "FRAGMENT")
    library.define(
        "keep_by_index(Tensor values, Tensor indices, int dim, float fill) -> Tensor"
    )

    def keep_by_index(values, indices, dim, fill):
        kept = torch.full_like(values, fill)
        return kept.scatter(dim, indices.long(), values.gather(dim, indices.long()))

    library.impl("keep_by_index", keep_by_index, "CPU")
    _register_cpu_keep_by_index.library = library


def tiny_gemma4_moe(monkeypatch):
    """``(stock HF model, adapter-prepared copy)`` of a tiny random Gemma 4 MoE."""
    _register_cpu_keep_by_index()
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "1")
    monkeypatch.delenv("SPYRE_MOE_UNION_CAP", raising=False)
    monkeypatch.delenv("SPYRE_GEMMA4_FP8", raising=False)
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config()).eval().to(torch.float32)
    adapted = copy.deepcopy(reference)
    _prepare_on_cpu(adapted)
    return reference, adapted


def _unwrap(model):
    """Run the compiled callables eagerly (CPU, fast, same math)."""
    for block in hf_gemma4_moe._gemma4_backbone(model).layers:
        for name in (
            "_compiled_decode",
            "_compiled_prefill_attn",
            "_compiled_prefill_ffn",
        ):
            compiled = getattr(block, name)
            setattr(
                block, name, getattr(compiled, "_torchdynamo_orig_callable", compiled)
            )
    head = model._spyre_lm_head_forward
    model._spyre_lm_head_forward = getattr(head, "_torchdynamo_orig_callable", head)


def run_eagerly(monkeypatch):
    """Make ``torch.compile`` the identity for the verifier's own graphs."""
    monkeypatch.setattr(_gemma4_tree.torch, "compile", lambda fn, **_: fn)
    monkeypatch.setattr(hf_gemma4_moe.torch, "compile", lambda fn, **_: fn)


def _reference_next(reference, ids):
    with torch.no_grad():
        return reference(torch.tensor([ids])).logits[0, -1]
