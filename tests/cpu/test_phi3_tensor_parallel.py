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

"""CPU tests for Phi-3 tensor-parallel weight placement (issue #710).

Transformers' Phi-3 TP plan shards the fused ``qkv_proj`` / ``gate_up_proj``
weights by plain row count (``colwise_gather_output``). The resulting shards
do not respect Q/K/V or gate/up boundaries, so the adapter's prepare-time
split produced empty K/V projections on some ranks and crashed the Spyre DMA
("Invalid device storage offset"). The fix keeps the fused weights unsharded
on CPU and slices whole heads per rank in ``prepare_for_spyre``.

Geometry is a scaled-down Phi-4-mini: 24 Q heads, 8 KV heads (GQA 3:1), TP=4.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Phi3Config, Phi3ForCausalLM

from hf_adapters import hf_phi3
from hf_adapters.hf_common import allocate_kv_caches, kv_cache_shapes, set_rope_dtype
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_CPU_REPLICATED,
    prepare_spyre_tp_plan,
)

NUM_Q = 24
NUM_KV = 8
HEAD_DIM = 8
HIDDEN = 16
INTERMEDIATE = 32
TP_SIZE = 4


def _fused_qkv_weight(num_q=NUM_Q, num_kv=NUM_KV, head_dim=HEAD_DIM, hidden=HIDDEN):
    rows = (num_q + 2 * num_kv) * head_dim
    return torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)


def _full_split(weight, num_q=NUM_Q, num_kv=NUM_KV, head_dim=HEAD_DIM):
    q_dim = num_q * head_dim
    k_dim = num_kv * head_dim
    return (
        weight[:q_dim],
        weight[q_dim : q_dim + k_dim],
        weight[q_dim + k_dim :],
    )


def _heads(weight, head_dim=HEAD_DIM):
    return weight.view(-1, head_dim, weight.shape[-1])


def _attn_with_qkv(weight):
    attn = nn.Module()
    attn.qkv_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    attn.qkv_proj.weight = nn.Parameter(weight, requires_grad=False)
    return attn


def _tiny_phi3_on_meta():
    cfg = Phi3Config(
        vocab_size=64,
        hidden_size=NUM_Q * HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=NUM_Q,
        num_key_value_heads=NUM_KV,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    with torch.device("meta"):
        return Phi3ForCausalLM(cfg)


# ---------------------------------------------------------------------------
# Reproduction of issue #710
# ---------------------------------------------------------------------------


def test_split_fused_qkv_rejects_row_sharded_weight():
    """A colwise-sharded fused QKV shard must not be split as if it were whole.

    This mirrors what each rank received at TP=4: an equal block of rows from
    the fused weight. Before the fix, rank 0's shard split into a Q that held
    the whole shard and empty K/V, which crashed the later Spyre DMA.
    """
    full = _fused_qkv_weight()
    rank0_shard = full.chunk(TP_SIZE, dim=0)[0]

    with pytest.raises(ValueError, match="qkv_proj"):
        hf_phi3._split_fused_qkv(_attn_with_qkv(rank0_shard), NUM_Q, NUM_KV, HEAD_DIM)


def test_split_fused_qkv_full_weight_unchanged():
    full = _fused_qkv_weight()
    q, k, v = hf_phi3._split_fused_qkv(_attn_with_qkv(full), NUM_Q, NUM_KV, HEAD_DIM)
    ref_q, ref_k, ref_v = _full_split(full)

    torch.testing.assert_close(q.weight, ref_q)
    torch.testing.assert_close(k.weight, ref_k)
    torch.testing.assert_close(v.weight, ref_v)


# ---------------------------------------------------------------------------
# TP placement policy
# ---------------------------------------------------------------------------


def test_cpu_replicated_policy_covers_fused_projections():
    model = _tiny_phi3_on_meta()

    modules = hf_phi3.spyre_tp_cpu_replicated_modules(model, TP_SIZE)

    expected = {
        f"model.layers.{i}.{name}"
        for i in range(2)
        for name in ("self_attn.qkv_proj", "mlp.gate_up_proj")
    }
    assert modules == expected


def test_cpu_replicated_policy_overrides_hf_gather_plan():
    model = _tiny_phi3_on_meta()
    modules = hf_phi3.spyre_tp_cpu_replicated_modules(model, TP_SIZE)

    plan = prepare_spyre_tp_plan(
        model, dict(model.tp_plan or {}), cpu_replicated_modules=modules
    )

    for name in modules:
        assert plan[name] == SPYRE_CPU_REPLICATED


@pytest.mark.parametrize(
    ("num_q", "num_kv", "tp_size"),
    [
        (24, 8, 3),  # KV heads do not divide across ranks
        (24, 2, 4),  # fewer KV heads than ranks (grouped replication not supported)
        (22, 4, 4),  # Q heads do not divide across ranks
    ],
)
def test_cpu_replicated_policy_rejects_unsupported_head_layouts(num_q, num_kv, tp_size):
    cfg = Phi3Config(
        vocab_size=64,
        hidden_size=num_q * HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=num_q,
        num_key_value_heads=num_kv,
        pad_token_id=0,
    )
    with torch.device("meta"):
        model = Phi3ForCausalLM(cfg)

    with pytest.raises(ValueError, match="heads"):
        hf_phi3.spyre_tp_cpu_replicated_modules(model, tp_size)


def test_cpu_replicated_policy_accepts_one_kv_head_per_rank():
    cfg = Phi3Config(
        vocab_size=64,
        hidden_size=20 * HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=4,
        pad_token_id=0,
    )
    with torch.device("meta"):
        model = Phi3ForCausalLM(cfg)

    assert hf_phi3.spyre_tp_cpu_replicated_modules(model, 4)


def test_cpu_replicated_policy_rejects_indivisible_intermediate():
    cfg = Phi3Config(
        vocab_size=64,
        hidden_size=NUM_Q * HEAD_DIM,
        intermediate_size=30,
        num_hidden_layers=1,
        num_attention_heads=NUM_Q,
        num_key_value_heads=NUM_KV,
        pad_token_id=0,
    )
    with torch.device("meta"):
        model = Phi3ForCausalLM(cfg)

    with pytest.raises(ValueError, match="gate_up_proj"):
        hf_phi3.spyre_tp_cpu_replicated_modules(model, TP_SIZE)


# ---------------------------------------------------------------------------
# Per-rank QKV slicing
# ---------------------------------------------------------------------------


def _all_rank_qkv(weight, tp_size=TP_SIZE):
    return [
        hf_phi3._local_qkv_weights(weight, NUM_Q, NUM_KV, HEAD_DIM, rank, tp_size)
        for rank in range(tp_size)
    ]


def test_local_qkv_shapes_are_whole_heads():
    for q, k, v in _all_rank_qkv(_fused_qkv_weight()):
        assert q.shape == (NUM_Q // TP_SIZE * HEAD_DIM, HIDDEN)
        assert k.shape == (NUM_KV // TP_SIZE * HEAD_DIM, HIDDEN)
        assert v.shape == (NUM_KV // TP_SIZE * HEAD_DIM, HIDDEN)


def test_local_qkv_reassembles_to_full_split():
    full = _fused_qkv_weight()
    ranks = _all_rank_qkv(full)
    ref_q, ref_k, ref_v = _full_split(full)

    torch.testing.assert_close(torch.cat([r[0] for r in ranks]), ref_q)
    torch.testing.assert_close(torch.cat([r[1] for r in ranks]), ref_k)
    torch.testing.assert_close(torch.cat([r[2] for r in ranks]), ref_v)


def test_local_qkv_preserves_gqa_pairing():
    """Each local Q head must see the same KV head it pairs with globally."""
    full = _fused_qkv_weight()
    _, ref_k, ref_v = _full_split(full)
    group = NUM_Q // NUM_KV
    local_q_heads = NUM_Q // TP_SIZE

    for rank, (_, k, v) in enumerate(_all_rank_qkv(full)):
        for local_q in range(local_q_heads):
            global_q = rank * local_q_heads + local_q
            torch.testing.assert_close(
                _heads(k)[local_q // group], _heads(ref_k)[global_q // group]
            )
            torch.testing.assert_close(
                _heads(v)[local_q // group], _heads(ref_v)[global_q // group]
            )


def test_local_qkv_tp1_is_full_split():
    full = _fused_qkv_weight()
    local = hf_phi3._local_qkv_weights(full, NUM_Q, NUM_KV, HEAD_DIM, 0, 1)

    for got, ref in zip(local, _full_split(full)):
        torch.testing.assert_close(got, ref)


def test_local_qkv_returns_independent_tensors():
    full = _fused_qkv_weight()
    before = full.clone()
    q, _, _ = hf_phi3._local_qkv_weights(full, NUM_Q, NUM_KV, HEAD_DIM, 1, TP_SIZE)

    q.zero_()

    torch.testing.assert_close(full, before)


def test_local_qkv_rejects_sharded_input():
    rank0_shard = _fused_qkv_weight().chunk(TP_SIZE, dim=0)[0]

    with pytest.raises(ValueError, match="qkv_proj"):
        hf_phi3._local_qkv_weights(rank0_shard, NUM_Q, NUM_KV, HEAD_DIM, 0, TP_SIZE)


@pytest.mark.parametrize("rank", [-1, TP_SIZE])
def test_local_qkv_rejects_out_of_range_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        hf_phi3._local_qkv_weights(
            _fused_qkv_weight(), NUM_Q, NUM_KV, HEAD_DIM, rank, TP_SIZE
        )


# ---------------------------------------------------------------------------
# Per-rank gate/up slicing
# ---------------------------------------------------------------------------


def _fused_gate_up_weight():
    rows = 2 * INTERMEDIATE
    return torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN)


def test_local_gate_up_reassembles_to_full_halves():
    full = _fused_gate_up_weight()
    ref_gate, ref_up = full[:INTERMEDIATE], full[INTERMEDIATE:]

    ranks = [
        hf_phi3._local_gate_up_weights(full, rank, TP_SIZE) for rank in range(TP_SIZE)
    ]

    for gate, up in ranks:
        assert gate.shape == (INTERMEDIATE // TP_SIZE, HIDDEN)
        assert up.shape == (INTERMEDIATE // TP_SIZE, HIDDEN)
    torch.testing.assert_close(torch.cat([r[0] for r in ranks]), ref_gate)
    torch.testing.assert_close(torch.cat([r[1] for r in ranks]), ref_up)


def test_local_gate_up_pairs_matching_rows():
    """Rank-local gate row i and up row i must come from the same neuron."""
    full = _fused_gate_up_weight()
    shard = INTERMEDIATE // TP_SIZE

    for rank in range(TP_SIZE):
        gate, up = hf_phi3._local_gate_up_weights(full, rank, TP_SIZE)
        start = rank * shard
        torch.testing.assert_close(gate, full[start : start + shard])
        torch.testing.assert_close(
            up, full[INTERMEDIATE + start : INTERMEDIATE + start + shard]
        )


def test_local_gate_up_rejects_indivisible_intermediate():
    with pytest.raises(ValueError, match="gate_up_proj"):
        hf_phi3._local_gate_up_weights(_fused_gate_up_weight(), 0, 3)


@pytest.mark.parametrize("rank", [-1, TP_SIZE])
def test_local_gate_up_rejects_out_of_range_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        hf_phi3._local_gate_up_weights(_fused_gate_up_weight(), rank, TP_SIZE)


# ---------------------------------------------------------------------------
# Whole-model TP simulation: every rank runs in a thread, and the compiled
# all-reduce is replaced by a barrier that sums the ranks' partial results.
# ---------------------------------------------------------------------------


class _FakeMesh:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    @staticmethod
    def get_group():
        return SimpleNamespace(group_name="test_group")


class _ThreadAllReduce:
    """Sum one tensor per rank thread, like a blocking collective."""

    def __init__(self, world_size):
        self._barrier = threading.Barrier(world_size, timeout=120)
        self._slots = [None] * world_size
        self._local = threading.local()
        self.calls = 0

    def bind_rank(self, rank):
        self._local.rank = rank

    def __call__(self, tensor, group_name):
        assert group_name == "test_group"
        rank = self._local.rank
        self._slots[rank] = tensor
        self._barrier.wait()
        total = torch.stack(self._slots).sum(dim=0)
        if rank == 0:
            self.calls += 1
        self._barrier.wait()
        return total


def _tiny_phi3(head_dim, partial_rotary_factor):
    torch.manual_seed(0)
    cfg = Phi3Config(
        vocab_size=192,
        hidden_size=8 * head_dim,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        partial_rotary_factor=partial_rotary_factor,
        tie_word_embeddings=False,
    )
    return Phi3ForCausalLM(cfg).float().eval()


def _as_loaded_on_rank(reference, rank, tp_size):
    """Mirror the TP loader: fused weights whole, rowwise weights column-sharded."""
    model = copy.deepcopy(reference)
    for layer in model.model.layers:
        for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
            shard = proj.weight.data.chunk(tp_size, dim=1)[rank].clone()
            proj.weight = nn.Parameter(shard, requires_grad=False)
            proj.in_features = shard.shape[1]
            proj._hf_device_mesh = _FakeMesh(tp_size, rank)
    return model


def _run_prepared(model, input_ids):
    # move_model_to_spyre sets this; without it the RoPE cache keeps its default.
    set_rope_dtype(model, torch.float32)
    seq_len = input_ids.shape[1]
    cache_len = 2 * seq_len
    key_caches, value_caches = allocate_kv_caches(
        model, 1, cache_len, torch.float32, device="cpu"
    )
    mask = torch.ones(seq_len, cache_len, dtype=torch.bool).tril()[None, None]
    with torch.no_grad():
        return hf_phi3._run_backbone_forward(
            model,
            input_ids,
            torch.arange(seq_len)[None],
            mask,
            key_caches,
            value_caches,
            torch.arange(seq_len),
        )


@pytest.mark.parametrize(
    ("head_dim", "partial_rotary_factor"),
    [
        pytest.param(128, 0.75, id="phi4-partial-rope"),
        pytest.param(96, 1.0, id="phi35-padded-head"),
    ],
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_tp_ranks_reproduce_stock_hf(
    monkeypatch, head_dim, partial_rotary_factor, tp_size
):
    monkeypatch.setattr(hf_phi3.torch, "compile", lambda fn, **_: fn)
    all_reduce = _ThreadAllReduce(tp_size)
    monkeypatch.setattr(hf_phi3, "spyre_compiled_all_reduce", all_reduce)

    reference = _tiny_phi3(head_dim, partial_rotary_factor)
    input_ids = torch.randint(1, 64, (1, 8))
    with torch.no_grad():
        expected = reference.model(input_ids).last_hidden_state

    def run_rank(rank):
        all_reduce.bind_rank(rank)
        model = _as_loaded_on_rank(reference, rank, tp_size)
        hf_phi3.prepare_for_spyre(model)
        return model, _run_prepared(model, input_ids)

    with ThreadPoolExecutor(max_workers=tp_size) as pool:
        results = list(pool.map(run_rank, range(tp_size)))

    # One reduction after o_proj and one after down_proj in each layer.
    assert all_reduce.calls == 2 * reference.config.num_hidden_layers
    for model, output in results:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        work_head_dim = model._spyre_head_dim
        assert (
            kv_cache_shapes(model)
            == [(4 // tp_size, work_head_dim, work_head_dim)]
            * reference.config.num_hidden_layers
        )
        for layer in model.model.layers:
            assert layer.self_attn.qkv_proj is None
            assert layer.mlp.gate_up_proj is None


def test_tp1_keeps_stock_modules_and_config_cache_shape(monkeypatch):
    monkeypatch.setattr(hf_phi3.torch, "compile", lambda fn, **_: fn)
    monkeypatch.setattr(
        hf_phi3,
        "spyre_compiled_all_reduce",
        lambda *_: pytest.fail("TP=1 must not all-reduce"),
    )
    model = _tiny_phi3(128, 0.75)
    input_ids = torch.randint(1, 64, (1, 8))
    with torch.no_grad():
        expected = model.model(input_ids).last_hidden_state

    hf_phi3.prepare_for_spyre(model)

    torch.testing.assert_close(
        _run_prepared(model, input_ids), expected, rtol=1e-5, atol=1e-5
    )
    assert not hasattr(model, "_spyre_kv_shapes")
    assert model.model.layers[0].self_attn.qkv_proj is not None
