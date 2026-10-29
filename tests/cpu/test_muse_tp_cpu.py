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

"""CPU coverage for Muse Glimmer tensor parallelism (``_muse_tp``) over gloo.

Two and four CPU ranks each shard the tiny model on the host; a prefill,
a tree verify, the full speculative decoder and the adapter's own
``generate`` (plain SDPA and the anchored sliding-window cache) must equal
the unsharded stock model (four ranks exercise KV heads replicated across
rank pairs).
"""

import os
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PARENTS = [-1, 0, 1, 2, 3, 4, 0, 1, 6, 2, 3, 8]


def _worker(rank, world, init_file, case, errors):
    loopback = "lo0" if os.uname().sysname == "Darwin" else "lo"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
        )
        # A fresh process: point hf_common at the CPU before anything reads it.
        from hf_adapters import hf_common

        hf_common.DEVICE = "cpu"
        _run_case(case, rank, world)
    except Exception:  # noqa: BLE001 - reported to the parent
        errors.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _Patch:
    """A minimal monkeypatch for the worker processes."""

    def __init__(self):
        self.undo = []

    def setattr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setenv(self, key, value):
        os.environ[key] = value

    def delenv(self, key, raising=True):
        os.environ.pop(key, None)


def _run_case(case, rank, world):
    from hf_adapters import _muse_dflash, _muse_tree, hf_common, hf_muse_glimmer
    from hf_adapters._muse_tree import MuseTreeVerifier
    from hf_adapters._spec_tree import tree_ancestors
    from hf_adapters.muse_spec_decode import TreeSpeculativeDecoder
    from tests._muse_helpers import (
        VOCAB,
        prepared_pair,
        stock_forward,
        tiny_assistant,
    )

    patch = _Patch()
    tol = dict(rtol=1e-4, atol=1e-4)
    # TP 4 needs 4 query-head groups; 2 KV heads are then replicated in pairs.
    heads, kv_heads = (8, 2) if world == 4 else (2, 1)
    inter = 384
    reference, adapted = prepared_pair(
        patch,
        _muse_tree,
        _muse_dflash,
        heads=heads,
        kv_heads=kv_heads,
        intermediate=inter,
        window=1024 if case == "generate" else 48,
    )
    assert adapted._spyre_tp.world == world
    cfg = hf_common.text_config(adapted.config)
    layer0 = hf_common.get_backbone(adapted).layers[0]
    assert (
        layer0.self_attn.q_proj.weight.shape[0]
        == cfg.num_attention_heads * 128 // world
    )
    assert layer0.mlp.down_proj.weight.shape[1] == cfg.intermediate_size // world
    g = torch.Generator().manual_seed(7)
    if case == "generate":
        # Plain generate through the shared Gemma 3 blocks, whose o / down are
        # all-reducing linears: SDPA (within the window) and anchored SWA
        # with 512-token prefill chunks (beyond it).
        for length, mode in ((40, None), (1100, "anchored")):
            prompt = torch.randint(2, VOCAB - 2, (1, length), generator=g)
            with torch.no_grad():
                want = reference.generate(
                    prompt, max_new_tokens=12, min_new_tokens=12, do_sample=False
                )
            got = hf_muse_glimmer.generate(adapted, prompt, max_new_tokens=12)
            assert getattr(adapted, "_spyre_swa_mode", None) == mode
            assert got.tolist() == want.tolist(), (length, rank)
        return
    prompt = torch.randint(2, VOCAB - 2, (1, 70), generator=g)
    ids = prompt[0].tolist()

    verifier = MuseTreeVerifier(
        adapted,
        len(PARENTS),
        prefill_chunk=64,
        capacity_multiple=512,
        tap_layers=(1, 4),
    )
    logits = verifier.prefill(prompt, max_new_tokens=32)
    # TP logits keep the padded vocabulary: Spyre cannot slice the gathered
    # tensor past the first shard on the device, so callers crop on the host.
    assert logits.shape[-1] > VOCAB and logits.shape[-1] % world == 0
    logits = logits[..., :VOCAB]
    expected = stock_forward(reference, ids).logits[0, -1]
    torch.testing.assert_close(logits[0, -1], expected, **tol)
    states = stock_forward(reference, ids, output_hidden_states=True).hidden_states
    torch.testing.assert_close(verifier.tap_features()[:, :256], states[2][0], **tol)
    tokens = [int(expected.argmax())] + torch.randint(
        2, VOCAB - 2, (len(PARENTS) - 1,), generator=g
    ).tolist()
    logits = verifier.verify(tokens, PARENTS)[..., :VOCAB]
    for node, path in enumerate(tree_ancestors(PARENTS)):
        want = stock_forward(reference, ids + [tokens[p] for p in path]).logits[0, -1]
        torch.testing.assert_close(logits[0, node], want, **tol)

    if case in ("decode", "decode_sharded_draft"):
        from tests._muse_helpers import HIDDEN, TAPS

        new = 40
        with torch.no_grad():
            greedy = reference.generate(
                prompt, max_new_tokens=new, min_new_tokens=new, do_sample=False
            )[0, prompt.shape[1] :]
        sharded = case == "decode_sharded_draft"
        # A sharded drafter needs KV heads for every rank: 4 q / 2 kv.
        assistant = tiny_assistant(heads=4, kv_heads=2) if sharded else tiny_assistant()
        if sharded:
            # The drafter's blocks must equal the unsharded assistant's.
            whole = tiny_assistant(heads=4, kv_heads=2)
            kwargs = dict(dtype=torch.float32, device="cpu", compile=False)
            one = _muse_dflash.MuseDFlashDrafter(whole, adapted, shard=False, **kwargs)
            two = _muse_dflash.MuseDFlashDrafter(
                tiny_assistant(heads=4, kv_heads=2), adapted, shard=True, **kwargs
            )
            assert two.n_heads == 2 and two.n_kv == 1
            rows = torch.randn(30, len(TAPS) * HIDDEN, generator=g)
            for d in (one, two):
                d.reset(64)
                d.append(rows)
            torch.testing.assert_close(two.draft(5), one.draft(5), **tol)
        decoder = TreeSpeculativeDecoder(
            adapted,
            assistant,
            width=8,
            top_k=2,
            prefill_chunk=64,
            drafter_kwargs=dict(
                dtype=torch.float32, device="cpu", compile=False, shard=sharded
            ),
        )
        decoder.verifier.capacity_multiple = 512
        out = decoder.generate(prompt, max_new_tokens=new)
        assert out[0].tolist() == greedy.tolist()


def _spawn(world, case):
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target=_worker, args=(r, world, f"{tmp}/init", case, errors))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=600)
    failures = []
    while not errors.empty():
        failures.append(errors.get())
    assert not failures, "\n".join(f"rank {r}:\n{tb}" for r, tb in failures)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


@pytest.mark.parametrize("world", [2, 4])
def test_tp_verify_matches_the_unsharded_reference(world):
    _spawn(world, "verify")


@pytest.mark.parametrize("world", [2, 4])
def test_tp_plain_generate_matches_the_unsharded_reference(world):
    _spawn(world, "generate")


def test_tp2_speculative_decoding_equals_greedy():
    _spawn(2, "decode")


def test_tp2_sharded_drafter_equals_the_whole_drafter_and_greedy():
    _spawn(2, "decode_sharded_draft")
