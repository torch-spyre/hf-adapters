# Copyright 2025 The Torch-Spyre Authors.
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

"""
CPU tests for the XPress drafter adapter (``hf_xpress_qwen3``) and its refiner
head (``_xpress_refiner``), on a tiny random-weight drafter.

Everything is compared against ``tests/_xpress_reference.py`` — a plain-PyTorch
transcription of the speculators training code — so the Spyre idioms (stick
padding, matmul RoPE, folded mixer, additive masks, CPU/device split) are
checked against the training semantics rather than against themselves.

Usage::

    pytest tests/cpu/test_xpress_cpu.py -v
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from tests._xpress_reference import (
    reference_allowed,
    reference_block_hidden,
    reference_jacobi,
    reference_refiner_bias,
    reference_sequential,
)

VOCAB = 1000  # not a stick multiple: exercises the padded readout / lm_head
HIDDEN = 256
BLOCK = 8
RANK = 32
MLP_HIDDEN = 64
TAPS = 2
WINDOW = 6
CTX_PAD = 16
MASK_ID = 4


def _config() -> Qwen3Config:
    return Qwen3Config(
        architectures=["Qwen3XPressModel"],
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=WINDOW,
        use_sliding_window=True,
        max_window_layers=3,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        tie_word_embeddings=True,
        block_size=BLOCK,
        dflash_config={"mask_token_id": MASK_ID, "target_layer_ids": [1, 2]},
        xpress_rank=RANK,
        xpress_mlp_hidden=MLP_HIDDEN,
        xpress_block_size=BLOCK,
        xpress_num_passes=6,
        xpress_topc=64,
        sample_from_anchor=False,
    )


def _training_head_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """Refiner weights in speculators' training layout, scaled so the bias
    actually moves argmaxes (std 0.02 would make every refinement a no-op)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, std=0.2):
        return torch.randn(*shape, generator=g) * std

    return {
        "token_embed.weight": rnd(VOCAB, RANK, std=1.0),
        "down_h.weight": rnd(RANK, HIDDEN, std=0.1),
        "down_g.weight": rnd(RANK, HIDDEN, std=0.1),
        "in_proj.weight": rnd(RANK, 3 * RANK),
        "mix_l": rnd(RANK, BLOCK, BLOCK, std=0.3),
        "mlp_gate.weight": rnd(2 * RANK, RANK),
        "mlp_up.weight": rnd(2 * RANK, RANK),
        "mlp_down.weight": rnd(RANK, 2 * RANK),
        "readout.weight": rnd(VOCAB, RANK, std=1.0),
    }


_TRAINING_TO_PUBLISHED = {
    "token_embed.weight": "w1.weight",
    "down_h.weight": "down_h.weight",
    "down_g.weight": "down_g.weight",
    "in_proj.weight": "in_proj.weight",
    "mix_l": "mix.L",
    "mlp_gate.weight": "mlp.gate_proj.weight",
    "mlp_up.weight": "mlp.up_proj.weight",
    "mlp_down.weight": "mlp.down_proj.weight",
    "readout.weight": "w2.weight",
}


def _published(head_w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"xpress_head.{_TRAINING_TO_PUBLISHED[k]}": v for k, v in head_w.items()}


def _backbone_weights(model, seed: int = 1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, p in model.named_parameters():
        if name.startswith(("xpress_head.", "embed_tokens.", "lm_head.")):
            continue
        if name.endswith("norm.weight"):
            out[name] = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        else:
            out[name] = 0.05 * torch.randn(p.shape, generator=g)
    return out


def _target_embedding(seed: int = 2) -> torch.Tensor:
    return torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(seed))


def _loaded_drafter(module, head_seed: int = 0):
    model = module.build_drafter(_config())
    weights = _backbone_weights(model) | _published(_training_head_weights(head_seed))
    module.load_drafter_weights(model, weights, _target_embedding())
    return model.float().eval()


def _unwrap_xpress(model) -> None:
    """Run the compiled callables eagerly on CPU (see conftest unwrap), with the
    fp32 RoPE cache that ``move_model_to_spyre`` would set for an fp32 model."""

    def orig(fn):
        return getattr(fn, "_orig_mod", getattr(fn, "_torchdynamo_orig_callable", fn))

    model._spyre_compiled_blocks = [orig(b) for b in model._spyre_compiled_blocks]
    model._spyre_xpress_fns = {k: orig(f) for k, f in model._spyre_xpress_fns.items()}
    model._spyre_rope.set_dtype(torch.float32)


def _context(ctx_len: int = 10, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(1, ctx_len, TAPS * HIDDEN, generator=g)
    return feats, torch.arange(ctx_len)


@pytest.fixture
def module():
    from hf_adapters import hf_xpress_qwen3

    return hf_xpress_qwen3


@pytest.fixture
def refiner_cls():
    from hf_adapters._xpress_refiner import XPressRefinerHead

    return XPressRefinerHead


def _head(refiner_cls, weights):
    head = refiner_cls(VOCAB, HIDDEN, BLOCK, rank=RANK, mlp_hidden=MLP_HIDDEN)
    head.load_checkpoint_weights(weights)
    return head.float().eval()


# ---------------------------------------------------------------------------
# Refiner head
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_refiner_bias_matches_training_formula(refiner_cls):
    w = _training_head_weights()
    head = _head(refiner_cls, {f"refiner_head.{k}": v for k, v in w.items()})
    g = torch.Generator().manual_seed(5)
    hidden = torch.randn(1, BLOCK, HIDDEN, generator=g)
    prev = torch.randint(0, VOCAB, (1, BLOCK), generator=g)

    # Folding reassociates fp32 sums: allow that rounding, not a formula change.
    torch.testing.assert_close(
        head.bias(prev, hidden),
        reference_refiner_bias(w, prev, hidden),
        rtol=1e-3,
        atol=1e-3,
    )


@torch.no_grad()
def test_published_and_training_keys_load_identically(refiner_cls):
    w = _training_head_weights()
    training = _head(refiner_cls, {f"refiner_head.{k}": v for k, v in w.items()})
    published = _head(refiner_cls, _published(w))
    for (name, a), (_, b) in zip(
        training.state_dict().items(), published.state_dict().items()
    ):
        torch.testing.assert_close(a, b, msg=name)


def test_loader_rejects_missing_or_mixed_keys(refiner_cls):
    w = _published(_training_head_weights())
    head = refiner_cls(VOCAB, HIDDEN, BLOCK, rank=RANK, mlp_hidden=MLP_HIDDEN)
    with pytest.raises(ValueError, match="missing"):
        head.load_checkpoint_weights(
            {k: v for k, v in w.items() if not k.endswith("w2.weight")}
        )
    with pytest.raises(ValueError, match="exactly one"):
        head.load_checkpoint_weights(w | {"refiner_head.mix_l": w["xpress_head.mix.L"]})


@torch.no_grad()
def test_refiner_mixer_is_causal(refiner_cls):
    """Changing slot j's previous token never moves the bias of a slot < j —
    the property that makes the Jacobi fixed point exist."""
    head = _head(refiner_cls, _published(_training_head_weights()))
    g = torch.Generator().manual_seed(6)
    hidden = torch.randn(1, BLOCK, HIDDEN, generator=g)
    prev = torch.randint(0, VOCAB, (1, BLOCK), generator=g)
    base = head.bias(prev, hidden)
    j = BLOCK // 2
    changed = prev.clone()
    changed[0, j] = (prev[0, j] + 1) % VOCAB
    moved = head.bias(changed, hidden)
    torch.testing.assert_close(moved[:, :j], base[:, :j])
    assert not torch.allclose(moved[:, j:], base[:, j:])


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_draft_mask_matches_training_pattern(module, layer_type):
    ctx_positions = torch.arange(10)
    kv_pad = 32
    masks = module.build_draft_masks(
        ctx_positions,
        ctx_pad=CTX_PAD,
        kv_pad=kv_pad,
        block_size=BLOCK,
        sliding_window=WINDOW,
        dtype=torch.float32,
    )
    allowed = masks[layer_type][0, 0] == 0
    window = WINDOW if layer_type == "sliding_attention" else None
    expected = reference_allowed(ctx_positions, BLOCK, window)

    assert masks[layer_type].shape == (1, 1, BLOCK, kv_pad)
    assert torch.equal(allowed[:, :10], expected[:, :10])  # real context
    assert not allowed[:, 10:CTX_PAD].any()  # context padding
    assert torch.equal(allowed[:, CTX_PAD : CTX_PAD + BLOCK], expected[:, 10:])
    assert not allowed[:, CTX_PAD + BLOCK :].any()  # kv tail padding
    assert torch.isfinite(masks[layer_type]).all()  # finite fill, not -inf


# ---------------------------------------------------------------------------
# Drafter end to end (prepare_for_spyre path, run eagerly on CPU)
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("ctx_len", [10, CTX_PAD, CTX_PAD + 5])
def test_draft_block_matches_reference(module, ctx_len):
    """Concat-KV blocks, matmul RoPE, padded K/V and the per-layer masks match
    the training semantics, including context padding, the sliding window and
    (ctx_len > ctx_pad) the fixed-width most-recent-context window."""
    model = _loaded_drafter(module)
    feats, positions = _context(ctx_len)
    kept = slice(max(0, ctx_len - CTX_PAD), ctx_len)
    expected = reference_block_hidden(model, feats[:, kept], positions[kept], 7)

    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    if ctx_len > CTX_PAD:
        with pytest.warns(UserWarning, match="truncated"):
            actual = module.run_draft_block(model, feats, positions, anchor_id=7)
    else:
        actual = module.run_draft_block(model, feats, positions, anchor_id=7)

    torch.testing.assert_close(actual.float(), expected, rtol=1e-4, atol=1e-4)


@torch.no_grad()
@pytest.mark.parametrize(
    "feats_shape, positions, anchor, match",
    [
        ((1, 0, TAPS * HIDDEN), torch.arange(0), 7, "at least one"),
        ((1, 10, TAPS * HIDDEN), torch.arange(9), 7, "one position per"),
        ((1, 10, HIDDEN), torch.arange(10), 7, "width"),
        ((2, 10, TAPS * HIDDEN), torch.arange(10), 7, r"\[1, L"),
        ((1, 3, TAPS * HIDDEN), torch.tensor([0, 2, 1]), 7, "increasing"),
        ((1, 10, TAPS * HIDDEN), torch.arange(10), VOCAB, "anchor_id"),
    ],
)
def test_draft_block_rejects_bad_context(module, feats_shape, positions, anchor, match):
    model = _loaded_drafter(module)
    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    with pytest.raises(ValueError, match=match):
        module.run_draft_block(model, torch.zeros(feats_shape), positions, anchor)


def test_prepare_keeps_cpu_only_modules_and_pads_readout(module):
    from hf_adapters.hf_common import padded_vocab_size

    model = _loaded_drafter(module)
    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    assert {"fc", "hidden_norm"} <= set(model._spyre_cpu_submodules)
    readout = model.xpress_readout.weight
    assert readout.shape == (padded_vocab_size(VOCAB, RANK, 4), RANK)
    assert torch.equal(readout[:VOCAB], model.xpress_head.readout.weight)
    assert not readout[VOCAB:].any()


@torch.no_grad()
@pytest.mark.parametrize("topc", [0, VOCAB, 64])
@pytest.mark.parametrize("num_passes", [0, 1, 6])
def test_propose_block_matches_reference(module, topc, num_passes):
    model = _loaded_drafter(module)
    feats, positions = _context()
    hidden = reference_block_hidden(model, feats, positions, 7)
    base = model.lm_head(hidden)
    w = _training_head_weights()

    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    draft = module.propose_block(
        model,
        feats,
        positions,
        anchor_id=7,
        predecessor_id=11,
        num_passes=num_passes,
        topc=topc,
    )

    assert draft.shape == (BLOCK - 1,)
    if topc in (0, VOCAB):
        expected = reference_jacobi(w, base, hidden, 7, 11, num_passes)[0]
        assert torch.equal(draft, expected)
    else:
        # Top-C restricts each pass to the base logits' top-C candidates.
        top = base[0, 1:].topk(topc, dim=-1).indices
        assert all(int(t) in set(top[i].tolist()) for i, t in enumerate(draft))


@torch.no_grad()
def test_refinement_is_not_a_noop(module):
    """Guard against vacuous comparisons: these weights do move the drafts."""
    model = _loaded_drafter(module)
    feats, positions = _context()
    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    kwargs = dict(anchor_id=7, predecessor_id=11, topc=0)
    base = module.propose_block(model, feats, positions, num_passes=0, **kwargs)
    refined = module.propose_block(model, feats, positions, num_passes=6, **kwargs)
    assert not torch.equal(base, refined)


@torch.no_grad()
def test_jacobi_reaches_sequential_fixed_point(module):
    """With block-1 passes, Jacobi equals left-to-right greedy refinement."""
    model = _loaded_drafter(module)
    feats, positions = _context()
    hidden = reference_block_hidden(model, feats, positions, 7)
    base = model.lm_head(hidden)

    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    draft = module.propose_block(
        model,
        feats,
        positions,
        anchor_id=7,
        predecessor_id=11,
        num_passes=BLOCK - 1,
        topc=0,
    )
    expected = reference_sequential(_training_head_weights(), base, hidden, 7, 11)
    assert torch.equal(draft, expected[0])


@torch.no_grad()
def test_predecessor_defaults_to_anchor(module):
    """Serving convention (vLLM#54448): the anchor is its own predecessor."""
    model = _loaded_drafter(module)
    feats, positions = _context()
    module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
    _unwrap_xpress(model)
    kwargs = dict(anchor_id=7, num_passes=6, topc=0)
    default = module.propose_block(model, feats, positions, **kwargs)
    explicit = module.propose_block(model, feats, positions, predecessor_id=7, **kwargs)
    assert torch.equal(default, explicit)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _write_checkpoint(tmp_path, module) -> None:
    model = module.build_drafter(_config())
    weights = _backbone_weights(model) | _published(_training_head_weights())
    save_file(
        {k: v.contiguous() for k, v in weights.items()}, tmp_path / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps(_config().to_dict()))


@torch.no_grad()
def test_load_model_from_checkpoint_dir(module, tmp_path):
    _write_checkpoint(tmp_path, module)
    embed = _target_embedding()
    model = module.load_model(
        str(tmp_path), target_embed_weight=embed, dtype=torch.float32
    )

    assert model.block_size == BLOCK
    assert model.mask_token_id == MASK_ID
    assert model.num_passes == 6 and model.topc == 64
    torch.testing.assert_close(model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.lm_head.weight, embed)  # tied to the target
    assert not any(p.isnan().any() for p in model.parameters())


@torch.no_grad()
def test_bf16_load_keeps_fp32_rope_and_one_vocab_matrix(module, tmp_path):
    """A bf16 inv_freq drifts RoPE angles by radians at large positions, and the
    dtype cast must not leave two copies of the target's vocab matrix."""
    _write_checkpoint(tmp_path, module)
    model = module.load_model(
        str(tmp_path), target_embed_weight=_target_embedding(), dtype=torch.bfloat16
    )
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.lm_head.weight is model.embed_tokens.weight

    pos = torch.tensor([[4000]])
    _, sin = model.rotary_emb(torch.zeros(1, 1, HIDDEN), pos)
    head_dim = _config().head_dim
    inv_freq = 1.0 / 1_000_000 ** (torch.arange(0, head_dim, 2).double() / head_dim)
    expected = torch.sin(4000 * inv_freq).float()
    torch.testing.assert_close(sin[0, 0, : head_dim // 2], expected, atol=1e-3, rtol=0)


def test_build_leaves_callers_config_untouched(module):
    config = _config()
    config._attn_implementation = "eager"
    before = config.to_dict()
    model = module.build_drafter(config)
    assert config._attn_implementation == "eager"
    assert config.to_dict() == before
    assert model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("sharded", [False, True])
def test_load_target_embedding_picks_text_decoder(module, tmp_path, sharded):
    """Gemma4-style multimodal targets carry several embeddings; take the text
    decoder's, reading only the shard that holds it."""
    embed = _target_embedding()
    tensors = {
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.embed_tokens_per_layer.weight": torch.zeros(3, 2),
        "model.vision_tower.embed_tokens.weight": torch.ones(3, 2),
    }
    if sharded:
        weight_map = {}
        for i, (name, t) in enumerate(tensors.items()):
            save_file({name: t}, tmp_path / f"shard-{i}.safetensors")
            weight_map[name] = f"shard-{i}.safetensors"
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(tensors, tmp_path / "model.safetensors")
    torch.testing.assert_close(module.load_target_embedding(str(tmp_path)), embed)


def _target_lm_head(seed: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, HIDDEN, generator=g) * 0.02


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_untied_target_keeps_its_own_lm_head(module, tmp_path, dtype):
    """Qwen3-8B-style targets (tie_word_embeddings=False) store a separate
    lm_head: the drafter's base logits must use it, not the input embedding."""
    _write_checkpoint(tmp_path, module)
    embed, head = _target_embedding(), _target_lm_head()
    model = module.load_model(
        str(tmp_path),
        target_embed_weight=embed,
        target_lm_head_weight=head,
        dtype=dtype,
    )
    torch.testing.assert_close(model.embed_tokens.weight, embed.to(dtype))
    torch.testing.assert_close(model.lm_head.weight, head.to(dtype))
    assert model.lm_head.weight is not model.embed_tokens.weight


def test_untied_lm_head_drives_proposals(module):
    """Swapping the target lm_head changes the drafts (it is really used)."""
    feats, positions = _context()
    drafts = []
    for head in (None, _target_lm_head()):
        model = module.build_drafter(_config())
        weights = _backbone_weights(model) | _published(_training_head_weights())
        module.load_drafter_weights(
            model, weights, _target_embedding(), target_lm_head_weight=head
        )
        model = model.float().eval()
        module.prepare_for_spyre(model, ctx_pad=CTX_PAD)
        _unwrap_xpress(model)
        drafts.append(module.propose_block(model, feats, positions, 7, num_passes=0))
    assert not torch.equal(drafts[0], drafts[1])


def test_load_rejects_mismatched_lm_head(module):
    model = module.build_drafter(_config())
    weights = _backbone_weights(model) | _published(_training_head_weights())
    with pytest.raises(ValueError, match="lm_head"):
        module.load_drafter_weights(
            model,
            weights,
            _target_embedding(),
            target_lm_head_weight=torch.zeros(VOCAB, HIDDEN + 1),
        )


@pytest.mark.parametrize("sharded", [False, True])
def test_load_target_lm_head(module, tmp_path, sharded):
    """Returns the stored lm_head of an untied target, None for a tied one."""
    embed, head = _target_embedding(), _target_lm_head()
    tensors = {"model.embed_tokens.weight": embed, "lm_head.weight": head}
    if sharded:
        weight_map = {}
        for i, (name, t) in enumerate(tensors.items()):
            save_file({name: t}, tmp_path / f"shard-{i}.safetensors")
            weight_map[name] = f"shard-{i}.safetensors"
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(tensors, tmp_path / "model.safetensors")
    torch.testing.assert_close(module.load_target_lm_head(str(tmp_path)), head)

    tied = tmp_path / "tied"
    tied.mkdir()
    save_file({"model.embed_tokens.weight": embed}, tied / "model.safetensors")
    assert module.load_target_lm_head(str(tied)) is None


def test_load_rejects_incomplete_backbone(module, tmp_path):
    model = module.build_drafter(_config())
    weights = _backbone_weights(model) | _published(_training_head_weights())
    weights.pop("layers.0.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="layers.0.self_attn.q_proj.weight"):
        module.load_drafter_weights(model, weights, _target_embedding())


def test_build_rejects_unsupported_layouts(module):
    for override in (
        {"sample_from_anchor": True},
        {"xpress_block_size": BLOCK * 2},
        {"layer_types": ["sliding_attention"] * 3, "sliding_window": None},
        {"layer_types": ["sliding_attention", "chunked_attention", "full_attention"]},
        {"layer_types": ["sliding_attention", "full_attention"]},
    ):
        cfg = _config()
        for k, v in override.items():
            setattr(cfg, k, v)
        with pytest.raises(ValueError):
            module.build_drafter(cfg)


def test_resolver_routes_xpress_checkpoint(module, tmp_path):
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    _write_checkpoint(tmp_path, module)
    assert resolve_adapter_module(str(tmp_path)) is module
