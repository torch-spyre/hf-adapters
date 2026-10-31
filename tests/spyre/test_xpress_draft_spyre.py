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
Spyre test for the XPress speculative-decoding DRAFT adapter (``hf_xpress_qwen3``).

Loads the real drafter checkpoint (vocab weights from its target), moves it to
the card, and runs ``propose_block`` against a synthetic target context. Checks
against the fp32 CPU oracle (``tests/_xpress_reference.py``, transcribed from
the speculators training code):

- block hidden states and base logits: per-row cosine >= ``MIN_COSINE``;
- refiner, isolated from backbone drift: the device refiner bias vs the oracle's
  (``reference_refiner_bias``) on the same inputs (the device hidden states and
  previous tokens from the oracle's Jacobi draft over the device base logits),
  per-slot cosine >= ``MIN_BIAS_COSINE``;
- proposals: finite, in-vocab, ``block_size - 1`` of them.

Top-1 draft agreement with the oracle is *reported*, not asserted: on a
synthetic context the base logits are nearly flat, so near-ties dominate and
the Jacobi passes amplify every flip. Measured on the pod (bf16/fp16, positions
0/4000, 5 seeds each), a correct refiner agreed on 9-15 of 15 slots even with
the backbone isolated, while deliberately broken refiners still reached 8 — no
threshold separates them. The bias cosine does (see ``MIN_BIAS_COSINE``). The
real-context acceptance check comes with the target integration (P0/P2).

The target embedding (and, for an untied target such as Qwen3-8B, its own
``lm_head``) is read from the target checkpoint in the HF cache
(``load_target_embedding`` / ``load_target_lm_head`` read only their shards),
or from the safetensors files in ``XPRESS_TARGET_EMBED`` /
``XPRESS_TARGET_LM_HEAD`` if set. Both the device model and the fp32 oracle
use the same matrices, so a wrong head cannot hide behind a matching oracle.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_xpress_draft_spyre.py
"""

import os
import time

import pytest
import torch
from safetensors.torch import load_file

from tests._xpress_reference import (
    reference_block_hidden,
    reference_jacobi,
    reference_refiner_bias,
)
from tests.model_registry import CAUSAL_LM_MODELS, XPRESS_PATHS

MIN_COSINE = 0.99
# Measured over 20 contexts per dtype: a correct refiner scored >= 0.99990 (bf16)
# and >= 0.99996 (fp16); the nearest bug (wrong predecessor token) <= 0.977 and
# zeroed/transposed mixer or zeroed global summary <= 0.82.
MIN_BIAS_COSINE = 0.995
CTX_LEN = 40
ANCHOR_ID = 1000


def _single_tensor(path: str) -> torch.Tensor:
    (tensor,) = load_file(path).values()
    return tensor


def _target_vocab(module, ckpt: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The target's embedding and its own lm_head (``None`` if tied)."""
    embed_file = os.environ.get("XPRESS_TARGET_EMBED")
    if embed_file:
        head_file = os.environ.get("XPRESS_TARGET_LM_HEAD")
        head = _single_tensor(head_file) if head_file else None
        return _single_tensor(embed_file), head
    target = next(i["target"] for i in CAUSAL_LM_MODELS.values() if i["path"] == ckpt)
    return module.load_target_embedding(target), module.load_target_lm_head(target)


def _context(model, start: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    width = len(model.target_layer_ids) * model.config.hidden_size
    positions = start + torch.arange(CTX_LEN)
    return torch.randn(1, CTX_LEN, width, generator=g), positions


def _min_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min()
    )


def _isolated_refiner_check(model, head_w, hidden, base):
    """Check the refiner on the device's own backbone outputs.

    Returns ``(bias_cos, ref_draft)``: the worst-slot cosine of the device
    refiner bias against ``reference_refiner_bias``, and the oracle's Jacobi
    draft over the device base logits (to report draft agreement against).
    """
    fns, head = model._spyre_xpress_fns, model.xpress_head
    vocab = model.config.vocab_size
    dev_hidden = hidden.to("cpu").float()
    ref_draft = reference_jacobi(
        head_w, base, dev_hidden, ANCHOR_ID, ANCHOR_ID, model.num_passes
    )[0]
    # Previous tokens as propose_block builds them: [predecessor, anchor, draft[:-1]].
    prev = torch.cat([torch.tensor([ANCHOR_ID, ANCHOR_ID]), ref_draft[:-1]])[None]
    projected = fns["project_hidden"](hidden)
    token_latent = torch.nn.functional.embedding(prev, head.token_embed.weight)
    latent = fns["latent"](projected, token_latent.to(hidden.device, projected.dtype))
    dev_bias = fns["readout"](latent).to("cpu").float()[..., :vocab]
    ref_bias = reference_refiner_bias(head_w, prev, dev_hidden)
    return _min_row_cosine(dev_bias[:, 1:], ref_bias[:, 1:]), ref_draft


# pos4000 checks the prebuilt RoPE cache far from position 0. It is not an
# inv_freq-precision check: attention only sees relative angles, so that error
# scales with the key span (<= kv_pad here), not the absolute position.
@pytest.mark.parametrize("start", [0, 4000], ids=["pos0", "pos4000"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("ckpt", XPRESS_PATHS)
@torch.no_grad()
def test_xpress_propose_block(ckpt, dtype, start):
    import torch_spyre  # noqa: F401

    from hf_adapters import hf_common
    from hf_adapters._xpress_refiner import canonical_head_weights
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    # hf_common.DEVICE (spyre:$LOCAL_RANK) is bound into the adapters at import;
    # select the same card instead of overriding it.
    torch.spyre.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    module = resolve_adapter_module(ckpt)
    assert module.__name__.endswith("hf_xpress_qwen3"), module.__name__

    embed, lm_head = _target_vocab(module, ckpt)
    config, raw = module.read_checkpoint(ckpt)  # read once, for both models
    oracle = module.build_model(
        config, raw, embed, dtype=torch.float32, target_lm_head_weight=lm_head
    )
    feats, positions = _context(oracle, start)
    ctx = slice(CTX_LEN - min(CTX_LEN, module.CTX_PAD), CTX_LEN)
    ref_hidden = reference_block_hidden(
        oracle, feats[:, ctx], positions[ctx], ANCHOR_ID
    )
    ref_base = oracle.lm_head(ref_hidden)
    head_w = {k: v.float() for k, v in canonical_head_weights(raw).items()}
    ref_draft = reference_jacobi(
        head_w, ref_base, ref_hidden, ANCHOR_ID, ANCHOR_ID, oracle.num_passes
    )[0]
    del oracle

    model = module.build_model(
        config, raw, embed, dtype=dtype, target_lm_head_weight=lm_head
    )
    del raw
    hf_common.move_model_to_spyre(model, module, dtype)

    hidden = module.run_draft_block(model, feats, positions, ANCHOR_ID)
    vocab = model.config.vocab_size
    base = model._spyre_xpress_fns["lm_head"](hidden).to("cpu").float()[..., :vocab]
    hidden_cos = _min_row_cosine(hidden.to("cpu"), ref_hidden)
    base_cos = _min_row_cosine(base, ref_base)
    bias_cos, iso_draft = _isolated_refiner_check(model, head_w, hidden, base)
    del head_w

    timings = {}
    for topc in (model.topc, 0):
        module.propose_block(model, feats, positions, ANCHOR_ID, topc=topc)  # warm up
        t0 = time.perf_counter()
        draft = module.propose_block(model, feats, positions, ANCHOR_ID, topc=topc)
        timings[topc] = time.perf_counter() - t0
        assert draft.shape == (model.block_size - 1,)
        assert bool(((draft >= 0) & (draft < vocab)).all())
        if topc == 0:
            agree = int((draft == ref_draft).sum())
            iso_agree = int((draft == iso_draft).sum())

    slots = model.block_size - 1
    print(
        f"\n[xpress {dtype} pos{start}+] hidden cos={hidden_cos:.5f} base-logit cos={base_cos:.5f} "
        f"refiner-bias cos={bias_cos:.5f} "
        f"draft agreement (topc=0) vs fp32 oracle {agree}/{slots}, "
        f"vs oracle refiner on device backbone {iso_agree}/{slots} "
        f"propose latency: "
        + ", ".join(f"topc={c}: {t * 1e3:.1f} ms" for c, t in timings.items())
    )
    assert torch.isfinite(hidden.to("cpu")).all()
    assert hidden_cos >= MIN_COSINE, f"block hidden cosine {hidden_cos:.5f}"
    assert base_cos >= MIN_COSINE, f"base logit cosine {base_cos:.5f}"
    assert bias_cos >= MIN_BIAS_COSINE, f"refiner bias cosine {bias_cos:.5f}"
