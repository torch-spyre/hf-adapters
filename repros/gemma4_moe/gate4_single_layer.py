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
Gate 4: Gemma 4 MoE decoder layer 0 parity on REAL 26B-A4B-it weights, on-card.

The prior gates proved the two device geometries (gate 1) and the device/host
route/permute/GEMM/combine round-trip (gate 2) on synthetic weights. This gate
closes the loop on the real checkpoint: it compares the adapter's split MoE
decoder block (compiled Gemma4 attention + compiled dense MLP +
host-orchestrated device FFN, ``hf_gemma4_moe._make_moe_block``) against a
pure-CPU fp32 reference built from the STOCK
``transformers`` ``Gemma4TextDecoderLayer.forward`` math, for layer 0.

Ordering rule (CLAUDE.md "Test ordering matters"): the CPU fp32 reference runs
BEFORE ``prepare_for_spyre`` (the Gemma4RMSNorm patch is global).

K=4, not the config's 8: ``prepare_for_spyre`` coerces ``top_k_experts=4`` for
bring-up, so this gate coerces the reference to K=4 too, for an apples-to-apples
compare. The divergence from the true-8 model is expected and tracked (spec §9).

Correctness criterion (matches gate 1/2 and docs/spyre-numerical-findings.md):
device fp16 output vs fp32 CPU reference, relative error with a clipped
denominator to avoid near-zero noise:

  denom = ref32.abs().clamp_min(1.0)
  rel   = |got_fp32 - ref32| / denom
  PASS iff  mean_rel < 0.02  and  max_rel < 0.5

A ``corrupted double-linked list`` SIGABRT on process teardown AFTER the
PASS/FAIL line is the known torch-spyre lifetime issue, not a compute failure.

Run:
  HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python3 repros/gemma4_moe/gate4_single_layer.py

Model loading: the full 26B (~49 GB fp16) is loaded on CPU via
``AutoModelForCausalLM.from_pretrained`` (host has ~1.3 TB free); only layer 0's
compiled block is exercised on-card.
"""

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/models/hf_cache/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Make ``hf_adapters`` importable when run directly (repo root is two levels up).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch_spyre

torch_spyre._autoload()

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from hf_adapters import hf_gemma4_moe
from hf_adapters.hf_common import (
    allocate_kv_caches,
    build_prefill_mask,
    get_model_dtype,
)
from hf_adapters.hf_gemma4 import _gemma4_backbone, _build_layer_masks

MODEL_PATH = "google/gemma-4-26B-A4B-it"
BLOCK_SIZE = 64
# The bring-up K the adapter pins; the CPU reference must match it.
K = hf_gemma4_moe._MOE_BRINGUP_K  # 4


def build_cpu_reference(layer, h_embed_fp32, position_ids, cfg, rope):
    """Run the STOCK Gemma4TextDecoderLayer.forward for layer 0 on CPU in fp32.

    ``layer`` is the real decoder layer (K already coerced to 4 on ``cfg``);
    ``h_embed_fp32`` is [B, L, H]. Builds the stock rotary position embeddings
    and an additive causal mask, then calls the layer's own ``forward`` (no
    KV cache — single prefill block), returning the fp32 layer output [B, L, H].
    """
    bsz, seq_len, _ = h_embed_fp32.shape
    layer_type = cfg.layer_types[0]

    # Stock rotary_emb returns (cos, sin) for this layer's type, fp32 here.
    cos, sin = rope(h_embed_fp32, position_ids, layer_type=layer_type)

    # Additive causal mask [B, 1, L, L] (fp32). Layer 0 is a sliding layer, but
    # for L <= sliding_window (64 <= 1024) the sliding band is a no-op, so a
    # plain causal mask is exact for this prefill.
    mask = torch.zeros(bsz, 1, seq_len, seq_len, dtype=torch.float32)
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )
    mask = mask + causal[None, None]

    out = layer(
        hidden_states=h_embed_fp32,
        position_embeddings=(cos, sin),
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=None,
        shared_kv_states={},
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def run_device_block(model, h_embed, position_ids):
    """Drive the adapter's compiled layer-0 MoE block on-card, prefill mode.

    Mirrors ``hf_gemma4._run_blocks_over_embeds`` but for block 0 only: per-type
    RoPE freqs from ``model._spyre_rope``, the per-layer-type mask dict from
    ``_build_layer_masks``, a KV cache sized to the model's per-layer shapes, and
    ``is_filling=False, token_index=0, cache_position=0``. Returns fp16 [B,L,H].
    """
    backbone = _gemma4_backbone(model)
    cfg = model.config.get_text_config()
    dtype = get_model_dtype(model)
    bsz, seq_len, _ = h_embed.shape

    freqs = {
        lt: rope(h_embed, position_ids)
        for lt, rope in model._spyre_rope.items()
    }
    max_cache_len = seq_len
    # Base causal prefill mask; the sliding-layer band intersects this. Built on
    # CPU (host tensor arithmetic in _build_layer_masks) then moved to device.
    prefill_mask = build_prefill_mask(
        bsz, seq_len, max_cache_len, prompt_offsets=0, dtype=dtype
    )
    masks = _build_layer_masks(
        model, prefill_mask, seq_len, bsz, token_index=0, cache_position=0
    )
    masks = {lt: m.to("spyre") for lt, m in masks.items()}

    key_caches, value_caches = allocate_kv_caches(
        model, bsz, max_cache_len, dtype
    )

    layer_type = cfg.layer_types[0]
    block = model._spyre_compiled_blocks[0]
    layer_scalar = backbone.layers[0].layer_scalar

    with torch.no_grad():
        h, _, _ = block(
            h_embed.to("spyre"),
            freqs[layer_type],
            masks[layer_type],
            key_caches[0],
            value_caches[0],
            False,  # is_filling
            0,  # token_index
            0,  # cache_position
            layer_scalar,
        )
    return h.cpu()


def main():
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    tcfg = cfg.get_text_config()
    print(f"model_type={tcfg.model_type} enable_moe_block={tcfg.enable_moe_block} "
          f"num_experts={tcfg.num_experts} top_k_experts(cfg)={tcfg.top_k_experts} "
          f"K(gate)={K}")

    # Spyre path dtype is fp16 (AutoSpyreModelForCausalLM default); mirror it.
    print("Loading full 26B model on CPU (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    model.requires_grad_(False)

    backbone = _gemma4_backbone(model)
    layer0 = backbone.layers[0]

    # Coerce K=4 on the config BEFORE the reference so the stock router selects
    # top-4 (apples-to-apples with the adapter's bring-up K). The router reads
    # config.top_k_experts at call time.
    tcfg = model.config.get_text_config()
    tcfg.top_k_experts = K
    layer0.router.config.top_k_experts = K

    # A short real prompt, left-padded to one BLOCK_SIZE prefill block.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    pad = BLOCK_SIZE - seq_len
    padded_ids = torch.cat(
        [ids.new_zeros((1, pad)), ids], dim=1
    ) if pad > 0 else ids
    position_ids = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
    position_ids[:, pad:] = torch.arange(seq_len)

    # Embeddings (scaled word embedding) shared by both paths, computed BEFORE
    # prepare_for_spyre so the CPU reference sees the unpatched RMSNorm.
    with torch.no_grad():
        h_embed = backbone.embed_tokens(padded_ids)  # [1, 64, H] fp16

    # --- CPU fp32 reference (stock layer.forward), BEFORE prepare_for_spyre ---
    rope = backbone.rotary_emb
    layer0_fp32 = layer0.float()
    with torch.no_grad():
        ref = build_cpu_reference(
            layer0_fp32,
            h_embed.float(),
            position_ids,
            tcfg,
            rope,
        )
    ref = ref[0].float()  # [64, H]
    print(f"ref fp32 shape={tuple(ref.shape)} min={ref.min():.4f} "
          f"max={ref.max():.4f}")

    # Restore layer weights to fp16 so prepare_for_spyre lays out fp16 experts.
    layer0.half()

    # --- adapter path: prepare_for_spyre + on-card compiled block 0 ----------
    print("prepare_for_spyre + move to spyre...")
    from hf_adapters.hf_common import move_model_to_spyre

    move_model_to_spyre(model, hf_gemma4_moe, torch.float16)

    got = run_device_block(model, h_embed, position_ids)[0].float()  # [64, H]
    print(f"got fp16->fp32 shape={tuple(got.shape)} min={got.min():.4f} "
          f"max={got.max():.4f}")

    # --- compare -------------------------------------------------------------
    denom = ref.abs().clamp_min(1.0)
    rel = (got - ref).abs() / denom
    abs_err = (got - ref).abs()
    mean_rel = rel.mean().item()
    max_rel = rel.max().item()
    print("=" * 70)
    print(f"mean_rel={mean_rel:.4%}  max_rel={max_rel:.4f}  "
          f"mean_abs={abs_err.mean():.4f}  max_abs={abs_err.max():.4f}")
    ok = mean_rel < 0.02 and max_rel < 0.5
    print(f"{'PASS' if ok else 'FAIL'} gemma4 MoE layer-0 parity "
          f"(mean_rel<0.02, max_rel<0.5)")
    print("=" * 70)
    assert ok, (
        f"layer-0 parity diverged: mean_rel={mean_rel:.4%} max_rel={max_rel:.4f}"
    )


if __name__ == "__main__":
    main()
