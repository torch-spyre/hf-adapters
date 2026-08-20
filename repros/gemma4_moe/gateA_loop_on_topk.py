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

"""On-card single-layer gate for the Gemma 4 MoE loop-on-topk path (Approach A).

Compares the compiled loop-on-topk MoE FFN (device fp16, experts HBM-resident,
on-device index_select, row-tiled spyre_hint) for decoder layer 0 against a
pure-CPU fp32 reference on the real google/gemma-4-26B-A4B-it weights. This is
the pass/fail oracle for the three Approach-A assumptions: on-device topk(k=4),
on-device windowed indirect gather, and the E-outermost + stick layout.

Criterion: mean_rel < 0.02, max_rel < 0.5 (device fp16 vs fp32 CPU truth).
If it aborts or diverges, STOP and report the exact failure (do not fall back).

Run:
  HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
    python3 -u repros/gemma4_moe/gateA_loop_on_topk.py
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import hf_adapters.hf_gemma4_moe as moe
from hf_adapters.hf_gemma4 import _gemma4_backbone
from hf_adapters.hf_gemma4_moe import _moe_ffn_loop_ref

MODEL = "google/gemma-4-26B-A4B-it"
K = 4


def main():
    moe._MOE_BRINGUP_K = K  # apples-to-apples with the reference
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    # gemma-4-26B loads as multimodal Gemma4ForConditionalGeneration; the decoder
    # stack lives at the text backbone (Gemma4TextModel), NOT model.model.layers.
    layer = _gemma4_backbone(model).layers[0]

    # fp32 CPU ground truth on a fixed random input, BEFORE prepare (the RMSNorm
    # patch is global; capture the reference first). K=4, N=T*K=256 (>1, tiled).
    torch.manual_seed(0)
    T = 64  # small token count; N = T*K = 256 rows
    H = cfg.text_config.hidden_size
    eps = cfg.text_config.rms_norm_eps
    x = torch.randn(T, H, dtype=torch.float32)

    # Router preprocessing must match the device region EXACTLY: scale-free
    # RMSNorm (eps inside sqrt) -> * scale[H] -> * scalar_root_size, THEN proj.
    # Feed that preprocessed input to the reference via its x_router seam so the
    # gate compares the expert FFN + on-device machinery, not router-math skew.
    W_router = layer.router.proj.weight.data.float()
    router_scale = layer.router.scale.data.float()  # [H] vector, NOT scalar
    root_size = float(layer.router.scalar_root_size)  # hidden_size ** -0.5
    var = x.pow(2).mean(-1, keepdim=True)
    x_router = x * torch.rsqrt(var + eps)  # scale-free RMSNorm
    x_router = x_router * router_scale * root_size
    # gate_up_proj is FUSED [E,2M,H]; transpose to [E,H,2M] (chunk stays post-bmm).
    gate_up_t = layer.experts.gate_up_proj.data.transpose(1, 2).contiguous().float()
    down_t = layer.experts.down_proj.data.transpose(1, 2).contiguous().float()
    per_expert_scale = layer.router.per_expert_scale.data.float()
    ref = _moe_ffn_loop_ref(
        x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=x_router
    )

    # Device path: force the loop-on-topk formulation, prepare, run the region.
    moe._MOE_LOOP_ON_TOPK = True
    moe.prepare_for_spyre(model)
    router = layer.router
    compiled_loop = torch.compile(moe._compiled_moe_loop_region, dynamic=False)
    x_dev = x.to(torch.float16).to("spyre")
    got = moe._moe_ffn_loop(
        x_dev, x_dev, router, compiled_loop,
        layer._spyre_gate_up_dev, layer._spyre_down_dev, K, moe._MOE_TILE, eps,
    )
    got = got.cpu().float()

    diff = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    mean_rel = (diff / denom).mean().item()
    max_rel = (diff / denom).max().item()
    print(f"ref fp32 shape={tuple(ref.shape)} min={ref.min():.4f} max={ref.max():.4f}")
    print(f"got fp16->fp32 min={got.min():.4f} max={got.max():.4f}")
    print(f"mean_rel={mean_rel*100:.4f}%  max_rel={max_rel:.4f}")
    ok = mean_rel < 0.02 and max_rel < 0.5
    print("PASS" if ok else "FAIL", "gemma4 MoE loop-on-topk layer-0 parity")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
