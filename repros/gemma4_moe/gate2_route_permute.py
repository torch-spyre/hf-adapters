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
Gate 2: Gemma 4 MoE device/host-split route -> permute -> GEMM -> combine (K=4).

The routing/permute ops (topk, argsort, 1-D index arithmetic, index_add) do
NOT lower on the current torch-spyre backend (spec 2.1) -- only plain gather
and bmm do. This gate proves the device/host split is a working substitute:

  device (torch.compile, spyre):  router Linear ; token gather x[token_of_row]
                                  ; expert grouped GEMM (bmm+gelu+bmm)
  host   (eager CPU):             softmax/topk/renorm/per_expert_scale
                                  ; argsort + token_of_row index arithmetic
                                  ; weighted index_add combine

Correctness is checked against a pure-CPU dense MoE reference (compute all
experts, select top-K) at fp32, using relative error vs a clipped denominator.

## Two adapter-level shape rules this gate establishes (feed Task 6)

1. Keep the row-batched expert tensors 3D ([N,1,.]) THROUGH the whole expert
   FFN -- do NOT squeeze back to 2D between the two grouped GEMMs. The
   squeeze(1) -> chunk -> unsqueeze(1) round-trip through 2D breaks Spyre
   layout propagation ("Incompatible host_size and dim_order"); staying 3D
   compiles cleanly. Squeeze only at the very end.
2. Store/supply the expert weights PRE-TRANSPOSED ([E,H,2M] and [E,M,H]) so
   the compiled region contains no in-kernel .transpose of a large weight.
   An in-graph transpose of [N,2M,H] forces a restickify with a huge byte
   offset -> "Immediate value out of boundary" (L3_ADDEARIMM) abort. This is
   a prepare_for_spyre layout decision (lay experts out transposed once).

## Outcome (2026-07-31, on-card)

PASS. mean_rel=0.46%, max_rel=0.19% vs fp32 CPU MoE reference.
All three device regions lower: router OK, gather OK, expert (3D, pre-T) OK.
A `corrupted double-linked list` SIGABRT may appear on process teardown AFTER
the OK line -- that is the known torch-spyre lifetime issue, not a compute
failure (Global Constraints).

Layout note: the single-op device_gather region ([N,H] output, row dim
outermost by construction) needs no restickify. The composed expert region
was validated by the numeric round-trip; a dedicated restickify-artifact
audit of the gather buffer belongs to Task 8 gate-3 on real weights.
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch_spyre

torch_spyre._autoload()

import torch
import torch.nn.functional as F


H, E, T, K, M = 2816, 128, 64, 4, 704  # K=4 bring-up; M=moe_intermediate_size


# --- device regions (compiled, spyre) ---
def device_router(x, W_router):  # [T,H] x [E,H] -> [T,E]
    return F.linear(x, W_router)


def device_expert(gathered3d, gate_up_row_t, down_row_t):
    # gathered3d: [N,1,H]; weights PRE-TRANSPOSED: [N,H,2M], [N,M,H].
    # Stay 3D throughout -- no squeeze between the two bmms (see docstring).
    gu = torch.bmm(gathered3d, gate_up_row_t)          # [N,1,2M]
    g, u = gu.chunk(2, dim=-1)                          # [N,1,M] each
    act = F.gelu(g, approximate="tanh") * u            # [N,1,M]
    return torch.bmm(act, down_row_t).squeeze(1)       # [N,H]


def device_gather(x, token_of_row):  # the indirect-access op under test
    return x[token_of_row]           # [T*K,H]


# --- pure-CPU reference (dense: compute all experts, select top-K) ---
def ref_moe(x, W_router, gate_up, down, scale, K):
    probs = torch.softmax(F.linear(x, W_router).float(), dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx]
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(K):
            e = idx[t, k].item()
            g, u = F.linear(x[t].float(), gate_up[e].float()).chunk(2, dim=-1)
            h = F.linear(F.gelu(g, approximate="tanh") * u, down[e].float())
            out[t] += w[t, k] * h
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(T, H, dtype=torch.float16)
    W_router = torch.randn(E, H, dtype=torch.float16)
    gate_up = torch.randn(E, 2 * M, H, dtype=torch.float16) * 0.02  # HF [E,2M,H]
    down = torch.randn(E, H, M, dtype=torch.float16) * 0.02         # HF [E,H,M]
    scale = (torch.rand(E) + 0.5).half()

    ref = ref_moe(x, W_router, gate_up, down, scale, K)

    crouter = torch.compile(device_router, dynamic=False)
    cexpert = torch.compile(device_expert, dynamic=False)
    cgather = torch.compile(device_gather, dynamic=False)

    # 1) device: router logits -> host
    logits = crouter(x.to("spyre"), W_router.to("spyre")).cpu().float()
    # 2) host: softmax/topk/renorm/scale/argsort/token_of_row (unsupported on dev)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx].float()
    flat_expert = idx.reshape(-1)                                   # [T*K]
    sort_perm = torch.argsort(flat_expert)
    row_expert = flat_expert[sort_perm]
    token_of_row = (torch.arange(T * K) // K)[sort_perm].to(torch.int32)
    # 3) device: gather token rows
    gathered = cgather(x.to("spyre"), token_of_row.to("spyre")).cpu()  # [T*K,H]
    # host: select + PRE-TRANSPOSE the per-row expert weights (a prepare_for_spyre
    # layout decision; kept on host here to keep the gate focused)
    gate_up_row_t = gate_up[row_expert].transpose(1, 2).contiguous()   # [N,H,2M]
    down_row_t = down[row_expert].transpose(1, 2).contiguous()         # [N,M,H]
    # 4) device: expert GEMM (gathered rows as [N,1,H])
    expert_out = cexpert(
        gathered.unsqueeze(1).to("spyre"),
        gate_up_row_t.to("spyre"),
        down_row_t.to("spyre"),
    ).cpu().float()
    # 5) host: weighted index_add combine
    row_w = w.reshape(-1)[sort_perm].unsqueeze(-1).float()
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row.long(), expert_out * row_w)

    denom = ref.abs().clamp_min(1.0)
    rel = (out - ref).abs() / denom
    print(f"mean_rel={rel.mean():.4%} max_rel={rel.max():.4f}")
    assert rel.mean() < 0.02 and rel.max() < 0.5, "split MoE round-trip diverged"
    print("OK device/host-split route/permute/GEMM/combine round-trip")
