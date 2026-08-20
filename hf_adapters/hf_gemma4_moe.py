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
HuggingFace Transformers adapter for the Gemma 4 **MoE** causal-LM on Spyre.

Targets the sparse ``google/gemma-4-26B-A4B-it`` variant, whose
``Gemma4TextDecoderLayer`` runs a dense MLP **in parallel** with a top-K
mixture-of-experts FFN when ``config.enable_moe_block=True`` (see stock
``transformers.models.gemma4.modeling_gemma4``). The dense attention half and
all attention-side Spyre prep (RMSNorm patch, per-type RoPE, KV shapes,
LM-head padding) are shared with the dense adapter ``hf_gemma4`` — this module
only adds the sparse FFN and the surrounding block/prepare/forward wiring.

The FFN has three selectable formulations (module flags, mutually exclusive):

  * ``_MOE_CHUNKED_ONDEVICE`` (Gate-A5-PROVEN, mean_rel=0.028) -- the WHOLE FFN
    runs on the device; host does only chunk-loop glue. Sidesteps the routing
    ops below via the topk-pad fix (pad logits to a non-pow2 width before topk,
    threshold on the kth VALUE -- no fp16 index) and mask-reduce weighting
    (``(w*onehot[e]).sum`` into a [T,H] device accumulator -- no gather/scatter).
    Experts are split into ``ceil(E/_MOE_EC)`` compiled chunks (>32 expert
    GEMM-chains in one sdsc program crashes the DDC scheduler). See the flag's
    definition for the full rationale.
  * ``_MOE_LOOP_ON_TOPK`` -- experts HBM-resident, on-device ``index_select``
    under a row-tiled ``spyre_hint``. Blocked at E=128 (topk pow2-width abort +
    P4 slab-gather overflow); kept as scaffold.
  * default (``_moe_ffn_split``) -- device/host split (spec §2.1, verified in
    ``repros/gemma4_moe/gate2_route_permute.py``). The routing ops (``topk``,
    ``argsort``, 1-D index arithmetic, ``index_add``) do not lower, so:

      device (torch.compile, spyre):  router projection ; token gather
                                      ``x[token_of_row]`` ; expert grouped GEMM
                                      (bmm + gelu_tanh SwiGLU + bmm)
      host   (eager CPU):             softmax / topk / renorm / per_expert_scale
                                      ; argsort + ``token_of_row`` arithmetic
                                      ; weighted ``index_add`` combine

Two load-bearing device-shape rules (verified on-card, gate 2; apply to the
split / loop bmm paths -- the chunked path uses plain 2D matmuls):

1. The row-batched expert tensors stay **3D ``[N,1,·]``** through the whole
   expert FFN — the ``squeeze(1)→chunk→unsqueeze(1)`` 2D round-trip breaks
   Spyre layout propagation ("Incompatible host_size and dim_order"). Squeeze
   only at the very end.
2. Expert weights are supplied **pre-transposed** (``gate_up`` as ``[E,H,2M]``,
   ``down`` as ``[E,M,H]``) so the compiled region has no in-kernel
   ``.transpose`` of a large weight (which forces a giant-offset restickify:
   ``L3_ADDEARIMM Immediate value out of boundary``). ``prepare_for_spyre``
   lays the experts out pre-transposed once.

``K`` is pinned to 4 for bring-up; ``prepare_for_spyre`` coerces/asserts
``config.top_k_experts == 4``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_adapters.hf_common import text_config
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _run_backbone_forward,  # re-exported: block-agnostic, drives _spyre_compiled_blocks
    _run_forward,  # re-exported: block-agnostic backbone + LM head + softcap
    _setup_gemma4_text_decoder,
)

# ``_run_forward`` / ``_run_backbone_forward`` are re-exported from ``hf_gemma4``
# unchanged: since #350 they drive ``model._spyre_compiled_blocks`` and read
# ``layer_scalar`` off each registered block, so they are block-AGNOSTIC — the
# MoE blocks slot in transparently. Kept in this module's namespace because
# ``resolve_adapter_module`` / ``generate`` look them up on the resolved adapter.
__all__ = ["prepare_for_spyre", "_run_forward", "_run_backbone_forward"]

# Top-K pinned to 4 for MoE bring-up (Global Constraints). The device grouped
# GEMM and the host route/permute path are validated at this K; a different K
# is a later-task concern.
_MOE_BRINGUP_K = 4

# Grouped-GEMM lowering selector (spec §4, Task 9 "Option 4B").
#
#   False (default) -> Option 4A: per-row expert-weight gather + row-batched
#       bmm. This is the SHIPPED, on-card-validated bring-up path (Tasks 2/6/8).
#   True            -> Option 4B: keep ``gathered`` contiguous (rows already
#       sorted by expert) and walk the expert segments given by
#       ``group_off = cumsum(bincount(row_expert, E))``, doing one slab GEMM per
#       segment (one weight load per expert instead of per row).
#
# 4B is EXPERIMENTAL and OFF by default: on the current backend the
# ``spyre_hint(tiles=...)`` vocabulary cannot express a per-tile operand switch
# (see ``_grouped_gemm_4b`` docstring), so the on-device weight-load reduction
# 4B targets is not achievable through the hint API today. 4B is retained as a
# numerically-identical CPU-reference variant (verified by
# ``tests/test_gemma4_moe_ffn.py::test_grouped_gemm_4a_4b_agree``) and as the
# scaffold for a future backend that grows a grouped-GEMM primitive. Flip this
# flag only after validating 4B end-to-end on-card.
_MOE_GEMM_4B = False

# Row-tile size for the loop-on-topk device region (spec Approach A). The
# spyre_hint(tiles={"row": _MOE_TILE}) tiles the N=T*K row axis so the backend
# loops over ceil(N/_MOE_TILE) tiles; a tuning knob (scratchpad window size).
_MOE_TILE = 32

# Device-FFN formulation selector (spec Approach A).
#   False (default) -> shipped host-split path (_moe_ffn_split): experts
#       host-resident, per-row weight select on CPU, [N,.] slices to device.
#   True            -> loop-on-topk path (_moe_ffn_loop): experts HBM-resident
#       on device, on-device index_select under a row-tiled spyre_hint.
# Flip to True only after gateA_loop_on_topk.py passes on-card.
_MOE_LOOP_ON_TOPK = False

# ALL-DEVICE chunked formulation selector (spec Approach A, "nothing but glue
# on host"). This is the on-card-PROVEN path (Gate A5, mean_rel=0.028 vs CPU
# fp32): the WHOLE FFN -- router (softmax/topk/renorm/scale), expert GEMMs,
# gelu-tanh SwiGLU, per-expert weight application, and the sum-over-experts
# accumulate -- lowers and runs on the device. Host does ONLY glue: a
# ``_MOE_NCHUNK``-iteration counter that threads a device-resident [T,H]
# accumulator back into the next chunk. Nothing gather/scatter/FFN runs on
# host. It sidesteps the two ops that abort ``_moe_ffn_loop`` at E=128:
#
#   * ``topk`` on a pow2 stick-multiple width (E=128 = 2 sticks) aborts
#     ``Incorrect chunk size`` (L3DlOpsScheduler.cpp:1714). FIX: pad the
#     router logits [T,E]->[T,_MOE_PADW] with -inf before topk, threshold on
#     the kth VALUE over the original [T,E] (no fp16-index materialize).
#   * the per-row on-device weight ``index_select`` (P4 L3_ADDEARIMM
#     immediate overflow) and the ``index_add`` scatter-combine (P5 silently
#     wrong). FIX: apply the router weight as arithmetic
#     ``we=(w*onehot[e]).sum(-1,keepdim=True)`` [T,1] and accumulate into a
#     [T,H] running buffer -- no gather, no scatter.
#
# One fused sdsc program with >32 expert GEMM-chains makes the DDC scheduler
# derive a non-stick-aligned chunk (same 1714 crash), so the E=128 experts are
# split into ``_MOE_NCHUNK`` compiled regions of ``_MOE_EC`` experts each,
# threading the device accumulator across them. Per-chunk expert weights are
# pre-materialized OFFSET-0 contiguous at load (a non-zero storage_offset
# device-tensor view passed as a compile input reads wrong storage -- see
# [[project-pr2426-storage-offset-review]]).
#
# Flip to True only after gateA5_chunked_ondevice.py passes on-card. Mutually
# exclusive with _MOE_LOOP_ON_TOPK (asserted in prepare_for_spyre).
_MOE_CHUNKED_ONDEVICE = False

# Experts per compiled chunk for the all-device path. >32 expert GEMM-chains in
# one fused sdsc program crashes the DDC scheduler (Incorrect chunk size); <=32
# compiles clean. E must be divisible by _MOE_EC.
_MOE_EC = 32

# topk-input pad width for the all-device router (topk-pad fix). The router
# logits width E is padded to this NON-pow2, non-stick-multiple width before
# topk so the backend's binary-tree tiling stays stick-aligned. For E=128 the
# proven value is 160 (=E+32). Pad columns are -inf so they never win top-K.
_MOE_PADW = 160
_MOE_PAD_NEG = -30000.0  # ~ -inf in fp16 (< any real softmax-prob logit)


def _moe_route(x, W_router, per_expert_scale, K):
    """Route tokens to top-K experts with softmax and per-expert scaling.

    Args:
        x: Token embeddings [T, H]
        W_router: Expert router weights [E, H]
        per_expert_scale: Expert scaling factors [E]
        K: Number of top experts per token

    Returns:
        w: Router weights after softmax, top-K selection, renormalization,
           and per-expert scaling [T, K]
        idx: Top-K expert indices [T, K]
    """
    logits = F.linear(x, W_router)  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K],[T,K]
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[idx]
    return w, idx


def _moe_permute(x, idx, K):
    """Sort token-expert pairs by expert and return gather/sort information.

    Args:
        x: Token embeddings [T, H]
        idx: Top-K expert indices [T, K]
        K: Number of experts per token (used for reconstruction)

    Returns:
        gathered: Token embeddings sorted by expert assignment [T*K, H]
        token_of_row: Token indices for each row in gathered [T*K]
        row_expert: Expert ID for each row in gathered [T*K]
        sort_perm: Permutation that sorts (token, expert) pairs by expert
                   [T*K]
    """
    flat_expert = idx.reshape(-1)  # [T*K]
    sort_perm = torch.argsort(flat_expert)  # [T*K]
    row_expert = flat_expert[sort_perm]  # [T*K] expert id per sorted row
    token_of_row = (
        torch.arange(idx.shape[0] * K, device=x.device) // K
    )[sort_perm]
    gathered = x[token_of_row]  # [T*K,H]
    return gathered, token_of_row, row_expert, sort_perm


def _grouped_gemm_4a(gathered, Wstack, row_expert):
    """Option 4A: gather per-row weight, row-batched matmul.

    Args:
        gathered: Token embeddings sorted by expert [N, in]
        Wstack: Expert weight matrices [E, out, in]
        row_expert: Expert ID for each row in gathered [N]

    Returns:
        out: Result of gather per-row weight @ gathered [N, out]
    """
    W_row = Wstack[row_expert]  # index_select on expert dim [N,out,in]
    out = torch.bmm(
        gathered.unsqueeze(1), W_row.transpose(1, 2)
    )  # [N,1,out]
    return out.squeeze(1)  # [N,out]


def _group_offsets(row_expert, E):
    """Segment boundaries of a by-expert-sorted row block.

    ``group_off[e]:group_off[e+1]`` is the contiguous row range assigned to
    expert ``e`` (empty when the expert got no rows). Equivalent to the
    DeepGEMM ``group_off = cumsum(bincount(row_expert, E))`` prefix sum used to
    drive a contiguous grouped GEMM.

    Args:
        row_expert: Expert ID per sorted row [N] (non-decreasing)
        E: Number of experts

    Returns:
        group_off: Prefix-sum segment boundaries [E+1] (int64), on
            ``row_expert``'s device.
    """
    counts = torch.bincount(row_expert, minlength=E)  # [E]
    group_off = torch.zeros(E + 1, dtype=torch.long, device=row_expert.device)
    group_off[1:] = torch.cumsum(counts, 0)
    return group_off


def _grouped_gemm_4b(gathered, Wstack, row_expert):
    """Option 4B: contiguous grouped GEMM over per-expert row segments.

    ``gathered`` is already sorted by expert (``_moe_permute``), so each
    expert owns one contiguous row range ``group_off[e]:group_off[e+1]``. This
    walks those ranges and does ONE slab matmul per expert
    (``seg @ Wstack[e].T``) instead of materializing a weight per row (4A) —
    the DeepGEMM contiguous-layout formulation. Numerically identical to 4A
    (verified by ``test_grouped_gemm_4a_4b_agree``); only the weight-load
    schedule differs.

    NOTE (on-device limitation, Task 9): the perf win 4B targets is "one expert
    weight load per tile, scheduled across cores via ``spyre_hint``". The
    torch-spyre ``spyre_hint(tiles={name: n})`` API tiles a *named dimension of
    a single op's own iteration space* (see
    ``torch_spyre/_inductor/propagate_hints.py`` + the flash-SDPA use in
    ``decompositions.py``); it has NO kwarg to switch the *weight operand* per
    tile from a host-supplied ``group_off``. A grouped GEMM whose weight slab
    changes per tile is a distinct backend primitive (the Triton ``group_id``
    pointer-arithmetic loop has no Spyre-hint equivalent today). This Python
    ``for``-over-experts form is therefore the faithful, numerically-correct
    reference; expressing it as a single fused device kernel needs a backend
    grouped-GEMM op, tracked as the Task 9 follow-up. Kept behind
    ``_MOE_GEMM_4B`` (OFF) so the shipped 4A device path is unaffected.

    Args:
        gathered: Token embeddings sorted by expert [N, in]
        Wstack: Expert weight matrices [E, out, in]
        row_expert: Expert ID for each row in gathered [N] (non-decreasing)

    Returns:
        out: Grouped ``gathered @ Wstack[expert].T`` [N, out]
    """
    E = Wstack.shape[0]
    out_dim = Wstack.shape[1]
    group_off = _group_offsets(row_expert, E)
    out = torch.empty(
        gathered.shape[0], out_dim, dtype=gathered.dtype, device=gathered.device
    )
    for e in range(E):
        lo = int(group_off[e])
        hi = int(group_off[e + 1])
        if hi > lo:
            # One contiguous slab load of expert e's weight for its whole
            # row segment (vs. 4A's per-row gather).
            out[lo:hi] = gathered[lo:hi] @ Wstack[e].transpose(0, 1)
    return out


def _grouped_gemm(gathered, Wstack, row_expert):
    """Grouped GEMM dispatcher (Option 4A default, 4B behind ``_MOE_GEMM_4B``).

    Both variants return the identical result: for each row ``r``,
    ``gathered[r] @ Wstack[row_expert[r]].T``. See ``_MOE_GEMM_4B`` and the
    per-variant docstrings for the schedule trade-off.

    Args:
        gathered: Token embeddings sorted by expert [N, in]
        Wstack: Expert weight matrices [E, out, in]
        row_expert: Expert ID for each row in gathered [N]

    Returns:
        out: Result [N, out]
    """
    if _MOE_GEMM_4B:
        return _grouped_gemm_4b(gathered, Wstack, row_expert)
    return _grouped_gemm_4a(gathered, Wstack, row_expert)


def _moe_ffn(x, W_router, gate_up_proj, down_proj, per_expert_scale, K):
    """MoE FFN forward: route, permute, grouped gate_up, gelu_tanh SwiGLU,
    grouped down, weight by w, scatter_add combine.

    Args:
        x: Token embeddings [T, H]
        W_router: Expert router weights [E, H]
        gate_up_proj: Gate-up projection per expert [E, 2*M, H]
        down_proj: Down projection per expert [E, H, M]
        per_expert_scale: Expert scaling factors [E]
        K: Number of top experts per token

    Returns:
        out: MoE FFN output [T, H]
    """
    T, H = x.shape
    w, idx = _moe_route(x, W_router, per_expert_scale, K)
    (
        gathered,
        token_of_row,
        row_expert,
        sort_perm,
    ) = _moe_permute(x, idx, K)
    gate_up = _grouped_gemm(
        gathered, gate_up_proj, row_expert
    )  # [N,2M]
    g, u = gate_up.chunk(2, dim=-1)
    act = F.gelu(g, approximate="tanh") * u  # [N,M]
    expert_out = _grouped_gemm(
        act, down_proj, row_expert
    )  # [N,H]
    expert_out = expert_out * w.reshape(-1)[sort_perm].unsqueeze(-1)
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out = out.index_add(0, token_of_row, expert_out)  # scatter_add combine
    return out


def _moe_ffn_loop_ref(
    x, W_router, gate_up_t, down_t, per_expert_scale, K, x_router=None
):
    """CPU fp32 reference for the loop-on-topk MoE FFN (no expert grouping).

    Mirrors the on-device dataflow: route to top-K, flatten the [T,K] result to
    N=T*K rows in topk order (NO argsort), per-row select the expert weight,
    two bmms with a gelu-tanh SwiGLU between, weight by the router weight, and
    scatter-add back to [T,H]. The numeric oracle for _compiled_moe_loop_region.

    Args:
        x: Expert-FFN input token embeddings [T, H] (fed to the experts).
        W_router: Router projection weights [E, H].
        gate_up_t: Pre-transposed gate_up per expert [E, H, 2M].
        down_t: Pre-transposed down per expert [E, M, H].
        per_expert_scale: Per-expert scale [E].
        K: Top-K experts per token.
        x_router: Router input [T, H] (already RMSNorm/scale-preprocessed). When
            None, defaults to x. Routing uses x_router; the experts always read
            x -- router and expert inputs are threaded separately (the double-
            normalization rule from the design spec).

    Returns:
        out: MoE FFN output [T, H].
    """
    T, H = x.shape
    if x_router is None:
        x_router = x
    w, idx = _moe_route(x_router, W_router, per_expert_scale, K)  # [T,K],[T,K]
    expert_of_row = idx.reshape(-1)  # [N] N=T*K, topk order (NOT sorted)
    token_of_row = (torch.arange(T * K, device=x.device) // K)  # [N]
    row_w = w.reshape(-1, 1)  # [N,1]

    gathered = x[token_of_row]  # [N,H]
    gu = torch.bmm(
        gathered.unsqueeze(1), gate_up_t[expert_of_row]
    )  # [N,1,2M]
    g, u = gu.chunk(2, dim=-1)  # [N,1,M]
    act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
    row_out = torch.bmm(act, down_t[expert_of_row]).squeeze(1)  # [N,H]
    row_out = row_out * row_w  # [N,1] broadcast

    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out = out.index_add(0, token_of_row.long(), row_out)
    return out


# ---------------------------------------------------------------------------
# Device (compiled, spyre) FFN region + host orchestrator + decoder block.
# ---------------------------------------------------------------------------


def _compiled_moe_device_region(gathered3d, gate_up_row_t, down_row_t):
    """The device portion of the expert FFN — the two grouped GEMMs + SwiGLU.

    This is the exact shape flow verified on-card in gate 2
    (``repros/gemma4_moe/gate2_route_permute.py``) and MUST stay byte-for-byte
    shape-identical to it. Everything is 3D ``[N,1,·]`` (shape rule 1) and the
    weights are pre-transposed (shape rule 2):

        gathered3d:    [N,1,H]   (per-row gathered token embedding)
        gate_up_row_t: [N,H,2M]  (per-row gate_up weight, PRE-transposed)
        down_row_t:    [N,M,H]   (per-row down weight, PRE-transposed)

    Do NOT squeeze between the two ``bmm``s — the 2D round-trip breaks Spyre
    layout propagation. Squeeze only on the final return.
    """
    gu = torch.bmm(gathered3d, gate_up_row_t)  # [N,1,2M]
    g, u = gu.chunk(2, dim=-1)  # [N,1,M] each
    act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
    return torch.bmm(act, down_row_t).squeeze(1)  # [N,H]


def _compiled_moe_loop_region(
    x_router,
    x_expert,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    gate_up_dev,
    down_dev,
    token_ids,
    K,
    tile,
    eps,
):
    """Whole MoE FFN on-device except the scatter-combine (spec Approach A).

    Router (inlined SCALE-FREE RMSNorm on the RAW residual x_router with eps
    INSIDE the sqrt, then * router_scale[H] * router_scalar_root_size, then
    proj) -> softmax -> topk(K) -> renorm -> per_expert_scale; then, under a
    single spyre_hint(tiles={"row": tile}) that tiles the N=T*K row axis (the
    hint IS the loop -- no Python for), gather the expert-input rows from
    x_expert, index_select the per-row expert weights from the HBM-resident
    E-outermost stacks, two bmms (3D [*,1,*] throughout) with a gelu-tanh
    SwiGLU, and weight by the router weight.

    Preflight-corrected router surface: the stock router.norm has NO .weight
    (Gemma4RMSNorm with_scale=False); the learnable gain is router_scale, an
    [H] vector applied AFTER the scale-free norm. router_scalar_root_size is a
    Python float (hidden_size ** -0.5). eps is config.rms_norm_eps.

    gate_up_dev: [E,H,2M] (stick=2M), down_dev: [E,M,H] (stick=H), E outermost.
    tile must be >= 2 (single-row P=1 gather SIGABRTs in dxp_standalone).
    Returns (row_out[N,H], token_of_row[N]) for the host index_add combine.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape
    N = T * K

    # --- router (all device-lowerable): SCALE-FREE RMSNorm (eps inside sqrt)
    # on the RAW residual, then the [H] scale vector and the root-size scalar.
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free (no gain in norm)
    normed = normed * router_scale * router_scalar_root_size  # scale is [H]
    logits = F.linear(normed, router_proj_w)  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    # topk returns int64 indices (the Spyre spyre_topk decomp converts the
    # fp16-encoded positions the dxp kernel writes to int64), so idx is ready
    # to feed the three indirect gathers below (per_expert_scale[idx],
    # gate_up_dev[expert_of_row], down_dev[...]) with no adapter-side cast.
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K],[T,K] int64  ASSUMED to lower
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[idx]  # [T,K]

    expert_of_row = idx.reshape(N)  # [N] topk order (no sort)
    token_of_row = (token_ids.reshape(T, 1).expand(T, K)).reshape(N)  # [N]
    row_w = w.reshape(N, 1)  # [N,1]

    with spyre_hint(tiles={"row": tile}):
        gathered = x_expert[token_of_row]  # [N,H]
        W_gu = gate_up_dev[expert_of_row]  # [N,H,2M] on-device index_select
        W_dn = down_dev[expert_of_row]  # [N,M,H]
        gu = torch.bmm(gathered.unsqueeze(1), W_gu)  # [N,1,2M] 3D throughout
        g, u = gu.chunk(2, dim=-1)  # [N,1,M]
        act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
        row_out = torch.bmm(act, W_dn).squeeze(1)  # [N,H]
        row_out = row_out * row_w  # [N,1] broadcast
    return row_out, token_of_row


def _compiled_device_gather(x, token_of_row):
    """On-device token gather ``x[token_of_row]`` (indirect-access op).

    Kept as its own compiled region (gate-2 template): a single-op ``[N,H]``
    gather whose row (indexed) dim is outermost by construction, so it needs no
    restickify. The routing tensor ``token_of_row`` is computed host-side and
    moved to the device for this call.
    """
    return x[token_of_row]  # [T*K, H]


def _moe_ffn_split(
    x_router,
    x_expert,
    router,
    compiled_gather,
    compiled_expert,
    gate_up_host_t,
    down_host_t,
    K,
):
    """Device/host-split MoE FFN (spec §2.1; gate-2 host-orchestration template).

    The router and the experts consume **two independent normalizations of the
    same raw flattened residual** (stock ``modeling_gemma4.py:1432-1435``):

      * ``x_router`` ``[T,H]`` — the **raw** flattened residual. The router's
        own internal ``self.norm`` (scale-free RMSNorm) is applied to THIS
        tensor. Do NOT pass a ``pre_feedforward_layernorm_2``-normed tensor
        here: that would double-normalize the router input
        (``router.norm ∘ pre_ff_ln_2``) and select the wrong experts.
      * ``x_expert`` ``[T,H]`` — the ``pre_feedforward_layernorm_2``-normed
        residual. The token gather (expert FFN input) reads THIS tensor.

    Both are **on the device**. Router weights are device-resident; the
    pre-transposed expert weights (``gate_up_host_t`` ``[E,H,2M]``,
    ``down_host_t`` ``[E,M,H]``) stay resident on the **HOST (CPU)** and are
    indexed there. Only the small per-row ``[N,·]`` expert slices, the routing
    tensors, and the ``[T*K,H]`` gathered / expert-out buffers cross the
    host/device boundary.

    Ordering (matches gate 2):
      device: router projection (on x_router) → cpu
      host:   softmax / topk(K) / renorm / per_expert_scale ; argsort +
              token_of_row arithmetic
      device: gather token rows (from x_expert) ; expert grouped GEMM
              (3D, pre-transposed)
      host:   weighted index_add combine

    ``topk`` / ``argsort`` / index arithmetic / ``index_add`` are eager host
    CPU (never inside a spyre ``torch.compile``, per Global Constraints).
    Returns the combined ``[T,H]`` output **on the device**.
    """
    T, H = x_expert.shape

    # --- device: router projection (router.norm + router.scale + proj) -> host
    # The router's norm/scale/proj are all device-lowerable (rmsnorm + mul +
    # linear); only the softmax/topk that follow must be host. Reproduce the
    # stock Gemma4TextRouter pre-softmax math on the RAW residual (x_router),
    # then bring logits to CPU. router.norm is applied to x_router here, NOT to
    # the pre_ff_ln_2-normed x_expert (avoids the double-normalization bug).
    normed = router.norm(x_router)
    normed = normed * router.scale * router.scalar_root_size
    logits = router.proj(normed)  # [T, E]
    logits = logits.cpu().float()

    # --- host: softmax / topk / renorm / per_expert_scale (unsupported on dev)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K], [T,K]
    w = w / w.sum(-1, keepdim=True)
    per_expert_scale = router.per_expert_scale.detach().cpu().float()
    w = w * per_expert_scale[idx]

    # --- host: argsort + token_of_row / row_expert index arithmetic
    flat_expert = idx.reshape(-1)  # [T*K]
    sort_perm = torch.argsort(flat_expert)
    row_expert = flat_expert[sort_perm]  # [T*K] expert id per sorted row
    token_of_row = (torch.arange(T * K) // K)[sort_perm].to(torch.int32)

    # --- device: gather token rows from the EXPERT input ([N,H])
    gathered = compiled_gather(x_expert, token_of_row.to(x_expert.device))

    # --- host: select per-row expert weights, then move the slice to device.
    # gate_up_host_t / down_host_t are the pre-transposed, expert-outermost
    # stacks ([E,H,2M] / [E,M,H]) kept on the HOST (prepare_for_spyre does not
    # move them to the device). The per-row select ``stack[row_expert]`` is an
    # eager ``aten::index.Tensor`` — NOT lowerable on the spyre backend — so it
    # runs on CPU here (gate 2's validated flow). Indexing on the expert dim
    # preserves the pre-transposed layout, so the compiled expert region sees no
    # in-kernel transpose (shape rule 2). Only the resulting small ``[N,·]``
    # slices cross to the device for the grouped GEMM.
    gate_up_row_t = gate_up_host_t[row_expert].to(x_expert.device)  # [N,H,2M]
    down_row_t = down_host_t[row_expert].to(x_expert.device)  # [N,M,H]

    # --- device: expert grouped GEMM (rows as [N,1,H], stay 3D)
    expert_out = compiled_expert(
        gathered.unsqueeze(1), gate_up_row_t, down_row_t
    )  # [N,H]
    expert_out = expert_out.cpu().float()

    # --- host: weighted index_add combine
    row_w = w.reshape(-1)[sort_perm].unsqueeze(-1)  # [N,1]
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row.long(), expert_out * row_w)
    return out.to(dtype=x_expert.dtype, device=x_expert.device)


def _moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_up_dev,
                  down_dev, K, tile, eps):
    """Loop-on-topk MoE FFN orchestrator (spec Approach A).

    Unpacks the router's tensors, calls the compiled loop region (router +
    gather + on-device expert-weight index_select + bmms, row-tiled), then does
    the host index_add scatter-combine (scatter does not lower on device). The
    expert stacks are DEVICE-resident here (unlike _moe_ffn_split's host-
    resident stacks) -- the whole point of Approach A is the on-device select.

    Router surface (preflight-corrected): router.norm has NO .weight; the
    region applies a scale-free RMSNorm (eps INSIDE sqrt) then the [H]
    router.scale vector and the router.scalar_root_size float. Pass
    eps=config.rms_norm_eps.

    Returns the combined [T,H] MoE output on x_expert's device.
    """
    T, H = x_expert.shape
    token_ids = torch.arange(T, device=x_expert.device, dtype=torch.int32)
    # The router's proj.weight / scale / per_expert_scale are moved onto the
    # device in prepare_for_spyre (Approach-A flag branch) so they are already
    # device-resident here -- pass them straight through as region inputs.
    row_out, token_of_row = compiled_loop(
        x_router,
        x_expert,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        gate_up_dev,
        down_dev,
        token_ids,
        K,
        tile,
        eps,
    )
    row_out = row_out.cpu().float()
    token_of_row = token_of_row.cpu().long()
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row, row_out)
    return out.to(dtype=x_expert.dtype, device=x_expert.device)


# ---------------------------------------------------------------------------
# ALL-DEVICE chunked FFN (spec Approach A, "nothing but glue on host").
# Gate-A5-proven (gateA5_chunked_ondevice.py, mean_rel=0.028). See the
# _MOE_CHUNKED_ONDEVICE flag docstring for the two backend workarounds this
# encodes (topk-pad + mask-reduce/accumulate instead of gather/scatter).
# ---------------------------------------------------------------------------


def _moe_route_padded(x_router, router_proj_w, router_scale,
                      router_scalar_root_size, per_expert_scale, K, eps,
                      pad_w, pad_neg):
    """All-device router: full dense [T,E] routing-weight (topk-pad fix).

    Router surface matches _compiled_moe_loop_region (preflight-corrected): the
    stock ``router.norm`` is a SCALE-FREE Gemma4RMSNorm (no ``.weight``; eps
    INSIDE the sqrt) applied to the RAW residual ``x_router``, then the [H]
    ``router_scale`` vector and the ``router_scalar_root_size`` float, then the
    router projection.

    topk over a pow2 stick-multiple width (E=128) aborts on-card
    (``Incorrect chunk size``), so the logits are padded to ``pad_w`` (non-pow2)
    with ``pad_neg`` (-inf) BEFORE topk; the pad columns never win top-K. Only
    the kth topk VALUE is used (``wv[..., -1:]``) as a threshold over the
    ORIGINAL [T,E] probs -- the fp16 index is never materialized (no
    ``customops.py`` fp16->int32 CPU fallback). The result is a DENSE [T,E]
    routing weight: zero for non-selected experts, ``renorm*per_expert_scale``
    for the top-K. Downstream expert chunks turn per-expert selection into
    arithmetic (``(w*onehot[e]).sum``), so no index/gather is needed.

    Args:
        x_router: RAW flattened residual [T,H] (router's own norm applied here).
        router_proj_w: Router projection weight [E,H].
        router_scale: Post-norm gain vector [H] (router.scale).
        router_scalar_root_size: hidden_size ** -0.5 (Python float).
        per_expert_scale: Per-expert scale [E] (router.per_expert_scale).
        K: Top-K experts per token.
        eps: config.rms_norm_eps (inside the sqrt).
        pad_w: Padded topk-input width (>E, non-pow2; e.g. 160 for E=128).
        pad_neg: Pad-column fill (~ -inf fp16; e.g. -30000.0).

    Returns:
        w: Dense routing weight [T,E] (renormed top-K * per_expert_scale, zero
           elsewhere), on x_router's device.
    """
    T, _ = x_router.shape
    E = router_proj_w.shape[0]
    # scale-free RMSNorm (eps inside sqrt) on the raw residual, then [H] gain.
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)
    normed = normed * router_scale * router_scalar_root_size
    probs = torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]
    # topk-pad: widen to a non-pow2 width so the backend tiling stays
    # stick-aligned; pad cols are -inf -> never selected.
    pad = torch.full((T, pad_w - E), pad_neg, dtype=probs.dtype,
                     device=probs.device)
    padded = torch.cat([probs, pad], dim=-1)  # [T,pad_w]
    wv, _ = torch.topk(padded, K, dim=-1)  # [T,K]; idx<E, never materialized
    kth = wv[..., -1:]  # [T,1] kth-largest VALUE threshold
    mask = torch.where(probs >= kth, probs, torch.zeros_like(probs))  # [T,E]
    w = mask / mask.sum(-1, keepdim=True)  # renorm top-K to sum 1
    return w * per_expert_scale  # [T,E] dense routing weight


def _moe_expert_chunk(x_expert, w, acc, gate_c, up_c, down_c, onehot_c):
    """All-device expert chunk: per-expert SwiGLU + mask-reduce weight + accum.

    Runs the ``_MOE_EC`` experts of ONE chunk and adds their weighted outputs
    into the running device accumulator ``acc``. For each chunk-local expert
    ``j`` (global expert ``lo+j``):

        a  = gelu(x @ gate_c[j], tanh) * (x @ up_c[j])   # [T,F] SwiGLU
        we = (w * onehot_c[j]).sum(-1, keepdim=True)      # [T,1] mask-reduce
        acc += (a @ down_c[j]) * we                       # [T,H]

    The mask-reduce ``(w*onehot_c[j]).sum`` picks this expert's dense routing
    weight out of the full [T,E] ``w`` as ARITHMETIC (onehot_c[j] is one at the
    global column lo+j), avoiding an index/gather that does not lower on-card.
    The [T,H] accumulator matches the down-projection output layout AND the
    reduction layout, so summing over experts needs no stack. Everything stays
    on the device; ``acc`` is threaded back in by the host glue loop.

    Args:
        x_expert: pre_ff_ln_2-normed residual [T,H] (expert FFN input).
        w: Dense routing weight [T,E] from _moe_route_padded.
        acc: Running device accumulator [T,H] (device-resident across chunks).
        gate_c: This chunk's gate weights [Ec,H,F] (offset-0 contiguous).
        up_c: This chunk's up weights [Ec,H,F] (offset-0 contiguous).
        down_c: This chunk's down weights [Ec,F,H] (offset-0 contiguous).
        onehot_c: This chunk's one-hot rows [Ec,E] (row j is one at col lo+j).

    Returns:
        acc: Updated accumulator [T,H] on the device.
    """
    Ec = gate_c.shape[0]
    for j in range(Ec):
        a = F.gelu(x_expert @ gate_c[j], approximate="tanh") * (x_expert @ up_c[j])
        we = (w * onehot_c[j]).sum(-1, keepdim=True)  # [T,1]
        acc = acc + (a @ down_c[j]) * we
    return acc


def _moe_ffn_chunked(x_router, x_expert, router, compiled_route,
                     compiled_chunk, chunks, K, eps):
    """All-device chunked MoE FFN orchestrator (spec Approach A).

    Runs the router ONCE (its own compiled region) to get the dense [T,E]
    routing weight, then walks the pre-materialized expert-weight chunks,
    threading a DEVICE-RESIDENT [T,H] accumulator through each compiled chunk
    region. The host does ONLY glue: the chunk-loop counter and passing the same
    device accumulator handle back in. Nothing gather/scatter/FFN runs on host,
    and the accumulator never round-trips to CPU until the final combine.

    ``chunks`` is the per-layer list built by prepare_for_spyre: each entry is
    ``(gate_c, up_c, down_c, onehot_c)`` of OFFSET-0 contiguous device tensors
    for one chunk of ``_MOE_EC`` experts (the storage-offset remat fix -- a
    non-zero-offset slice passed as a compile input reads wrong storage).

    Args:
        x_router: RAW flattened residual [T,H] (router input).
        x_expert: pre_ff_ln_2-normed residual [T,H] (expert input).
        router: stock Gemma4TextRouter (proj/scale/per_expert_scale device-
            resident; scalar_root_size is a Python float).
        compiled_route: torch.compile(_moe_route_padded).
        compiled_chunk: torch.compile(_moe_expert_chunk).
        chunks: list of (gate_c, up_c, down_c, onehot_c) device-tensor tuples.
        K: Top-K experts per token.
        eps: config.rms_norm_eps.

    Returns:
        out: MoE FFN output [T,H] on x_expert's device.
    """
    T, H = x_expert.shape
    w = compiled_route(
        x_router,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        K,
        eps,
        _MOE_PADW,
        _MOE_PAD_NEG,
    )  # [T,E] on device
    acc = torch.zeros(T, H, dtype=x_expert.dtype, device=x_expert.device)
    for gate_c, up_c, down_c, onehot_c in chunks:  # HOST GLUE: loop counter only
        acc = compiled_chunk(x_expert, w, acc, gate_c, up_c, down_c, onehot_c)
    return acc


class Gemma4MoEBlock(nn.Module):
    """Registered Gemma 4 **MoE** decoder block used by the Spyre adapter.

    Mirrors the dense ``hf_gemma4.Gemma4Block`` (same class shape, same 7-arg
    ``cache_index`` call signature, same ``layer_scalar`` buffer idiom) but its
    ``forward`` reproduces the ``enable_moe_block=True`` branch of the stock
    ``Gemma4TextDecoderLayer.forward``: a dense MLP **in parallel** with a
    top-K MoE FFN, combined ``post_feedforward_layernorm(h_dense + h_moe)``.

    Attention is the upstream ``Gemma4Attention`` module composed VERBATIM
    (exactly as ``Gemma4Block`` does), so KV handling (in-place
    ``kv_cache_update`` indirect scatter, #330) is the same code path the dense
    adapter is tested on. The block is NOT a single ``torch.compile`` — the MoE
    routing is host-side in the default/split mode — so it composes several
    per-region ``torch.compile`` handles built once in ``__init__``:

        # attention half -> post-attn-norm sandwich -> residual add
        residual = h
        h_dense = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
        flat    = residual.reshape(-1, H)          # RAW residual
        # router reads flat (its own scale-free norm inside the FFN region);
        # experts read a SEPARATE pre_ff_ln_2 norm of the same flat.
        h_moe   = post_feedforward_layernorm_2(<ffn mode>(flat, pre_ff_ln_2(flat), ...))
        h       = post_feedforward_layernorm(h_dense + h_moe)
        h       = residual + h
        h       = h * layer_scalar

    The pre/post ``_2`` norms run on-device on the flattened ``[T,H]`` tensor;
    the router's internal norm runs on the raw ``flat`` (NOT the pre_ff_ln_2
    output), so the host only ever sees the small routing tensors plus the
    ``[T*K,H]`` gathered / expert-out buffers. Mode-specific expert weights
    (set by ``prepare_for_spyre``) are read fresh from ``self`` at call time,
    the same call-time-read rule the dense block uses for ``layer_scalar``.
    """

    def __init__(self, layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        # MoE-branch submodules / norms (stock Gemma4TextDecoderLayer names).
        self.router = layer.router
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
        self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        # Captured knobs. K + eps are per-layer scalars; the mode-specific
        # expert stacks are read fresh off ``self`` at call time (populated by
        # prepare_for_spyre), NOT captured here.
        self._moe_k = layer._spyre_moe_k
        # Gemma4RMSNorm exposes ``.eps`` (== config.rms_norm_eps).
        self._moe_rms_eps = self.pre_feedforward_layernorm_2.eps

        # Compiled device regions (built once per block). The dense MLP is
        # compiled as its own region so the dense branch lowers; the router,
        # gather, expert GEMM, loop, and per-chunk regions are each compiled
        # for whichever FFN mode is active.
        self._compiled_mlp = torch.compile(self.mlp, dynamic=False)
        self._compiled_gather = torch.compile(
            _compiled_device_gather, dynamic=False
        )
        self._compiled_expert = torch.compile(
            _compiled_moe_device_region, dynamic=False
        )
        self._compiled_loop = torch.compile(
            _compiled_moe_loop_region, dynamic=False
        )
        self._compiled_route = torch.compile(_moe_route_padded, dynamic=False)
        self._compiled_chunk = torch.compile(_moe_expert_chunk, dynamic=False)
        self.train(layer.training)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            h,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        # Sandwich: norm the attention output BEFORE adding the residual.
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        bsz, seq_len, hidden = h.shape

        # Dense branch: pre_ff_ln -> mlp -> post_ff_ln_1.
        h_dense = self.post_feedforward_layernorm_1(
            self._compiled_mlp(self.pre_feedforward_layernorm(residual))
        )

        # Sparse branch: the router reads the RAW flattened residual (its own
        # scale-free norm is applied inside the FFN region), while the experts
        # consume a SEPARATE pre_ff_ln_2 normalization of that same residual
        # (stock modeling_gemma4.py). Thread them as two tensors so the router
        # input is not double-normalized. The per-layer expert weights
        # (mode-specific layout, set by prepare_for_spyre) are read fresh off
        # ``self`` at call time (like the dense block's layer_scalar).
        flat = residual.reshape(-1, hidden)  # [T,H] RAW -> router
        x_moe = self.pre_feedforward_layernorm_2(flat)  # [T,H] normed -> experts
        if _MOE_CHUNKED_ONDEVICE:
            # ALL-DEVICE: router + expert GEMMs + weight + sum-over-experts all
            # lower; host does only the chunk-loop glue (inside _moe_ffn_chunked)
            # threading a device-resident accumulator. Per-chunk offset-0 expert
            # weights were pre-materialized in prepare_for_spyre.
            moe_out = _moe_ffn_chunked(
                flat,
                x_moe,
                self.router,
                self._compiled_route,
                self._compiled_chunk,
                self._spyre_moe_chunks,
                self._moe_k,
                self._moe_rms_eps,
            )  # [T,H]
        elif _MOE_LOOP_ON_TOPK:
            moe_out = _moe_ffn_loop(
                flat,
                x_moe,
                self.router,
                self._compiled_loop,
                self._spyre_gate_up_dev,
                self._spyre_down_dev,
                self._moe_k,
                _MOE_TILE,
                self._moe_rms_eps,
            )  # [T,H]
        else:
            moe_out = _moe_ffn_split(
                flat,
                x_moe,
                self.router,
                self._compiled_gather,
                self._compiled_expert,
                self._spyre_gate_up_t,
                self._spyre_down_t,
                self._moe_k,
            )  # [T,H]
        moe_out = moe_out.reshape(bsz, seq_len, hidden)
        h_moe = self.post_feedforward_layernorm_2(moe_out)

        # Combine dense + MoE, final sandwich norm, residual, per-layer scalar.
        h = self.post_feedforward_layernorm(h_dense + h_moe)
        h = residual + h
        return h * layer_scalar, key_cache, value_cache


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Gemma 4 **MoE** causal-LM model in-place.

    Reuses the shared attention-side prep (``_setup_gemma4_text_decoder``:
    RMSNorm patch, per-type RoPE, KV shapes, ``pad_lm_head``) and adds the MoE
    layout / bring-up steps:

      * assert ``enable_moe_block=True`` and coerce/assert ``top_k_experts == 4``
        (K pinned for bring-up, Global Constraints);
      * lay each layer's packed expert weights **expert-dim-outermost and
        pre-transposed** (``gate_up`` ``[E,2M,H]`` -> ``[E,H,2M]``, ``down``
        ``[E,H,M]`` -> ``[E,M,H]``; shape rule 2 + spec §3.5). The layout THEN
        depends on the active FFN mode:
          - ``_MOE_CHUNKED_ONDEVICE``: de-fuse gate_up into gate/up ``[E,H,M]``
            halves, slice into ``ceil(E/_MOE_EC)`` chunks, move each chunk +
            its one-hot rows to the device as OFFSET-0 contiguous tensors
            (``layer._spyre_moe_chunks``); router weights moved device-resident;
          - ``_MOE_LOOP_ON_TOPK``: whole stacks device-resident
            (``_spyre_gate_up_dev`` / ``_spyre_down_dev``), router device-resident;
          - default: stacks stay HOST-resident plain CPU attributes
            (``_spyre_gate_up_t`` / ``_spyre_down_t``) so ``model.to("spyre")``
            never sweeps them;
        the original ``gate_up_proj`` / ``down_proj`` parameters are deleted;
      * build ``model._spyre_compiled_blocks`` from the MoE block factory.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    # K pinned to 4 for bring-up. Coerce the config then assert so the router's
    # host topk and the device grouped-GEMM path both run at the validated K.
    cfg.top_k_experts = _MOE_BRINGUP_K
    assert cfg.top_k_experts == _MOE_BRINGUP_K, (
        f"MoE bring-up pins top_k_experts to {_MOE_BRINGUP_K}; "
        f"got {cfg.top_k_experts}."
    )
    # The expert SwiGLU hardcodes gelu(approximate="tanh") to match this
    # checkpoint's hidden_activation. Guard so a variant with a different
    # activation fails loudly instead of computing silently-wrong output.
    act_fn = getattr(cfg, "hidden_activation", None)
    assert act_fn == "gelu_pytorch_tanh", (
        "hf_gemma4_moe expert SwiGLU is fixed to gelu(approximate='tanh'); "
        f"config hidden_activation={act_fn!r} is unsupported."
    )

    # The three FFN modes lay experts out differently and are mutually
    # exclusive; guard so a mis-set pair fails loudly at load rather than
    # reading a stack the active mode's forward never populated.
    assert not (_MOE_CHUNKED_ONDEVICE and _MOE_LOOP_ON_TOPK), (
        "_MOE_CHUNKED_ONDEVICE and _MOE_LOOP_ON_TOPK are mutually exclusive "
        "FFN modes; enable at most one."
    )
    if _MOE_CHUNKED_ONDEVICE:
        E = cfg.num_experts
        assert E % _MOE_EC == 0, (
            f"all-device chunked MoE needs num_experts ({E}) divisible by "
            f"_MOE_EC ({_MOE_EC})."
        )
        assert _MOE_PADW > E and (_MOE_PADW & (_MOE_PADW - 1)) != 0, (
            f"_MOE_PADW ({_MOE_PADW}) must exceed num_experts ({E}) and be "
            "non-power-of-two (topk-pad fix)."
        )

    num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer = (
        _setup_gemma4_text_decoder(model, allow_moe=True)
    )

    # Lay out expert weights: expert-dim-outermost (already, [E,...]) and
    # PRE-TRANSPOSED so the compiled expert region needs no in-kernel transpose
    # of a large weight (shape rule 2).
    #
    # The pre-transposed expert stacks are kept on the HOST (CPU), NOT moved to
    # the device, for two reasons that both surfaced on-card at 26B:
    #
    #   1. The per-row weight select ``gate_up_t[row_expert]`` is an eager
    #      ``aten::index.Tensor`` — unsupported on the spyre backend
    #      (``NotImplementedError``). Only plain ``gather``/``bmm`` lower
    #      (spec §2.1); fancy indexing must run on CPU. Gate 2 selects on host
    #      for exactly this reason, then moves the ``[N,·]`` slice to the device.
    #   2. Even if the select lowered, keeping all 128 experts × 30 layers
    #      resident is ~46 GB fp16; the [N,H,2M]/[N,M,H] per-row gathers plus
    #      the rest of the model exhaust the card (FlexAllocator OOM).
    #
    # They are stored as PLAIN ATTRIBUTES (not ``register_buffer``) so
    # ``_move_to_spyre_with_layout``'s ``named_buffers()`` sweep never moves them
    # to the device — they stay on CPU. ``_moe_ffn_split`` selects the per-row
    # weights here on the host and moves only the small ``[N,·]`` slices to the
    # device for the compiled expert GEMM. The ORIGINAL ``gate_up_proj`` /
    # ``down_proj`` parameters are deleted (the stock ``Gemma4TextExperts.forward``
    # never runs; the split FFN uses only the transposed CPU stacks), so the
    # experts are not paid for twice on the host either.
    # One pass per layer: build the MoE block (composing upstream Gemma4Attention
    # verbatim), attach its mode-specific expert weights, register it back into
    # ``backbone.layers[i]`` (so ``_run_blocks_over_embeds`` can read
    # ``layer_scalar`` off it, exactly as ``prepare_gemma4_blocks`` does for the
    # dense path), and compile it. The expert-weight layout below targets the
    # BLOCK instance (``block._spyre_*``), read fresh by ``Gemma4MoEBlock.forward``
    # — the same call-time-read rule the dense block uses for ``layer_scalar``.
    compiled_blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        # Stash the pinned K on the layer so Gemma4MoEBlock.__init__ can capture
        # it (cfg.top_k_experts was coerced to _MOE_BRINGUP_K above).
        layer._spyre_moe_k = _MOE_BRINGUP_K
        block = Gemma4MoEBlock(
            layer,
            num_q_heads_per_layer[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
        )
        experts = layer.experts
        # gate_up_proj: [E,2M,H] -> [E,H,2M]; down_proj: [E,H,M] -> [E,M,H].
        gate_up_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()
        down_t = experts.down_proj.data.transpose(1, 2).contiguous()
        del experts.gate_up_proj
        del experts.down_proj
        # The router is shared between ``layer`` and ``block`` (composed as
        # ``self.router = layer.router``), so router-weight moves below apply to
        # the block's router too.
        router = block.router
        if _MOE_CHUNKED_ONDEVICE:
            # ALL-DEVICE chunked path (spec Approach A, "nothing but glue on
            # host"). De-fuse the packed gate_up [E,H,2M] into separate gate /
            # up [E,H,M] halves (matching x@Wg (gate) and x@Wu (up) in
            # _moe_expert_chunk: the fused SwiGLU is gelu(g)*u with
            # g,u = chunk(2, dim=-1), so cols [:M] are gate, [M:] are up), then
            # slice the E experts into ceil(E/_MOE_EC) chunks of _MOE_EC and
            # move each chunk to the device as an OFFSET-0 CONTIGUOUS tensor.
            #
            # The offset-0 remat is load-bearing: a non-zero storage_offset
            # device-tensor view (e.g. gate_dev[lo:lo+Ec]) passed as a compile
            # input reads the WRONG storage on-card (proven in
            # slice_input_iso.py; related to [[project-pr2426-storage-offset-
            # review]]). Slicing on the HOST tensor then ``.to("spyre")`` gives
            # each chunk its own offset-0 device tensor -- pure glue, done once
            # here, not per-forward.
            E = gate_up_t.shape[0]
            M = gate_up_t.shape[2] // 2
            gate_t = gate_up_t[:, :, :M].contiguous()  # [E,H,M] gate half
            up_t = gate_up_t[:, :, M:].contiguous()  # [E,H,M] up half
            onehot = torch.eye(E, dtype=gate_t.dtype)  # [E,E] one-hot rows
            chunks = []
            for lo in range(0, E, _MOE_EC):
                hi = lo + _MOE_EC
                chunks.append((
                    gate_t[lo:hi].contiguous().to("spyre"),   # [Ec,H,M]
                    up_t[lo:hi].contiguous().to("spyre"),      # [Ec,H,M]
                    down_t[lo:hi].contiguous().to("spyre"),    # [Ec,M,H]
                    onehot[lo:hi].contiguous().to("spyre"),    # [Ec,E]
                ))
            block._spyre_moe_chunks = chunks
            # The router runs entirely on-device (scale-free norm + [H] scale +
            # proj + softmax + padded topk + per_expert_scale), so its weights
            # must be device-resident. Reassign the Parameter (a cross-backend
            # ``param.data = ...`` raises on the type change).
            router.proj.weight = torch.nn.Parameter(
                router.proj.weight.data.to("spyre"), requires_grad=False
            )
            router.scale = torch.nn.Parameter(
                router.scale.data.to("spyre"), requires_grad=False
            )
            router.per_expert_scale = torch.nn.Parameter(
                router.per_expert_scale.data.to("spyre"), requires_grad=False
            )
        elif _MOE_LOOP_ON_TOPK:
            # Approach A: experts HBM-RESIDENT on device, E outermost. Row-major
            # [E,H,2M]/[E,M,H] is E-outermost (enforce_indirect_access: indexed
            # dim at device position 0) AND stick-correct for the bmm weight
            # operand (2M / H is the generated dim on the stick) -> zero
            # restickify. Move explicitly (plain attrs are not in the buffer
            # sweep).
            block._spyre_gate_up_dev = gate_up_t.to("spyre")  # [E,H,2M]
            block._spyre_down_dev = down_t.to("spyre")  # [E,M,H]
            # The whole router runs on-device in the loop region (scale-free
            # norm + [H] scale + proj + topk + per_expert_scale gather), so its
            # weights must be device-resident too. Move them here rather than
            # rely on a model.to("spyre") sweep -- standalone callers (the gate)
            # never sweep the model, and the region's normed activation is
            # device-side. Reassign the Parameter object (a cross-backend
            # ``param.data = ...`` set_data raises on the type change); the
            # spyre move stickifies proj.weight [E,H] like any 2D matmul weight.
            # router.scalar_root_size is a Python float (no move).
            router.proj.weight = torch.nn.Parameter(
                router.proj.weight.data.to("spyre"), requires_grad=False
            )
            router.scale = torch.nn.Parameter(
                router.scale.data.to("spyre"), requires_grad=False
            )
            router.per_expert_scale = torch.nn.Parameter(
                router.per_expert_scale.data.to("spyre"), requires_grad=False
            )
        else:
            block._spyre_gate_up_t = gate_up_t.cpu()  # [E,H,2M] host-resident
            block._spyre_down_t = down_t.cpu()  # [E,M,H] host-resident

        # Register the block back into the backbone (so _run_blocks_over_embeds
        # reads layer_scalar off it), then append the EAGER block to the
        # compiled-blocks list. Unlike the dense Gemma4Block (a pure-device
        # forward that prepare_gemma4_blocks wraps in torch.compile), the MoE
        # block's forward is a HYBRID: eager host orchestration (topk/argsort/
        # index_add/chunk-loop glue, all outside any spyre graph per the
        # all-device Global Constraint) around several inner torch.compile'd
        # device regions built in __init__. Wrapping the whole block in
        # torch.compile would try to trace that host work into one spyre graph.
        # So _run_blocks_over_embeds invokes the eager block, which dispatches to
        # its inner compiled regions.
        backbone.layers[i] = block
        compiled_blocks.append(block)

    model._spyre_compiled_blocks = compiled_blocks


# ``_run_backbone_forward`` and ``_run_forward`` are NOT defined here — they are
# re-exported from ``hf_gemma4`` (see the import block at the top of this file).
# Since #350 both are block-AGNOSTIC: they drive ``model._spyre_compiled_blocks``
# and read ``layer_scalar`` off each registered block, so the MoE blocks slot in
# with no MoE-specific forward. This removes the last carrier of the pre-#330
# ``is_filling/token_index/cache_position`` triple and guarantees the MoE forward
# tracks any future dense-forward change.
