#!/usr/bin/env python3
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

"""Benchmark the Gemma 4 decode MoE region, without attention or LM head.

Run on a Spyre host with the repository on PYTHONPATH::

    python scripts/benchmark_gemma4_decode_moe.py --layout chunked --batches 1 \
        --output /tmp/gemma4-decode-moe.json --trace /tmp/gemma4-decode-moe-trace.json

Use ``--layout baseline`` with the same seed and batch for a matched comparison.
The timed measurements exclude setup and warmup. The separate profiling call
exports a Chrome trace.
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch_spyre.ops.fallbacks import FallbackWarning

from hf_adapters.hf_common import (
    moe_decode_selected_experts,
    moe_topk,
    prepare_moe_expert_weights,
)
from hf_adapters.hf_gemma4_moe import _compiled_moe_loop_region, _router_probs

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, STICK = 128, 2816, 704, 8, 64


def _prepare_weights(seed, layout):
    generator = torch.Generator().manual_seed(seed)
    gate_up = (
        torch.randn(
            EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.float16, generator=generator
        )
        * 0.05
    )
    down = (
        torch.randn(
            EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.float16, generator=generator
        )
        * 0.05
    )
    router = (
        torch.randn(EXPERTS, HIDDEN, dtype=torch.float16, generator=generator) * 0.05
    )
    scales = torch.rand(EXPERTS, dtype=torch.float16, generator=generator) + 0.5
    experts = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    prepare_moe_expert_weights(
        experts, pad_to_multiple=STICK, decode_chunks=4 if layout == "chunked" else 1
    )
    scale_stick = scales[:, None].expand(-1, STICK).contiguous().to("spyre")
    return experts, gate_up, down, router, scales, scale_stick


def _reference(x, gate_up, down, router, scales):
    # Independent dense CPU arithmetic, with the same fp16 boundaries as routing.
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    normalized = normalized.half()
    probs = torch.softmax(F.linear(normalized, router), dim=-1)
    values, indices = torch.topk(probs, TOP_K, dim=-1)
    weights = values / values.sum(-1, keepdim=True)
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(TOP_K):
            expert = int(indices[token, slot])
            gate = row @ gate_up[expert, :INTERMEDIATE].float().T
            up = row @ gate_up[expert, INTERMEDIATE:].float().T
            active = F.gelu(gate, approximate="tanh") * up
            partial = active @ down[expert].float().T
            output[token] += (
                partial * float(weights[token, slot]) * float(scales[expert])
            )
    return output


def _run(batch, prepared, warmup, iterations, trace, layout):
    experts, gate_up, down, router, scales, scale_stick = prepared
    generator = torch.Generator().manual_seed(20 + batch)
    x = torch.randn(batch, HIDDEN, dtype=torch.float16, generator=generator) * 0.5
    expected = _reference(x, gate_up, down, router, scales)
    x_device = x.to("spyre")
    router_device = router.to("spyre")

    def region(hidden):
        if layout == "baseline":
            probs = _router_probs(hidden, router_device, 1.0, 1.0, 1e-6)
            values, indices = moe_topk(probs, TOP_K)
            return moe_decode_selected_experts(
                hidden,
                values / values.sum(-1, keepdim=True),
                indices,
                experts.gate_proj,
                experts.up_proj,
                experts.down_proj,
                TOP_K,
                32,
                STICK,
                "gelu_tanh",
                per_expert_scale_stick=scale_stick,
            )
        return _compiled_moe_loop_region(
            hidden,
            hidden,
            router_device,
            1.0,
            1.0,
            scale_stick,
            experts.decode_gate_proj,
            experts.decode_up_proj,
            experts.decode_down_proj,
            TOP_K,
            32,
            STICK,
            1e-6,
        )

    compiled = torch.compile(region, dynamic=False, fullgraph=True)
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always", FallbackWarning)
        actual = compiled(x_device)
        torch.spyre.synchronize()
        torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)
        for _ in range(warmup):
            compiled(x_device)
            torch.spyre.synchronize()
        samples = []
        for _ in range(iterations):
            torch.spyre.synchronize()
            start = time.perf_counter_ns()
            compiled(x_device)
            torch.spyre.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            compiled(x_device)
            torch.spyre.synchronize()
    fallbacks = [
        str(item.message)
        for item in caught
        if issubclass(item.category, FallbackWarning)
    ]
    if fallbacks:
        raise RuntimeError(f"decode MoE fell back to CPU: {fallbacks}")
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [
        {"name": event["name"], "duration_ms": event["dur"] / 1000}
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X" and event.get("dur")
    ]
    if not kernels:
        raise RuntimeError("no Spyre kernel events in the decode MoE trace")
    return {
        "layout": layout,
        "batch": batch,
        "warmup": warmup,
        "iterations": iterations,
        "wall_ms": samples,
        "wall_median_ms": statistics.median(samples),
        "kernel_event_sum_ms": sum(item["duration_ms"] for item in kernels),
        "kernels": kernels,
        "correctness": "matches independent dense CPU expert calculation",
        "trace": str(trace),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=["baseline", "chunked"], default="chunked")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace", type=Path, default=Path("/tmp/gemma4-decode-moe-trace.json")
    )
    args = parser.parse_args()
    if not args.batches or any(b < 1 or b & (b - 1) for b in args.batches):
        parser.error("batch sizes must be positive powers of two")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if args.output and args.trace == args.output:
        parser.error("output and trace must be different paths")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_weights(args.seed, args.layout)
    results = []
    for batch in args.batches:
        trace = args.trace.with_name(f"{args.trace.stem}-b{batch}{args.trace.suffix}")
        result = _run(batch, prepared, args.warmup, args.iterations, trace, args.layout)
        results.append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "layout",
                        "batch",
                        "wall_median_ms",
                        "kernel_event_sum_ms",
                        "correctness",
                        "trace",
                    )
                }
            ),
            flush=True,
        )
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
