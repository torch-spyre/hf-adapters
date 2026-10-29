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

"""Benchmark Muse Glimmer 30B decode on one Spyre card: plain greedy and tree spec.

Decode tokens per second at batch 1, greedy: generated tokens after the first,
divided by decode wall time (prefill excluded). Each prompt runs once as a
warmup (compiles every graph), then ``--repeats`` measured times.

``--mode ar`` times the adapter's plain ``generate``; ``--mode spec`` the
DFlash tree decoder (and, with ``--compare-ar``, the first position where it
differs from plain greedy). Plain ``generate`` follows ``hf_muse_glimmer``'s
per-request policy (plain SDPA when the cache fits the sliding window, the
anchored sliding-window cache beyond it; 512-token prefill chunks past 512
tokens). ``SPYRE_MUSE_FP8=attn,mlp,lm_head`` (read at model preparation)
switches the target's text linears to W8A8.

Under ``torchrun --nproc-per-node=N`` the target runs tensor-parallel over N
cards (``_muse_tp``); rank 0 reports.

Usage (on the Spyre pod, project root on PYTHONPATH)::

    SPYRE_DEVICES=3 python scripts/bench_muse_spec_decode_spyre.py \\
        --model /home/galbloch/hub/muse-glimmer-30b \\
        --assistant /home/galbloch/hub/muse-glimmer-30b-assistant \\
        --prompts-pt prompts.pt --mode spec --compare-ar --output out.json
"""

import argparse
import json
import os
import statistics
import time

import torch


def _first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    p.add_argument("--assistant", default="meta-models/Muse-Glimmer-30B-assistant")
    p.add_argument("--prompt-file", action="append", default=[])
    p.add_argument("--prompts-pt", help="torch.load-able list of 1-D token-id tensors")
    p.add_argument("--mode", choices=("ar", "spec"), default="spec")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--capacity", type=int, help="fixed verify KV rows (one compile)")
    p.add_argument(
        "--prefill-chunk",
        type=int,
        help="tree prefill chunk rows (default: the adapter's prompt-length policy)",
    )
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--compare-ar", action="store_true")
    p.add_argument(
        "--reference",
        help="JSON from a --mode ar run: its outputs are the stock greedy",
    )
    p.add_argument(
        "--warmup-first-only",
        action="store_true",
        help="warm up on the first prompt only (fixed graph shapes: one compile)",
    )
    p.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="target weight dtype (fp8 linears need float16)",
    )
    p.add_argument("--output", help="write the per-run results as JSON here")
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        # TP: the adapter shards the text decoder at preparation (_muse_tp).
        import torch.distributed as dist

        dist.init_process_group("cpu:gloo,spyre:spyreccl")
    log = print if rank == 0 else (lambda *a, **k: None)

    from transformers import AutoTokenizer, MuseGlimmerAssistantModel

    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.hf_common import encode_prompts
    from hf_adapters.muse_spec_decode import TreeSpeculativeDecoder

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        encode_prompts(tok, [open(f).read()])["input_ids"] for f in args.prompt_file
    ]
    if args.prompts_pt:
        prompts += [
            torch.as_tensor(t).long().reshape(1, -1)
            for t in torch.load(args.prompts_pt)
        ]
    if not prompts:
        p.error("pass --prompt-file or --prompts-pt")

    t = time.time()
    target = AutoSpyreModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    )
    decoder = None
    if args.mode == "spec":
        assistant = MuseGlimmerAssistantModel.from_pretrained(
            args.assistant, dtype=torch.float32
        ).eval()
        decoder = TreeSpeculativeDecoder(
            target,
            assistant,
            width=args.width,
            top_k=args.top_k,
            prefill_chunk=args.prefill_chunk,
        )
        decoder.verifier.fixed_capacity = args.capacity
        if args.capacity:
            decoder.draft_capacity = args.capacity - 2 * args.width - 64
        del assistant  # the drafter holds its own device copies
    log(f"loaded in {time.time() - t:.0f} s", flush=True)
    env = {
        k: os.getenv(k)
        for k in ("SPYRE_MUSE_FP8", "SPYRE_MUSE_DRAFT_FP8", "SPYRE_MUSE_FP8_ACT_MAX")
    } | {"dtype": args.dtype}
    # Muse ends a turn with <|eot|> (200008) as well as <|end_of_text|>.
    eos = target.generation_config.eos_token_id
    stock = {}
    if args.reference:
        stock = {
            r["prompt"]: r["output_ids"]
            for r in json.load(open(args.reference))
            if r["phase"] == "measured"
        }
    results = []
    for index, ids in enumerate(prompts):
        reference = stock.get(index)
        if args.compare_ar and reference is None:
            reference = target.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
            )[0, ids.shape[1] :].tolist()
        first = 0 if (index == 0 or not args.warmup_first_only) else 1
        for repeat in range(first, args.repeats + 1):
            row = dict(
                prompt=index,
                prompt_tokens=int(ids.shape[1]),
                phase="warmup" if repeat == 0 else "measured",
                mode=args.mode,
                env=env,
            )
            if args.mode == "ar":
                t0 = time.perf_counter()
                target.generate(ids, max_new_tokens=1, do_sample=False)
                t1 = time.perf_counter()
                out = target.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos,
                )[0, ids.shape[1] :].tolist()
                t2 = time.perf_counter()
                decode_s = (t2 - t1) - (t1 - t0)
                row |= dict(
                    generated=len(out),
                    decode_tps=round((len(out) - 1) / max(decode_s, 1e-9), 3),
                    prefill_ms=round(1000 * (t1 - t0), 1),
                )
            else:
                out = decoder.generate(ids, args.max_new_tokens, eos_token_id=eos)[
                    0
                ].tolist()
                s = decoder.stats
                row |= dict(
                    generated=len(out),
                    decode_tps=round(s.decode_tps(), 3),
                    tokens_per_step=round(s.mean_accepted, 3),
                    draft_ms=round(statistics.median(s.draft_ms), 2),
                    verify_ms=round(statistics.median(s.verify_ms), 2),
                    commit_ms=round(statistics.median(s.commit_ms), 2),
                    prefill_ms=round(s.prefill_ms, 1),
                )
            if reference is not None:
                row["first_difference_vs_ar"] = _first_difference(out, reference)
            row["text"] = tok.decode(out[:64])
            log(json.dumps(row), flush=True)
            results.append(row | {"output_ids": out, "reference_ids": reference})
    measured = [r for r in results if r["phase"] == "measured"]
    summary = dict(
        mode=args.mode,
        mean_decode_tps=round(statistics.mean(r["decode_tps"] for r in measured), 3),
        runs=len(measured),
        env=env,
    )
    if args.mode == "spec":
        summary["mean_tokens_per_step"] = round(
            statistics.mean(r["tokens_per_step"] for r in measured), 3
        )
    summary["tp"] = world
    log(json.dumps(summary), flush=True)
    if args.output and rank == 0:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
