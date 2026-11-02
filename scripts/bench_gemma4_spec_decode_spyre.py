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

"""Benchmark Gemma 4 26B-A4B tree speculative decoding (XPress) on one Spyre card.

Decode tokens per second at batch 1, greedy, with the definition used for the
plain-decode baseline: the generated tokens after the first, divided by the
decode wall time (prefill / TTFT excluded). Each prompt runs once as a warmup
(compiles every graph), then ``--repeats`` measured times.

Reports per run: decode TPS, mean tokens per step, median draft / verify /
commit ms, and the first position where the output differs from a plain
greedy ``generate`` of the same prompt (``--compare-ar``).

The lossy options are environment variables read at model preparation:
``SPYRE_MOE_UNION_CAP=16`` and ``SPYRE_GEMMA4_FP8=attn,dense,lm_head``.

Usage (on the Spyre pod, project root on PYTHONPATH)::

    SPYRE_DEVICES=0 SPYRE_MOE_UNION_CAP=16 SPYRE_GEMMA4_FP8=attn,dense,lm_head \\
        python scripts/bench_gemma4_spec_decode_spyre.py \\
        --prompt-file prompt.txt --max-new-tokens 256 --compare-ar

``--prompts-pt`` takes a list of token-id tensors (e.g. pre-tokenized
LongBench prompts) instead; ``--draft-vocab`` a 1-D tensor of the token ids
the drafter may propose.
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
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--drafter", default="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16")
    p.add_argument("--prompt-file", action="append", default=[])
    p.add_argument("--prompts-pt", help="torch.load-able list of 1-D token-id tensors")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--frontier", type=int, default=4)
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--draft-vocab", help="torch.load-able 1-D tensor of token ids")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--compare-ar", action="store_true")
    p.add_argument("--output", help="write the per-run results as JSON here")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3
    from hf_adapters.gemma4_spec_decode import TreeSpeculativeDecoder
    from hf_adapters.hf_common import encode_prompts

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
    target = AutoSpyreModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    drafter = hf_xpress_qwen3.load_model(
        args.drafter, target_path=args.model, dtype=torch.float32
    )
    vocab = torch.load(args.draft_vocab) if args.draft_vocab else None
    decoder = TreeSpeculativeDecoder(
        target,
        drafter,
        width=args.width,
        draft_vocab=vocab,
        frontier=args.frontier,
        branching=args.branching,
        depth=args.depth,
    )
    print(f"loaded in {time.time() - t:.0f} s", flush=True)
    env = {k: os.getenv(k) for k in ("SPYRE_MOE_UNION_CAP", "SPYRE_GEMMA4_FP8")}
    eos = tok.eos_token_id
    results = []
    for index, ids in enumerate(prompts):
        reference = None
        if args.compare_ar:
            reference = target.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )[0, ids.shape[1] :].tolist()
        for repeat in range(args.repeats + 1):
            out = decoder.generate(ids, args.max_new_tokens, eos_token_id=eos)[
                0
            ].tolist()
            s = decoder.stats
            row = dict(
                prompt=index,
                prompt_tokens=int(ids.shape[1]),
                phase="warmup" if repeat == 0 else "measured",
                generated=len(out),
                decode_tps=round(s.decode_tps(), 2),
                tokens_per_step=round(s.mean_accepted, 3),
                draft_ms=round(statistics.median(s.draft_ms), 2),
                verify_ms=round(statistics.median(s.verify_ms), 2),
                commit_ms=round(statistics.median(s.commit_ms), 2),
                prefill_ms=round(s.prefill_ms, 1),
                first_difference_vs_ar=(
                    _first_difference(out, reference) if reference is not None else None
                ),
                env=env,
                text=tok.decode(out[:64]),
            )
            print(json.dumps(row), flush=True)
            results.append(row | {"output_ids": out, "reference_ids": reference})
    measured = [r for r in results if r["phase"] == "measured"]
    print(
        json.dumps(
            dict(
                mean_decode_tps=round(
                    statistics.mean(r["decode_tps"] for r in measured), 2
                ),
                mean_tokens_per_step=round(
                    statistics.mean(r["tokens_per_step"] for r in measured), 3
                ),
                runs=len(measured),
                env=env,
            )
        ),
        flush=True,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
