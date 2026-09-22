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

"""Profile an end-to-end generation loop on Spyre.

Runs the same load + generate path as the e2e smoke test twice:

  1. A warmup run (no profiler) that triggers torch.compile so the Spyre
     Inductor backend codegens and caches every kernel the workload hits.
  2. A profiled run under torch.profiler with CPU + PrivateUse1 (Spyre-side)
     activities. Because the cache is warm, this measures execution, not
     compile time. The trace is written to a single Chrome-trace JSON.

Usage (on the Spyre pod, with the project root on PYTHONPATH)::

    python scripts/profile_e2e_spyre.py --model ministral3 \\
        --hf-home /mnt/models/hf_cache

To reproduce the Granite batch-4, 1K aggregate prompt-token workload used by
the shape worker (4 rows x 256 tokens, 512 total tokens per row)::

    SPYRE_DEVICES=0 python scripts/profile_e2e_spyre.py --model granite8b \\
        --hf-home /mnt/models/hf_cache --batch 4 --prompt-tokens 256 \\
        --max-length 512

The HF hub cache lives at ``<HF_HOME>/hub``; pass ``--hf-home`` (or set the
``HF_HOME`` env var) to point at a shared cache instead of the default.

For tensor parallelism, launch one process per card. Each rank writes its own
trace so concurrent profilers do not overwrite one another::

    SPYRE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 \\
        scripts/profile_e2e_spyre.py --model gemma4_26b_a4b \\
        --hf-home /mnt/models/hf_cache

Pass ``--with-stack`` to annotate each trace event with its Python source
stack and module hierarchy, so an op (e.g. the per-layer ``torch.full``) can
be traced back to its origin in the viewer. It is off by default because it
adds per-event overhead and enlarges the trace.

Open the resulting trace in https://ui.perfetto.dev/ or chrome://tracing.
"""

import argparse
import os
import sys
import time


def _apply_hf_home(argv: "list[str] | None") -> None:
    """Set HF_HOME before any HF import so the hub cache path takes effect.

    ``huggingface_hub`` computes ``HF_HUB_CACHE`` (``<HF_HOME>/hub``) at import
    time, so the env var must be set before ``transformers``/``hf_adapters``
    are imported below. We peek at ``--hf-home`` here rather than in the main
    argparse pass (which runs after those imports). Precedence: explicit
    ``--hf-home`` > an ``HF_HOME`` already in the environment > unset (HF's
    own default, typically ``~/.cache/huggingface``).
    """
    argv = sys.argv[1:] if argv is None else argv
    hf_home = None
    for i, tok in enumerate(argv):
        if tok == "--hf-home" and i + 1 < len(argv):
            hf_home = argv[i + 1]
        elif tok.startswith("--hf-home="):
            hf_home = tok.split("=", 1)[1]
    if hf_home:
        os.environ["HF_HOME"] = hf_home


# Bootstrap the cache path from --hf-home / HF_HOME before the HF imports run.
_apply_hf_home(None)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hf_adapters import AutoSpyreModelForCausalLM  # noqa: E402

# Registry key -> (HF path, Spyre-safe dtype). Kept inline so this script has
# no dependency on the tests/ package. These match the checkpoints' configured
# dtypes resolved by hf_adapters.auto_spyre_model.dtype_for_model_path.
MODELS: dict[str, tuple[str, "torch.dtype"]] = {
    "gemma4_26b_a4b": ("google/gemma-4-26B-A4B-it", torch.float16),
    "ministral3": ("mistralai/Ministral-3-14B-Instruct-2512", torch.bfloat16),
    "granite8b": ("ibm-granite/granite-3.3-8b-instruct", torch.float16),
}

DEFAULT_PROMPT = "The capital of France is"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ministral3",
        choices=sorted(MODELS),
        help="Registry key to profile (default: ministral3).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Chrome-trace output path (default: <model>_trace.json).",
    )
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="New-token budget (default: 5, matching the smoke test).",
    )
    length_group.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Total prompt + generated token budget (mutually exclusive with --max-new-tokens).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to generate from (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Force each input row to exactly N tokens by repeating/truncating "
            "the tokenized prompt (default: use its natural length)."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Number of identical prompt rows (default: 1).",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME"),
        help=(
            "HF_HOME dir; the hub cache is <HF_HOME>/hub (e.g. "
            "/mnt/models/hf_cache -> /mnt/models/hf_cache/hub). Applied before "
            "the HF imports. Defaults to $HF_HOME, else HF's own default."
        ),
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help=(
            "Record the Python source stack (file:line) and module hierarchy "
            "for each op so you can trace an event (e.g. the per-layer "
            "torch.full) back to its origin in the trace viewer. Off by "
            "default: it adds per-event overhead and enlarges the trace."
        ),
    )
    return parser


def build_inputs(
    tokenizer,
    prompt: str,
    batch_size: int,
    prompt_tokens: int | None,
) -> dict[str, "torch.Tensor"]:
    """Tokenize *prompt* and return an exact, uniformly sized input batch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prompt_tokens is not None and prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if prompt_tokens is not None:
        if input_ids.shape[1] < prompt_tokens:
            # Repeat ordinary prompt tokens rather than inserting padding: the
            # shape worker's seqlen represents real prompt tokens in every row.
            seed = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            if seed.numel() == 0:
                seed = input_ids
            missing = prompt_tokens - input_ids.shape[1]
            repeats = (missing + seed.shape[1] - 1) // seed.shape[1]
            input_ids = torch.cat((input_ids, seed.repeat(1, repeats)), dim=1)
        input_ids = input_ids[:, :prompt_tokens]

    input_ids = input_ids.repeat(batch_size, 1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def run_profile(
    model_path: str,
    dtype: "torch.dtype",
    prompt: str,
    max_new_tokens: int | None,
    max_length: int | None,
    batch_size: int,
    prompt_tokens: int | None,
    out_path: str,
    with_stack: bool = False,
) -> None:
    """Load *model_path*, warm the compile cache, then profile one generate."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp_plan = "auto" if world_size > 1 else None
    if world_size > 1:
        stem, suffix = os.path.splitext(out_path)
        out_path = f"{stem}.rank{rank}{suffix or '.json'}"

    print(f"{'=' * 70}")
    print(f"  profiling {model_path}  (dtype={dtype}, rank={rank}/{world_size})")
    print(f"{'=' * 70}")

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, tp_plan=tp_plan
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Load time: {time.time() - t0:.1f}s")
    print(f"  Prompt seed: {prompt!r}")

    encoded = build_inputs(tokenizer, prompt, batch_size, prompt_tokens)
    actual_prompt_len = encoded["input_ids"].shape[1]
    if max_length is not None and max_length <= actual_prompt_len:
        raise ValueError(
            f"max_length ({max_length}) must exceed prompt length ({actual_prompt_len})"
        )
    print(
        f"  Input shape: {list(encoded['input_ids'].shape)} "
        f"({encoded['input_ids'].numel()} aggregate prompt tokens)"
    )
    gen_kwargs = dict(
        **encoded,
        do_sample=False,
        timing=True,
    )
    if max_length is not None:
        gen_kwargs["max_length"] = max_length
    else:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    # --- Warmup: first generate triggers torch.compile; NOT profiled. This
    #     is what "warms the compiler cache" so the profiled run below measures
    #     steady-state execution rather than compile time.
    print("\n[warmup] compiling (this run is not profiled)...")
    model.generate(**gen_kwargs)

    # --- Profiled run: cache is warm. CPU + PrivateUse1 (Spyre device) activity.
    #     NOTE(#114): do NOT call prof.events() / prof.key_averages().table() —
    #     reading the event buffer hits the libaiupti/kineto ABI decode crash.
    #     export_chrome_trace() writes the trace without walking that buffer.
    print(f"[profile] capturing trace... (with_stack={with_stack})")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        # --with-stack: attach each event's Python source stack (file:line)
        # and module hierarchy, so an op can be traced back to its origin
        # (e.g. the per-layer torch.full -> torch_spyre's SDPA decomposition).
        # Both are gated because they add per-event overhead and grow the trace.
        with_stack=with_stack,
        with_modules=with_stack,
    ) as prof:
        outputs = model.generate(**gen_kwargs)

    prof.export_chrome_trace(out_path)

    output_texts = tokenizer.batch_decode(
        outputs[:, actual_prompt_len:], skip_special_tokens=True
    )
    print(f"\n  Output shape: {list(outputs.shape)}")
    for i, output_text in enumerate(output_texts):
        print(f"  Output[{i}]: {output_text!r}")
    print(f"  trace → {out_path}")
    print("  open in https://ui.perfetto.dev/ or chrome://tracing")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens is None and args.max_length is None:
        args.max_new_tokens = 5
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.prompt_tokens is not None and args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.hf_home:
        # Normally already applied by _apply_hf_home() at import time; re-set
        # for the programmatic main(argv=[...]) path where import ran first.
        os.environ["HF_HOME"] = args.hf_home
    from huggingface_hub import constants

    print(f"  HF hub cache: {constants.HF_HUB_CACHE}")
    path, dtype = MODELS[args.model]
    out = args.out or f"{args.model}_trace.json"
    run_profile(
        model_path=path,
        dtype=dtype,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        batch_size=args.batch,
        prompt_tokens=args.prompt_tokens,
        out_path=out,
        with_stack=args.with_stack,
    )


if __name__ == "__main__":
    main()
