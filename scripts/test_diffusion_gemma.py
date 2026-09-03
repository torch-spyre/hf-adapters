import argparse
import os
import tempfile
import time

import torch
from transformers import AutoProcessor, AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/models/diffusiongemma-26B-A4B-it")
parser.add_argument("--prompt", default="Why is the sky blue?")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--max-denoising-steps", type=int, default=48,
                    help="Denoising steps per canvas (default 48). "
                         "Early stopping usually fires well before this — "
                         "lower values trade quality for speed.")
parser.add_argument("--tp", action="store_true")
parser.add_argument("--no-warmup", action="store_true",
                    help="Skip warmup run (use when inductor cache is already warm).")
args = parser.parse_args()

# TP: give each rank its own inductor cache to avoid bundle-path collisions.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank_cache = os.path.join(tempfile.gettempdir(), f"torchinductor_rank{local_rank}")
os.makedirs(rank_cache, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", rank_cache)

model = AutoSpyreModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    tp_plan="auto" if args.tp else None,
)
try:
    tokenizer = AutoProcessor.from_pretrained(args.model).tokenizer
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

# Apply chat template if available (instruct models require it), then tokenize.
# Reuse the same tensors for warmup and timed run.
if getattr(tokenizer, "chat_template", None) is not None:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    prompt_text = args.prompt
enc = tokenizer([prompt_text], return_tensors="pt", padding=True)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
gen_kwargs = dict(
    max_new_tokens=args.max_new_tokens,
    max_denoising_steps=args.max_denoising_steps,
)

# Warmup: first call compiles all torch.compile graphs (encoder + decoder
# blocks). Must be excluded from the throughput measurement.
if not args.no_warmup:
    if local_rank == 0:
        print(f"Warming up (compiling graphs, max_denoising_steps={args.max_denoising_steps})...", flush=True)
    model.generate(input_ids, attention_mask, **gen_kwargs)
    if local_rank == 0:
        print("Warmup done.", flush=True)

# Timed run — all graphs already compiled.
if local_rank == 0:
    print("Running timed generate...", flush=True)
t0 = time.perf_counter()
output_ids = model.generate(input_ids, attention_mask, **gen_kwargs)
elapsed = time.perf_counter() - t0

# generate() returns only the generated tokens (prompt already stripped).
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# Count non-pad tokens: output tensor is zero-padded to uniform length, so
# nonzero entries are actual generated token ids (including EOS).
output_token_count = (output_ids[0] != 0).sum().item()
toks_per_sec = output_token_count / elapsed

# Only rank 0 prints to avoid duplicate lines under TP.
if local_rank == 0:
    print(output_text)
    print()
    print(f"--- throughput ---")
    print(f"  max_denoising_steps : {args.max_denoising_steps}")
    print(f"  generated tokens    : {output_token_count}")
    print(f"  wall time           : {elapsed:.2f}s")
    print(f"  throughput          : {toks_per_sec:.1f} tok/s")
    print()
    print("NOTE: throughput is dominated by Spyre↔CPU MoE round-trips.")
    print(f"  {args.max_denoising_steps} steps × 30 layers × 2 transfers/layer")
    print(f"  = {args.max_denoising_steps * 30 * 2} PCIe transfers per canvas.")
    print("  Reduce --max-denoising-steps to trade quality for speed.")
