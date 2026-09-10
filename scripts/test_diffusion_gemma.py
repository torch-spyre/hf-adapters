import argparse
import os
import tempfile
import time

import torch
from transformers import AutoProcessor, AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM

parser = argparse.ArgumentParser(
    description="Block-diffusion inference script for DiffusionGemma on Spyre.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--model", default="/models/diffusiongemma-26B-A4B-it",
                    help="HuggingFace repo ID or local model path.")
parser.add_argument("--prompt", default="Why is the sky blue?")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--max-denoising-steps", type=int, default=48,
                    help="Denoising steps per canvas (default 48). "
                         "Early stopping usually fires well before this — "
                         "lower values trade quality for speed.")
parser.add_argument("--tp", action="store_true",
                    help="Enable tensor parallelism via DistributedConfig. "
                         "Requires torchrun --nproc_per_node >= 2.")
parser.add_argument("--no-warmup", action="store_true",
                    help="Skip warmup run (use when inductor cache is already warm).")
parser.add_argument("--batch-size", type=int, default=1,
                    help="Number of identical prompts to batch together (default 1).")
parser.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"],
                    help="Model dtype (default bfloat16).")
args = parser.parse_args()

# Give each rank its own inductor cache to avoid bundle-path collisions.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank_cache = os.path.join(tempfile.gettempdir(), f"torchinductor_rank{local_rank}")
os.makedirs(rank_cache, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", rank_cache)

_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
model = AutoSpyreModelForCausalLM.from_pretrained(
    args.model,
    dtype=_dtype,
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
enc = tokenizer([prompt_text] * args.batch_size, return_tensors="pt", padding=True)
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
# Count non-pad tokens: output tensor is zero-padded to uniform length, so
# nonzero entries are actual generated token ids (including EOS).
if local_rank == 0:
    from hf_adapters.hf_common import text_config
    num_layers = text_config(model.config).num_hidden_layers
    for b in range(args.batch_size):
        output_text = tokenizer.decode(output_ids[b], skip_special_tokens=True)
        output_token_count = (output_ids[b] != 0).sum().item()
        toks_per_sec = output_token_count / elapsed
        print(f"=== batch item {b} ===")
        print(output_text)
        print()
        print(f"--- throughput (batch item {b}) ---")
        print(f"  batch_size          : {args.batch_size}")
        print(f"  max_denoising_steps : {args.max_denoising_steps}")
        print(f"  generated tokens    : {output_token_count}")
        print(f"  wall time           : {elapsed:.2f}s")
        print(f"  throughput          : {toks_per_sec:.1f} tok/s")
        print()
    print("NOTE: throughput is dominated by Spyre<->CPU MoE round-trips.")
    print(f"  {args.max_denoising_steps} steps x {num_layers} layers x 2 transfers/layer")
    print(f"  = {args.max_denoising_steps * num_layers * 2} PCIe transfers per canvas.")
    print("  Reduce --max-denoising-steps to trade quality for speed.")
