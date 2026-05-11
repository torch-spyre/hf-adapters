#!/usr/bin/env python3
"""GSM8K evaluation on Spyre using hf_adapters generate().

Usage (on Spyre pod):
    PYTHONPATH=$PROJECT_ROOT python3 eval_gsm8k_spyre.py \
        --model Qwen/Qwen3-0.6B --num-samples 50 --output results_spyre.json
"""

import argparse
import time

import torch_spyre  # noqa: F401
from gsm8k_common import (
    extract_answer,
    extract_ground_truth,
    format_prompt,
    load_gsm8k,
    save_results,
)
from transformers import AutoTokenizer

from hf_adapters.hf_qwen3 import generate, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="results_spyre.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model} for Spyre ...")
    t0 = time.time()
    model = load_model(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    samples = load_gsm8k(num_samples=args.num_samples, seed=args.seed)
    print(f"Loaded {len(samples)} GSM8K samples")

    # Warmup: compile the block graphs
    print("Warmup (compiling graphs) ...")
    t0 = time.time()
    _ = generate(model, tokenizer, ["What is 1+1?"], max_new_tokens=10)
    print(f"Warmup done in {time.time() - t0:.1f}s")

    results = []
    correct = 0

    for i, sample in enumerate(samples):
        prompt_text = format_prompt(sample["question"], tokenizer)

        t0 = time.time()
        outputs = generate(
            model,
            tokenizer,
            [prompt_text],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.time() - t0

        gen_text = outputs[0]

        predicted = extract_answer(gen_text)
        ground_truth = extract_ground_truth(sample["answer"])
        is_correct = (predicted == ground_truth) if predicted else False
        if is_correct:
            correct += 1

        gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)

        results.append(
            {
                "index": i,
                "question": sample["question"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "generated_text": gen_text,
                "generated_token_ids": gen_token_ids,
                "time_s": round(elapsed, 2),
                "prompt_tokens": len(tokenizer.encode(prompt_text)),
                "gen_tokens": len(gen_token_ids),
            }
        )

        status = "OK" if is_correct else "WRONG"
        print(
            f"[{i+1}/{len(samples)}] {status}  pred={predicted}  "
            f"gt={ground_truth}  ({elapsed:.1f}s)"
        )

    accuracy = correct / len(results) if results else 0
    metadata = {
        "backend": "spyre_adapter",
        "model": args.model,
        "num_samples": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }

    save_results(results, metadata, args.output)
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
