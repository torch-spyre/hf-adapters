#!/usr/bin/env python3
"""GSM8K evaluation on GPU using stock HuggingFace generate().

Usage:
    python eval_gsm8k_gpu.py --model Qwen/Qwen3-0.6B --num-samples 50 \
        --output results_gpu.json
"""

import argparse
import time

import torch
from gsm8k_common import (
    extract_answer,
    extract_ground_truth,
    format_prompt,
    load_gsm8k,
    save_results,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="results_gpu.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    samples = load_gsm8k(num_samples=args.num_samples, seed=args.seed)
    print(f"Loaded {len(samples)} GSM8K samples")
    print(f"Model: {args.model}")

    results = []
    correct = 0

    for i, sample in enumerate(samples):
        prompt_text = format_prompt(sample["question"], tokenizer)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            model.device
        )

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        gen_ids = output_ids[0, input_ids.shape[1] :]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        predicted = extract_answer(gen_text)
        ground_truth = extract_ground_truth(sample["answer"])
        is_correct = (predicted == ground_truth) if predicted else False
        if is_correct:
            correct += 1

        results.append(
            {
                "index": i,
                "question": sample["question"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "generated_text": gen_text,
                "generated_token_ids": gen_ids.tolist(),
                "time_s": round(elapsed, 2),
                "prompt_tokens": int(input_ids.shape[1]),
                "gen_tokens": int(len(gen_ids)),
            }
        )

        status = "OK" if is_correct else "WRONG"
        print(
            f"[{i+1}/{len(samples)}] {status}  pred={predicted}  "
            f"gt={ground_truth}  ({elapsed:.1f}s)"
        )

    accuracy = correct / len(results) if results else 0
    metadata = {
        "backend": "gpu_hf",
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
