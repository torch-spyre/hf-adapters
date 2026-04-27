#!/usr/bin/env python3
"""Minimal MMLU eval — works on both GPU (stock HF) and Spyre (adapter).

Each question needs only 1 token (A/B/C/D), so max_new_tokens=5 is plenty.
Runs fast even on Spyre since all generation stays within one 64-token block.

Usage:
  GPU:   python mmlu_eval.py --backend gpu --num-samples 50
  Spyre: PYTHONPATH=$PROJECT_ROOT python3 mmlu_eval.py --backend spyre --num-samples 50
"""

import argparse
import json
import re
import time
from pathlib import Path


SAMPLES_FILE = Path(__file__).parent / "mmlu_samples.json"


def load_mmlu(num_samples=50, seed=42):
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE) as f:
            data = json.load(f)
        if len(data) >= num_samples:
            return data[:num_samples]

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    choices = ["A", "B", "C", "D"]
    data = []
    for ex in ds:
        data.append({
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": choices[ex["answer"]],
            "subject": ex["subject"],
        })

    with open(SAMPLES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data[:num_samples]


def format_prompt(sample):
    q = sample["question"]
    opts = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", sample["choices"]))
    return f"Question: {q}\n{opts}\nAnswer:"


def extract_choice(text):
    text = text.strip()
    match = re.match(r"[^A-Da-d]*([A-Da-d])", text)
    if match:
        return match.group(1).upper()
    return None


def run_eval(args, backend_label, generate_fn):
    """Shared eval loop. generate_fn(prompt) -> (gen_text, gen_token_ids, elapsed)."""
    samples = load_mmlu(args.num_samples)
    print(f"Loaded {len(samples)} MMLU samples")

    results = []
    correct = 0

    for i, sample in enumerate(samples):
        prompt = format_prompt(sample)
        gen_text, gen_token_ids, elapsed = generate_fn(prompt)

        predicted = extract_choice(gen_text)
        gt = sample["answer"]
        is_correct = predicted == gt
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "subject": sample["subject"],
            "ground_truth": gt,
            "predicted": predicted,
            "correct": is_correct,
            "generated_text": gen_text,
            "generated_token_ids": gen_token_ids,
            "time_s": round(elapsed, 2),
        })

        status = "OK" if is_correct else "WRONG"
        print(f"[{i+1}/{len(samples)}] {status}  pred={predicted}  gt={gt}  ({elapsed:.1f}s)  {sample['subject']}")

    accuracy = correct / len(results) if results else 0
    metadata = {
        "backend": backend_label,
        "model": args.model,
        "num_samples": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }
    with open(args.output, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}")
    print(f"Saved to {args.output}")


def run_gpu(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"Model: {args.model} (GPU)")

    def generate_fn(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        elapsed = time.time() - t0
        gen_ids = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text, gen_ids.tolist(), elapsed

    run_eval(args, "gpu_hf", generate_fn)


def run_spyre(args):
    import torch  # noqa: F401
    import torch_spyre  # noqa: F401
    from transformers import AutoTokenizer
    from hf_adapters.hf_qwen3 import load_model, generate

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model} for Spyre ...")
    model = load_model(args.model)

    print("Warmup ...")
    t0 = time.time()
    _ = generate(model, tokenizer, ["Hello"], max_new_tokens=3)
    print(f"Warmup: {time.time()-t0:.1f}s")

    def generate_fn(prompt):
        t0 = time.time()
        outputs = generate(model, tokenizer, [prompt], max_new_tokens=5, do_sample=False)
        elapsed = time.time() - t0
        gen_text = outputs[0]
        gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        return gen_text, gen_token_ids, elapsed

    run_eval(args, "spyre_adapter", generate_fn)


def compare(file_a, file_b):
    with open(file_a) as f:
        da = json.load(f)
    with open(file_b) as f:
        db = json.load(f)

    ra = {r["index"]: r for r in da["results"]}
    rb = {r["index"]: r for r in db["results"]}
    common = sorted(set(ra.keys()) & set(rb.keys()))

    la = da["metadata"]["backend"]
    lb = db["metadata"]["backend"]

    both_ok = a_only = b_only = both_wrong = same_pred = 0
    for idx in common:
        a_ok = ra[idx]["correct"]
        b_ok = rb[idx]["correct"]
        if ra[idx]["predicted"] == rb[idx]["predicted"]:
            same_pred += 1
        if a_ok and b_ok: both_ok += 1
        elif a_ok: a_only += 1
        elif b_ok: b_only += 1
        else: both_wrong += 1

    n = len(common)
    print(f"\n{'='*60}")
    print(f"  MMLU: {la} vs {lb}  ({n} samples)")
    print(f"{'='*60}")
    print(f"  {la} accuracy: {da['metadata']['accuracy']:.1%}")
    print(f"  {lb} accuracy: {db['metadata']['accuracy']:.1%}")
    print(f"  Both correct:    {both_ok:>3} ({both_ok/n:.1%})")
    print(f"  {la} only:       {a_only:>3} ({a_only/n:.1%})")
    print(f"  {lb} only:       {b_only:>3} ({b_only/n:.1%})")
    print(f"  Both wrong:      {both_wrong:>3} ({both_wrong/n:.1%})")
    print(f"  Same prediction: {same_pred:>3} ({same_pred/n:.1%})")

    diffs = [(idx, ra[idx], rb[idx]) for idx in common if ra[idx]["predicted"] != rb[idx]["predicted"]]
    if diffs:
        print(f"\n  Prediction disagreements (first 10):")
        print(f"  {'Idx':<5} {'GT':<4} {la:<6} {lb:<6} {'Subject'}")
        print(f"  {'-'*40}")
        for idx, a, b in diffs[:10]:
            print(f"  {idx:<5} {a['ground_truth']:<4} {str(a['predicted']):<6} {str(b['predicted']):<6} {a['subject']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "spyre", "compare"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--file-a", default=None)
    parser.add_argument("--file-b", default=None)
    args = parser.parse_args()

    if args.backend == "gpu":
        if args.output is None:
            args.output = "mmlu_gpu.json"
        run_gpu(args)
    elif args.backend == "spyre":
        if args.output is None:
            args.output = "mmlu_spyre.json"
        run_spyre(args)
    elif args.backend == "compare":
        compare(args.file_a or "mmlu_gpu.json", args.file_b or "mmlu_spyre.json")
