#!/usr/bin/env python3
"""Evaluate deepset/bert-base-cased-squad2 (or any extractive-QA model).

Two backends are supported:
  gpu    — stock HuggingFace AutoModelForQuestionAnswering on CPU/GPU
  spyre  — Spyre encoder via AutoSpyreModelForQuestionAnswering + CPU head

Two datasets are supported via --dataset:
  amazon_qa — sentence-transformers/amazon-qa  (conversational product Q&A)
  squad2    — rajpurkar/squad_v2 validation    (the model's native benchmark;
              streams only the samples needed, answerable examples only by default)

Usage:
  # SQuAD 2 on GPU (matches published BERT-SQuAD2 numbers):
  python eval_f1_em_qa.py --model deepset/bert-base-cased-squad2 --dataset squad2 --backend gpu --num-samples 200 \
      --output results_squad2_gpu.json

  # SQuAD 2 on Spyre:
  python eval_f1_em_qa.py --model deepset/bert-base-cased-squad2 --dataset squad2 --backend spyre --num-samples 200 --output results_squad2_spyre.json

  # Amazon QA:
  python eval_f1_em_qa.py --model deepset/bert-base-cased-squad2 --dataset amazon_qa --backend gpu --num-samples 200 \
      --output results_amazon_qa_gpu.json
"""

import argparse
import sys
import time

import torch
from eval_f1_em_qa_common import (
    compute_f1,
    extract_answer_span,
    load_amazon_qa,
    load_squad2,
    normalize_answer,
    save_results,
)
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Default model — deepset/bert-base-cased-squad2 is the SQuAD 2-fine-tuned
# BERT model registered in tests/model_registry.py as "bert_qa".
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "deepset/bert-base-cased-squad2"


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def _load_gpu(model_path: str):
    """Load the model with stock HuggingFace on CPU/GPU."""
    from transformers import AutoModelForQuestionAnswering

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    ).eval()
    return model, device


def _load_spyre(model_path: str):
    """Load the model with the Spyre adapter."""
    import torch_spyre  # noqa: F401

    from hf_adapters import AutoSpyreModelForQuestionAnswering

    model = AutoSpyreModelForQuestionAnswering.from_pretrained(model_path)
    return model, "spyre"


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------


def _run_inference(model, tokenizer, question: str, context: str, device: str):
    """Tokenize one (question, context) pair and return (start_logits, end_logits, encoding)."""
    encoding = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    if device not in ("spyre", "cpu"):
        encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding, return_dict=True)

    return outputs.start_logits.cpu(), outputs.end_logits.cpu(), encoding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an extractive-QA model on Amazon QA or SQuAD 2."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon_qa", "squad2"],
        default="amazon_qa",
        help=(
            "Dataset to evaluate on: 'amazon_qa' (sentence-transformers/amazon-qa) or "
            "'squad2' (rajpurkar/squad_v2 validation, answerable examples only). "
            "Use 'squad2' to reproduce published BERT-SQuAD2 benchmark numbers. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["gpu", "spyre"],
        default="gpu",
        help="Inference backend: 'gpu' uses stock HF, 'spyre' uses the Spyre adapter.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of Amazon QA samples to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--max-answer-len",
        type=int,
        default=50,
        help="Maximum answer span length in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output JSON file path (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling (default: %(default)s)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model ({args.backend} backend) ...")
    t0 = time.time()
    if args.backend == "spyre":
        model, device = _load_spyre(args.model)
    else:
        model, device = _load_gpu(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s  [device={device}]")

    # ------------------------------------------------------------------
    # Spyre warmup
    # ------------------------------------------------------------------
    if args.backend == "spyre":
        print("Warmup (compiling graphs) ...")
        t0 = time.time()
        _run_inference(
            model, tokenizer, "What is the color?", "The sky is blue.", device
        )
        print(f"Warmup done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if args.dataset == "squad2":
        print(
            f"Loading {args.num_samples} SQuAD 2 validation samples "
            f"(answerable only, seed={args.seed}) ..."
        )
        samples = load_squad2(
            num_samples=args.num_samples,
            seed=args.seed,
            answerable_only=True,
        )
    else:
        print(f"Loading {args.num_samples} Amazon QA samples (seed={args.seed}) ...")
        samples = load_amazon_qa(num_samples=args.num_samples, seed=args.seed)
    print(f"Loaded {len(samples)} samples from '{args.dataset}'")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    results = []
    total_f1 = 0.0
    exact_match_count = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        context = sample["context"]
        ground_truth = sample["answer"]

        t0 = time.time()
        try:
            start_logits, end_logits, encoding = _run_inference(
                model, tokenizer, question, context, device
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i+1}/{len(samples)}] ERROR: {exc}", file=sys.stderr)
            results.append(
                {
                    "index": i,
                    "question": question,
                    "context": context,
                    "ground_truth": ground_truth,
                    "predicted": "",
                    "exact_match": False,
                    "f1": 0.0,
                    "error": str(exc),
                    "time_s": 0.0,
                }
            )
            continue
        elapsed = time.time() - t0

        # Decode answer span from token indices
        input_ids_cpu = encoding["input_ids"].cpu()
        predicted = extract_answer_span(
            input_ids_cpu,
            start_logits[0],
            end_logits[0],
            tokenizer,
            max_answer_len=args.max_answer_len,
        )

        em = normalize_answer(predicted) == normalize_answer(ground_truth)
        f1 = compute_f1(predicted, ground_truth)

        if em:
            exact_match_count += 1
        total_f1 += f1

        results.append(
            {
                "index": i,
                "question": question,
                "context": context,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": em,
                "f1": round(f1, 4),
                "time_s": round(elapsed, 3),
            }
        )

        status = "EM" if em else f"F1={f1:.2f}"
        print(
            f"[{i+1:>{len(str(len(samples)))}}/{len(samples)}] {status:10s}  "
            f"pred={predicted!r:.50s}  ({elapsed:.2f}s)"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(results)
    em_score = exact_match_count / n if n else 0.0
    avg_f1 = total_f1 / n if n else 0.0

    metadata = {
        "backend": args.backend,
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": n,
        "exact_match": round(em_score, 4),
        "f1": round(avg_f1, 4),
        "seed": args.seed,
        "max_answer_len": args.max_answer_len,
    }

    save_results(results, metadata, args.output)

    print("\n" + "=" * 60)
    print(f"  Model         : {args.model}")
    print(f"  Backend       : {args.backend}")
    print(f"  Samples       : {n}")
    print(f"  Exact Match   : {exact_match_count}/{n} = {em_score:.1%}")
    print(f"  Token-level F1: {avg_f1:.4f}")
    print(f"  Results saved : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
