"""Shared utilities for extractive-QA evaluation (Amazon QA / SQuAD 2): data loading, answer extraction, metrics, result saving."""

import json
import re
from pathlib import Path

SAMPLES_FILE = Path(__file__).parent / "amazon_qa_samples.json"
SQUAD2_SAMPLES_FILE = Path(__file__).parent / "squad2_samples.json"

# Standard Parquet dataset — no loading script, no trust_remote_code required.
# Columns: query (str), answer (str).
HF_DATASET_NAME = "sentence-transformers/amazon-qa"
HF_DATASET_CONFIG = "pair"


def load_amazon_qa(num_samples: int = 200, seed: int = 42) -> list[dict]:
    """Load Amazon QA samples from ``sentence-transformers/amazon-qa``.

    Each returned sample has keys:
      - ``question``: the product question string
      - ``context``: the answer text used as the passage (extractive target)
      - ``answer``: the ground-truth answer string (same as context)

    Saves to a local JSON cache on first call for reproducibility.
    """
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE) as f:
            data = json.load(f)
        if len(data) >= num_samples:
            return data[:num_samples]

    from datasets import load_dataset

    # sentence-transformers/amazon-qa is a plain Parquet dataset.
    # Columns: query, answer.  No loading script → no trust_remote_code.
    ds = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    data = []
    for ex in ds:
        question = (ex.get("query") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if not question or not answer:
            continue
        # Use the answer as the passage so the model can extract the span.
        data.append(
            {
                "question": question,
                "context": answer,
                "answer": answer,
            }
        )
        if len(data) >= num_samples:
            break

    with open(SAMPLES_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return data[:num_samples]


def load_squad2(
    num_samples: int = 200, seed: int = 42, answerable_only: bool = True
) -> list[dict]:
    """Load SQuAD 2.0 validation samples from ``rajpurkar/squad_v2``.

    Streams only the number of samples needed — never downloads the full 130K
    training split.  Uses the ``validation`` split (~11K examples) so results
    are directly comparable to published BERT-SQuAD2 benchmarks.

    Each returned sample has keys:
      - ``question``: the question string
      - ``context``: the Wikipedia passage (the real extractive context)
      - ``answer``: the first listed ground-truth answer span

    Args:
        num_samples:     How many examples to return (default 200).
        seed:            Shuffle seed for reproducibility (default 42).
        answerable_only: If True (default), skip examples with no answer —
                         keeps evaluation focused on span extraction quality.
                         Set to False to include unanswerable examples
                         (model should predict empty string for those).

    Saves to a local JSON cache (``squad2_samples.json``) on first call.
    """
    # Return from cache if already collected with same settings
    if SQUAD2_SAMPLES_FILE.exists():
        with open(SQUAD2_SAMPLES_FILE) as f:
            cached = json.load(f)
        # Cache stores metadata so we can detect setting mismatches
        if (
            cached.get("answerable_only") == answerable_only
            and len(cached.get("data", [])) >= num_samples
        ):
            return cached["data"][:num_samples]

    from datasets import load_dataset

    # Stream the validation split — only pulls Parquet row-groups as needed,
    # never materialises the full dataset in memory.
    ds = load_dataset(
        "rajpurkar/squad_v2",
        split="validation",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5_000)

    data = []
    for ex in ds:
        answers = ex.get("answers", {})
        answer_texts = answers.get("text", [])

        if answerable_only and not answer_texts:
            # Skip unanswerable questions when answerable_only=True
            continue

        question = ex.get("question", "").strip()
        context = ex.get("context", "").strip()
        answer = answer_texts[0].strip() if answer_texts else ""

        if not question or not context:
            continue

        data.append(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "title": ex.get("title", ""),
            }
        )
        if len(data) >= num_samples:
            break

    # Persist with metadata so cache can be invalidated on setting change
    with open(SQUAD2_SAMPLES_FILE, "w") as f:
        json.dump({"answerable_only": answerable_only, "data": data}, f, indent=2)

    return data[:num_samples]


def extract_answer_span(
    input_ids,
    start_logits,
    end_logits,
    tokenizer,
    max_answer_len: int = 50,
) -> str:
    """Decode the highest-scoring valid answer span from start/end logits."""
    import torch

    seq_len = input_ids.shape[-1]
    start_idx = int(torch.argmax(start_logits).item())
    end_idx = int(torch.argmax(end_logits).item())

    # Enforce start <= end and reasonable answer length
    if end_idx < start_idx or (end_idx - start_idx) > max_answer_len:
        # Brute-force best valid pair
        best_score = float("-inf")
        for s in range(seq_len):
            for e in range(s, min(s + max_answer_len, seq_len)):
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score = score
                    start_idx, end_idx = s, e

    token_ids = input_ids[0, start_idx : end_idx + 1]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles for EM comparison."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between predicted and ground-truth answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    num_common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def save_results(results: list[dict], metadata: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
