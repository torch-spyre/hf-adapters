"""Shared utilities for GSM8K evaluation: data loading, prompts, answer extraction."""

import json
import re
from pathlib import Path


SAMPLES_FILE = Path(__file__).parent / "gsm8k_samples.json"

FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
]


def load_gsm8k(num_samples=50, seed=42):
    """Load GSM8K test samples. Saves to JSON on first call for cross-platform consistency."""
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE) as f:
            data = json.load(f)
        if len(data) >= num_samples:
            return data[:num_samples]

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    data = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

    with open(SAMPLES_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return data[:num_samples]


def format_prompt(question, tokenizer=None):
    """Format a GSM8K question with few-shot examples."""
    parts = []
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    parts.append(f"Question: {question}\nAnswer:")
    prompt = "\n\n".join(parts)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    return prompt


def _normalize(s):
    return s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")


def extract_answer(text):
    """Extract predicted numeric answer from model output."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return _normalize(match.group(1))

    match = re.search(r"[Tt]he (?:final )?answer is[:\s]*(-?[\d,.$]+)", text)
    if match:
        return _normalize(match.group(1))

    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return _normalize(numbers[-1])

    return None


def extract_ground_truth(answer_text):
    """Extract ground truth from GSM8K answer field."""
    return _normalize(answer_text.split("####")[-1])


def save_results(results, metadata, output_path):
    with open(output_path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
