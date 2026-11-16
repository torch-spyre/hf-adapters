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

"""Run command of a quality case: score the evals/ question samples on the Spyre adapter and record the result.

For the engine it writes `metrics.json` (the score) into --repeat-dir; for the generic benchmarks ingest it writes
`<case>.json` into --raw-dir. The questions, prompts and answer extraction are the ones evals/ already uses
(greedy decoding); the model is loaded with the current public API (`AutoSpyreModelForCausalLM`). Nothing here
judges or gates. Per-answer text stays in --repeat-dir/answers.json and is not part of the recorded result.

    run_eval_case.py --task slot-12|slot-14 --case NAME --model PATH --repeat-dir DIR --raw-dir DIR
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = "hf-adapters-evals"
QUIET = "BENCH_QUIET"
# Identity inputs of each task; the Benchmark's identity block must list the same values.
TASKS = {
    "slot-12": {
        "scorer": "numeric",
        "dir": "gsm8k",
        "samples": "gsm8k_samples.json",
        "module": "gsm8k_common",
        "num_fewshot": 2,
        "max_new_tokens": 512,
    },
    "slot-14": {
        "scorer": "choice",
        "dir": "mmlu",
        "samples": "mmlu_samples.json",
        "module": "mmlu_eval",
        "num_fewshot": 0,
        "max_new_tokens": 5,
    },
}


def identity_props(task: str, model: str, num_questions: int) -> dict[str, str]:
    """The identity of the case as the generic ingest hashes it: every prop is a discriminator."""
    return {
        "dataset": task,
        "harness": HARNESS,
        "model": model,
        "num_fewshot": str(TASKS[task]["num_fewshot"]),
        "num_questions": str(num_questions),
        "record_type": "quality",
        "task": task,
    }


def stderr_binomial(p: float, n: int) -> float:
    return (max(p * (1 - p), 0.0) / n) ** 0.5 if n else 0.0


def build(
    case: str, task: str, model: str, correct: int, total: int, samples_digest: str
):
    """(metrics.json document, benchmarks file document) for one case."""
    acc = correct / total if total else 0.0
    samples = {"llm.eval.exact_match": [acc]}
    props = {
        "run.quality.harness": HARNESS,
        "run.quality.dataset_digest": samples_digest,
    }
    row = {
        "name": case,
        "tags": [],
        "backend": "spyre",
        "props": identity_props(task, model, total),
        "measurements": samples,
        "iterations": total,
        "run_props": props,
    }
    return {"samples": samples, "props": props}, {"benchmarks": [row]}


def load_task(task: str):
    """(helper module, samples, digest) of the vendored question file, scored in file order."""
    directory = ROOT / "evals" / TASKS[task]["dir"]
    sys.path.insert(0, str(directory))
    mod = importlib.import_module(TASKS[task]["module"])
    path = directory / TASKS[task]["samples"]
    return (
        mod,
        json.loads(path.read_text()),
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def spyre_generate(model_path: str, max_new_tokens: int):
    """generate(prompt_text) -> text, greedy, on the Spyre adapter."""
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path)

    def generate(prompt: str) -> str:
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

    generate.tokenizer = tokenizer
    return generate


def score(task: str, mod, samples: list, generate) -> tuple[int, list]:
    """(correct, answers) over the samples."""
    correct, answers = 0, []
    for i, sample in enumerate(samples):
        if TASKS[task]["scorer"] == "numeric":
            text = generate(
                mod.format_prompt(
                    sample["question"], getattr(generate, "tokenizer", None)
                )
            )
            predicted, truth = mod.extract_answer(text), mod.extract_ground_truth(
                sample["answer"]
            )
            ok = predicted is not None and predicted == truth
        else:
            text = generate(mod.format_prompt(sample))
            predicted, truth = mod.extract_choice(text), sample["answer"]
            ok = predicted == truth
        correct += bool(ok)
        answers.append(
            {
                "index": i,
                "predicted": predicted,
                "correct": bool(ok),
                "generated_text": text,
            }
        )
    return correct, answers


def main(argv=None, generate=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    mod, samples, digest = load_task(args.task)
    generate = generate or spyre_generate(
        args.model, TASKS[args.task]["max_new_tokens"]
    )
    correct, answers = score(args.task, mod, samples, generate)
    metrics, bench = build(
        args.case, args.task, args.model, correct, len(samples), digest
    )
    args.repeat_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    (args.repeat_dir / "metrics.json").write_text(json.dumps(metrics))
    (args.repeat_dir / "answers.json").write_text(json.dumps(answers))
    (args.raw_dir / f"{args.case}.json").write_text(json.dumps(bench))
    shown = "" if os.environ.get(QUIET, "1") != "0" else f": {correct}/{len(samples)}"
    print(f"recorded {args.case}{shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
