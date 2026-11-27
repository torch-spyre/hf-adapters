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

"""Run command of a quality case: record the score that the repository's own eval script produces.

This file measures nothing. `evals/gsm8k/eval_gsm8k_spyre.py` and `evals/mmlu/mmlu_eval.py` already
load the model through the adapter, warm the graphs up, score the vendored question file in order and
write `{"metadata": {...}, "results": [...]}`. Scoring the questions a second time here would mean a
second implementation of the same eval -- a different model-loading call, a different generate path,
no warmup -- and the number the catalog filed would stop being the number evals/ reports. So the
script is run as-is and only its result is translated:

  * `metrics.json` in --repeat-dir -- the score, which the engine reads.
  * `<case>.json` in --raw-dir -- the same score as a row for the generic benchmarks ingest.
  * `answers.json` in --repeat-dir -- the script's own per-answer output, kept out of the result.

--limit is the question count the Benchmark declares. It is passed to the script AND checked against
what came back, so a case cannot quietly record a different number of questions than it promised
(which would change its identity and restart its series).

    run_eval_case.py --task gsm8k|mmlu --case NAME --model PATH --limit N
                     --repeat-dir DIR --raw-dir DIR
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = "hf-adapters-evals"
QUIET = "BENCH_QUIET"

# The eval each task delegates to, and the identity inputs that go with it. `num_fewshot` describes
# the prompt the script builds (gsm8k prepends two worked examples, mmlu is zero-shot); the
# Benchmark's identity block must list the same value, which test_run_eval_case.py checks.
TASKS = {
    "gsm8k": {
        "dir": "gsm8k",
        "script": "eval_gsm8k_spyre.py",
        "samples": "gsm8k_samples.json",
        "num_fewshot": 2,
        "args": lambda model, limit, out: [
            "--model", model,
            "--num-samples", str(limit),
            "--max-new-tokens", "512",
            "--output", out,
        ],  # fmt: skip
    },
    "mmlu": {
        "dir": "mmlu",
        "script": "mmlu_eval.py",
        "samples": "mmlu_samples.json",
        "num_fewshot": 0,
        "args": lambda model, limit, out: [
            "--backend", "spyre",
            "--model", model,
            "--num-samples", str(limit),
            "--output", out,
        ],  # fmt: skip
    },
}


def samples_digest(task: str) -> str:
    """The digest of the vendored question file, recorded so an edited file is visible in the row."""
    path = ROOT / "evals" / TASKS[task]["dir"] / TASKS[task]["samples"]
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


def script_argv(task: str, model: str, limit: int, out: Path) -> list[str]:
    """The command line of the repository's own eval script for this case."""
    spec = TASKS[task]
    script = ROOT / "evals" / spec["dir"] / spec["script"]
    return [sys.executable, str(script), *spec["args"](model, limit, str(out))]


class EvalError(RuntimeError):
    """The eval script failed, or wrote no usable result. Carries a message, not a traceback."""


def run_script(task: str, model: str, limit: int, out: Path) -> dict:
    """Run the eval script and return its metadata block."""
    env = dict(os.environ)
    # The scripts import hf_adapters; their own directory is already sys.path[0].
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH", "")])
    )
    argv = script_argv(task, model, limit, out)
    done = subprocess.run(argv, env=env, check=False)
    if done.returncode:
        raise EvalError(f"{Path(argv[1]).name} exited {done.returncode}")
    try:
        return json.loads(out.read_text())["metadata"]
    except (OSError, ValueError, KeyError) as e:
        raise EvalError(
            f"{Path(argv[1]).name} wrote no usable result to {out} ({type(e).__name__})"
        ) from None


def build(case: str, task: str, model: str, correct: int, total: int, digest: str):
    """(metrics.json document, benchmarks file document) for one case."""
    acc = correct / total if total else 0.0
    samples = {"llm.eval.exact_match": [acc]}
    props = {"run.quality.harness": HARNESS, "run.quality.dataset_digest": digest}
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


def main(argv=None, run=run_script) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--limit",
        required=True,
        type=int,
        help="the question count the Benchmark declares; the result must match it",
    )
    ap.add_argument("--repeat-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.limit < 1:
        ap.error("--limit must be at least 1")
    args.repeat_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    answers = args.repeat_dir / "answers.json"
    try:
        meta = run(args.task, args.model, args.limit, answers)
    except EvalError as e:
        print(f"{args.case}: {e}", file=sys.stderr)
        return 1
    total, correct = int(meta["num_samples"]), int(meta["correct"])
    if total != args.limit:
        print(
            f"{args.case}: the eval scored {total} questions but the benchmark declares "
            f"{args.limit}; the recorded identity would not be the declared one",
            file=sys.stderr,
        )
        return 3

    metrics, bench = build(
        args.case, args.task, args.model, correct, total, samples_digest(args.task)
    )
    (args.repeat_dir / "metrics.json").write_text(json.dumps(metrics))
    (args.raw_dir / f"{args.case}.json").write_text(json.dumps(bench))
    shown = "" if os.environ.get(QUIET, "1") != "0" else f": {correct}/{total}"
    print(f"recorded {args.case}{shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
