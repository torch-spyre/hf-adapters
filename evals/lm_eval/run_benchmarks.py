"""Run a configured lm-eval suite on Spyre or native Transformers.

The supervisor imports neither torch nor hf_adapters. Each benchmark runs in a
fresh process.
"""

from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evals.lm_eval.dashboard import dashboard_results, metric_iterations, sample_outputs

ROOT = Path(__file__).resolve().parents[2]
# Validation rules, not defaults: values come only from the YAML/JSON config.
BENCHMARK_SETTINGS = {
    "limit": lambda v: type(v) is int and v > 0,
    "num_fewshot": lambda v: type(v) is int and v >= 0,
    "max_length": lambda v: type(v) is int and v > 0,
    "batch_size": lambda v: type(v) is int and v > 0,
    "dtype": lambda v: v in ("bfloat16", "float16", "float32", "auto"),
    "apply_chat_template": lambda v: type(v) is bool,
    "fewshot_as_multiturn": lambda v: type(v) is bool,
    "seed": lambda v: type(v) is int and v >= 0,
    "timeout_seconds": lambda v: type(v) is int and v > 0,
    "generation": lambda v: v is None or isinstance(v, dict),
}
RESULT_SCHEMA_VERSION = 1
PROTOCOL_PACKAGES = (
    "lm_eval",
    "transformers",
    "datasets",
    "evaluate",
    "nltk",
    "langdetect",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def model_key(model):
    """Filesystem/artifact-safe name without org/name collisions."""
    return re.sub(r"[^A-Za-z0-9_.-]", "--", model)[:100] + "-" + digest(model)[:10]


def write_json(path, value):
    def serialize(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        # Harness metadata includes torch.dtype and serialized task callables.
        return str(item)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=serialize, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def check_keys(value, allowed, context, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {context} fields: {sorted(unknown)}")
    missing = set(required) - value.keys()
    if missing:
        raise ValueError(f"{context}: missing required settings: {sorted(missing)}")


def load_suite(path):
    suite = yaml.safe_load(Path(path).read_text())
    check_keys(suite, {"schema_version", "name", "defaults", "benchmarks"}, "suite")
    if suite.get("schema_version") != 1 or not isinstance(suite.get("name"), str):
        raise ValueError("Suite requires schema_version: 1 and a name")
    defaults = suite.get("defaults", {})
    check_keys(defaults, BENCHMARK_SETTINGS, "defaults")
    entries = suite.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmarks must be a nonempty list")
    resolved, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each benchmark must be a mapping")
        spec = {**defaults, **entry}
        name = spec.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Each benchmark needs a filesystem-safe id")
        if name in seen:
            raise ValueError(f"Duplicate benchmark id: {name}")
        seen.add(name)
        check_keys(
            spec,
            BENCHMARK_SETTINGS.keys() | {"id", "tasks"},
            name,
            required=BENCHMARK_SETTINGS,
        )
        for key, valid in BENCHMARK_SETTINGS.items():
            if not valid(spec[key]):
                raise ValueError(f"{name}: invalid {key}: {spec[key]!r}")
        tasks = spec.get("tasks")
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(
                not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", t)
                for t in tasks
            )
            or len(set(tasks)) != len(tasks)
        ):
            raise ValueError(f"{name}: tasks must be unique, literal task/group IDs")
        if spec["fewshot_as_multiturn"] and not spec["apply_chat_template"]:
            raise ValueError(
                f"{name}: fewshot_as_multiturn requires apply_chat_template"
            )
        if spec["generation"] is not None:
            generation = spec["generation"]
            check_keys(
                generation,
                {"max_gen_toks", "do_sample", "until"},
                f"{name} generation",
                required={"max_gen_toks", "do_sample"},
            )
            budget = generation["max_gen_toks"]
            if type(budget) is not int or not 0 < budget < spec["max_length"]:
                raise ValueError(
                    f"{name}: max_gen_toks must be positive and below max_length"
                )
            if generation["do_sample"] is not False:
                raise ValueError(
                    f"{name}: periodic comparisons require greedy generation"
                )
            if "until" in generation and (
                not isinstance(generation["until"], list)
                or any(not isinstance(s, str) for s in generation["until"])
            ):
                raise ValueError(f"{name}: until must be a list of stop strings")
        resolved.append(spec)
    return {"schema_version": 1, "name": suite["name"], "benchmarks": resolved}


def load_models(path, group):
    data = yaml.safe_load(Path(path).read_text())
    models = data.get(group, {}).get("models")
    return validate_models(models)


def validate_models(models):
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(m, str) or not m.strip() for m in models)
    ):
        raise ValueError("Expected a nonempty list of model paths")
    if len(set(models)) != len(models):
        raise ValueError("Model paths must be unique")
    return models


def select_sample_indices(task_name, num_docs, spec):
    """Choose a fixed subset independently of model, backend, and task order."""
    if not num_docs:
        # An empty samples list makes lm-eval fall back to evaluating everything.
        raise ValueError(f"{task_name}: no evaluation documents")
    count = min(spec["limit"], num_docs)
    # A private RNG leaves task preprocessing and few-shot RNGs untouched. Use a
    # stable digest rather than Python's process-randomized hash(task_name).
    rng = random.Random(digest([spec["seed"], task_name]))
    # lm-eval iterates the subset in dataset order and uses this list to recover
    # original doc IDs. Unsorted indices would mislabel the logged documents.
    return sorted(rng.sample(range(num_docs), count))


def summarize_result(raw, spec, versions):
    """Keep references small, while retaining evidence of equivalent workloads."""
    manifests = {}
    for task, rows in raw["samples"].items():
        identities = set()
        for row in rows:
            fields = ("doc_id", "doc_hash", "prompt_hash", "target_hash")
            if any(row.get(field) is None for field in fields):
                raise ValueError(f"{task}: sample provenance is missing")
            identities.add(
                json.dumps({field: row[field] for field in fields}, sort_keys=True)
            )
        if not identities:
            raise ValueError(f"{task}: no evaluated samples")
        manifests[task] = {
            "count": len(identities),
            "doc_ids": sorted({row["doc_id"] for row in rows}),
            "sha256": digest(sorted(identities)),
        }
    if not manifests:
        raise ValueError("Harness returned no sample provenance")
    metrics = []
    for task, values in raw["results"].items():
        for metric, value in values.items():
            if "," not in metric or "_stderr," in metric:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{task}/{metric}: non-finite result")
            metrics.append(
                {
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "iterations": metric_iterations(raw, task, metric),
                    "num_fewshot": raw.get("n-shot", {}).get(task, spec["num_fewshot"]),
                    "higher_is_better": raw.get("higher_is_better", {})
                    .get(task, {})
                    .get(metric.split(",")[0]),
                }
            )
    if not metrics:
        raise ValueError("Harness returned no metrics")
    result = {
        "status": "completed",
        "metrics": metrics,
        "protocol": {
            "settings": {
                k: v for k, v in spec.items() if k not in ("id", "timeout_seconds")
            },
            "sampling": "uniform_without_replacement_v1",
            "packages": {p: versions[p] for p in PROTOCOL_PACKAGES},
            "task_versions": raw["versions"],
            "samples": manifests,
            "model_sha": raw["config"].get("model_sha"),
            "model_dtype": (
                str(raw["config"]["model_dtype"])
                if raw["config"].get("model_dtype")
                else None
            ),
        },
        "versions": versions,
        "sample_counts": raw.get("n-samples", {}),
    }
    # Record only preprocessing actually used by loaded tasks. Unrelated
    # benchmarks retain their existing protocol and reusable references.
    from evals.lm_eval.gpqa import METADATA_KEY

    preprocessing = {
        task: {METADATA_KEY: cfg["metadata"][METADATA_KEY]}
        for task, cfg in raw.get("configs", {}).items()
        if METADATA_KEY in (cfg.get("metadata") or {})
    }
    if preprocessing:
        result["protocol"]["preprocessing"] = preprocessing
    return result


def run_worker(args, spec):
    output = Path(args.output_dir)
    started = time.monotonic()
    try:
        import lm_eval
        from langdetect import DetectorFactory
        from lm_eval.tasks import TaskManager

        from evals.lm_eval.gpqa import install_gpqa_preprocessing

        samples = {}

        class SamplingTaskManager(TaskManager):
            def load(self, task_list):
                # Let simple_evaluate seed and load tasks exactly once. Populate
                # its samples argument here, after groups expand to leaf tasks
                # and before requests are built; preserve all group metadata.
                loaded = super().load(task_list)
                for name, task in loaded["tasks"].items():
                    # ConfigurableTask already processed these during loading.
                    # Avoid repeating potentially randomized preprocessing.
                    docs = getattr(task, "task_docs", None)
                    if docs is None:
                        docs = task.eval_docs
                    samples[name] = select_sample_indices(name, len(docs), spec)
                return loaded

        # IFEval's language checks use a separate RNG, outside the harness seeds.
        DetectorFactory.seed = spec["seed"]
        manager = SamplingTaskManager()
        install_gpqa_preprocessing(manager, spec["seed"])
        missing = set(spec["tasks"]) - set(manager.all_tasks)
        if missing:
            raise ValueError(f"Tasks not registered in lm_eval: {sorted(missing)}")
        model_args = {
            "pretrained": args.model[0],
            "backend": "causal",
            "dtype": spec["dtype"],
            "max_length": spec["max_length"],
            "batch_size": spec["batch_size"],
        }
        if args.backend == "spyre":
            from evals.lm_eval.run_spyre_lm_eval import SpyreHFLM

            model = SpyreHFLM(**model_args)
        else:
            from evals.lm_eval.native_hflm import NativeHFLM

            model = NativeHFLM(**model_args, device=args.device)
        seed = spec["seed"]
        raw = lm_eval.simple_evaluate(
            model=model,
            tasks=spec["tasks"],
            task_manager=manager,
            samples=samples,
            num_fewshot=spec["num_fewshot"],
            apply_chat_template=spec["apply_chat_template"],
            fewshot_as_multiturn=spec["fewshot_as_multiturn"],
            gen_kwargs=copy.deepcopy(spec["generation"]),
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=True,
            bootstrap_iters=0,
        )
        if raw is None:
            raise RuntimeError("Harness returned no results")
        # Use the loaded checkpoint's revision, not a fresh Hub lookup of main
        # after inference (which could have moved while the benchmark ran).
        raw["config"]["model_sha"] = getattr(
            model.model.config, "_commit_hash", None
        ) or getattr(model.config, "_commit_hash", None)
        # Samples are needed for provenance hashes, but may contain gated
        # dataset text. Do not publish those documents in CI artifacts.
        write_json(
            output / "results.json", {k: v for k, v in raw.items() if k != "samples"}
        )
        write_json(output / "samples.json", sample_outputs(raw))
        versions = {
            p: importlib.metadata.version(p) for p in (*PROTOCOL_PACKAGES, "torch")
        }
        if args.backend == "spyre":
            versions["torch-spyre"] = importlib.metadata.version("torch-spyre")
        result = summarize_result(raw, spec, versions)
    except Exception as error:
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / ".worker-result.json", result)
    return int(result["status"] != "completed")


def compare(current, reference):
    """Attach comparison data directly to metric rows; leave both inputs intact."""
    current = copy.deepcopy(current)
    if current["status"] != "completed":
        return current
    if reference is None or reference.get("status") != "completed":
        return {**current, "comparison": "missing_reference"}
    protocol = current["protocol"]
    previous = reference.get("protocol", {})
    differences = [key for key, value in protocol.items() if previous.get(key) != value]
    # Comparing two unknown checkpoint revisions is not evidence of equality.
    for key in ("model_sha", "model_dtype"):
        if not protocol.get(key) and key not in differences:
            differences.append(key)
    if differences:
        return {
            **current,
            "comparison": "incompatible_reference",
            "differences": differences,
        }
    baselines = {(r["task"], r["metric"]): r for r in reference["metrics"]}
    current["comparison"] = "compatible"
    for row in current["metrics"]:
        baseline = baselines.get((row["task"], row["metric"]))
        if baseline is None:
            row["comparison"] = "missing_metric"
            continue
        original = baseline["value"]
        if type(original) not in (int, float) or not math.isfinite(original):
            raise ValueError(f"Invalid reference metric: {row['task']}/{row['metric']}")
        delta = row["value"] - original
        row.update(
            comparison="zero_reference" if original == 0 else "compared",
            reference=original,
            delta=delta,
            percent_change=100 * delta / abs(original) if original else None,
        )
    return current


def stop_worker(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The worker exited between the timeout and the signal.
    else:
        process.kill()
    process.wait()


def wait_with_output(process, log_path, timeout):
    """Tail the worker's log without blocking timeout checks on silent output.

    Keep the worker writing straight to disk: a pipe or readline() can block on
    silent compilation, partial lines, or carriage-return-only progress bars.
    """
    deadline = time.monotonic() + timeout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with log_path.open("rb") as stream:
        try:
            while True:
                chunk = stream.read(65536)
                if chunk:
                    sys.stdout.write(decoder.decode(chunk))
                    sys.stdout.flush()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                try:
                    return process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_worker(process)
            raise
        finally:
            # The process has exited or been killed. Include the final output,
            # including partial lines and any pending multibyte character.
            while chunk := stream.read(65536):
                sys.stdout.write(decoder.decode(chunk))
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()


def run_benchmark(args, model, spec, output):
    output.mkdir()
    command = [
        sys.executable,
        "-u",
        "-m",
        "evals.lm_eval.run_benchmarks",
        "--config",
        str(Path(args.config).resolve()),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--model",
        model,
        "--output-dir",
        str(output),
        "--worker",
        spec["id"],
    ]
    log_path = output / "run.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = wait_with_output(process, log_path, spec["timeout_seconds"])
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": f"Timed out after {spec['timeout_seconds']} seconds",
            }
    path = output / ".worker-result.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "failed", "error": "Worker produced no result"}
    )
    path.unlink(missing_ok=True)  # Consumed worker handoff, not a second report.
    result["exit_code"] = code
    if code:
        result["status"] = "failed"
        result.setdefault("error", f"Worker exited with code {code}; see run.log")
    return result


def print_benchmark_result(result):
    """Print scores immediately, including available reference comparisons."""
    elapsed = result.get("elapsed_seconds")
    duration = f" in {elapsed:.1f}s" if elapsed is not None else ""
    detail = result.get("error", result.get("comparison", ""))
    if result.get("differences"):
        detail += ": " + ", ".join(result["differences"])
    # Progress bars and compiler messages need not end with a newline.
    print(
        f"\n  {result['status']}{duration}" + (f": {detail}" if detail else ""),
        flush=True,
    )
    for row in result.get("metrics", []):
        line = f"    {row['task']} | {row['metric']}: {row['value']:.6g}"
        if "reference" in row:
            change = row.get("percent_change")
            change = f"{change:+.2f}%" if change is not None else "N/A (zero reference)"
            line += f" (reference: {row['reference']:.6g}; change: {change})"
        elif row.get("comparison"):
            line += f" ({row['comparison']})"
        print(line, flush=True)


def markdown_report(report):
    lines = [
        f"# {report['suite']['name']} ({report['backend']})",
        "",
        "Relative change = 100 × (current − reference) / |reference|. N/A for zero or unavailable references.",
        "",
        "| Model | Benchmark / task | Metric | Score | Reference | Change | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for model, benchmarks in report["models"].items():
        for name, result in benchmarks.items():
            status = result.get("comparison", result["status"])
            if result["status"] != "completed":
                lines.append(
                    f"| {cell(model)} | {name} | — | — | — | — | {cell(status + ': ' + result.get('error', ''))} |"
                )
                continue
            if result.get("differences"):
                status += ": " + ", ".join(result["differences"])
            for row in result["metrics"]:
                reference = f"{row['reference']:.6g}" if "reference" in row else "—"
                change = (
                    f"{row['percent_change']:+.2f}%"
                    if row.get("percent_change") is not None
                    else "N/A"
                )
                lines.append(
                    f"| {cell(model)} | {name} / {cell(row['task'])} | {cell(row['metric'])} | {row['value']:.6g} | {reference} | {change} | {cell(row.get('comparison', status))} |"
                )
    return "\n".join(lines) + "\n"


def run_suite(args, suite, models):
    references = {}
    reference_dir = Path(args.reference_dir)
    for model in models:
        path = reference_dir / f"{model_key(model)}.json"
        if args.write_references and path.exists() and not args.overwrite_references:
            raise ValueError(
                f"Reference exists: {path}; use --overwrite-references to replace it"
            )
        if path.exists() and not args.write_references:
            reference = json.loads(path.read_text())
            if (
                reference.get("schema_version") != RESULT_SCHEMA_VERSION
                or reference.get("backend") != "hf"
                or model not in reference.get("models", {})
            ):
                raise ValueError(f"Invalid native reference for {model}: {path}")
            references[model] = reference["models"][model]
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    attachments = output / "attachments"
    attachments.mkdir()
    # Freeze resolved settings once; edits to the input file during a long run
    # must not change later workers' workloads.
    write_json(attachments / "suite.json", suite)
    args = copy.copy(args)
    args.config = str(attachments / "suite.json")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "backend": args.backend,
        "device": args.device if args.backend == "hf" else "spyre",
        "suite": suite,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "models": {},
    }
    failed = False
    for model in models:
        model_dir = attachments / model_key(model)
        model_dir.mkdir()
        benchmarks = report["models"][model] = {}
        for spec in suite["benchmarks"]:
            print(f"{model}: {spec['id']}", flush=True)
            result = run_benchmark(args, model, spec, model_dir / spec["id"])
            failed |= result["status"] != "completed"
            if not args.write_references:
                result = compare(result, references.get(model, {}).get(spec["id"]))
            benchmarks[spec["id"]] = result
            write_json(attachments / "report.json", report)
            write_json(output / "benchmarks.json", dashboard_results(report))
            (attachments / "summary.md").write_text(markdown_report(report))
            print_benchmark_result(result)
        successful = {
            name: r for name, r in benchmarks.items() if r["status"] == "completed"
        }
        if args.write_references and successful:
            reference_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                reference_dir / f"{model_key(model)}.json",
                {**report, "models": {model: successful}},
            )
    print(f"Results: {attachments / 'summary.md'}", flush=True)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "evals/lm_eval/benchmarks.yaml"))
    parser.add_argument("--backend", choices=("spyre", "hf"), default="spyre")
    parser.add_argument(
        "--model",
        action="append",
        help="Repeat to run multiple models; defaults to the model list.",
    )
    parser.add_argument(
        "--model-list", default=str(ROOT / "tests/model_lists/perf.yaml")
    )
    parser.add_argument("--model-group", default="nightly")
    parser.add_argument(
        "--device", default="cuda", help="Native HFLM device; ignored for Spyre."
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "evals/lm_eval/runs"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        ),
    )
    parser.add_argument(
        "--reference-dir", default=str(ROOT / "evals/lm_eval/references")
    )
    parser.add_argument(
        "--write-references",
        action="store_true",
        help="Save successful native results as references.",
    )
    parser.add_argument("--overwrite-references", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without loading models or datasets.",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print a GitHub Actions include matrix; no inference.",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        suite = load_suite(args.config)
        models = (
            validate_models(args.model)
            if args.model
            else load_models(args.model_list, args.model_group)
        )
        if args.write_references and args.backend != "hf":
            raise ValueError("--write-references requires --backend hf")
        if args.overwrite_references and not args.write_references:
            raise ValueError("--overwrite-references requires --write-references")
        if args.worker:
            if len(models) != 1:
                raise ValueError("A worker requires exactly one model")
            spec = next(s for s in suite["benchmarks"] if s["id"] == args.worker)
            return run_worker(args, spec)
        if args.print_matrix:
            print(
                json.dumps(
                    {"include": [{"model": m, "key": model_key(m)} for m in models]}
                )
            )
            return 0
        if args.dry_run:
            print(
                json.dumps(
                    {"backend": args.backend, "models": models, "suite": suite},
                    indent=2,
                )
            )
            return 0
        return run_suite(args, suite, models)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
