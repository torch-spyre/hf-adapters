"""Translate lm-eval results into the dashboard's benchmarks/quality payload."""

import hashlib
import json


def metric_iterations(raw, task, metric):
    """Count actual observations, not the requested limit or extraction filters."""
    if task not in raw["samples"]:
        return None  # Group aggregates are retained in the diagnostic report only.
    name, filter_name = metric.rsplit(",", 1)
    values = {
        row["doc_id"]: row[name]
        for row in raw["samples"][task]
        if row.get("filter") == filter_name and name in row
    }
    # IFEval aggregates these lists across instructions, not across prompts.
    count = (
        sum(len(value) for value in values.values())
        if name in ("inst_level_strict_acc", "inst_level_loose_acc")
        else len(values)
    )
    if count <= 0:
        raise ValueError(f"{task}/{metric}: no metric observations")
    return count


def sample_outputs(raw):
    """Keep model outputs and scores, without publishing gated source text."""
    fields = (
        "doc_id",
        "doc_hash",
        "prompt_hash",
        "target_hash",
        "filter",
        "metrics",
        "resps",
        "filtered_resps",
    )
    return {
        task: [
            {key: row[key] for key in (*fields, *row.get("metrics", [])) if key in row}
            for row in rows
        ]
        for task, rows in raw["samples"].items()
    }


def dashboard_results(report):
    """Export successful leaf-task scores; backend is never part of identity."""
    backend = (
        "spyre" if report["backend"] == "spyre" else report["device"].split(":", 1)[0]
    )
    entries = []
    for model, benchmarks in report["models"].items():
        for name, result in benchmarks.items():
            if result["status"] != "completed":
                continue
            grouped = {}
            denominators = {}
            for row in result["metrics"]:
                count = row["iterations"]
                if count is None:
                    continue
                task = row["task"]
                metric, filter_name = row["metric"].rsplit(",", 1)
                sample_unit = (
                    "instruction"
                    if metric.startswith("inst_level_")
                    else "prompt" if metric.startswith("prompt_level_") else None
                )
                # Match on the leaf's workload, not unrelated tasks in its group
                # or backend-specific runtime versions, device ordinals, or dates.
                protocol = result["protocol"]
                identity = {
                    **protocol,
                    "settings": {
                        **{
                            k: v
                            for k, v in protocol["settings"].items()
                            if k != "tasks"
                        },
                        "num_fewshot": row["num_fewshot"],
                    },
                    "task_versions": {task: protocol["task_versions"][task]},
                    "samples": {task: protocol["samples"][task]},
                }
                preprocessing = identity.pop("preprocessing", {})
                if task in preprocessing:
                    identity["preprocessing"] = {task: preprocessing[task]}
                fingerprint = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
                population = (task, filter_name, sample_unit)
                denominators.setdefault(population, set()).add(count)
                key = (*population, count)
                if key not in grouped:
                    grouped[key] = {
                        "name": name,
                        "props": {
                            "model": model,
                            "task": task,
                            "num_fewshot": row["num_fewshot"],
                            "filter": filter_name,
                            "protocol": fingerprint,
                            **({"sample_unit": sample_unit} if sample_unit else {}),
                        },
                        "backend": backend,
                        "measurements": {},
                        "iterations": count,
                    }
                grouped[key]["measurements"][metric] = row["value"]
            for (*population, count), entry in grouped.items():
                # The ingest writer merges equal identities and sums iterations.
                # Distinct denominators must differ in props, not just iterations.
                if len(denominators[tuple(population)]) > 1:
                    entry["props"]["observation_count"] = count
            entries.extend(grouped.values())
    return {"benchmarks": entries}
