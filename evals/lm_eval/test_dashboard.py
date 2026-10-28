"""Dashboard payload identity, metric denominators, and attachment boundaries."""

import copy
import json

import pytest

from evals.lm_eval.dashboard import dashboard_results, metric_iterations, sample_outputs


@pytest.fixture
def report():
    return {
        "backend": "spyre",
        "device": "spyre",
        "models": {
            "ibm-granite/granite-4.1-8b": {
                "gsm8k": {
                    "status": "completed",
                    "protocol": {
                        "settings": {
                            "tasks": ["gsm8k"],
                            "num_fewshot": 5,
                            "seed": 1234,
                        },
                        "sampling": "uniform_without_replacement_v1",
                        "packages": {"lm_eval": "0.4.13"},
                        "task_versions": {"gsm8k": 3},
                        "samples": {"gsm8k": {"count": 2, "sha256": "sample-hash"}},
                        "model_sha": "checkpoint",
                        "model_dtype": "torch.bfloat16",
                    },
                    "versions": {"torch": "2.13"},
                    "metrics": [
                        {
                            "task": "gsm8k",
                            "metric": "exact_match,strict-match",
                            "value": 0.5,
                            "iterations": 2,
                            "num_fewshot": 5,
                        },
                        {
                            "task": "gsm8k",
                            "metric": "exact_match,flexible-extract",
                            "value": 1.0,
                            "iterations": 2,
                            "num_fewshot": 5,
                        },
                    ],
                },
                "broken": {"status": "failed", "error": "compile failed"},
            },
        },
    }


def test_payload_schema_and_filter_identity(report):
    payload = dashboard_results(report)
    assert set(payload) == {"benchmarks"}
    strict, flexible = payload["benchmarks"]
    assert set(strict) == {"name", "props", "backend", "measurements", "iterations"}
    assert strict["name"] == "gsm8k"
    assert strict["props"]["model"] == "ibm-granite/granite-4.1-8b"
    assert strict["props"]["task"] == "gsm8k" and strict["props"]["num_fewshot"] == 5
    assert strict["props"]["filter"] == "strict-match"
    assert flexible["props"]["filter"] == "flexible-extract"
    assert strict["measurements"] == {"exact_match": 0.5}
    assert flexible["measurements"] == {"exact_match": 1.0}
    assert strict["iterations"] == flexible["iterations"] == 2
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize(
    "device, backend", [("cpu", "cpu"), ("cuda:1", "cuda"), ("mps", "mps")]
)
def test_backend_is_separate_from_benchmark_identity(report, device, backend):
    spyre = dashboard_results(report)["benchmarks"]
    report.update(backend="hf", device=device)
    native = dashboard_results(report)["benchmarks"]
    assert all(r["backend"] == backend for r in native)
    assert [r["props"] for r in native] == [r["props"] for r in spyre]


def test_only_relevant_protocol_changes_affect_identity(report):
    original = dashboard_results(report)
    result = report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"]
    result["versions"]["torch"] = "other-runtime"
    protocol = result["protocol"]
    protocol["settings"]["tasks"].append("unrelated")
    protocol["task_versions"]["unrelated"] = 99
    protocol["samples"]["unrelated"] = {"count": 50}
    protocol["preprocessing"] = {"unrelated": {"version": 7}}
    assert dashboard_results(report) == original
    protocol["settings"]["seed"] += 1
    changed = dashboard_results(report)
    assert (
        changed["benchmarks"][0]["props"]["protocol"]
        != original["benchmarks"][0]["props"]["protocol"]
    )


def test_reference_deltas_are_not_ingested_as_quality_scores(report):
    original = dashboard_results(report)
    result = report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"]
    result["comparison"] = "compatible"
    for row in result["metrics"]:
        row.update(reference=0.8, percent_change=-37.5, delta=-0.3)
    assert dashboard_results(report) == original


def test_metrics_with_same_filter_and_count_share_an_entry(report):
    result = report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"]
    result["metrics"].append(
        {**result["metrics"][0], "metric": "acc_norm,strict-match", "value": 0.7}
    )
    entries = dashboard_results(report)["benchmarks"]
    assert len(entries) == 2
    assert entries[0]["measurements"] == {"exact_match": 0.5, "acc_norm": 0.7}


def test_all_failed_runs_produce_an_empty_payload(report):
    report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"] = {"status": "failed"}
    assert dashboard_results(report) == {"benchmarks": []}


def test_group_aggregates_are_not_exported_as_leaf_tasks(report):
    result = report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"]
    result["metrics"].append(
        {"task": "group", "metric": "acc,none", "value": 0.8, "iterations": None}
    )
    assert len(dashboard_results(report)["benchmarks"]) == 2
    assert metric_iterations({"samples": {}}, "group", "acc,none") is None


def test_iterations_count_selected_metric_and_filter_without_duplicates():
    raw = {
        "samples": {
            "task": [
                {"doc_id": 8, "filter": "none", "acc": 1, "acc_norm": 1},
                {"doc_id": 9, "filter": "none", "acc": 0},
                {"doc_id": 8, "filter": "none", "acc": 1, "acc_norm": 1},
                {"doc_id": 8, "filter": "other", "acc": 1},
            ]
        }
    }
    assert metric_iterations(raw, "task", "acc,none") == 2
    assert metric_iterations(raw, "task", "acc_norm,none") == 1
    assert metric_iterations(raw, "task", "acc,other") == 1
    with pytest.raises(ValueError, match="no metric observations"):
        metric_iterations(raw, "task", "missing,none")


def test_ifeval_prompt_and_instruction_denominators(report):
    raw = {
        "samples": {
            "ifeval": [
                {
                    "doc_id": 3,
                    "filter": "none",
                    "prompt_level_strict_acc": True,
                    "inst_level_strict_acc": [True, True],
                },
                {
                    "doc_id": 9,
                    "filter": "none",
                    "prompt_level_strict_acc": False,
                    "inst_level_strict_acc": [True, False, False],
                },
            ]
        }
    }
    prompt_count = metric_iterations(raw, "ifeval", "prompt_level_strict_acc,none")
    instruction_count = metric_iterations(raw, "ifeval", "inst_level_strict_acc,none")
    assert (prompt_count, instruction_count) == (2, 5)
    result = copy.deepcopy(report["models"]["ibm-granite/granite-4.1-8b"]["gsm8k"])
    for key in ("task_versions", "samples"):
        result["protocol"][key] = {"ifeval": result["protocol"][key]["gsm8k"]}
    result["metrics"] = [
        {
            "task": "ifeval",
            "metric": "prompt_level_strict_acc,none",
            "value": 0.5,
            "iterations": prompt_count,
            "num_fewshot": 0,
        },
        {
            "task": "ifeval",
            "metric": "inst_level_strict_acc,none",
            "value": 0.6,
            "iterations": instruction_count,
            "num_fewshot": 0,
        },
    ]
    report["models"] = {"model": {"ifeval": result}}
    prompt, instruction = dashboard_results(report)["benchmarks"]
    assert prompt["props"] == instruction["props"]
    assert prompt["iterations"] == 2 and instruction["iterations"] == 5
    assert prompt["measurements"] == {"prompt_level_strict_acc": 0.5}
    assert instruction["measurements"] == {"inst_level_strict_acc": 0.6}


def test_sample_attachments_exclude_dataset_text():
    raw = {
        "samples": {
            "task": [
                {
                    "doc_id": 7,
                    "filter": "none",
                    "metrics": ["acc"],
                    "acc": 1,
                    "resps": [["generated output"]],
                    "filtered_resps": ["generated output"],
                    "doc": {"question": "gated source text"},
                    "arguments": ["private prompt"],
                    "target": "private target",
                    "doc_hash": "hash",
                }
            ]
        }
    }
    row = sample_outputs(raw)["task"][0]
    assert row["doc_id"] == 7 and row["acc"] == 1
    assert row["resps"] == [["generated output"]]
    assert not ({"doc", "arguments", "target"} & row.keys())
    assert "private" not in json.dumps(row) and "gated" not in json.dumps(row)
