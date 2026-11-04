"""Offline checks of CI scheduling, planning, and tested-software provenance."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def workflow(name):
    # BaseLoader preserves GitHub's `on` key rather than treating it as True.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def step(job, name):
    return next(item for item in job["steps"] if item.get("name") == name)


def condition(expression, values):
    """Evaluate the boolean/string subset used by the checked-in job gates."""
    expression = expression.removeprefix("${{").removesuffix("}}")
    for name, value in sorted(values.items(), key=lambda item: -len(item[0])):
        expression = expression.replace(name, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return eval(
        f"({expression.strip()})",
        {"__builtins__": {}},
        {"contains": lambda text, needle: needle in text, "format": str.format},
    )


@pytest.mark.parametrize(
    "tier,selected",
    [
        ("regression", True),
        ("trunk", True),
        ("quality", True),
        ("smoke quality", True),
        ("smoke", False),
        ("integration", False),
        ("unit", False),
        ("perf", False),
        ("quality_extra", False),
    ],
)
@pytest.mark.parametrize("resolve_result", ["success", "skipped", "failure"])
def test_quality_job_selection(tier, selected, resolve_result):
    job = workflow("_test_matrix.yaml")["jobs"]["spyre-quality-tests"]
    values = {
        "cancelled()": False,
        "needs.resolve-torch-spyre.result": resolve_result,
        "needs.resolve-test-type.outputs.test_type": tier,
        "inputs.skip_tests": False,
        "inputs.edge_cases_only": False,
    }
    assert condition(job["if"], values) == (selected and resolve_result != "failure")
    for blocker in ("cancelled()", "inputs.skip_tests", "inputs.edge_cases_only"):
        assert not condition(job["if"], {**values, blocker: True})


@pytest.mark.parametrize(
    "quality_result", ["success", "skipped", "failure", "cancelled"]
)
def test_quality_failure_reaches_required_check(quality_result):
    job = workflow("_test_matrix.yaml")["jobs"]["run-spyre-unit-tests"]
    assert "spyre-quality-tests" in job["needs"]
    values = {f"needs.{name}.result": "success" for name in job["needs"]}
    values["needs.spyre-quality-tests.result"] = quality_result
    script = job["steps"][0]["run"]
    rendered = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(condition(match.group(), values)).lower(),
        script,
        flags=re.DOTALL,
    )
    result = subprocess.run(["bash", "-e", "-c", rendered], capture_output=True)
    assert result.returncode == (1 if quality_result in {"failure", "cancelled"} else 0)


def test_callers_share_evaluator_and_forward_tested_stack():
    matrix = workflow("_test_matrix.yaml")["jobs"]["spyre-quality-tests"]
    standalone = workflow("lm-eval.yaml")["jobs"]["quality"]
    assert matrix["uses"] == standalone["uses"] == "./.github/workflows/_lm-eval.yaml"
    assert matrix["secrets"] == standalone["secrets"] == "inherit"
    for name in ("ref", "repository", "models", "image_label", "prebaked_image"):
        assert matrix["with"][name] == "${{ inputs." + name + " }}"
    assert matrix["with"]["torch_spyre_sha"] == (
        "${{ needs.resolve-torch-spyre.outputs.torch_spyre_sha }}"
    )
    core = workflow("_lm-eval.yaml")["jobs"]
    build = next(s for s in core["evaluate"]["steps"] if s.get("id") == "build")
    assert build["with"]["ref"] == "${{ needs.plan.outputs.source_sha }}"
    for name in ("repository", "prebaked_image", "torch_spyre_sha"):
        assert build["with"][name] == "${{ inputs." + name + " }}"
    assert core["evaluate"]["runs-on"][-1] == "${{ inputs.image_label }}"
    checkout = next(
        s for s in core["plan"]["steps"] if "checkout@" in s.get("uses", "")
    )
    assert checkout["with"]["ref"] == "${{ inputs.ref || github.sha }}"
    assert (
        checkout["with"]["repository"] == "${{ steps.repository.outputs.repository }}"
    )
    dispatch = workflow("integration-tests.yaml")["on"]["workflow_dispatch"]
    assert "quality" in dispatch["inputs"]["test_type"]["options"]


def run_shell(script, tmp_path, **env):
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "GITHUB_OUTPUT": str(output),
            **env,
        },
        capture_output=True,
        text=True,
    )
    return result, output


@pytest.mark.parametrize("models", ["", "org/model-a org/model-b"])
def test_plan_executes_with_default_or_explicit_models(tmp_path, models):
    plan = step(
        workflow("_lm-eval.yaml")["jobs"]["plan"], "Validate config and select models"
    )
    result, output = run_shell(
        plan["run"],
        tmp_path,
        EVAL_CONFIG="evals/lm_eval/benchmarks.yaml",
        EVAL_MODELS=models,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    matrix = json.loads(outputs["matrix"])["include"]
    expected = (
        models.split()
        or yaml.safe_load((ROOT / "tests/model_lists/perf.yaml").read_text())[
            "nightly"
        ]["models"]
    )
    assert [row["model"] for row in matrix] == expected
    assert len({row["key"] for row in matrix}) == len(expected)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert outputs["source_sha"] == sha


@pytest.mark.parametrize(
    "repository", ["user/fork", "https://github.com/user/fork.git"]
)
def test_plan_accepts_repository_formats(tmp_path, repository):
    plan = workflow("_lm-eval.yaml")["jobs"]["plan"]
    result, output = run_shell(
        step(plan, "Resolve repository for checkout")["run"],
        tmp_path,
        EVAL_REPOSITORY=repository,
        GITHUB_SERVER_URL="https://github.com",
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == "repository=user/fork"


@pytest.mark.parametrize(
    "prebaked,pin,ok",
    [("true", "", True), ("false", "b" * 40, True), ("false", "c" * 40, False)],
)
def test_identity_records_installed_stack_and_rejects_wrong_pin(
    tmp_path, prebaked, pin, ok
):
    capture = step(
        workflow("_lm-eval.yaml")["jobs"]["evaluate"],
        "Capture tested artifact identity",
    )
    # Exercise the workflow shell with a mock image checkout and identity CLI.
    # Hash derivation itself belongs to torch-spyre; this checks its installed input.
    mocks = """
git() { printf '%s\\n' "$MOCK_SPYRE_SHA"; }
python3() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --installed) shift; delta="$1" ;;
      --output-file) shift; output="$1" ;;
    esac
    shift
  done
  printf 'derived|base|%s\\n' "$delta" > "$output"
}
"""
    identity = tmp_path / "identity"
    result, _ = run_shell(
        mocks + capture["run"],
        tmp_path,
        EVAL_IDENTITY_DIR=str(identity),
        IDENTITY_ROOT=str(tmp_path / "library"),
        HF_ADAPTERS_SHA="a" * 40,
        PREBAKED_IMAGE=prebaked,
        TORCH_SPYRE_SHA=pin,
        MOCK_SPYRE_SHA="b" * 40,
        GITHUB_STEP_SUMMARY=str(tmp_path / "summary"),
    )
    assert (result.returncode == 0) == ok, result.stderr
    record = identity / "artifact-id.txt"
    if not ok:
        assert not record.exists()
        return
    installed = "hf-adapters@" + "a" * 12
    if prebaked == "false":
        installed += " torch-spyre@" + "b" * 12
        assert (identity / "torch-spyre-sha.txt").read_text().strip() == "b" * 40
    else:
        assert not (identity / "torch-spyre-sha.txt").exists()
    assert record.read_text().strip() == f"derived|base|{installed}"


def test_failed_quality_jobs_preserve_bundles_and_identity():
    evaluate = workflow("_lm-eval.yaml")["jobs"]["evaluate"]
    assert "continue-on-error" not in evaluate
    assert evaluate["strategy"]["fail-fast"] == "false"
    for name in (
        "Preserve artifact identity with results",
        "Publish metric summary",
        "Upload results and per-benchmark logs",
    ):
        assert step(evaluate, name)["if"] == "always()"
    upload = step(evaluate, "Upload results and per-benchmark logs")
    assert upload["with"]["name"] == "lm-eval-${{ matrix.key }}"
