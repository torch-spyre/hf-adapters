"""Exercise aggregate Make dispatch without running hardware tests or inference."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_make(tmp_path, monkeypatch):
    calls_path = tmp_path / "calls.jsonl"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "with open(os.environ['QUALITY_MAKE_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(7 if sys.argv[1] == os.environ.get('QUALITY_MAKE_FAIL') else 0)\n"
    )
    monkeypatch.setenv("QUALITY_MAKE_CALLS", str(calls_path))
    for name in ("TEST_TYPE", "MAKEFLAGS", "MFLAGS", "QUALITY_MAKE_FAIL"):
        monkeypatch.delenv(name, raising=False)

    def run(*args):
        calls_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-s",
                "tests",
                f"MAKE={shlex.join([sys.executable, str(probe), 'suite'])}",
                f"QUALITY_RUNNER={shlex.join([sys.executable, str(probe), 'quality'])}",
                f"RESULTS_DIR={tmp_path / 'results with spaces'}",
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        return result, [call[1:] for call in calls if call[0] == "quality"]

    return run


@pytest.mark.parametrize(
    "tier, expected",
    [
        (None, 1),
        ("", 1),
        ("regression", 1),
        ("trunk", 1),
        ("quality", 1),
        ("unit", 0),
        ("integration", 0),
    ],
)
def test_quality_runs_once_in_regression_and_trunk(run_make, tier, expected):
    result, quality = run_make(*([] if tier is None else [f"TEST_TYPE={tier}"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(quality) == expected
    if quality:
        # With no override, leave model selection to the runner's nightly list.
        assert "--model" not in quality[0]
        assert quality[0][quality[0].index("--backend") + 1] == "spyre"


def test_regression_forwards_quality_options(run_make, tmp_path):
    models = ["ibm-granite/granite-4.1-8b", "google/gemma-4-26B-A4B-it"]
    config = str(tmp_path / "suite override.yaml")
    result, quality = run_make(
        "TEST_TYPE=regression",
        f"MODEL_PATH={' '.join(models)}",
        f"QUALITY_CONFIG={config}",
        "QUALITY_ARGS=--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(quality) == 1
    args = quality[0]
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--model"] == models
    assert args[args.index("--config") + 1] == config
    assert args[args.index("--output-dir") + 1] == str(
        tmp_path / "results with spaces" / "quality"
    )
    assert "--dry-run" in args


@pytest.mark.parametrize("failure", ["quality", "suite"])
def test_regression_reports_failures_and_still_attempts_quality(
    run_make, monkeypatch, failure
):
    monkeypatch.setenv("QUALITY_MAKE_FAIL", failure)
    result, quality = run_make("TEST_TYPE=regression")
    assert result.returncode != 0
    assert len(quality) == 1
