"""Validate the nightly model selection without importing the Spyre runtime."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from generate_prefill_matrix import read_nightly_models

_SCRIPT = Path(__file__).with_name("generate_prefill_matrix.py")


def test_cli_emits_only_nightly_models_in_file_order(tmp_path: Path) -> None:
    config = tmp_path / "perf.yaml"
    config.write_text(
        "nightly:\n  models:\n    - org/model-b\n    - org/model-a\n"
        "weekly:\n  models:\n    - org/weekly-only\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(config)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == ["org/model-b", "org/model-a"]


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "[]",
        "weekly: {models: [org/weekly-only]}",
        "nightly: []",
        "nightly: {}",
        "nightly: {models: []}",
        "nightly: {models: org/model-a}",
        "nightly: {models: [org/model-a, null]}",
        "nightly: {models: [org/model-a, 42]}",
        "nightly: {models: [org/model-a, '']}",
        "nightly: {models: [' org/model-a ']}",
        "nightly: {models: [org/model-a, org/model-a]}",
    ],
)
def test_invalid_model_lists_fail(tmp_path: Path, contents: str) -> None:
    config = tmp_path / "perf.yaml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_nightly_models(config)


@pytest.mark.parametrize("contents", ["nightly: [", "nightly: {models: []}"])
def test_cli_fails_without_emitting_a_matrix(tmp_path: Path, contents: str) -> None:
    config = tmp_path / "perf.yaml"
    config.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(config)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error:" in result.stderr
