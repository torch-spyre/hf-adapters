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

"""Targeted regression tests for ppc64le model exclusions.

Tests:
1. Schema validation for tests/model_lists/exclude.yaml in manage_cache.py.
2. Cache download list under simulated ppc64le vs x86_64, including extra_models.
3. --model-path overrides (mixed, all-excluded, and positive x86 control) in conftest.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_CONFIG_DIR = REPO_ROOT / ".github" / "cache_config"
if str(CACHE_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CACHE_CONFIG_DIR))

import manage_cache  # noqa: E402


def test_manage_cache_rejects_non_mapping_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exclude.yaml must be a YAML mapping."""
    bad_file = tmp_path / "exclude.yaml"
    bad_file.write_text("- not-a-dict\n")
    monkeypatch.setattr(manage_cache, "EXCLUDE_FILE_PATH", bad_file)

    with pytest.raises(ValueError, match="did not parse to a mapping"):
        manage_cache._load_excluded_paths()


def test_manage_cache_rejects_unexpected_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exclude.yaml must only contain allowed sections."""
    bad_file = tmp_path / "exclude.yaml"
    bad_file.write_text("s390x-models:\n  - model/foo\n")
    monkeypatch.setattr(manage_cache, "EXCLUDE_FILE_PATH", bad_file)

    with pytest.raises(ValueError, match="contains unexpected section"):
        manage_cache._load_excluded_paths()


@pytest.mark.parametrize("key", ["models", "ppc64le_models"])
@pytest.mark.parametrize(
    "invalid_val",
    [
        None,
        "just-a-string",
        123,
        {"nested": "dict"},
        ["valid-model", 456],
    ],
)
def test_manage_cache_validates_section_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, invalid_val: object
) -> None:
    """Each exclusion section must be a list of strings."""
    bad_file = tmp_path / "exclude.yaml"
    bad_file.write_text(yaml.dump({key: invalid_val}))
    monkeypatch.setattr(manage_cache, "EXCLUDE_FILE_PATH", bad_file)

    with pytest.raises(ValueError, match=f"'{key}' section must be a list of strings"):
        manage_cache._load_excluded_paths()


def test_manage_cache_ppc64le_excludes_registry_and_extra_models() -> None:
    """Under simulated ppc64le, Qwen models from registry and extra_models are excluded."""
    extra = ["Qwen/Qwen3-0.6B", "custom-org/valid-extra-model"]

    with patch("platform.machine", return_value="ppc64le"):
        models = manage_cache.build_download_list(extra_models=extra)

        assert not any("Qwen" in m or "qwen" in m for m in models)
        assert "custom-org/valid-extra-model" in models
        assert "Qwen/Qwen3-0.6B" not in models


def test_manage_cache_x86_includes_qwen_models() -> None:
    """Under simulated x86_64, Qwen models remain present."""
    with patch("platform.machine", return_value="x86_64"):
        models = manage_cache.build_download_list()

        assert any("Qwen" in m or "qwen" in m for m in models)


def test_pytest_generate_tests_mixed_overrides_ppc64le() -> None:
    """When some models are excluded on ppc64le, only the valid ones are parametrized."""
    import tests.conftest as conftest_mod

    metafunc = MagicMock()
    metafunc.config.getoption.return_value = [
        "Qwen/Qwen3-0.6B",
        "ibm-granite/granite-4.1-8b",
    ]
    metafunc.fixturenames = ["model_path"]
    metafunc.definition.iter_markers.return_value = []
    metafunc.definition.own_markers = []
    metafunc.definition.get_closest_marker.return_value = MagicMock(args=["causal"])

    with patch("platform.machine", return_value="ppc64le"):
        import model_registry as mr

        importlib.reload(mr)
        assert sys.modules["model_registry"] is mr

        conftest_mod.pytest_generate_tests(metafunc)

        call_args = metafunc.parametrize.call_args
        assert call_args is not None
        assert call_args[1]["ids"] == ["ibm-granite/granite-4.1-8b"]


def test_pytest_generate_tests_all_excluded_overrides_ppc64le() -> None:
    """When ALL overrides are excluded on ppc64le, tests are parametrized as skipped."""
    import tests.conftest as conftest_mod

    metafunc = MagicMock()
    metafunc.config.getoption.return_value = ["Qwen/Qwen3-0.6B"]
    metafunc.fixturenames = ["model_path"]
    metafunc.definition.iter_markers.return_value = []
    metafunc.definition.own_markers = []
    metafunc.definition.get_closest_marker.return_value = MagicMock(args=["causal"])

    with patch("platform.machine", return_value="ppc64le"):
        import model_registry as mr

        importlib.reload(mr)
        assert sys.modules["model_registry"] is mr

        conftest_mod.pytest_generate_tests(metafunc)

        call_args = metafunc.parametrize.call_args
        assert call_args is not None
        assert call_args[1]["ids"] == ["Qwen/Qwen3-0.6B"]
        # The single param should have a skip marker
        param_obj = call_args[0][1][0]
        assert hasattr(param_obj, "marks")
        assert any(m.name == "skip" for m in param_obj.marks)


def test_pytest_generate_tests_x86_64_retains_qwen() -> None:
    """On x86_64, Qwen overrides are not excluded and are parametrized normally."""
    import tests.conftest as conftest_mod

    metafunc = MagicMock()
    metafunc.config.getoption.return_value = [
        "Qwen/Qwen3-0.6B",
        "ibm-granite/granite-4.1-8b",
    ]
    metafunc.fixturenames = ["model_path"]
    metafunc.definition.iter_markers.return_value = []
    metafunc.definition.own_markers = []
    metafunc.definition.get_closest_marker.return_value = MagicMock(args=["causal"])

    with patch("platform.machine", return_value="x86_64"):
        import model_registry as mr

        importlib.reload(mr)
        assert sys.modules["model_registry"] is mr

        conftest_mod.pytest_generate_tests(metafunc)

        call_args = metafunc.parametrize.call_args
        assert call_args is not None
        assert call_args[1]["ids"] == [
            "Qwen/Qwen3-0.6B",
            "ibm-granite/granite-4.1-8b",
        ]


def test_perf_models_filtered_on_ppc64le() -> None:
    """Perf models are filtered on ppc64le to exclude Qwen models, but kept on x86_64."""
    perf_models = [
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.6-35B-A3B",
        "ibm-granite/granite-3.3-2b-instruct",
    ]

    with patch("platform.machine", return_value="ppc64le"):
        import model_registry as mr

        importlib.reload(mr)
        filtered = mr._exclude(perf_models)
        assert filtered == ["ibm-granite/granite-3.3-2b-instruct"]

    with patch("platform.machine", return_value="x86_64"):
        import model_registry as mr

        importlib.reload(mr)
        filtered = mr._exclude(perf_models)
        assert filtered == perf_models


def test_load_tests_model_path_exclusion_resolution() -> None:
    """Explicitly excluded models in MODEL_PATH resolve to empty, while valid ones resolve."""
    with patch("platform.machine", return_value="ppc64le"):
        import model_registry as mr

        importlib.reload(mr)
        # All excluded
        to_run, invalid = mr.resolve_load_test_models(["Qwen/Qwen3-0.6B"])
        assert to_run == []
        assert invalid == []

        # Mixed valid + excluded
        to_run, invalid = mr.resolve_load_test_models(
            ["Qwen/Qwen3-0.6B", "ibm-granite/granite-3.3-2b-instruct"]
        )
        assert to_run == ["ibm-granite/granite-3.3-2b-instruct"]
        assert invalid == []

        # Mixed valid + invalid
        to_run, invalid = mr.resolve_load_test_models(
            ["ibm-granite/granite-3.3-2b-instruct", "invalid-nonexistent-model"]
        )
        assert to_run == ["ibm-granite/granite-3.3-2b-instruct"]
        assert invalid == ["invalid-nonexistent-model"]


@pytest.mark.skipif(
    importlib.import_module("platform").machine() != "ppc64le",
    reason="Only runs on native ppc64le",
)
def test_makefile_load_tests_all_excluded_skips_ppc64le(tmp_path: Path) -> None:
    """make load-tests with an excluded model on ppc64le exits 0 and records a skip."""
    import subprocess

    junit_file = tmp_path / "junit-load.xml"
    result = subprocess.run(
        [
            "make",
            "load-tests",
            "MODEL_PATH=Qwen/Qwen3-0.6B",
            f"JUNIT_XML={junit_file}",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "skipping load-tests" in result.stdout
    assert junit_file.exists()


@pytest.mark.skipif(
    importlib.import_module("platform").machine() != "ppc64le",
    reason="Only runs on native ppc64le",
)
def test_makefile_load_tests_invalid_model_fails(tmp_path: Path) -> None:
    """make load-tests with an unlisted/invalid model exits non-zero (preserves failure)."""
    import subprocess

    result = subprocess.run(
        [
            "make",
            "load-tests",
            "MODEL_PATH=nonexistent-invalid-model-foo",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Invalid model path(s)" in result.stderr


@pytest.mark.skipif(
    importlib.import_module("platform").machine() != "ppc64le",
    reason="Only runs on native ppc64le",
)
def test_makefile_load_tests_mixed_valid_and_invalid_fails(tmp_path: Path) -> None:
    """make load-tests with mixed valid and invalid models fails on the invalid model."""
    import subprocess

    result = subprocess.run(
        [
            "make",
            "load-tests",
            "MODEL_PATH=ibm-granite/granite-3.3-2b-instruct nonexistent-invalid-model-foo",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Invalid model path(s)" in result.stderr


@pytest.mark.skipif(
    importlib.import_module("platform").machine() != "ppc64le",
    reason="Only runs on native ppc64le",
)
def test_makefile_perf_tests_recording_stub_ppc64le(tmp_path: Path) -> None:
    """make perf-tests invokes spyre-perf-suite only for non-excluded models on ppc64le."""
    import subprocess

    log_file = tmp_path / "invoked_models.log"
    stub_script = tmp_path / "mock_spyre_perf_suite.sh"
    stub_content = f"""#!/bin/bash
if [[ "$1" == "--help" ]]; then
    exit 0
fi
model=""
report=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-model) model="$2"; shift 2 ;;
        --report) report="$2"; shift 2 ;;
        *) shift ;;
    esac
done
echo "$model" >> "{log_file}"
if [[ -n "$report" ]]; then
    xml_report="${{report%.txt}}.xml"
    echo '<testsuites><testsuite name="perf"/></testsuites>' > "$xml_report"
fi
exit 0
"""
    stub_script.write_text(stub_content)
    stub_script.chmod(0o755)

    results_dir = tmp_path / "results"
    result = subprocess.run(
        [
            "make",
            "perf-tests",
            f"PERF_SUITE={stub_script}",
            "MODEL_PATH=Qwen/Qwen3-0.6B ibm-granite/granite-3.3-2b-instruct",
            f"RESULTS_DIR={results_dir}",
            "PERF_REPORT=split",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    invoked = log_file.read_text().strip().splitlines()
    assert invoked == ["ibm-granite/granite-3.3-2b-instruct"]


@pytest.mark.skipif(
    importlib.import_module("platform").machine() != "ppc64le",
    reason="Only runs on native ppc64le",
)
def test_makefile_perf_tests_all_excluded_skips_ppc64le(tmp_path: Path) -> None:
    """make perf-tests with only excluded models skips without invoking the harness."""
    import subprocess

    results_dir = tmp_path / "results"
    result = subprocess.run(
        [
            "make",
            "perf-tests",
            "PERF_SUITE=/bin/false",
            "MODEL_PATH=Qwen/Qwen3-0.6B Qwen/Qwen3.5-2B",
            f"RESULTS_DIR={results_dir}",
            "PERF_REPORT=combined",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "skipping perf-tests" in result.stdout
    assert (results_dir / "report.xml").exists()
