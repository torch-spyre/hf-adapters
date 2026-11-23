# Copyright 2025 The Torch-Spyre Authors.
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

"""Guard that ``tests/model_lists/perf.yaml`` stays in sync with the registry.

The ``perf`` test tier benchmarks an explicit model list (perf.yaml) rather than
the full registry. The ``weekly.models`` entry of that list is meant to mirror
``WEEKLY_PERF_PATHS`` -- the causal-LM registry entries marked ``weekly_perf``.
When that metadata or a marked model's path changes, ``WEEKLY_PERF_PATHS`` may
shift while the checked-in perf.yaml does not, so this test fails and points at
``tests/generate_perf_yaml.py`` to regenerate it.

Like ``test_adapter_coverage.py``, this test only needs ``pytest`` + ``PyYAML``
and no torch fixtures, so run it with ``pytest --noconftest`` to bypass the
torch-importing root conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tests.generate_perf_yaml as perf_generator
from tests.model_registry import WEEKLY_PERF_PATHS

PERF_YAML_PATH: Path = Path(__file__).resolve().parent / "model_lists" / "perf.yaml"

REGEN_HINT: str = (
    "Run `python -m tests.generate_perf_yaml` (from the repo root) to "
    "regenerate tests/model_lists/perf.yaml from the model registry, then "
    "commit the result."
)


def _load_weekly_models() -> list[str]:
    """Return ``weekly.models`` from perf.yaml (raising with a hint if absent)."""
    assert PERF_YAML_PATH.is_file(), f"{PERF_YAML_PATH} is missing. {REGEN_HINT}"
    with open(PERF_YAML_PATH) as f:
        data: object = yaml.safe_load(f)

    assert isinstance(
        data, dict
    ), f"{PERF_YAML_PATH} did not parse to a mapping. {REGEN_HINT}"
    weekly: object = data.get("weekly")
    assert isinstance(
        weekly, dict
    ), f"{PERF_YAML_PATH} has no `weekly:` section. {REGEN_HINT}"
    models: object = weekly.get("models")
    assert isinstance(
        models, list
    ), f"{PERF_YAML_PATH} `weekly.models` is not a list. {REGEN_HINT}"
    return [str(m) for m in models]


def test_generator_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """An existing perf file with the wrong top-level shape must not be replaced."""
    path: Path = tmp_path / "perf.yaml"
    path.write_text("- not-a-mapping\n")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        perf_generator._load_existing(path)


def test_generator_rejects_invalid_nightly_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed nightly data must not be silently replaced with an empty list."""
    path: Path = tmp_path / "perf.yaml"
    path.write_text("nightly:\n  models: not-a-list\n")
    monkeypatch.setattr(perf_generator, "PERF_YAML_PATH", path)

    with pytest.raises(ValueError, match=r"nightly\.models must be a list"):
        perf_generator.build_perf_config()


def test_weekly_models_match_weekly_perf_paths() -> None:
    """``perf.yaml`` weekly list must hold the same models as ``WEEKLY_PERF_PATHS``.

    Compared as sets: order does not matter, but every path must be present and
    none may be stale.
    """
    weekly_models: set[str] = set(_load_weekly_models())
    expected: set[str] = set(WEEKLY_PERF_PATHS)

    if weekly_models != expected:
        missing: list[str] = sorted(expected - weekly_models)
        extra: list[str] = sorted(weekly_models - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing from perf.yaml: {missing}")
        if extra:
            details.append(f"stale in perf.yaml: {extra}")
        raise AssertionError(
            "tests/model_lists/perf.yaml `weekly.models` is out of sync with "
            f"WEEKLY_PERF_PATHS in tests/model_registry.py ({'; '.join(details)}). "
            + REGEN_HINT
        )
