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

"""Regenerate ``tests/model_lists/perf.yaml`` from ``model_registry.py``.

The ``weekly.models`` list contains the causal-LM registry entries marked
``weekly_perf`` (exposed through ``WEEKLY_PERF_PATHS``), so it is generated here
rather than hand-maintained. The ``nightly.models`` list is a hand-picked subset
of large models and is preserved verbatim from the existing file.

Run this whenever the registry changes and ``test_perf_coverage.py`` fails:

    python -m tests.generate_perf_yaml

``tests/test_perf_coverage.py`` asserts that ``weekly.models`` matches
``WEEKLY_PERF_PATHS``; this script is what makes that assertion pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from tests.model_registry import WEEKLY_PERF_PATHS

PERF_YAML_PATH: Path = Path(__file__).resolve().parent / "model_lists" / "perf.yaml"

_HEADER: str = (
    "# Models benchmarked by TEST_TYPE=perf (one GHA matrix leg each); must be "
    "causal-LM paths in tests/model_registry.py.\n"
    "# weekly.models is generated from WEEKLY_PERF_PATHS by "
    "tests/generate_perf_yaml.py -- do not edit by hand.\n"
)


def _load_existing(path: Path) -> dict[str, object]:
    """Load the current perf.yaml (empty mapping if it does not exist)."""
    if not path.is_file():
        return {}
    with open(path) as f:
        data: object = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def build_perf_config() -> dict[str, dict[str, list[str]]]:
    """Build the perf.yaml document: generated ``weekly``, preserved ``nightly``."""
    existing: dict[str, object] = _load_existing(PERF_YAML_PATH)
    nightly: object = existing.get("nightly")
    nightly_models: list[str] = []
    if nightly is not None:
        if not isinstance(nightly, dict):
            raise ValueError(f"{PERF_YAML_PATH} nightly must be a mapping")
        models: object = nightly.get("models")
        if not isinstance(models, list):
            raise ValueError(f"{PERF_YAML_PATH} nightly.models must be a list")
        nightly_models = [str(model) for model in models]

    return {
        "nightly": {"models": nightly_models},
        "weekly": {"models": list(WEEKLY_PERF_PATHS)},
    }


def render_perf_yaml(config: dict[str, dict[str, list[str]]]) -> str:
    """Render the perf.yaml document text (header comment + YAML body)."""
    body: str = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    return _HEADER + body


def main() -> None:
    config: dict[str, dict[str, list[str]]] = build_perf_config()
    text: str = render_perf_yaml(config)
    with open(PERF_YAML_PATH, "w") as f:
        f.write(text)
    rel: str = os.path.relpath(PERF_YAML_PATH, Path.cwd())
    print(f"Wrote {rel} ({len(config['weekly']['models'])} weekly models).")


if __name__ == "__main__":
    main()
