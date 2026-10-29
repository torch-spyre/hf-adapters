# /// script
# dependencies = [
#   "huggingface-hub",
#   "pyyaml",
# ]
# ///

# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "tests" / "model_registry.py"
EXCLUDE_FILE_PATH = REPO_ROOT / "tests" / "model_lists" / "exclude.yaml"
# Matches every `"path": "org/repo"` entry across the registry's model dicts.
_PATH_ENTRY_RE = re.compile(r'"path":\s*"([^"]+)"')


def _load_excluded_paths() -> frozenset[str]:
    """Load excluded model paths from tests/model_lists/exclude.yaml."""
    if not EXCLUDE_FILE_PATH.is_file():
        raise FileNotFoundError(f"Missing exclusions file: {EXCLUDE_FILE_PATH}")
    with open(EXCLUDE_FILE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{EXCLUDE_FILE_PATH} did not parse to a mapping")

    allowed_sections = {"models", "ppc64le_models"}
    unexpected = set(data.keys()) - allowed_sections
    if unexpected:
        raise ValueError(
            f"{EXCLUDE_FILE_PATH} contains unexpected section(s): {sorted(unexpected)}; "
            f"allowed: {sorted(allowed_sections)}"
        )

    for key, val in data.items():
        if not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
            raise ValueError(
                f"{EXCLUDE_FILE_PATH} '{key}' section must be a list of strings (got {type(val).__name__})"
            )

    excluded: set[str] = set(data.get("models") or [])
    if platform.machine() == "ppc64le":
        excluded |= set(data.get("ppc64le_models") or [])
    return frozenset(excluded)


def _registry_model_paths(
    excluded: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Every HF repo referenced by tests/model_registry.py, excluding architecture-specific exclusions."""
    text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    paths = set(_PATH_ENTRY_RE.findall(text))
    excluded_set = set(_load_excluded_paths()) if excluded is None else set(excluded)
    return sorted(paths - excluded_set)


def build_download_list(extra_models: list[str] | None = None) -> list[str]:
    """Assemble and return the final filtered list of models to download."""
    excluded = set(_load_excluded_paths())
    extra = set(extra_models or [])
    return sorted((set(_registry_model_paths(excluded=excluded)) | extra) - excluded)


def main():
    token = os.getenv("HF_TOKEN")
    force = os.getenv("FORCE_DOWNLOAD") == "true"
    config_file = os.getenv("CACHE_CONFIG_FILE_PATH")
    if not token:
        print("❌ Error: HF_TOKEN secret is not available or empty.")
        sys.exit(1)
    print("the HF_TOKEN is non-empty, length:", len(token))
    if not os.path.exists(config_file):
        print(f"❌ Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    with open(config_file, encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
            extra_models = config.get("models", []) or []
        except Exception as e:
            print(f"❌ Error parsing {config_file}: {e}")
            sys.exit(1)
    models = build_download_list(extra_models=extra_models)
    if not models:
        print(f"⚠️ Warning: No models found in the registry or {config_file}.")
        sys.exit(0)
    print(f"📋 Found {len(models)} model(s) to cache:", models)
    failed_models = []
    for repo_id in models:
        print(f"\n🚀 Processing: {repo_id}...")
        try:
            # snapshot_download automatically reads and uses the HF_HOME env var
            snapshot_download(
                repo_id=repo_id,
                token=token,
                force_download=force,
                ignore_patterns=["*.pt", "*.bin"],
            )
            print(f"✅ Success: {repo_id} cache verified!")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")
            failed_models.append(repo_id)
    if failed_models:
        print(f"\n❌ Pipeline completed with errors. Failed models: {failed_models}")
        sys.exit(1)
    print("\n🎉 All models successfully processed and cached!")


if __name__ == "__main__":
    main()
