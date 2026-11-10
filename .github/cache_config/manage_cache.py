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
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "tests" / "model_registry.py"
DEFAULT_CACHE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "hf_models_and_datasets.yaml"
)
EXCLUDE_FILE_PATH = REPO_ROOT / "tests" / "model_lists" / "exclude.yaml"
# Matches every `"path": "org/repo"` entry across the registry's model dicts.
_PATH_ENTRY_RE = re.compile(r'"path":\s*"([^"]+)"')


def _load_excluded_paths() -> frozenset[str]:
    """Load globally excluded model paths ('models' section) from tests/model_lists/exclude.yaml.

    This script only runs on the x86 CI/CD cluster, so arch-specific sections
    (e.g. 'ppc64le_models') are intentionally ignored here.
    """
    if not EXCLUDE_FILE_PATH.is_file():
        raise FileNotFoundError(f"Missing exclusions file: {EXCLUDE_FILE_PATH}")
    with open(EXCLUDE_FILE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{EXCLUDE_FILE_PATH} did not parse to a mapping")

    if "models" in data:
        val = data["models"]
        if not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
            raise ValueError(
                f"{EXCLUDE_FILE_PATH} 'models' section must be a list of strings (got {type(val).__name__})"
            )
    return frozenset(data.get("models") or [])


def _registry_model_paths(
    excluded: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Every HF repo referenced by tests/model_registry.py, minus globally excluded models."""
    text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    paths = set(_PATH_ENTRY_RE.findall(text))
    excluded_set = set(_load_excluded_paths()) if excluded is None else set(excluded)
    return sorted(paths - excluded_set)


def _read_config(config_file: str | Path) -> dict[str, Any]:
    with open(config_file, encoding="utf-8") as f:
        config: Any = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Cache configuration must be a mapping")
    return config


def _config_extra_models(config_file: str | Path) -> list[str]:
    models: Any = _read_config(config_file).get("models", []) or []
    if not isinstance(models, list) or any(
        not isinstance(model, str) or not model for model in models
    ):
        raise ValueError("models must be a list of non-empty repository IDs")
    return models


def build_download_list(extra_models: list[str] | None = None) -> list[str]:
    """Assemble and return the final filtered list of models to download."""
    excluded = set(_load_excluded_paths())
    extra = set(extra_models or [])
    return sorted((set(_registry_model_paths(excluded=excluded)) | extra) - excluded)


def cache_warmed_model_paths(
    config_file: str | Path = DEFAULT_CACHE_CONFIG_PATH,
) -> list[str]:
    """Every repo this job keeps warm; weekly_test.py never deletes their weights."""
    extra_models = _config_extra_models(config_file)
    return build_download_list(extra_models=extra_models)


def cache_warmed_dataset_files(
    config_file: str | Path = DEFAULT_CACHE_CONFIG_PATH,
) -> list[tuple[str, str]]:
    """Exact (dataset repo, filename) pairs needed by offline tests."""
    datasets: Any = _read_config(config_file).get("datasets", []) or []
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a list of repositories with files")
    result: list[tuple[str, str]] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset must declare repo_id and files")
        repo_id: Any = dataset.get("repo_id")
        filenames: Any = dataset.get("files")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("Each dataset must have a non-empty repo_id")
        if not isinstance(filenames, list) or not filenames:
            raise ValueError(f"Dataset {repo_id}: files must be a non-empty list")
        for filename in filenames:
            if not isinstance(filename, str) or not filename:
                raise ValueError(
                    f"Dataset {repo_id}: filenames must be non-empty strings"
                )
            result.append((repo_id, filename))
    return result


def cache_dataset_files(
    dataset_files: list[tuple[str, str]],
    *,
    token: str | None,
    force_download: bool = False,
) -> list[str]:
    """Download named files into HF_HOME and verify cache-only resolution.

    Return failed artifact names so the caller can finish other downloads and
    fail the job. Use the default revision, matching the tests' Hub lookups.
    """
    failures: list[str] = []
    for repo_id, filename in dataset_files:
        artifact = f"dataset:{repo_id}/{filename}"
        print(f"\n🚀 Processing: {artifact}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                force_download=force_download,
            )
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                local_files_only=True,
            )
            print(f"✅ Success: {artifact} available offline!")
        except Exception as e:
            print(f"❌ Failed to cache {artifact}: {e}")
            failures.append(artifact)
    return failures


def main() -> None:
    token = os.getenv("HF_TOKEN")
    force = os.getenv("FORCE_DOWNLOAD") == "true"
    config_file = Path(os.getenv("CACHE_CONFIG_FILE_PATH") or DEFAULT_CACHE_CONFIG_PATH)
    if not token:
        print("❌ Error: HF_TOKEN secret is not available or empty.")
        sys.exit(1)
    print("the HF_TOKEN is non-empty, length:", len(token))
    if not os.path.exists(config_file):
        print(f"❌ Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    try:
        models = cache_warmed_model_paths(config_file)
        dataset_files = cache_warmed_dataset_files(config_file)
    except Exception as e:
        print(f"❌ Error parsing {config_file}: {e}")
        sys.exit(1)
    if not models and not dataset_files:
        print(f"⚠️ Warning: No artifacts found in the registry or {config_file}.")
        sys.exit(0)
    print(f"📋 Found {len(models)} model(s) to cache:", models)
    print(f"📋 Found {len(dataset_files)} dataset file(s) to cache:")
    for repo_id, filename in dataset_files:
        print(f"  - {repo_id}/{filename}")
    failed_artifacts = cache_dataset_files(
        dataset_files, token=token, force_download=force
    )
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
            failed_artifacts.append(repo_id)
    if failed_artifacts:
        print(
            f"\n❌ Pipeline completed with errors. Failed artifacts: {failed_artifacts}"
        )
        sys.exit(1)
    print("\n🎉 All models and dataset files successfully processed and cached!")


if __name__ == "__main__":
    main()
