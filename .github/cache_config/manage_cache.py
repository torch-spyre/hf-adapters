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

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "model_registry.py"
)
VISION_HELPERS_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "_vision_helpers.py"
)
# Matches every `"path": "org/repo"` entry across the registry's model dicts.
_PATH_ENTRY_RE = re.compile(r'"path":\s*"([^"]+)"')
# Matches each dict literal that carries `"repo_type": "dataset"`, so the sample
# images the vision tests download are derived from the tests themselves.
_DATASET_DICT_RE = re.compile(r'\{[^{}]*?"repo_type":\s*"dataset"[^{}]*?\}', re.S)
_REPO_ID_RE = re.compile(r'"repo_id":\s*"([^"]+)"')
_FILENAME_RE = re.compile(r'"filename":\s*"([^"]+)"')


def _registry_model_paths() -> list[str]:
    """Every HF repo referenced by tests/model_registry.py, so the cache can't drift from what CI actually tests."""
    text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    return sorted(set(_PATH_ENTRY_RE.findall(text)))


def _dataset_files() -> list[tuple[str, str]]:
    """Every (repo_id, filename) the vision tests fetch from a dataset repo.

    These are not models, so snapshot_download's model-only default never
    cached them. With HF_HUB_OFFLINE=1 on the test runs, an uncached one is a
    hard failure rather than a download, so warm them here.
    """
    text = VISION_HELPERS_PATH.read_text(encoding="utf-8")
    pairs = set()
    for block in _DATASET_DICT_RE.findall(text):
        repo_id = _REPO_ID_RE.search(block)
        filename = _FILENAME_RE.search(block)
        if repo_id and filename:
            pairs.add((repo_id.group(1), filename.group(1)))
    return sorted(pairs)


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
    models = sorted(set(_registry_model_paths()) | set(extra_models))
    if not models:
        # Warn and carry on rather than exit: the dataset files below are a
        # separate source and still need warming, and an empty model list is
        # itself a symptom (a registry parse that found nothing) rather than a
        # reason to skip the rest of the job.
        print(f"⚠️ Warning: No models found in the registry or {config_file}.")
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
    # Tallied separately from failed_models: a model that is gated or renamed
    # upstream fails every night, and lumping these together would let that
    # standing noise hide a missing sample image, which fails the vision tests
    # outright under HF_HUB_OFFLINE=1.
    dataset_files = _dataset_files()
    failed_datasets = []
    print(f"\n📋 Found {len(dataset_files)} dataset file(s) to cache:", dataset_files)
    for repo_id, filename in dataset_files:
        print(f"\n🚀 Processing dataset file: {repo_id}/{filename}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                force_download=force,
            )
            print(f"✅ Success: {repo_id}/{filename} cache verified!")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}/{filename}: {e}")
            failed_datasets.append(f"{repo_id}/{filename}")
    if failed_models:
        print(f"\n⚠️ Models that failed to cache: {failed_models}")
    if failed_datasets:
        print(f"\n❌ Dataset files that failed to cache: {failed_datasets}")
    if failed_models or failed_datasets:
        sys.exit(1)
    print("\n🎉 All models and dataset files successfully processed and cached!")


if __name__ == "__main__":
    main()
