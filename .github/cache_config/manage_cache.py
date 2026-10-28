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
from huggingface_hub import snapshot_download

MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "model_registry.py"
)
# Matches every `"path": "org/repo"` entry across the registry's model dicts.
_PATH_ENTRY_RE = re.compile(r'"path":\s*"([^"]+)"')


def _registry_model_paths() -> list[str]:
    """Every HF repo referenced by tests/model_registry.py, so the cache can't drift from what CI actually tests."""
    text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    return sorted(set(_PATH_ENTRY_RE.findall(text)))


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
