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

"""
Test that all adapter files in hf_adapters/ are registered in model_registry.py.

This test ensures that every hf_*.py adapter file has at least one corresponding
entry in either CAUSAL_LM_MODELS or EMBEDDING_MODELS dictionaries.
"""

from pathlib import Path

from tests.model_registry import (
    CAUSAL_LM_MODELS,
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    VISION_MODELS,
)


def get_adapter_files():
    """
    Collect all hf_*.py adapter files from the hf_adapters directory.

    Excludes hf_common.py which is a shared utilities module, not an adapter.

    Returns:
        set: Set of adapter filenames (e.g., {'hf_granite.py', 'hf_llama.py', ...})
    """
    # Get the hf_adapters directory path
    hf_adapters_dir = Path(__file__).parent.parent / "hf_adapters"

    # Collect all files matching hf_*.py pattern, excluding hf_common.py
    adapter_files = set()
    for file_path in hf_adapters_dir.glob("hf_*.py"):
        # Skip hf_common.py as it's a utilities module, not an adapter
        if file_path.name != "hf_common.py":
            adapter_files.add(file_path.name)

    return adapter_files


def get_registered_adapters():
    """
    Extract all adapter filenames referenced in the model registries.

    Returns:
        set: Set of adapter filenames referenced in CAUSAL_LM_MODELS or EMBEDDING_MODELS
    """
    registered_adapters = set()

    # Collect adapters from CAUSAL_LM_MODELS
    for model_info in CAUSAL_LM_MODELS.values():
        adapter = model_info.get("adapter")
        if adapter:
            registered_adapters.add(adapter)

    # Collect adapters from EMBEDDING_MODELS
    for model_info in EMBEDDING_MODELS.values():
        adapter = model_info.get("adapter")
        if adapter:
            registered_adapters.add(adapter)

    # Collect adapters from VISION_MODELS (vision towers + multimodal VLMs)
    for model_info in VISION_MODELS.values():
        adapter = model_info.get("adapter")
        if adapter:
            registered_adapters.add(adapter)

    # Collect adapters from RERANKER_MODELS
    for model_info in RERANKER_MODELS.values():
        adapter = model_info.get("adapter")
        if adapter:
            registered_adapters.add(adapter)

    return registered_adapters


def test_all_adapters_are_registered():
    """
    Test that every hf_*.py file in hf_adapters/ is registered in model_registry.py.

    This ensures that:
    1. All adapter files are discoverable
    2. Each adapter has at least one test model
    3. No orphaned adapters exist without test coverage
    """
    # Get all adapter files from the filesystem
    adapter_files = get_adapter_files()

    # Get all adapters referenced in the registries
    registered_adapters = get_registered_adapters()

    # Find adapters that exist as files but are not registered
    unregistered_adapters = adapter_files - registered_adapters

    # Assert that all adapter files are registered
    assert not unregistered_adapters, (
        f"The following adapter files exist in hf_adapters/ but are not registered "
        f"in CAUSAL_LM_MODELS or EMBEDDING_MODELS in tests/model_registry.py:\n"
        f"{sorted(unregistered_adapters)}\n\n"
        f"Please add at least one model entry for each adapter to ensure test coverage."
    )


def test_no_invalid_adapter_references():
    """
    Test that all adapter references in model_registry.py point to existing files.

    This is the inverse check: ensure no broken references exist in the registry.
    """
    # Get all adapter files from the filesystem
    adapter_files = get_adapter_files()

    # Get all adapters referenced in the registries
    registered_adapters = get_registered_adapters()

    # Find registered adapters that don't have corresponding files
    invalid_references = registered_adapters - adapter_files

    # Assert that all registered adapters have corresponding files
    assert not invalid_references, (
        f"The following adapters are referenced in model_registry.py but do not "
        f"exist as files in hf_adapters/:\n"
        f"{sorted(invalid_references)}\n\n"
        f"Please either create the adapter files or remove the invalid references."
    )


def test_adapter_coverage_details():
    """
    Provide detailed information about adapter coverage for debugging.

    This test always passes but prints useful information about the coverage.
    """
    adapter_files = get_adapter_files()
    registered_adapters = get_registered_adapters()

    # Count how many models use each adapter
    adapter_usage = {}
    for model_info in (
        list(CAUSAL_LM_MODELS.values())
        + list(EMBEDDING_MODELS.values())
        + list(VISION_MODELS.values())
    ):
        adapter = model_info.get("adapter")
        if adapter:
            adapter_usage[adapter] = adapter_usage.get(adapter, 0) + 1

    print("\n" + "=" * 70)
    print("ADAPTER COVERAGE REPORT")
    print("=" * 70)
    print(f"\nTotal adapter files found: {len(adapter_files)}")
    print(f"Total adapters registered: {len(registered_adapters)}")
    print("\nAdapter usage breakdown:")
    for adapter in sorted(adapter_files):
        count = adapter_usage.get(adapter, 0)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {adapter}: {count} model(s)")
    print("=" * 70)


# Made with Bob
