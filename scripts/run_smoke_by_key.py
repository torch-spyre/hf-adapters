#!/usr/bin/env python3
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

"""Run the smoke test for any model registered in CAUSAL_LM_MODELS by key name.

Bypasses pytest parametrization entirely, so any registry key works — including
models that are not in CAUSAL_PATHS because they share an adapter with a smaller
representative (e.g. granite8b, gemma4_31b).

Usage::

    python scripts/run_smoke_by_key.py granite8b
    python scripts/run_smoke_by_key.py gemma4_moe
    python scripts/run_smoke_by_key.py --list        # show all available keys
"""

from __future__ import annotations

import argparse
import os
import sys

# Make sure ``tests/`` is importable regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
sys.path.insert(0, _REPO_ROOT)

from model_registry import CAUSAL_LM_MODELS  # noqa: E402


def _list_keys() -> None:
    print(f"{'KEY':<30}  {'PATH':<55}  {'SIZE':>6}  {'GATED'}")
    print("-" * 110)
    for key, info in CAUSAL_LM_MODELS.items():
        gated = "yes" if info.get("is_gated") else ""
        kind = f"  [{info['kind']}]" if "kind" in info else ""
        print(f"{key:<30}  {info['path']:<55}  {info['size']:>6}  {gated}{kind}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the smoke test for a specific model registry key."
    )
    parser.add_argument(
        "key",
        nargs="?",
        help="Registry key from CAUSAL_LM_MODELS (e.g. granite8b, gemma4_moe).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all registered keys and exit.",
    )
    args = parser.parse_args()

    if args.list:
        _list_keys()
        return 0

    if not args.key:
        parser.print_help()
        return 1

    key = args.key
    if key not in CAUSAL_LM_MODELS:
        print(f"ERROR: '{key}' is not a registered key in CAUSAL_LM_MODELS.")
        print("Run with --list to see all available keys.")
        return 1

    info = CAUSAL_LM_MODELS[key]

    if info.get("is_gated"):
        hf_token = os.getenv("HF_TOKEN", "")
        if not hf_token:
            print(
                f"WARNING: '{key}' is a gated model ({info['path']}).\n"
                "Set HF_TOKEN in your environment to avoid authentication errors.\n"
            )

    if info.get("kind") == "dspark_draft":
        print(
            f"WARNING: '{key}' is a DSpark draft model — it uses a block-propose\n"
            "path rather than generate(). The smoke test may not be meaningful.\n"
        )

    model_path = info["path"]
    print(f"Key:   {key}")
    print(f"Path:  {model_path}")
    print(f"Size:  {info['size']}")
    print()

    # Import here so we don't pay the cost unless we're actually running.
    from tests.spyre.test_e2e_smoke_spyre import run_smoke_test  # noqa: E402

    result = run_smoke_test(model_path)

    print("\n## Smoke Test Result\n")
    print("| Model | Status | Tokens | Generated Text | Load (s) | Gen (s) |")
    print("|-------|--------|--------|----------------|----------|---------|")
    print(
        f"| {result['model']} | {result['status']} | {result['tokens']} "
        f"| {result['text']!r} | {result['load_s']:.1f} | {result['gen_s']:.1f} |"
    )

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
