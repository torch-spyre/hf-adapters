#!/usr/bin/env python3
"""
Generate dynamic test matrices for GitHub Actions workflows.

This script reads the model registry and generates JSON matrices for different
test suites. It supports manual exclusions passed as command-line arguments.

Usage:
    python generate_test_matrix.py [--exclude MODEL_KEY ...]

Example:
    python generate_test_matrix.py --exclude granite-vision phi4
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from tests/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.model_registry import select_representative_models  # noqa: E402


def generate_matrices(exclude_models=None):
    """
    Generate test matrices from the model registry.

    Args:
        exclude_models: List of model keys to exclude from all matrices

    Returns:
        dict: Dictionary with 'causal', 'embed', and 'combined' matrix lists
    """
    exclude_models = set(exclude_models or [])

    # Get representative models (one per adapter)
    causal_keys, embed_keys = select_representative_models()

    # Apply exclusions
    causal_keys = [k for k in causal_keys if k not in exclude_models]
    embed_keys = [k for k in embed_keys if k not in exclude_models]

    # Combine for jobs that test both types
    combined_keys = causal_keys + embed_keys

    return {
        "causal": causal_keys,
        "embed": embed_keys,
        "combined": combined_keys,
    }


def format_for_github_actions(matrices):
    """
    Format matrices as GitHub Actions JSON output.

    Args:
        matrices: Dictionary with matrix lists

    Returns:
        dict: Dictionary with JSON-stringified matrices
    """
    return {
        "causal_matrix": json.dumps(matrices["causal"]),
        "embed_matrix": json.dumps(matrices["embed"]),
        "combined_matrix": json.dumps(matrices["combined"]),
    }


def write_github_output(outputs):
    """
    Write outputs to GitHub Actions output file.

    Args:
        outputs: Dictionary of output_name -> output_value
    """
    import os

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        # Not running in GitHub Actions, print to stdout for debugging
        print("Not running in GitHub Actions. Output would be:")
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    with open(github_output, "a") as f:
        for key, value in outputs.items():
            # GitHub Actions multiline output format
            f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic test matrices for GitHub Actions"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Model keys to exclude from all matrices (e.g., granite-vision phi4)",
    )

    args = parser.parse_args()

    # Generate matrices
    matrices = generate_matrices(exclude_models=args.exclude)

    # Print summary for workflow logs
    print("Generated test matrices:")
    print(
        f"  Causal models ({len(matrices['causal'])}): {', '.join(matrices['causal'])}"
    )
    print(
        f"  Embedding models ({len(matrices['embed'])}): {', '.join(matrices['embed'])}"
    )
    print(
        f"  Combined ({len(matrices['combined'])}): {', '.join(matrices['combined'])}"
    )

    if args.exclude:
        print(f"\nExcluded models: {', '.join(args.exclude)}")

    # Format for GitHub Actions
    outputs = format_for_github_actions(matrices)

    # Write to GitHub Actions output
    write_github_output(outputs)

    print("\nMatrices written to GitHub Actions output.")


if __name__ == "__main__":
    main()

# Made with Bob
