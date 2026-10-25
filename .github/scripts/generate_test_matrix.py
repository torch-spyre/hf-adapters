#!/usr/bin/env python3
"""
Generate dynamic test matrices for GitHub Actions workflows.

This script reads the model registry and generates JSON matrices for different
test suites. It supports manual exclusions and an allowlist filter passed as
command-line arguments.

Usage:
    python generate_test_matrix.py [--exclude MODEL_PATH ...] [--only MODEL_PATH ...] [--strict-perf]

Example:
    python generate_test_matrix.py --exclude granite-vision phi4
    python generate_test_matrix.py --only Qwen/Qwen3-0.6B ministral/Ministral-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from tests/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import tests.model_registry  # noqa: E402


def generate_matrices(
    exclude_models: list[str] | None = None,
    only_models: list[str] | None = None,
    strict_perf: bool = False,
) -> dict[str, list[str]]:
    """
    Generate test matrices from the model registry.

    Args:
        exclude_models: List of model paths to exclude from all matrices
        only_models: If non-empty, restrict all matrices to just these model
            paths (applied after exclusions). Empty/None = no restriction.

    Returns:
        dict: Dictionary containing the per-suite and combined matrix lists
    """
    exclude_models = set(exclude_models or [])
    only_models = set(only_models or [])

    # Every category has a smallest-per-adapter representative list (default)
    # and an unreduced "every registered path" list (ALL_*_PATHS, used by
    # --only so a caller can target a non-representative checkpoint, e.g. a
    # larger model sharing an adapter with a smaller default). Adding a new
    # category later just means adding a row here.
    registry = tests.model_registry
    categories = {
        "causal": (registry.CAUSAL_PATHS, registry.ALL_CAUSAL_PATHS),
        "multicard_smoke": (
            registry.MULTICARD_SMOKE_PATHS,
            registry.MULTICARD_SMOKE_PATHS,
        ),
        "embed": (registry.EMBED_PATHS, registry.ALL_EMBED_PATHS),
        "vision": (registry.VISION_PATHS, registry.ALL_VISION_PATHS),
        "multimodal_embed_compare": (
            registry.MULTIMODAL_EMBED_PATHS,
            registry.ALL_MULTIMODAL_EMBED_PATHS,
        ),
        "masked_lm": (registry.MASKED_LM_PATHS, registry.ALL_MASKED_LM_PATHS),
        "question_answering": (
            registry.QUESTION_ANSWERING_PATHS,
            registry.ALL_QUESTION_ANSWERING_PATHS,
        ),
        "reranker": (registry.RERANKER_PATHS, registry.ALL_RERANKER_PATHS),
        "seq_classification": (
            registry.SEQ_CLASSIFICATION_PATHS,
            registry.ALL_SEQ_CLASSIFICATION_PATHS,
        ),
        "token_classification": (
            registry.TOKEN_CLASSIFICATION_PATHS,
            registry.ALL_TOKEN_CLASSIFICATION_PATHS,
        ),
    }

    if only_models:
        available_paths = {
            path for _, all_paths in categories.values() for path in all_paths
        }
        unavailable_paths = sorted(only_models - available_paths)
        if unavailable_paths:
            formatted_paths = "\n  - ".join(unavailable_paths)
            raise ValueError(
                "--only contains model paths that are not available for matrix "
                f"generation (unregistered or excluded):\n  - {formatted_paths}"
            )

    paths = {}
    for name, (representative_paths, all_paths) in categories.items():
        source = all_paths if only_models else representative_paths
        selected = [p for p in source if p not in exclude_models]
        if only_models:
            selected = [p for p in selected if p in only_models]
        paths[name] = selected

    # Feeds spyre-load-tests' matrix: test_load_spyre.py's five model_path suites.
    combined_paths = (
        paths["causal"]
        + paths["embed"]
        + paths["masked_lm"]
        + paths["question_answering"]
        + paths["seq_classification"]
        + paths["token_classification"]
    )

    # Perf is generative-only: spyre-perf-suite's inference.py imports
    # AutoSpyreModelForCausalLM and measures prefill+decode, so a non-causal path has no
    # decode phase. Dropping one silently emptied perf_matrix -> legs skipped -> SUCCESS on
    # a non-gating cell. Only perf tiers pass strict_perf; other tiers want these paths.
    if strict_perf and only_models:
        unbenchmarkable = sorted(set(only_models) - set(paths["causal"]))
        if unbenchmarkable:
            formatted_paths = "\n  - ".join(unbenchmarkable)
            raise ValueError(
                "perf can only benchmark causal-LM paths (spyre-perf-suite measures "
                "prefill+decode); these are registered but not causal-LM:\n  - "
                f"{formatted_paths}"
            )
        if not paths["causal"]:
            raise ValueError(
                "perf resolved an empty model list; every leg would be skipped"
            )

    return {
        "causal": paths["causal"],
        "multicard_smoke": paths["multicard_smoke"],
        "embed": paths["embed"],
        "vision": paths["vision"],
        "multimodal_embed_compare": paths["multimodal_embed_compare"],
        "masked_lm": paths["masked_lm"],
        "question_answering": paths["question_answering"],
        "combined": combined_paths,
        "reranker": paths["reranker"],
        "seq_classification": paths["seq_classification"],
        "token_classification": paths["token_classification"],
        # Perf only benchmarks an explicit list (tests/model_lists/perf.yaml), never the whole registry.
        "perf": paths["causal"] if only_models else [],
    }


def format_for_github_actions(matrices: dict[str, list[str]]) -> dict[str, str]:
    """
    Format matrices as GitHub Actions JSON output.

    Args:
        matrices: Dictionary with matrix lists

    Returns:
        dict: Dictionary with JSON-stringified matrices
    """
    return {
        "causal_matrix": json.dumps(matrices["causal"]),
        "multicard_smoke_matrix": json.dumps(matrices["multicard_smoke"]),
        "embed_matrix": json.dumps(matrices["embed"]),
        "vision_matrix": json.dumps(matrices["vision"]),
        "multimodal_embed_compare_matrix": json.dumps(
            matrices["multimodal_embed_compare"]
        ),
        "masked_lm_matrix": json.dumps(matrices["masked_lm"]),
        "question_answering_matrix": json.dumps(matrices["question_answering"]),
        "combined_matrix": json.dumps(matrices["combined"]),
        "reranker_matrix": json.dumps(matrices["reranker"]),
        "seq_classification_matrix": json.dumps(matrices["seq_classification"]),
        "token_classification_matrix": json.dumps(matrices["token_classification"]),
        "perf_matrix": json.dumps(matrices["perf"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dynamic test matrices for GitHub Actions"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Model keys to exclude from all matrices (e.g., granite-vision phi4)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="If given, restrict all matrices to just these model paths "
        "(e.g., Qwen/Qwen3-0.6B ministral/Ministral-3B-Instruct)",
    )

    parser.add_argument(
        "--strict-perf",
        action="store_true",
        help="Fail instead of dropping non-causal paths from the perf matrix "
        "(set by the workflow when test_type is perf)",
    )

    args = parser.parse_args()

    # Generate matrices
    matrices = generate_matrices(
        exclude_models=args.exclude,
        only_models=args.only,
        strict_perf=args.strict_perf,
    )

    # Print summary for workflow logs
    print("Generated test matrices:")
    print(
        f"  Causal models ({len(matrices['causal'])}): {', '.join(matrices['causal'])}"
    )
    print(
        f"  Multicard-smoke models ({len(matrices['multicard_smoke'])}): "
        f"{', '.join(matrices['multicard_smoke'])}"
    )
    print(
        f"  Embedding models ({len(matrices['embed'])}): {', '.join(matrices['embed'])}"
    )
    print(
        f"  Vision models ({len(matrices['vision'])}): {', '.join(matrices['vision'])}"
    )
    print(
        f"  Multimodal embedding models ({len(matrices['multimodal_embed_compare'])}): {', '.join(matrices['multimodal_embed_compare'])}"
    )
    print(
        f"  Masked-LM models ({len(matrices['masked_lm'])}): {', '.join(matrices['masked_lm'])}"
    )
    print(
        f"  Question-answering models ({len(matrices['question_answering'])}): "
        f"{', '.join(matrices['question_answering'])}"
    )
    print(
        f"  Combined ({len(matrices['combined'])}): {', '.join(matrices['combined'])}"
    )
    print(
        f"  Reranker models ({len(matrices['reranker'])}): {', '.join(matrices['reranker'])}"
    )
    print(
        f"  Seq-classification models ({len(matrices['seq_classification'])}): "
        f"{', '.join(matrices['seq_classification'])}"
    )
    print(
        f"  Token-classification models ({len(matrices['token_classification'])}): "
        f"{', '.join(matrices['token_classification'])}"
    )
    print(f"  Perf models ({len(matrices['perf'])}): {', '.join(matrices['perf'])}")

    if args.exclude:
        print(f"\nExcluded models: {', '.join(args.exclude)}")
    if args.only:
        print(f"\nRestricted to models: {', '.join(args.only)}")

    # Format for GitHub Actions
    outputs = format_for_github_actions(matrices)

    # Write to GitHub Actions output
    write_github_output(outputs)

    print("\nMatrices written to GitHub Actions output.")


if __name__ == "__main__":
    main()
