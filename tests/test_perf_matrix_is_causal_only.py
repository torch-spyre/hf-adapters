"""The perf matrix must never silently drop a model it cannot benchmark.

spyre-perf-suite's generative harness imports AutoSpyreModelForCausalLM and measures
prefill+decode (utils/hf_adapters/models/inference.py), so an encoder or embedding path has
no decode phase. generate_matrices() restricts "perf" to causal paths; without
strict_perf a non-causal entry vanished from the matrix, and an empty perf_matrix makes
spyre-perf-tests' `if:` skip every leg, which reads as SUCCESS on a non-gating perf cell.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

from tests.model_registry import (
    ALL_CAUSAL_PATHS,
    CAUSAL_PATHS,
    EMBED_PATHS,
    EMBEDDING_MODELS,
    MASKED_LM_PATHS,
    MULTIMODAL_EMBED_PATHS,
    MULTIMODAL_EMBEDDING_MODELS,
    QUESTION_ANSWERING_PATHS,
    TOKEN_CLASSIFICATION_PATHS,
    VISION_MODELS,
    VISION_PATHS,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "generate_test_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location("generate_test_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
generate_test_matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_test_matrix)

# The weekly perf scope in spyre-perf-suite#229: one representative per adapter across
# generative, embedding and encoder tasks.
_WEEKLY_SCOPE = sorted(
    set(CAUSAL_PATHS)
    | set(EMBED_PATHS)
    | set(TOKEN_CLASSIFICATION_PATHS)
    | set(QUESTION_ANSWERING_PATHS)
    | set(MASKED_LM_PATHS)
    | set(VISION_PATHS)
)


def test_multimodal_registry_is_part_of_embedding_not_vision() -> None:
    assert MULTIMODAL_EMBEDDING_MODELS
    assert MULTIMODAL_EMBEDDING_MODELS.keys() <= EMBEDDING_MODELS.keys()
    assert MULTIMODAL_EMBEDDING_MODELS.keys().isdisjoint(VISION_MODELS)


@pytest.mark.parametrize("checkpoint", MULTIMODAL_EMBED_PATHS)
def test_multimodal_models_use_embedding_and_load_matrices(checkpoint: str) -> None:
    matrices: dict[str, list[str]] = generate_test_matrix.generate_matrices(
        only_models=[checkpoint]
    )
    assert matrices["embed"] == [checkpoint]
    assert matrices["combined"] == [checkpoint]
    assert matrices["vision"] == []
    assert matrices["perf"] == []
    assert "clip" not in matrices
    outputs: dict[str, str] = generate_test_matrix.format_for_github_actions(matrices)
    assert "clip_matrix" not in outputs


def test_causal_only_perf_list_is_accepted() -> None:
    """Positive control: the locked nightly shape resolves one leg per model."""
    only = set(CAUSAL_PATHS[:3])
    matrices = generate_test_matrix.generate_matrices(
        only_models=only, strict_perf=True
    )
    assert set(matrices["perf"]) == only


def test_non_causal_perf_entry_raises_instead_of_being_dropped() -> None:
    """Negative control: the exact weekly set that silently lost 18 of 44 models."""
    with pytest.raises(ValueError, match="only benchmark causal-LM paths"):
        generate_test_matrix.generate_matrices(
            only_models=set(_WEEKLY_SCOPE), strict_perf=True
        )


def test_the_dropped_models_are_named_in_the_error() -> None:
    """The error has to say which paths to move, not just that something is wrong."""
    dropped = sorted(set(_WEEKLY_SCOPE) - set(CAUSAL_PATHS))
    assert dropped, "registry has no non-causal representative to test with"
    with pytest.raises(ValueError) as excinfo:
        generate_test_matrix.generate_matrices(
            only_models=set(_WEEKLY_SCOPE), strict_perf=True
        )
    for path in dropped:
        assert path in str(excinfo.value)


def test_an_all_non_causal_perf_list_does_not_resolve_empty() -> None:
    """Without the guard this gave perf_matrix == [] and a green run that measured nothing."""
    embedding_only = set(EMBED_PATHS)
    assert embedding_only
    unguarded = generate_test_matrix.generate_matrices(only_models=embedding_only)
    assert unguarded["perf"] == [], "precondition: these are dropped from perf"
    with pytest.raises(ValueError):
        generate_test_matrix.generate_matrices(
            only_models=embedding_only, strict_perf=True
        )


def test_other_tiers_still_accept_non_causal_paths() -> None:
    """strict_perf is opt-in: one generate-matrix job serves every tier's 12 matrices."""
    only = set(EMBED_PATHS)
    matrices = generate_test_matrix.generate_matrices(only_models=only)
    assert set(matrices["embed"]) == only
    assert matrices["perf"] == []


def test_no_only_models_leaves_perf_empty_without_raising() -> None:
    """A registry-wide run (no --only) has no perf legs by design and must not fail."""
    matrices = generate_test_matrix.generate_matrices(strict_perf=True)
    assert matrices["perf"] == []


def test_every_perf_list_file_is_benchmarkable() -> None:
    """Every model in a committed perf list must be one perf can actually measure.

    This is the test-time counterpart of the workflow's --strict-perf: it catches a
    non-causal entry when the list changes, rather than at dispatch. Reads whatever perf
    lists exist, and handles both the flat `models:` shape and the nested
    `daily:`/`weekly:` sections, so it keeps working as the file's shape evolves.
    """
    list_dir = Path(__file__).resolve().parent / "model_lists"
    perf_lists = sorted(list_dir.glob("perf*.yaml"))
    assert perf_lists, f"no perf list files found in {list_dir}"

    # ALL_CAUSAL_PATHS, not CAUSAL_PATHS: a perf list names specific large checkpoints
    # (granite-4.1-8b, Ministral-3-14B) that are registered causal-LMs but are not their
    # adapter's smallest representative. This is the set --only resolves against.
    causal = set(ALL_CAUSAL_PATHS)
    offenders: dict[str, list[str]] = {}
    for path in perf_lists:
        with open(path) as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict), f"{path.name} did not parse to a mapping"
        # Flat `models:`, or one list per schedule section (daily:/weekly:/...).
        sections = {"": data} if isinstance(data.get("models"), list) else data
        for key, section in sections.items():
            if not isinstance(section, dict) or not isinstance(
                section.get("models"), list
            ):
                continue
            bad = sorted({str(m) for m in section["models"]} - causal)
            if bad:
                offenders[f"{path.name}:{key or 'models'}"] = bad

    assert not offenders, (
        "perf benchmarks prefill+decode, so every entry must be a causal-LM path in "
        f"tests/model_registry.py; these are not: {offenders}"
    )
