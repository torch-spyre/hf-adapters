"""Excluded models must not cost an adapter its representative (issue #680).

``_select_representative_paths`` picks the smallest model per adapter. If that
model is listed in ``tests/model_lists/exclude.yaml`` the next-smallest eligible
model must take its place, instead of the adapter vanishing after the outer
``_exclude()`` pass.
"""

from tests.model_registry import (
    _EXCLUDED_PATHS,
    CAUSAL_PATHS,
    _select_representative_paths,
)

_MODELS: dict[str, dict] = {
    "small": {"path": "org/small", "adapter": "hf_fake.py", "size": "1B"},
    "medium": {"path": "org/medium", "adapter": "hf_fake.py", "size": "2B"},
    "large": {"path": "org/large", "adapter": "hf_fake.py", "size": "7B"},
    "other": {"path": "org/other", "adapter": "hf_other.py", "size": "3B"},
}


def test_excluded_smallest_falls_back_to_next_smallest() -> None:
    paths: list[str] = _select_representative_paths(
        _MODELS, include_gated=True, excluded=frozenset({"org/small"})
    )
    assert sorted(paths) == ["org/medium", "org/other"]


def test_adapter_with_all_models_excluded_is_dropped() -> None:
    paths: list[str] = _select_representative_paths(
        _MODELS,
        include_gated=True,
        excluded=frozenset({"org/small", "org/medium", "org/large"}),
    )
    assert paths == ["org/other"]


def test_no_exclusions_picks_smallest() -> None:
    paths: list[str] = _select_representative_paths(
        _MODELS, include_gated=True, excluded=frozenset()
    )
    assert sorted(paths) == ["org/other", "org/small"]


def test_causal_paths_contain_no_excluded_models() -> None:
    assert not set(CAUSAL_PATHS) & _EXCLUDED_PATHS
