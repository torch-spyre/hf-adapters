"""Discover full VLMs under the Hub's image-text-to-text pipeline tag.

Ranking counts are taken after the usual checkpoint/access filters. Adapter
support is deliberately not a filter: unsupported families are weekly results.
Standalone towers are supplied by the registry, not this VLM ranking.
"""

import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from utils.fetch_curated_models_metadata import _create_fetch_metadata, keep_all
from utils.fetch_top_generative_models import keep as keep_generative
from utils.hf_model_catalog import (
    EXPAND_FIELDS,
    RESOURCES_DIR,
    HubRetryExhaustedError,
    Ranking,
    build_catalog,
    catalog_max_workers,
    with_transient_retry,
)

VISION_PIPELINE_TAG = "image-text-to-text"


def _create_fetch(ranking: Ranking) -> Callable[[HfApi, int], list[ModelInfo]]:
    def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
        return with_transient_retry(
            lambda: api.list_models(
                pipeline_tag=VISION_PIPELINE_TAG,
                sort=ranking,
                limit=int(limit * 3.5),
                expand=EXPAND_FIELDS,
            ),
            description=f"list_models[{VISION_PIPELINE_TAG}, sort={ranking}]",
        )

    return _fetch


def keep(model: ModelInfo, token: str | bool) -> tuple[bool, str]:
    """Apply the same weight, access, and remote-code policy as causal models."""
    if model.library_name not in (None, "transformers"):
        return False, "not a Transformers checkpoint"
    return keep_generative(model, token)


def _fetch_vision_models(
    fetcher: Callable[[HfApi, int], list[ModelInfo]],
    keeper: Callable[[ModelInfo, str | bool], tuple[bool, str]],
    limit: int,
    output_csv: Path | str | None = None,
    *,
    fill_ranked: bool = False,
) -> list[dict[str, object]]:
    token: str | bool = os.environ.get("HF_TOKEN") or False
    api = HfApi(token=token)
    decisions: dict[str, tuple[bool, str]] = {}

    def decision(model: ModelInfo) -> tuple[bool, str]:
        if model.id not in decisions:
            try:
                decisions[model.id] = keeper(model, token)
            except HubRetryExhaustedError:
                raise
            except Exception:
                decisions[model.id] = (False, "exception during vision filtering")
        return decisions[model.id]

    def candidates(count: int) -> list[ModelInfo]:
        if not fill_ranked:
            return fetcher(api, count)
        if count <= 0:
            return []
        candidate_limit = count
        while True:
            models = fetcher(api, candidate_limit)
            with ThreadPoolExecutor(max_workers=catalog_max_workers()) as executor:
                accepted = sum(flag for flag, _ in executor.map(decision, models))
            if accepted >= count or len(models) < int(candidate_limit * 3.5):
                return models
            # Vision rankings contain many converted or remote-code checkpoints.
            # Expand until top-K eligible models are found or the Hub is exhausted.
            candidate_limit *= 2
            print(f"Only {accepted}/{count} eligible VLMs; expanding ranking query.")

    return build_catalog(
        fetch_fn=candidates,
        filter_fn=decision,
        limit=limit,
        output_csv=output_csv,
        label="vision",
        allow_millions=True,
        token=token,
    )


def fetch_top_vision_models(
    limit: int,
    output_csv: Path | str | None = None,
    ranking: Ranking = "downloads",
) -> list[dict[str, object]]:
    """Return up to limit eligible VLMs ordered by downloads or trending score."""
    return _fetch_vision_models(
        _create_fetch(ranking), keep, limit, output_csv, fill_ranked=True
    )


def fetch_curated_vision_models_metadata(
    model_ids: list[str], output_csv: Path | str | None = None
) -> list[dict[str, object]]:
    """Fetch named checkpoints without silently filtering requested entries."""
    return _fetch_vision_models(
        _create_fetch_metadata(model_ids), keep_all, len(model_ids), output_csv
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_vision_models(
        limit=limit_, output_csv=RESOURCES_DIR / "top_vision_models.csv"
    )
