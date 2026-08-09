"""Fetch the top generative models from Hugging Face, ranked by downloads."""

import os
import sys
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from utils.fetch_curated_models_metadata import _create_fetch_metadata, keep_all
from utils.hf_model_catalog import (
    EXPAND_FIELDS,
    RESOURCES_DIR,
    build_catalog,
    contains_remote_code,
    has_loadable_weights,
    is_baseline_keep,
    with_transient_retry,
)
from utils.utilities import ts


def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
    """Top text-generation models by downloads (over-fetched to absorb the
    GGUF/MLX entries dropped by the filter).

    Retries transient 5xx gateway errors with exponential backoff via
    ``with_transient_retry``; permanent failures propagate.
    """
    print(f"{ts()} Fetching top {limit} text-generation models by downloads...")
    return with_transient_retry(
        lambda: api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=int(limit * 2),
            expand=EXPAND_FIELDS,
        ),
        description="list_models[text-generation]",
    )


def keep(model: ModelInfo, token: str | bool) -> bool:
    """Keep predicate for the generative fetcher.

    Ordering matters: the cheap metadata-only checks run first.
    """
    if not is_baseline_keep(model):
        return False
    if model.gated:
        return False
    if not has_loadable_weights(model):
        return False
    if contains_remote_code(model):
        return False
    return True


def fetch_generative_models(
    fetcher: Callable[[HfApi, int], list[ModelInfo]],
    keeper: Callable[[ModelInfo, str | bool], bool],
    limit: int,
    output_csv: Path | str | None = None,
) -> list[dict]:
    token: str | bool = os.environ.get("HF_TOKEN") or False
    api: HfApi = HfApi(token=token)
    return build_catalog(
        fetch_fn=lambda lim: fetcher(api, lim),
        filter_fn=lambda m: keeper(m, token),
        limit=limit,
        output_csv=output_csv,
        label="generative",
        token=token,
    )


def fetch_top_generative_models(
    limit: int, output_csv: Path | str | None = None
) -> list[dict[str, object]]:
    return fetch_generative_models(
        fetcher=_fetch, keeper=keep, limit=limit, output_csv=output_csv
    )


def fetch_curated_generative_models_metadata(
    model_ids: list[str], output_csv: Path | str | None = None
) -> list[dict[str, object]]:
    return fetch_generative_models(
        fetcher=_create_fetch_metadata(model_ids),
        keeper=keep_all,
        limit=len(model_ids),
        output_csv=output_csv,
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_generative_models(
        limit=limit_, output_csv=RESOURCES_DIR / "top_generative_models.csv"
    )
