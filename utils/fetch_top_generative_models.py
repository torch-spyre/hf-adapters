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
            limit=int(limit * 2.5),
            expand=EXPAND_FIELDS,
        ),
        description="list_models[text-generation]",
    )


def keep(model: ModelInfo, token: str | bool) -> tuple[bool, str]:
    """Keep predicate for the generative fetcher.

    Ordering matters: the cheap metadata-only checks run first.
    Returns a (keep, reason) tuple where reason describes why the model was
    rejected (empty string when kept).
    """
    try:
        baseline_keep, baseline_reason = is_baseline_keep(model)
        if not baseline_keep:
            return False, baseline_reason
    except Exception:
        return False, "exception during is_baseline_keep"
    try:
        if model.gated:
            return False, "model is gated"
    except Exception:
        return False, "exception during model.gated"
    try:
        if not has_loadable_weights(model):
            return False, "no loadable weights"
    except Exception:
        return False, "exception during has_loadable_weights"
    try:
        if contains_remote_code(model):
            return False, "requires trust_remote_code"
    except Exception:
        return False, "exception during contains_remote_code"
    return True, ""


def _fetch_generative_models(
    fetcher: Callable[[HfApi, int], list[ModelInfo]],
    keeper: Callable[[ModelInfo, str | bool], tuple[bool, str]],
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
    return _fetch_generative_models(
        fetcher=_fetch, keeper=keep, limit=limit, output_csv=output_csv
    )


def fetch_curated_generative_models_metadata(
    model_ids: list[str], output_csv: Path | str | None = None
) -> list[dict[str, object]]:
    return _fetch_generative_models(
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
