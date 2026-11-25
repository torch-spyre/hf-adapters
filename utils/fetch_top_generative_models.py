"""Fetch the top generative models from Hugging Face, ranked by downloads or trending score."""

import os
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Callable

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from transformers import AutoConfig, PretrainedConfig

from utils.fetch_curated_models_metadata import _create_fetch_metadata, keep_all
from utils.hf_model_catalog import (
    EXPAND_FIELDS,
    RESOURCES_DIR,
    HubRetryExhaustedError,
    Ranking,
    _guarded,
    _retry_hub_call,
    build_catalog,
    has_loadable_weights,
    is_baseline_keep,
    with_transient_retry,
)
from utils.utilities import ts


def _create_fetch(ranking: Ranking) -> Callable[[HfApi, int], list[ModelInfo]]:
    """Build a ``_fetch``-shaped callable that ranks text-generation models by *ranking*."""

    def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
        """Top text-generation models by *ranking* (over-fetched to absorb the
        GGUF/MLX entries dropped by the filter).

        Retries HTTP 429 and transient gateway errors via ``with_transient_retry``;
        permanent failures propagate.
        """
        print(f"{ts()} Fetching top {limit} text-generation models by {ranking}...")
        return with_transient_retry(
            lambda: api.list_models(
                pipeline_tag="text-generation",
                sort=ranking,
                limit=int(limit * 3.5),
                expand=EXPAND_FIELDS,
            ),
            description=f"list_models[text-generation, sort={ranking}]",
        )

    return _fetch


def _keep_generation_config(
    model: ModelInfo,
    token: str | bool,
    config_types: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Reject remote-code and encoder-only configs without loading model weights.

    Use the same adapter selection and encoder-only flag as the causal-LM loader.
    Do not resolve the adapter through ``resolve_adapter_module`` here: dimension
    validation and unsupported-adapter verdicts still belong to the worker.
    """
    from hf_adapters.auto_spyre_model import select_adapter_module

    try:
        config: PretrainedConfig = _retry_hub_call(
            lambda: AutoConfig.from_pretrained(
                model.id, token=token, trust_remote_code=False
            ),
            description=f"config[{model.id}]",
        )
    except (ValueError, OSError):
        return False, "requires trust_remote_code"

    if config_types is not None:
        config_types[model.id] = type(config).__name__

    module: ModuleType | None = select_adapter_module(config)

    if module is not None and getattr(module, "_is_encoder_only", False):
        return False, "encoder-only architecture does not support generation"
    return True, ""


def keep(
    model: ModelInfo,
    token: str | bool,
    *,
    config_types: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Keep predicate for the generative fetcher.

    Ordering matters: the cheap metadata-only checks run first.
    Returns a (keep, reason) tuple where reason describes why the model was
    rejected (empty string when kept).
    When provided, ``config_types`` collects loaded config class names for
    catalog enrichment after filtering has finished.
    """
    try:
        keep_flag, reason = is_baseline_keep(model)
        if not keep_flag:
            return False, reason
    except Exception:
        return False, "exception during is_baseline_keep"

    checks: list[tuple[Callable[[], bool], str, str]] = [
        (lambda: bool(model.gated), "model.gated", "model is gated"),
        (
            lambda: not has_loadable_weights(model),
            "has_loadable_weights",
            "no loadable weights",
        ),
    ]
    for predicate, name, rejection in checks:
        result, err = _guarded(predicate, name)
        if err:
            return False, err
        if result:
            return False, rejection

    # The config lookup also checks remote code, so rejecting encoder-only
    # adapters does not add a second Hub request per candidate.
    try:
        return _keep_generation_config(model, token, config_types)
    except HubRetryExhaustedError:
        raise
    except Exception:
        return False, "exception during generation config"


def _fetch_generative_models(
    fetcher: Callable[[HfApi, int], list[ModelInfo]],
    keeper: Callable[[ModelInfo, str | bool], tuple[bool, str]],
    limit: int,
    output_csv: Path | str | None = None,
) -> list[dict]:
    token: str | bool = os.environ.get("HF_TOKEN") or False
    api: HfApi = HfApi(token=token)
    # Per-fetch metadata only: do not retain live configs or cache Hub results
    # across scans. Curated requests use keep_all and fall back to enrichment.
    config_types: dict[str, str] = {}
    if keeper is keep:
        keeper = partial(keep, config_types=config_types)
    return build_catalog(
        fetch_fn=lambda lim: fetcher(api, lim),
        filter_fn=lambda m: keeper(m, token),
        limit=limit,
        output_csv=output_csv,
        label="generative",
        token=token,
        known_config_types=config_types,
    )


def fetch_top_generative_models(
    limit: int,
    output_csv: Path | str | None = None,
    ranking: Ranking = "downloads",
) -> list[dict[str, object]]:
    """Top *limit* generative models by *ranking*, counted after ``keep``."""
    return _fetch_generative_models(
        fetcher=_create_fetch(ranking),
        keeper=keep,
        limit=limit,
        output_csv=output_csv,
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
