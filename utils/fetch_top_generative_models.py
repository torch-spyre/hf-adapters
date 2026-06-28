"""Fetch the top generative models from Hugging Face, ranked by downloads."""

import os
import sys
from pathlib import Path

from hf_model_catalog import (
    EXPAND_FIELDS,
    RESOURCES_DIR,
    build_catalog,
    is_baseline_keep,
    is_moe,
)
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo


def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
    """Top text-generation models by downloads (over-fetched to absorb the
    GGUF/MLX entries dropped by the filter)."""
    print(f"Fetching top {limit} text-generation models by downloads...")
    return list(
        api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=int(limit * 2),
            expand=EXPAND_FIELDS,
        )
    )


def _keep(model: ModelInfo) -> bool:
    if not is_baseline_keep(model):
        return False
    if is_moe(model):
        return False
    if model.gated:
        return False
    return True


def fetch_top_generative_models(
    limit: int, output_csv: Path | str | None = None
) -> None:
    if output_csv is None:
        output_csv = RESOURCES_DIR / "top_generative_models.csv"
    token: str | bool = os.environ.get("HF_TOKEN", True)
    api: HfApi = HfApi(token=token)
    build_catalog(
        fetch_fn=lambda lim: _fetch(api, lim),
        filter_fn=_keep,
        limit=limit,
        output_csv=output_csv,
        label="generative",
        token=token,
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_generative_models(limit=limit_)
