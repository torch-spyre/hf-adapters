"""Fetch the top generative models from Hugging Face, ranked by downloads."""

import os
import sys
from pathlib import Path

from hf_model_catalog import EXPAND_FIELDS, RESOURCES_DIR, build_catalog
from huggingface_hub import HfApi


def _fetch(api, limit):
    """Top text-generation models by downloads (over-fetched to absorb the
    GGUF/MLX entries dropped by the filter)."""
    print(f"Fetching top {limit} text-generation models by downloads...")
    return list(
        api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=int(limit * 1.5),
            expand=EXPAND_FIELDS,
        )
    )


def _keep(model):
    return bool(model.config) and model.library_name not in ("gguf", "mlx")


def fetch_top_generative_models(limit, output_csv: Path | str | None = None):
    if output_csv is None:
        output_csv = RESOURCES_DIR / "top_generative_models.csv"
    token = os.environ.get("HF_TOKEN", True)
    api = HfApi(token=token)
    build_catalog(
        fetch_fn=lambda lim: _fetch(api, lim),
        filter_fn=_keep,
        limit=limit,
        output_csv=output_csv,
        label="generative",
        token=token,
    )


if __name__ == "__main__":
    limit_ = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_generative_models(limit=limit_)
