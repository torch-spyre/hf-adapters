#!/usr/bin/env python3
"""
Fetch the top generative and embedding models from Hugging Face for use as
GitHub Actions matrix inputs.

Reuses the fetch/filter logic from ``utils/fetch_top_generative_models.py`` and
``utils/fetch_top_embedding_models.py`` so the candidate sets stay aligned with
the offline catalog scripts. MoE models are excluded (no adapter support yet).

For each kept model the script emits a small metadata object — not the full
catalog row — so downstream matrix jobs can branch on ``is_supported`` /
``is_gated`` without re-querying the Hub.

Usage:
    python fetch_top_models.py --limit N

Outputs (to ``$GITHUB_OUTPUT`` when set, otherwise stdout):
    generative_matrix=<json array of objects>
    embedding_matrix=<json array of objects>
"""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.hf_api import ModelInfo  # noqa: E402

from utils.fetch_top_embedding_models import (  # noqa: E402
    _fetch as _fetch_embedding,
)
from utils.fetch_top_embedding_models import (
    _is_multimodal,
)
from utils.fetch_top_embedding_models import (
    _keep as _keep_embedding,
)
from utils.fetch_top_generative_models import (  # noqa: E402
    _fetch as _fetch_generative,
)
from utils.fetch_top_generative_models import (
    _keep as _keep_generative,
)
from utils.hf_model_catalog import (  # noqa: E402
    extract_model_size_from_model_name,
    format_number_to_billions_smart,
    get_param_count,
    is_custom_code,
    is_moe,
    is_supported_config,
)

# Make ``utils/`` and the project root importable so we can reuse the
# existing fetchers and the shared catalog helpers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))


def _param_str(model: ModelInfo, allow_millions: bool) -> str | None:
    """Best-effort parameter-size token (e.g. "7B") without an AutoConfig fetch."""
    name_match: str | None = extract_model_size_from_model_name(
        model.id, allow_millions
    )
    if name_match is not None:
        return name_match
    param_count: int | None = get_param_count(model)
    if param_count is not None:
        return format_number_to_billions_smart(param_count)
    return None


def _model_entry(
    model: ModelInfo, *, allow_millions: bool, is_multimodal: bool | None = None
) -> dict:
    """Compact dict describing one model for the GHA matrix."""
    config: dict = model.config or {}
    architectures: list[str] | None = config.get("architectures")
    config_class: str | None = architectures[0] if architectures else None
    entry: dict = {
        "model_id": model.id,
        "downloads": model.downloads,
        "likes": model.likes,
        "model_type": config.get("model_type"),
        "parameters": _param_str(model, allow_millions=allow_millions),
        "library": model.library_name,
        "is_gated": bool(model.gated),
        "is_custom_code": is_custom_code(model),
        "config_class": config_class,
        "is_supported": is_supported_config(config_class),
    }
    if is_multimodal is not None:
        entry["is_multimodal"] = is_multimodal
    return entry


def _collect(
    models: list[ModelInfo],
    keep_fn,
    limit: int,
    *,
    allow_millions: bool,
    multimodal_fn=None,
) -> list[dict]:
    """Filter (with MoE exclusion), truncate to ``limit``, and shape for GHA."""
    kept: list[ModelInfo] = [m for m in models if keep_fn(m) and not is_moe(m)]
    kept = kept[:limit]
    return [
        _model_entry(
            m,
            allow_millions=allow_millions,
            is_multimodal=multimodal_fn(m) if multimodal_fn else None,
        )
        for m in kept
    ]


def fetch_matrices(limit: int, token: str | bool) -> dict[str, list[dict]]:
    api: HfApi = HfApi(token=token)

    generative_raw: list[ModelInfo] = list(_fetch_generative(api, limit))
    embedding_raw: list[ModelInfo] = list(_fetch_embedding(api, limit))

    generative: list[dict] = _collect(
        generative_raw, _keep_generative, limit, allow_millions=False
    )
    embedding: list[dict] = _collect(
        embedding_raw,
        _keep_embedding,
        limit,
        allow_millions=True,
        multimodal_fn=_is_multimodal,
    )

    return {"generative": generative, "embedding": embedding}


def write_github_output(outputs: dict[str, str]) -> None:
    github_output: str | None = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        print("Not running in GitHub Actions. Output would be:")
        for key, value in outputs.items():
            print(f"{key}={value}")
        return
    with open(github_output, "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch top generative and embedding models for GHA matrices"
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of models per matrix (after filtering)",
    )
    args = parser.parse_args()

    token: str | bool = os.environ.get("HF_TOKEN", True)
    matrices: dict[str, list[dict]] = fetch_matrices(args.limit, token=token)

    print(f"Generative models kept: {len(matrices['generative'])}")
    print(f"Embedding  models kept: {len(matrices['embedding'])}")

    write_github_output(
        {
            "generative_matrix": json.dumps(matrices["generative"]),
            "embedding_matrix": json.dumps(matrices["embedding"]),
        }
    )


if __name__ == "__main__":
    main()
