"""Fetch the top embedding models from Hugging Face, ranked by downloads.

Embedding (sentence/text-embedding) models on the Hub almost always carry
exactly one ``pipeline_tag`` — and for the most popular models (BGE, E5,
Qwen3-Embedding, mxbai, jina) that primary tag is ``feature-extraction``, NOT
``sentence-similarity``. To get full coverage we therefore query BOTH pipeline
tags and merge.

``feature-extraction`` is noisy: it also surfaces audio encoders (wav2vec2,
hubert, mimi, encodec, ...), vision encoders (clip, vit, ...) and rerankers.
We separate genuine text embedders from this noise by requiring a
"sentence-transformers signal" — the model is published with
``library_name == "sentence-transformers"`` or carries the
``sentence-similarity`` / ``sentence-transformers`` tag. This single filter
removes essentially all pure audio/vision/reranker entries while keeping
transformers-library embedders such as ``jinaai/jina-embeddings-v3``.

Rerankers (cross-encoders) are excluded. Multimodal embedders (jina-clip,
jina-embeddings-v5-omni, ...) are kept but flagged via ``is_multimodal``.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from utils.hf_model_catalog import (
    EXPAND_FIELDS,
    RESOURCES_DIR,
    build_catalog,
    contains_remote_code,
    has_loadable_weights,
    is_baseline_keep,
    tags,
    with_transient_retry,
)
from utils.utilities import ts

# Pipeline tags that embedding models are filed under. They are mutually
# exclusive (one primary tag per model), so we query both and union.
EMBEDDING_PIPELINE_TAGS: tuple[str, ...] = ("feature-extraction", "sentence-similarity")

# Substrings (in model_type, architectures or tags) that mark a model as
# multimodal — i.e. it consumes images/audio/video in addition to (or instead
# of) text. These are kept but flagged, not dropped.
MULTIMODAL_SUBSTRINGS: list[str] = [
    "clip",
    "vision",
    "_vl",
    "vl_",
    "vit",
    "blip",
    "audio",
    "wav2vec",
    "hubert",
    "wavlm",
    "mimi",
    "encodec",
    "whisper",
    "speecht5",
    "sew",
    "unispeech",
    "data2vec-audio",
    "videoprism",
    "video",
    "owlvit",
    "groupvit",
    "_omni",
    "omni",
]

# Markers that identify rerankers / cross-encoders (excluded entirely — they
# are not bi-encoder embedders).
RERANKER_SUBSTRINGS: list[str] = ["rerank", "cross-encoder", "cross_encoder"]


def _has_embedding_signal(model: ModelInfo) -> bool:
    """True if the model looks like a sentence-transformers / embedding model.

    This is the core inclusion gate: it separates genuine text embedders from
    the audio/vision/reranker noise that shares the feature-extraction tag.
    """
    if model.library_name == "sentence-transformers":
        return True
    t: set[str] = tags(model)
    return "sentence-similarity" in t or "sentence-transformers" in t


def _is_reranker(model: ModelInfo) -> bool:
    """True if the model is a reranker / cross-encoder (excluded)."""
    if any(any(sub in t for sub in RERANKER_SUBSTRINGS) for t in tags(model)):
        return True
    return any(sub in model.id.lower() for sub in RERANKER_SUBSTRINGS)


def _is_multimodal(model: ModelInfo, _config_class: str | None = None) -> bool:
    """True if the embedder also handles images/audio/video (flagged, kept)."""
    config: dict = model.config or {}
    model_type: str = (config.get("model_type") or "").lower()
    if any(sub in model_type for sub in MULTIMODAL_SUBSTRINGS):
        return True

    architectures: list[str] = config.get("architectures") or []
    arch_lower: str = " ".join(architectures).lower()
    if any(sub in arch_lower for sub in MULTIMODAL_SUBSTRINGS):
        return True

    multimodal_tags: set[str] = {
        "image-feature-extraction",
        "multimodal",
        "vision",
        "audio",
    }
    return bool(tags(model) & multimodal_tags)


def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
    """Query both embedding pipeline tags and return a deduplicated list,
    sorted by downloads descending. Over-fetched (x2) to absorb the noise +
    rerankers + GGUF/MLX entries removed by the filter.

    Each per-tag call is wrapped in ``with_transient_retry`` so a mid-fetch
    504 from the HF gateway does not abort the run.
    """
    print(f"{ts()} Fetching top {limit} text-embedding models by downloads...")
    per_tag_limit: int = int(limit * 2)
    by_id: dict[str, ModelInfo] = {}
    for tag in EMBEDDING_PIPELINE_TAGS:
        print(f"Fetching up to {per_tag_limit} '{tag}' models by downloads...")
        models: list[ModelInfo] = with_transient_retry(
            lambda t=tag: api.list_models(
                pipeline_tag=t,
                sort="downloads",
                limit=per_tag_limit,
                expand=EXPAND_FIELDS,
            ),
            description=f"list_models[{tag}]",
        )
        for m in models:
            # First tag wins on dupes; they carry identical metadata anyway.
            by_id.setdefault(m.id, m)

    return sorted(by_id.values(), key=lambda m: (m.downloads or 0), reverse=True)


def keep(model: ModelInfo, token: str | bool) -> bool:
    """Keep predicate for the embedding fetcher.

    Ordering matters: the cheap metadata-only checks run first so we only
    spend the ``has_loadable_weights`` HTTP call on the ~1k candidates that
    would otherwise survive.
    """
    if not is_baseline_keep(model):
        return False
    if not _has_embedding_signal(model):
        return False
    if _is_reranker(model):
        return False
    if model.gated:
        return False
    if contains_remote_code(model):
        return False
    if not has_loadable_weights(model, token):
        return False
    return True


def fetch_top_embedding_models(
    limit: int, output_csv: Path | str | None = None
) -> list[dict[str, object]]:
    # `or True` (not `.get(..., True)`): GHA sets HF_TOKEN to an empty string
    # rather than omitting it when the secret doesn't exist, so `.get` alone
    # would return "" and send a malformed auth header on every request,
    # causing has_loadable_weights() to 401 (and thus filter out) every model.
    token: str | bool = os.environ.get("HF_TOKEN") or True
    api: HfApi = HfApi(token=token)
    return build_catalog(
        fetch_fn=lambda lim: _fetch(api, lim),
        filter_fn=lambda m: keep(m, token),
        limit=limit,
        output_csv=output_csv,
        label="embedding",
        extra_columns=[("is_multimodal", _is_multimodal)],
        allow_millions=True,
        token=token,
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_embedding_models(
        limit=limit_, output_csv=RESOURCES_DIR / "top_embedding_models.csv"
    )
