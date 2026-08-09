"""Fetch a model type's top-K catalog from the HuggingFace Hub.

One indirection over the two ``utils.fetch_top_*_models`` functions, so callers
select a catalog with a ``ModelType`` instead of branching on a string. Both
producers (``generate_weekly_shards`` and ``weekly_test --fetch``) reach the Hub
through here, which is also the single place the rows get normalized for the
JSON round-trip they are about to take.
"""

from collections.abc import Callable

from tests.spyre.weekly_generation.model_type import ModelType
from utils.fetch_top_embedding_models import (
    fetch_curated_embedding_models_metadata,
    fetch_top_embedding_models,
)
from utils.fetch_top_generative_models import (
    fetch_curated_generative_models_metadata,
    fetch_top_generative_models,
)
from utils.hf_model_catalog import (
    load_curated_embedding_models,
    load_curated_generative_models,
)
from utils.utilities import ts

all_fetchers: dict[ModelType, Callable[..., list[dict]]] = {
    ModelType.GENERATIVE: fetch_top_generative_models,
    ModelType.EMBEDDING: fetch_top_embedding_models,
}

all_metadata_fetchers: dict[ModelType, Callable[..., list[dict]]] = {
    ModelType.GENERATIVE: fetch_curated_generative_models_metadata,
    ModelType.EMBEDDING: fetch_curated_embedding_models_metadata,
}

all_curated_loaders: dict[ModelType, Callable[..., list[str]]] = {
    ModelType.GENERATIVE: load_curated_generative_models,
    ModelType.EMBEDDING: load_curated_embedding_models,
}
"""Catalog fetcher per model type. Module-level and mutable so tests can
monkeypatch it and exercise the pipeline without hitting the network."""


def fetch(model_type: ModelType, top_k: int) -> list[dict]:
    """Return the top *top_k* models of *model_type*, ordered by downloads.

    Descending download order is a contract, not an incidental: the tier router
    and shard chunker downstream both assume it, and the pre-filter preserves it.

    Each row is a plain dict keyed as the catalog CSV header is (``model_id``,
    ``downloads``, ``parameters``, ``is_supported``, ``is_moe``,
    ``config_class``, ``model_type``, ``architectures``) — JSON-serializable, so
    it survives being written to a shard file and read back by another process.
    """
    models: list[dict] = all_fetchers[model_type](limit=top_k)

    # model_info is a live huggingface_hub.ModelInfo object attached by
    # build_catalog — not JSON-serializable, and no longer needed since
    # is_moe is precomputed onto each model (see utils/hf_model_catalog.py).
    # Dropped here rather than at each call site because every consumer either
    # JSON-dumps these rows into a shard file or hands them to a spawned child
    # (which pickles them); both break on a ModelInfo.
    # we also indicate that the model is not a curated model
    for model in models:
        model.pop("model_info", None)
        model["curated"] = False

    print(f"{ts()} Fetched {len(models)} {model_type} model(s).")

    return models


def load_curated(model_type: ModelType) -> list[dict]:
    """Return catalog rows for the curated *model_type* ids, same shape as ``fetch``.

    The curated ids are known up front, so this skips the ``list_models`` ranking
    query the two fetchers use and asks the Hub for each repo directly via
    ``model_info``. Everything after that — config class, param count,
    architectures, is_moe, is_multimodal — is the shared ``build_catalog`` path, so
    a curated row and a fetched row are interchangeable to the pre-filter
    downstream.

    Deliberately *not* filtered: a curated id is one someone asked for by name, so the
    gates the ranked scan applies (embedding signal, gated, remote code, loadable
    weights) must not silently drop it — the curated callers pass an accept-everything
    keep predicate. The terminal gates in ``prefilter_models`` still apply downstream,
    and record why they skipped it.

    Ids the Hub does not return (typo, private, deleted) are dropped with a
    warning rather than raising: one bad line in a hand-maintained file should not
    abort the weekly scan.
    """
    model_ids: list[str] = all_curated_loaders[model_type]()
    print(
        f"{ts()} Found {len(model_ids)} curated {model_type} model(s) in pre-defined list."
    )
    if not model_ids:
        return []

    models: list[dict] = all_metadata_fetchers[model_type](model_ids=model_ids)

    # Same contract as fetch(): drop the non-serializable ModelInfo, and mark
    # these rows as curated so the sink can record where they came from.
    for model in models:
        model.pop("model_info", None)
        model["curated"] = True

    print(f"{ts()} Fetched metadata for {len(models)} curated {model_type} model(s).")
    return models
