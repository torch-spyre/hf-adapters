"""Fetch a model type's top-K catalog from the HuggingFace Hub.

One indirection over the two ``utils.fetch_top_*_models`` functions, so callers
select a catalog with a ``ModelType`` instead of branching on a string. Both
producers (``generate_weekly_shards`` and ``weekly_test --fetch``) reach the Hub
through here, which is also the single place the rows get normalized for the
JSON round-trip they are about to take.
"""

from collections.abc import Callable

from tests.spyre.weekly_generation.model_type import ModelType
from utils.fetch_top_embedding_models import fetch_top_embedding_models
from utils.fetch_top_generative_models import fetch_top_generative_models
from utils.utilities import ts

all_fetchers: dict[ModelType, Callable[..., list[dict]]] = {
    ModelType.GENERATIVE: fetch_top_generative_models,
    ModelType.EMBEDDING: fetch_top_embedding_models,
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
    for model in models:
        model.pop("model_info", None)

    print(f"{ts()} Fetched {len(models)} {model_type} model(s).")

    return models
