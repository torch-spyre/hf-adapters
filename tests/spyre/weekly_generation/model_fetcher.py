"""Fetch a model type's weekly-scan catalog from the HuggingFace Hub.

One indirection over the two ``utils.fetch_top_*_models`` functions, so callers
select a catalog with a ``ModelType`` instead of branching on a string. Both
producers (``generate_weekly_shards`` and ``weekly_test --fetch``) reach the Hub
through here, which is also the single place the rows get normalized for the
JSON round-trip they are about to take.

A model type's catalog is the union of two kinds of source:

* **Ranked** (``fetch``): the top-K by a ``Ranking`` -- downloads over the last 30
  days, or the Hub's trending score -- counted *after* the fetcher's ``keep``
  predicate.
* **By name** (``load_by_name``): the hand-maintained curated file plus every
  checkpoint registered in ``tests/model_registry.py`` for that type. These bypass
  ``keep`` and are flagged ``curated=True``.
"""

from collections.abc import Callable

from tests.model_registry import (
    WEEKLY_SCAN_EMBEDDING_PATHS,
    WEEKLY_SCAN_GENERATIVE_PATHS,
)
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
    Ranking,
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

all_registry_paths: dict[ModelType, list[str]] = {
    ModelType.GENERATIVE: WEEKLY_SCAN_GENERATIVE_PATHS,
    ModelType.EMBEDDING: WEEKLY_SCAN_EMBEDDING_PATHS,
}
"""Catalog sources per model type. Module-level and mutable so tests can
monkeypatch them and exercise the pipeline without hitting the network."""


def fetch(
    model_type: ModelType, top_k: int, ranking: Ranking = "downloads"
) -> list[dict]:
    """Return the top *top_k* models of *model_type*, ordered by *ranking*.

    Order is preserved through the pre-filter; the tier router and shard chunker
    downstream keep each source's relative order.

    Each row is a plain dict containing ``model_id``, ``downloads``,
    ``parameters``, ``is_moe``, ``config_class``, ``model_type``, and
    ``architectures``. It is JSON-serializable, so it survives being written to
    a shard file and read back by another process.
    """
    if top_k <= 0:
        print(f"{ts()} Skipping {model_type} ranking by {ranking} (top_k={top_k}).")
        return []

    models: list[dict] = all_fetchers[model_type](limit=top_k, ranking=ranking)

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

    print(f"{ts()} Fetched {len(models)} {model_type} model(s) by {ranking}.")

    return models


def _curated_and_registry_ids(model_type: ModelType) -> list[str]:
    """Curated-file ids, then registry paths not already listed, in that order."""
    seen: set[str] = set()
    model_ids: list[str] = []
    for model_id in (
        *all_curated_loaders[model_type](),
        *all_registry_paths[model_type],
    ):
        if model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
    return model_ids


def load_by_name(model_type: ModelType) -> list[dict]:
    """Return catalog rows for the by-name *model_type* ids, same shape as ``fetch``.

    "By name" is the curated file plus every checkpoint registered for this type
    in ``tests/model_registry.py`` (``WEEKLY_SCAN_*_PATHS``). Both are treated the
    same way: someone asked for them explicitly, so they are flagged ``curated``.

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
    curated_count: int = len(all_curated_loaders[model_type]())
    model_ids: list[str] = _curated_and_registry_ids(model_type)
    print(
        f"{ts()} Found {len(model_ids)} by-name {model_type} model(s): "
        f"{curated_count} curated + {len(model_ids) - curated_count} more "
        f"from the model registry."
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
