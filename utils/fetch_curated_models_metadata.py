"""Resolve an explicit list of model ids to ``ModelInfo``, for curated catalogs.

The counterpart to each fetcher's ``_fetch``: same ``(api, limit) -> list[ModelInfo]``
shape, so it drops into ``fetch_generative_models`` / ``fetch_embedding_models`` in
place of the ranking query and reuses the whole ``build_catalog`` enrichment path.
The ids are known up front, so there is nothing to rank or over-fetch — ``limit`` is
accepted only to satisfy that signature and is deliberately ignored.

Those wrappers take their keep predicate as a parameter, and the curated callers pass
an accept-everything one. That is deliberate: a curated id was requested by name, so
the gates the ranked scan applies (embedding signal, gated, remote code, loadable
weights) must not silently drop it. The terminal gates in ``prefilter_models``
(no adapter, too large, MoE) still apply downstream and record why they skipped it.
"""

from typing import Callable

from huggingface_hub import HfApi, ModelInfo

from utils.hf_model_catalog import EXPAND_FIELDS, with_transient_retry
from utils.utilities import ts


def keep_all(_model: ModelInfo, _token: str | bool) -> bool:
    """Keep predicate for curated catalogs: accept every id that resolved.

    Named rather than an inline ``lambda`` so the call sites read as a decision
    ("curated ids bypass the ranked scan's gates") instead of an unexplained
    constant. Matches the ``keep(model, token)`` signature the fetchers expect.
    """
    return True


def _create_fetch_metadata(
    model_ids: list[str],
) -> Callable[[HfApi, int], list[ModelInfo]]:
    """Build a ``_fetch``-shaped callable that resolves *model_ids* in order."""
    _model_ids = model_ids

    def _fetch_metadata(api: HfApi, limit: int) -> list[ModelInfo]:
        """Resolve each curated id to a ModelInfo, preserving file order.

        ``expand`` mirrors the fetchers' ``EXPAND_FIELDS`` so downstream helpers
        (is_moe, get_param_count, is_custom_code) see the same populated
        attributes they do on a ``list_models`` result.
        """
        infos: list[ModelInfo] = []
        for model_id in _model_ids:
            try:
                infos.append(
                    with_transient_retry(
                        lambda mid=model_id: [
                            api.model_info(mid, expand=EXPAND_FIELDS)
                        ],
                        description=f"model_info[{model_id}]",
                    )[0]
                )
            except Exception as e:
                print(f"{ts()} WARNING: curated model '{model_id}' skipped — {e}")
        return infos

    return _fetch_metadata
