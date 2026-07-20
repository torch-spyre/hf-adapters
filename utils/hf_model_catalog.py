"""Shared building blocks for the Hugging Face model-catalog fetchers.

Both ``fetch_top_generative_models.py`` and ``fetch_top_embedding_models.py``
pull models from the Hub, enrich them with config/param metadata, and write a
ranked CSV. Everything they have in common lives here; each script only has to
supply how it *sources* candidates, how it *filters* them, and any *extra
columns* it wants on top of the shared schema.
"""

import csv
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import ModelInfo
from tqdm import tqdm
from transformers import AutoConfig

# Import the mapping to get supported config classes dynamically
from hf_adapters.auto_spyre_model import CONFIG_TO_ADAPTER_MODULE_MAPPING

logging.getLogger("transformers").setLevel(logging.ERROR)


# Get the resources directory (parent of resources/__init__.py)
RESOURCES_DIR: Path = Path(__file__).resolve().parent.parent / "resources"

# Metadata fields requested from list_models for every fetcher.
EXPAND_FIELDS: list[str] = [
    "config",
    "safetensors",
    "gated",
    "likes",
    "downloads",
    "createdAt",
    "library_name",
    "tags",
]

# HF-API gateway 5xx statuses. Anything outside this set (400/401/403/404/...)
# is a permanent failure and must not be retried.
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({500, 502, 503, 504})
_MAX_FETCH_ATTEMPTS: int = 5
_MAX_BACKOFF_SECONDS: float = 60.0

_T = TypeVar("_T")


def with_transient_retry(
    call: Callable[[], Iterator[_T] | Iterable[_T]],
    description: str,
) -> list[_T]:
    """Materialize an HF paginated call, retrying transient 5xx failures.

    *call* is expected to return an iterable of ``ModelInfo`` (or similar)
    from a ``huggingface_hub`` API method. It is fully consumed into a list
    on each attempt because the pagination endpoint's failure mode is a
    mid-stream 504, and there is no way to resume — the whole traversal has
    to restart. Transient statuses (500/502/503/504) trigger up to
    ``_MAX_FETCH_ATTEMPTS`` retries with exponential backoff capped at
    ``_MAX_BACKOFF_SECONDS``; any other error propagates immediately.
    """
    last_error: HfHubHTTPError | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            return list(call())
        except HfHubHTTPError as e:
            status: int | None = (
                e.response.status_code if e.response is not None else None
            )
            if status not in _TRANSIENT_HTTP_STATUSES:
                raise
            last_error = e
            backoff: float = min(_MAX_BACKOFF_SECONDS, 2.0**attempt)
            print(
                f"    {description}: HF API returned {status} "
                f"(attempt {attempt}/{_MAX_FETCH_ATTEMPTS}); "
                f"retrying in {backoff:.0f}s..."
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


MOE_MODEL_TYPES: set[str] = {
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "gpt_oss",
}

MOE_MODEL_TYPE_PREFIXES: tuple[str, ...] = ("deepseek_v2", "deepseek_v3", "deepseek_v4")

MOE_ARCH_SUBSTRINGS: list[str] = [
    "mixtral",
    "moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "deepseek",
    "gptoss",
]

# Get supported config class names dynamically from the mapping
SUPPORTED_CONFIG_CLASSES: set[str] = {
    config_class.__name__ for config_class in CONFIG_TO_ADAPTER_MODULE_MAPPING.keys()
}


def tags(model: ModelInfo) -> set[str]:
    """Lower-cased set of a model's tags (empty set if none)."""
    return {t.lower() for t in (getattr(model, "tags", None) or [])}


def is_supported_config(config_class_name: str | None) -> bool:
    """Check if the config class is supported by our adapter code."""
    if config_class_name is None:
        return False
    return config_class_name in SUPPORTED_CONFIG_CLASSES


def is_moe(model: ModelInfo) -> bool:
    if any("moe" in t for t in tags(model)):
        return True

    config: dict = model.config or {}
    model_type: str = (config.get("model_type") or "").lower()
    if model_type in MOE_MODEL_TYPES:
        return True
    if model_type.startswith(MOE_MODEL_TYPE_PREFIXES):
        return True

    architectures: list[str] = config.get("architectures") or []
    arch_lower: str = " ".join(architectures).lower()
    return any(sub in arch_lower for sub in MOE_ARCH_SUBSTRINGS)


def is_custom_code(model: ModelInfo) -> bool:
    if "custom_code" in tags(model):
        return True
    config: dict = model.config or {}
    return bool(config.get("auto_map"))


def is_nsfw(model: ModelInfo) -> bool:
    if "nsfw" in tags(model):
        return True
    return False


# Repo-id substrings marking non-native conversions (ONNX/GGUF/MLX), dropped.
NON_NATIVE_ID_SUBSTRINGS: tuple[str, ...] = ("onnx", "gguf", "mlx")


def is_baseline_keep(model: ModelInfo) -> bool:
    """Shared inclusion gate: drop config-less, and ONNX/GGUF/MLX id checkpoints."""
    if not model.config:
        return False
    if model.library_name in NON_NATIVE_ID_SUBSTRINGS:
        return False
    model_id_lower: str = model.id.lower()
    if any(sub in model_id_lower for sub in NON_NATIVE_ID_SUBSTRINGS):
        return False
    if "nsfw" in tags(model):
        return False
    return True


def contains_remote_code(model: ModelInfo) -> bool:
    """Return False if the model requires trust_remote_code=True to load its config."""
    try:
        AutoConfig.from_pretrained(model.id, trust_remote_code=False)
        return False
    except (ValueError, OSError):
        return True


# Session-scoped cache for _has_loadable_weights. Keyed by repo_id; values are
# the bool result. The fetchers run twice a week in a fresh process, and repo
# file lists rarely change within a single run, so a plain dict is enough — no
# TTL or on-disk persistence needed.
_LOADABLE_WEIGHTS_CACHE: dict[str, bool] = {}

# Filenames transformers' AutoModel.from_pretrained recognizes as native
# weights (single-file or sharded via the matching index.json).
_NATIVE_WEIGHT_FILES: frozenset[str] = frozenset(
    {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
)


def has_loadable_weights(model: ModelInfo, token: str | bool) -> bool:
    """True if the repo ships weights AutoModel.from_pretrained can consume.

    Detects three unloadable classes without downloading any weight files
    — one ``list_repo_files`` call per repo:

    * adapter-only repos (LoRA/PEFT, `adapter_config.json` but no full model),
    * GGUF/MLX/ONNX-only repos that slipped past the id-substring filter,
    * abandoned uploads with a config but no weight files at all.

    Cached in-process by repo_id: transformers repos are effectively immutable
    within a fetcher run, and each fetcher process is short-lived.
    """
    from huggingface_hub import HfApi

    cached = _LOADABLE_WEIGHTS_CACHE.get(model.id)
    if cached is not None:
        return cached

    api: HfApi = HfApi(token=token)
    try:
        files: list[str] = with_transient_retry(
            lambda: api.list_repo_files(model.id, token=token),
            description=f"list_repo_files[{model.id}]",
        )
    except Exception:
        # Any permanent failure (404, gated without token, ...) — treat as
        # not loadable rather than raising into the fetcher's filter path.
        _LOADABLE_WEIGHTS_CACHE[model.id] = False
        return False

    lower_files: set[str] = {f.lower() for f in files}

    # Adapter-only repos ship adapter_config.json + adapter_model.safetensors
    # and expect PeftModel.from_pretrained(base, ...), not AutoModel directly.
    if "adapter_config.json" in lower_files:
        _LOADABLE_WEIGHTS_CACHE[model.id] = False
        return False

    result: bool = any(name in lower_files for name in _NATIVE_WEIGHT_FILES)
    _LOADABLE_WEIGHTS_CACHE[model.id] = result
    return result


def format_number_to_billions_smart(num: int | float) -> str:
    """Smart formatting that adjusts precision based on magnitude."""
    billions: float = num / 1_000_000_000

    if billions >= 10:
        # For numbers >= 10B, round to nearest integer
        result: int | float = round(billions)
        return f"{result}B"
    elif billions >= 1:
        # For 1B-10B, show 1 decimal place
        result = round(billions, 1)
        return f"{result}B" if result != int(result) else f"{int(result)}B"
    else:
        # For < 1B, show 1-2 decimal places
        result = round(billions, 2)
        return f"{result}B"


def parse_number_suffix(value: str) -> int:
    value = value.strip().upper()

    multipliers: dict[str, int] = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }

    suffix: str = value[-1]

    if suffix in multipliers:
        number: float = float(value[:-1])
        return int(number * multipliers[suffix])

    # No suffix → return as integer
    return int(float(value))


def extract_model_size_from_model_name(
    model_name: str, allow_millions: bool = False
) -> str | None:
    """Pull a parameter-size token (e.g. "7B", "33M") out of a model id.

    ``allow_millions`` also matches an ``M`` suffix — useful for embedding
    models, which are frequently sized in the tens/hundreds of millions.
    Returns the token only if exactly one match is found (avoids ambiguity).
    """
    units: str = "MBmb" if allow_millions else "Bb"
    pattern: str = rf"\b\d+(?:\.\d+)?[{units}]\b"
    matches: list[str] = re.findall(pattern, model_name)
    return matches[0] if len(matches) == 1 else None


def get_param_count(model: ModelInfo) -> int | None:
    if model.safetensors and model.safetensors.parameters:
        return sum(model.safetensors.parameters.values())
    return None


def get_config_type(model_id: str, token: str | bool) -> str | None:
    try:
        model_config = AutoConfig.from_pretrained(
            model_id, token=token, trust_remote_code=False
        )
        return type(model_config).__name__
    except Exception:
        return None


# Columns written for every catalog, in order. Each entry is
# (header, value_fn) where value_fn(model, config_class) -> cell value. The
# rank/config-class/param columns are interleaved by build_catalog because they
# depend on per-row computed state.
def _resolve_param_columns(
    model: ModelInfo, allow_millions: bool
) -> tuple[str | None, int | None]:
    """Return (param_str, param_int) for a model, name first then safetensors."""
    param_str: str | None = extract_model_size_from_model_name(model.id, allow_millions)
    param_int: int | None
    if param_str is None:
        param_int = get_param_count(model)
        if param_int is not None:
            param_str = format_number_to_billions_smart(param_int)
    else:
        param_int = parse_number_suffix(param_str)
    return param_str, param_int


def build_catalog(
    *,
    fetch_fn: Callable[[int], Iterable[ModelInfo]],
    filter_fn: Callable[[ModelInfo], bool],
    limit: int,
    output_csv: Path | str | None,
    label: str,
    extra_columns: (
        list[tuple[str, Callable[[ModelInfo, str | None], object]]] | None
    ) = None,
    allow_millions: bool = False,
    token: str | bool,
) -> list[dict[str, object]]:
    """Fetch → filter → enrich → return a ranked model catalog as a list of dicts.

    Args:
        fetch_fn: callable(limit) -> list of raw model objects (over-fetched).
        filter_fn: callable(model) -> bool, keep the model if True.
        limit: number of rows to write after filtering.
        output_csv: destination path, or None to skip writing.
        label: human-readable noun for log lines (e.g. "generative").
        extra_columns: optional list of (header, value_fn) where
            value_fn(model, config_class) -> cell. Inserted before config_class.
        allow_millions: pass-through to the size-name extractor.
        token: HF token (or True) for AutoConfig downloads.

    Returns:
        List of dicts, one per model, keyed by column name.
    """
    extra_columns = extra_columns or []

    def _safe_filter(model: ModelInfo) -> bool:
        try:
            return filter_fn(model)
        except Exception as e:
            logging.warning("filter_fn failed for %s: %s", model.id, e)
            return False

    candidates: list[ModelInfo] = list(fetch_fn(limit))
    print(f"Retrieved {len(candidates)} raw {label} candidates.")

    with ThreadPoolExecutor(max_workers=16) as ex:
        keep_flags: list[bool] = list(
            tqdm(
                ex.map(_safe_filter, candidates),
                total=len(candidates),
                desc="Filtering candidates",
            )
        )
    models: list[ModelInfo] = [m for m, keep in zip(candidates, keep_flags) if keep]
    print(f"Kept {len(models)} {label} models after filtering.")

    models = models[:limit]

    base_head: list[str] = [
        "rank",
        "model_id",
        "downloads",
        "likes",
        "model_type",
        "architectures",
        "parameters (str)",
        "parameters",
        "library",
        # "is_gated",
        # "is_moe",
    ]
    extra_head: list[str] = [h for h, _ in extra_columns]
    tail_head: list[str] = ["is_custom_code", "config_class", "is_supported", "Year"]
    header: list[str] = base_head + extra_head + tail_head

    with ThreadPoolExecutor(max_workers=16) as ex:
        config_classes: list[str | None] = list(
            tqdm(
                ex.map(lambda m: get_config_type(m.id, token), models),
                total=len(models),
                desc="Fetching config classes",
            )
        )

    rows: list[dict[str, object]] = []
    for rank, (m, config_class) in enumerate(zip(models, config_classes), start=1):
        architectures: list[str] | None = (m.config or {}).get("architectures")
        arch_str: str | None = ";".join(architectures) if architectures else None
        param_str, param_int = _resolve_param_columns(m, allow_millions)
        extra_vals: list[object] = [fn(m, config_class) for _, fn in extra_columns]
        rows.append(
            dict(
                zip(
                    header,
                    [
                        rank,
                        m.id,
                        m.downloads,
                        m.likes,
                        (m.config or {}).get("model_type"),
                        arch_str,
                        param_str,
                        param_int,
                        m.library_name,
                        # bool(m.gated),
                        # is_moe(m),
                        *extra_vals,
                        is_custom_code(m),
                        config_class,
                        is_supported_config(config_class),
                        m.created_at.year if m.created_at else None,
                    ],
                )
            )
        )

    if output_csv is not None:
        print(f"Writing top {len(rows)} to {output_csv}")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    # Attach the source ModelInfo to each row AFTER the CSV write. It is a
    # runtime-only field (not serializable, and never part of the schema),
    # useful for callers that need metadata the row dict does not expose —
    # e.g. safetensors.parameters, gated, sha, siblings.
    for row, m in zip(rows, models):
        row["model_info"] = m

    return rows
