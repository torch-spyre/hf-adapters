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
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    return True


def contains_remote_code(model: ModelInfo) -> bool:
    """Return False if the model requires trust_remote_code=True to load its config."""
    try:
        AutoConfig.from_pretrained(model.id, trust_remote_code=False)
        return False
    except (ValueError, OSError):
        return True


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

    candidates: list[ModelInfo] = list(fetch_fn(limit))
    print(f"Retrieved {len(candidates)} raw {label} candidates.")

    with ThreadPoolExecutor(max_workers=16) as ex:
        keep_flags: list[bool] = list(
            tqdm(
                ex.map(filter_fn, candidates),
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

    return rows
