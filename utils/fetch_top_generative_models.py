"""Fetch the top 1000 generative models from Hugging Face, ranked by likes."""

import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoConfig

# Import the mapping to get supported config classes dynamically
from hf_adapters.auto_spyre_model import CONFIG_TO_ADAPTER_MODULE_MAPPING

MOE_MODEL_TYPES = {
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "gpt_oss",
}

MOE_MODEL_TYPE_PREFIXES = ("deepseek_v2", "deepseek_v3", "deepseek_v4")

MOE_ARCH_SUBSTRINGS = [
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
SUPPORTED_CONFIG_CLASSES = {
    config_class.__name__ for config_class in CONFIG_TO_ADAPTER_MODULE_MAPPING.keys()
}


def _is_supported_config(config_class_name):
    """Check if the config class is supported by our adapter code."""
    if config_class_name is None:
        return False
    return config_class_name in SUPPORTED_CONFIG_CLASSES


def _is_moe(model):
    has_moe_tags = any("moe" in t.lower() for t in (getattr(model, "tags") or []))
    if has_moe_tags:
        return True

    config = model.config
    model_type = (config.get("model_type") or "").lower()
    if model_type in MOE_MODEL_TYPES:
        return True
    if model_type.startswith(MOE_MODEL_TYPE_PREFIXES):
        return True

    architectures = config.get("architectures") or []
    arch_lower = " ".join(architectures).lower()
    return any(sub in arch_lower for sub in MOE_ARCH_SUBSTRINGS)


def _is_custom_code(model):
    tags = {t.lower() for t in (getattr(model, "tags") or [])}
    if "custom_code" in tags:
        return True
    config = model.config or {}
    return bool(config.get("auto_map"))


def _format_number_to_billions_smart(num: int | float) -> str:
    """Smart formatting that adjusts precision based on magnitude."""
    billions = num / 1_000_000_000

    if billions >= 10:
        # For numbers >= 10B, round to nearest integer
        result = round(billions)
        return f"{result}B"
    elif billions >= 1:
        # For 1B-10B, show 1 decimal place
        result = round(billions, 1)
        return f"{result}B" if result != int(result) else f"{int(result)}B"
    else:
        # For < 1B, show 1-2 decimal places
        result = round(billions, 2)
        return f"{result}B"


def _parse_number_suffix(value: str) -> int:
    value = value.strip().upper()

    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }

    suffix = value[-1]

    if suffix in multipliers:
        number = float(value[:-1])
        return int(number * multipliers[suffix])

    # No suffix → return as integer
    return int(float(value))


def _extract_model_size_from_model_name(model_name):
    pattern = r"\b\d+(?:\.\d+)?[Bb]\b"
    matches = re.findall(pattern, model_name)
    # Return size only if exactly one match found
    return matches[0] if len(matches) == 1 else None


def _get_param_count(model, token):
    if model.safetensors and model.safetensors.parameters:
        return sum(model.safetensors.parameters.values())
    # try:
    #     meta = get_safetensors_metadata(model.id, token=token)
    #     if meta and meta.parameter_count:
    #         return sum(meta.parameter_count.values())
    # except Exception:
    #     pass
    return None


def _get_config_type(model_id, token):
    try:
        model_config = AutoConfig.from_pretrained(
            model_id, token=token, trust_remote_code=True
        )
        return type(model_config).__name__
    except Exception:
        return None


def fetch_top_generative_models(limit, output_csv="top_generative_models.csv"):
    token = os.environ.get("HF_TOKEN", True)
    api = HfApi(token=token)

    print(f"Fetching top {limit} text-generation models by downloads...")
    models = list(
        api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=int(limit * 1.5),  # We take more since, we will remove some of them
            expand=[
                "config",
                "safetensors",
                "gated",
                "likes",
                "downloads",
                "createdAt",
                "library_name",
                "tags",
            ],
        )
    )

    models = [m for m in models if m.library_name not in ["gguf", "mlx"] and m.config]
    print(
        f"Retrieved {len(models)} models (after filtering GGUF-only/MLX). "
        f"Writing to {output_csv}"
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "model_id",
                "downloads",
                "likes",
                "model_type",
                "architectures",
                "parameters (str)",
                "parameters",
                "library",
                "is_gated",
                "is_moe",
                "is_custom_code",
                "config_class",
                "is_supported",
                "Year",
            ]
        )

        models = models[:limit]
        with ThreadPoolExecutor(max_workers=16) as ex:
            config_classes = list(
                tqdm(
                    ex.map(lambda m: _get_config_type(m.id, token), models),
                    total=len(models),
                    desc="Fetching config classes",
                )
            )

        for rank, (m, config_class) in enumerate(zip(models, config_classes), start=1):
            model_type = m.config.get("model_type") if m.config else None
            architectures = m.config.get("architectures") if m.config else None
            arch_str = ";".join(architectures) if architectures else None
            param_count_str = _extract_model_size_from_model_name(model_name=m.id)
            if param_count_str is None:
                param_count = _get_param_count(m, token)
                if param_count is not None:
                    param_count_str = _format_number_to_billions_smart(param_count)
            else:  # Param Count is not None
                param_count = _parse_number_suffix(param_count_str)
            is_supported = _is_supported_config(config_class)
            writer.writerow(
                [
                    rank,
                    m.id,
                    m.downloads,
                    m.likes,
                    model_type,
                    arch_str,
                    param_count_str,
                    param_count,
                    m.library_name,
                    bool(m.gated),
                    _is_moe(model=m),
                    _is_custom_code(m),
                    config_class,
                    is_supported,
                    m.created_at.year if m.created_at else None,
                ]
            )

    print("Done. Top 5 models:")
    for i, m in enumerate(models[:5], start=1):
        print(f"  {i}. {m.id} — {m.downloads:,} downloads")


if __name__ == "__main__":
    limit_ = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_generative_models(limit=limit_)
