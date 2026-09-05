"""Add an ``is_moe`` column to a model catalog CSV."""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelInfo
from tqdm import tqdm

# from utils.hf_model_catalog import EXPAND_FIELDS, is_moe, with_transient_retry

WORKERS: int = 4

# HF-API gateway 5xx statuses. Anything outside this set (400/401/403/404/...)
# is a permanent failure and must not be retried.
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({500, 502, 503, 504})
_MAX_FETCH_ATTEMPTS: int = 5
_MAX_BACKOFF_SECONDS: float = 60.0


def with_transient_retry(
    call,
    description: str,
):
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


# Metadata fields requested from list_models/model_info for every fetcher.
# Typed as the hub's own ExpandModelProperty_T literal rather than list[str], so a
# typo here is a type error instead of a runtime 400 from the API.
EXPAND_FIELDS: list[ExpandModelProperty_T] = [
    "config",
    "safetensors",
    "gated",
    "likes",
    "downloads",
    "createdAt",
    "library_name",
    "tags",
    "siblings",
]


def _get_is_moe(api: HfApi, model_name: str) -> str:
    """Fetch one model's metadata and return its MoE status."""
    try:
        model: ModelInfo = with_transient_retry(
            lambda: [api.model_info(model_name, expand=EXPAND_FIELDS)],
            description=f"model_info[{model_name}]",
        )[0]
        return str(is_moe(model))
    except Exception as error:
        print(f"WARNING: could not determine is_moe for '{model_name}': {error}")
        return ""


def tags(model: ModelInfo) -> set[str]:
    """Lower-cased set of a model's tags (empty set if none)."""
    return {t.lower() for t in (getattr(model, "tags", None) or [])}


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


def main(csv_file: Path) -> None:
    """Populate ``is_moe`` for each model in *csv_file* using Hub metadata."""
    token: str | bool = os.environ.get("HF_TOKEN") or False
    api: HfApi = HfApi(token=token)

    print(f"Reading models from {csv_file}...")
    with csv_file.open(newline="", encoding="utf-8-sig") as input_file:
        reader: csv.DictReader = csv.DictReader(input_file)
        fieldnames: list[str] = list(reader.fieldnames or [])
        print(f"Fields names are {fieldnames}")
        if "model_name" not in fieldnames:
            raise ValueError(f"{csv_file} does not contain a model_name column")
        rows: list[dict[str, str]] = list(reader)

    if "is_moe" not in fieldnames:
        fieldnames.append("is_moe")

    model_names: list[str] = [row["model_name"] for row in rows]
    authentication: str = "authenticated" if token else "unauthenticated"
    print(
        f"Checking {len(model_names):,} models with {WORKERS} workers "
        f"({authentication} Hugging Face API)..."
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        moe_values: list[str] = list(
            tqdm(
                executor.map(lambda name: _get_is_moe(api, name), model_names),
                total=len(model_names),
                desc="Checking MoE status",
            )
        )

    for row, moe_value in zip(rows, moe_values):
        row["is_moe"] = moe_value

    moe_count: int = sum(value == "True" for value in moe_values)
    failed_count: int = sum(not value for value in moe_values)
    print(
        f"Finished checking models: {moe_count:,} MoE, "
        f"{len(rows) - moe_count - failed_count:,} non-MoE, "
        f"{failed_count:,} failed."
    )

    output_csv: Path = csv_file.with_name(f"{csv_file.stem}_with_moe.csv")
    print(f"Writing results to {output_csv}...")
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer: csv.DictWriter = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Wrote {len(rows):,} rows to {output_csv}.")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="Input CSV catalog")
    args: argparse.Namespace = parser.parse_args()
    main(args.csv_file)
